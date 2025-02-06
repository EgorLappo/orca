use clap::Parser;
use color_eyre::eyre::{bail, Result, WrapErr};
use csv;
use jemallocator::Jemalloc;
use log::debug;
use polars::prelude::*;
use std::path::Path;

use orca::cli::Opts;
use orca::marker_set::MarkerSet;
use orca::orca::{OrcaFull, OrcaSim};
use orca::writer::{OutRow, OutputWriter};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    assert!(opts.nmarkers >= opts.r);

    // read input csv
    let data = read_input_csv(&opts.input).wrap_err("failed to read input csv")?;
    let data = cut_data(data, &opts).wrap_err("failed to truncate the input dataset")?;

    // first we try to read output and load its contents...
    let resume = try_resume(&opts);

    // ... so now it doesn't matter if it is overwritten
    // writing is cheap here as there are rarely more than ~couple hundred lines,
    // and we don't have to bother with appending
    let mut writer =
        OutputWriter::new(&opts.output).wrap_err("failed to initialize output writer")?;

    let orca_full = OrcaFull;
    let orca_sim = OrcaSim::new(&opts);

    // try to resume computation from already existing file.
    let mut markers = if let Some(rows) = resume {
        let initial_set = rows.iter().map(|x| &x.id as &str).collect();
        let initial_orcas = rows.iter().map(|x| x.orca_marker).collect();

        let markers = MarkerSet::from_initial_set(data, &opts, initial_set, Some(initial_orcas))?;

        // rewrite the output file immediately (see comment above)
        writer
            .write_rows(rows.iter().cloned())
            .wrap_err("failed to re-write rows after resuming")?;

        markers
    }
    // if resuming is not available, check if we are provided with an initial set
    else if let Some(initial_set_str) = opts.initial_set.as_ref() {
        debug!("using provided initial marker set {}", initial_set_str);
        let initial_set = initial_set_str.split(',').collect::<Vec<_>>();
        let markers = MarkerSet::from_initial_set(data, &opts, initial_set, None)
            .wrap_err("failed to use provided initial marker set")?;

        // in this case, we have to evaluate each iterative set
        let full_evals = markers.evaluate_sets(orca_full, Some(opts.full_orca_eval_limit));
        let sim_evals = markers.evaluate_sets(orca_sim, None);

        // and record it immediately
        for i in 0..markers.cur.len() {
            writer
                .write_row(OutRow {
                    id: markers.ids[markers.cur[i] as usize].clone(),
                    orca_marker: markers.orcas[i],
                    orca_full: full_evals[i],
                    orca_sim: sim_evals[i],
                })
                .wrap_err("failed to write rows when starting from initial set")?;
        }

        markers
    }
    // finally, if we cannot resume or use a provided set,
    // run exhaustive computation from scratch
    else {
        let mut markers = MarkerSet::new(data, &opts).wrap_err("failed to initialize MarkerSet")?;
        debug!(
            "searching for {} best among {} markers in {} populations",
            opts.nmarkers, markers.n, markers.k
        );

        // first search exhaustive for first r
        debug!("searching for first {} markers exhaustively", opts.r);

        markers.search_exhaustive(OrcaFull);

        // here also evaluate all subsets
        // NOTE: these evaluations are purely for consistency
        // as they make no sense: the whole set was found with exhaustive calculation,
        // so it is unordered and evaluating addition of each marker is meaningless
        let eval_limit = opts.full_orca_eval_limit;
        let full_evals = markers.evaluate_sets(orca_full, Some(eval_limit));
        let sim_evals = markers.evaluate_sets(orca_sim, None);

        // and write all rows

        for i in 0..markers.cur.len() {
            writer
                .write_row(OutRow {
                    id: markers.ids[markers.cur[i] as usize].clone(),
                    orca_marker: markers.orcas[i],
                    orca_full: full_evals[i],
                    orca_sim: sim_evals[i],
                })
                .wrap_err("failed to write result after exhaustive search")?;
        }

        markers
    };

    // then add the rest of the markers greedily

    for _ in 0..(opts.nmarkers - markers.cur.len()) {
        let (id, orca_marker) = if markers.cur.len() < opts.full_orca_search_limit {
            debug!(
                "adding marker {}/{} using exhaustive f_ORCA computation",
                markers.cur.len() + 1,
                opts.nmarkers
            );
            markers.add_greedy(orca_full)
        } else {
            debug!(
                "adding marker {}/{} using simulation f_ORCA computation",
                markers.cur.len() + 1,
                opts.nmarkers
            );
            markers.add_greedy(orca_sim)
        };

        let orca_full = if markers.cur.len() <= opts.full_orca_eval_limit {
            Some(markers.evaluate_current(orca_full))
        } else {
            None
        };
        let orca_sim = Some(markers.evaluate_current(orca_sim));

        // write as soon as next marker is selected
        writer.write_row(OutRow {
            id,
            orca_marker,
            orca_full,
            orca_sim,
        })?;
    }

    Ok(())
}

// IO functions

fn read_input_csv(input: &Path) -> Result<DataFrame> {
    let reader = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(10))
        .try_into_reader_with_file_path(Some(input.to_path_buf()))?;

    // check that the right columns are present, and no others
    let df = reader.finish().wrap_err(format!(
        "couldn't read input csv file {} into dataframe",
        input.display()
    ))?;
    let columns: Vec<&str> = df.get_columns().iter().map(|x| x.name().as_str()).collect();
    let k = columns.len() - 2;

    // check that id column is present
    if !columns.iter().any(|&x| x == "id") {
        bail!("cound not find column 'id' in input file");
    }
    // check freq column
    if !columns.iter().any(|&x| x == "freq") {
        bail!("cound not find column 'freq' in input file");
    }

    // check that the rest of the columns are in the format `freq{i}`
    let freqi_re = regex::Regex::new(r"^freq\d+").unwrap();

    for x in columns {
        if (x != "id") & (x != "freq") & !freqi_re.is_match(x) {
            bail!("unknown column in input file: {}", x);
        }
    }

    // check that all freq{i} columns are present and in order
    let mut cols = vec!["id".to_string(), "freq".to_string()];
    for i in 0..k {
        cols.push(format!("freq{}", i + 1));
    }

    let df = df
        .select(cols)
        .wrap_err("freq{i} columns were not in order starting with i=1 in input file")?;

    Ok(df)
}

fn try_resume(opts: &Opts) -> Option<Vec<OutRow>> {
    // the logic is that we return an option where None means we do not resume,
    // and Some provides data to resume: current markers and current orcas

    // don't resume if not asked for
    if !opts.resume {
        None
    }
    // don't resume if file does not exist
    // note that we do this before initializing the writer in main
    else if !opts.output.exists() {
        debug!(
            "failed to resume computation: output file {} does not exist",
            opts.output.display()
        );
        None
    } else {
        // try to make file reader
        if let Ok(mut reader) = csv::Reader::from_path(opts.output.clone()) {
            let rows: Vec<_> = reader.deserialize().filter_map(|r| r.ok()).collect();

            if !rows.is_empty() {
                debug!("resuming computation with {} rows", rows.len());
                Some(rows)
            } else {
                debug!("failed to resume computation: empty table");
                None
            }
        } else {
            debug!("failed to resume computation: cannot create reader");
            None
        }
    }
}

fn cut_data(data: DataFrame, opts: &Opts) -> Result<DataFrame> {
    if let Some(cut) = opts.cut {
        // fail if cut < nmarkers
        if cut < opts.nmarkers {
            return Err(color_eyre::eyre::eyre!(
                "cut must be greater than or equal to nmarkers"
            ));
        }

        // sort data by freq and keep the top cut rows
        Ok(data
            .sort(
                ["freq"],
                SortMultipleOptions::new().with_order_descending(true),
            )?
            .head(Some(cut)))
    } else {
        Ok(data)
    }
}
