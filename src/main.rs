use clap::Parser;
use color_eyre::eyre::{bail, Result, WrapErr};
use jemallocator::Jemalloc;
use log::debug;
use polars::prelude::*;
use std::path::Path;

use orca::cli::Opts;
use orca::marker_set::MarkerSet;
use orca::orca::{OrcaFull, OrcaSim};
use orca::writer::OutputWriter;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    assert!(opts.nmarkers >= opts.r);

    // read input csv
    let data = read_input_csv(&opts.input).wrap_err("failed to read input csv")?;
    let data = cut_data(data, &opts).wrap_err("failed to truncate the input dataset")?;

    let mut writer = OutputWriter::new(&opts.output.with_extension("markers.csv"))
        .wrap_err("failed to initialize output writer")?;

    // create marker set
    let mut markers = if let Some(initial_set_str) = opts.initial_set.as_ref() {
        debug!("using provided initial marker set {}", initial_set_str);
        let initial_set = initial_set_str.split(',').collect::<Vec<_>>();
        let markers = MarkerSet::from_initial_set(data, &opts, initial_set)
            .wrap_err("failed to use provided initial marker set")?;

        // save the initial markers to file immediately
        markers
            .current_ids()
            .into_iter()
            .zip(markers.orcas.iter())
            .map(|(id, orca)| writer.write_row(id, *orca))
            .collect::<Result<Vec<()>>>()?;

        markers
    } else {
        let mut markers = MarkerSet::new(data, &opts);
        debug!(
            "searching for {} best among {} markers in {} populations",
            opts.nmarkers, markers.n, markers.k
        );

        // first search exhaustive for first r
        debug!("searching for first {} markers exhaustively", opts.r);

        // save the markers to file immediately
        markers
            .search_exhaustive(OrcaFull)
            .map(|(id, set_orca)| writer.write_row(id, set_orca))
            .collect::<Result<Vec<()>>>()?; // force write for all

        markers
    };

    // then add the rest of the markers greedily
    let orca_full = OrcaFull;
    let orca_sim = OrcaSim::new(&opts);

    for _ in 0..(opts.nmarkers - opts.r) {
        let (id, marker_orca) = if markers.cur.len() < opts.full_orca_limit {
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

        // write as soon as next marker is selected
        writer.write_row(id, marker_orca)?;
    }

    // done, now evaluate
    if !opts.no_eval {
        #[derive(serde::Serialize)]
        struct EvalRow {
            id: String,
            orca_full: f64,
            orca_sim: f64,
        }

        let full_evals = markers.evaluate(orca_full);
        let sim_evals = markers.evaluate(orca_sim);

        let mut writer = csv::Writer::from_path(opts.output.with_extension("eval.csv"))?;

        (0..markers.cur.len()).try_for_each(|i| -> Result<()> {
            writer.serialize(EvalRow {
                id: markers.ids[markers.cur[i] as usize].clone(),
                orca_full: full_evals[i],
                orca_sim: sim_evals[i],
            })?;
            Ok(())
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
