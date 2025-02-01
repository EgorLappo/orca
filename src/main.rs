use clap::Parser;
use color_eyre::eyre::{bail, OptionExt, Result, WrapErr};
use intmap::IntMap;
use itertools::Itertools;
use jemallocator::Jemalloc;
use log::debug;
use parking_lot::Mutex;
use polars::prelude::*;
use rand::{prelude::*, rngs::SmallRng};
use rayon::prelude::*;
use std::fs::File;
use std::marker::Sync;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about = "Compute f_ORCA for folklore markers", long_about = None)]
struct Opts {
    #[arg(short, long, value_name = "PATH", help = "input csv file")]
    input: PathBuf,
    #[arg(short, long, value_name = "PREFIX", help = "output file prefix")]
    output: PathBuf,
    #[arg(
        short,
        default_value_t = 2,
        value_name = "R",
        help = "number of markers for exhaustive search"
    )]
    r: usize,
    #[arg(
        short,
        long,
        default_value_t = 20,
        help = "number of best markers to return"
    )]
    nmarkers: usize,
    #[arg(
        short,
        long,
        help = "compute population prior based on proportion of present markers"
    )]
    prior: bool,
    #[arg(
        short,
        long,
        help = "number of most frequent markers to keep from the input"
    )]
    cut: Option<usize>,
    #[arg(
        long,
        default_value_t = 10000,
        value_name = "NSIM",
        help = "number of simulations for the simulation-based f_ORCA"
    )]
    nsim: usize,
    #[arg(short, long, help = "do not evaluate best marker set")]
    no_eval: bool,
    #[arg(
        long,
        value_name = "SET",
        help = "ids of initial set of markers to use, comma-separated"
    )]
    initial_set: Option<String>,
    #[arg(
        long,
        default_value_t = 24,
        value_name = "LIMIT",
        help = "maximal number of markers for full ORCA computation"
    )]
    full_orca_limit: usize,
    #[arg(
        long,
        value_name = "PATH",
        help = "compute and save evaluation of each single marker to a provided file"
    )]
    single_marker_eval: Option<PathBuf>,
}

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
            .iter()
            .zip(markers.orcas.iter())
            .map(|(id, orca)| OutRow {
                id: id.clone(),
                orca: *orca,
            })
            .map(|row| writer.write_row(&row))
            .collect::<Result<Vec<()>>>()?;

        markers
    } else {
        let mut markers = MarkerSet::new(data, &opts);
        debug!(
            "searching for {} best among {} markers in {} populations",
            opts.nmarkers,
            markers.markers.len(),
            markers.k
        );

        // first search exhaustive for first r
        debug!("searching for first {} markers exhaustively", opts.r);

        // save the markers to file immediately
        markers
            .search_exhaustive(OrcaFull)
            .map(|row| writer.write_row(&row))
            .collect::<Result<Vec<()>>>()?; // force write for all

        markers
    };

    // then add the rest of the markers greedily
    let orca_full = OrcaFull;
    let orca_sim = OrcaSim::new(&opts);

    for _ in 0..(opts.nmarkers - opts.r) {
        let row = if markers.cur.len() < opts.full_orca_limit {
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
        writer.write_row(&row)?;
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

#[derive(Debug, Clone)]
struct MarkerSet {
    // actual marker labels for printing/output
    ids: Vec<String>,
    // current best set of informative markers
    //   (in order of addition to the set)
    cur: Vec<u64>,
    // f_orca scores for each stage of the search
    //   (in the same order as cur)
    orcas: Vec<f64>,
    // remaining markers to search over
    rest: Vec<u64>,
    // marker frequencies in each of the populations
    markers: IntMap<u64, Vec<f64>>,
    // number of markers to consider for exhaustive searches
    r: usize,
    // number of populations
    k: usize,
    // prior probabilities of each population
    k_prior: Vec<f64>,
}

impl MarkerSet {
    fn new(data: DataFrame, opts: &Opts) -> Self {
        let k = data.shape().1 - 2;
        let k_prior = if opts.prior {
            debug!("computing non-uniform prior for the population distribution");
            cluster_proportions(&data, k)
        } else {
            debug!("using uniform prior for the population distribution");
            vec![1.0 / k as f64; k]
        };

        // store string ids sepatarely from data
        let ids = data
            .column("id")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Vec<_>>();

        // store marker frequencies in a map
        let mut markers: IntMap<u64, Vec<f64>> = IntMap::with_capacity(data.height());
        for (j, row) in (0..data.height()).enumerate() {
            let freqs = (0..k)
                .map(|i| {
                    data.column(&format!("freq{}", i + 1))
                        .unwrap()
                        .f64()
                        .unwrap()
                        .get(row)
                        .unwrap()
                })
                .collect::<_>();
            markers.insert(j as u64, freqs);
        }

        // make sure all entries in markers are 0 <= x <= 1
        for (_, freqs) in markers.iter_mut() {
            for f in freqs.iter_mut() {
                *f = f.clamp(0.0, 1.0);
            }
        }

        let marker_set = MarkerSet {
            ids,
            cur: vec![],
            orcas: vec![],
            rest: markers.keys().collect(),
            markers,
            r: opts.r,
            k,
            k_prior,
        };

        if let Some(single_marker_eval) = opts.single_marker_eval.as_ref() {
            debug!(
                "saving single marker evaluation to {}",
                single_marker_eval.display()
            );
            let orca = OrcaFull;
            let mut writer = csv::Writer::from_path(single_marker_eval).unwrap();
            for (i, id) in marker_set.ids.iter().enumerate() {
                let orca = orca.init().compute(
                    marker_set.markers.get_many(&[i as u64]),
                    &marker_set.k_prior,
                );
                writer.serialize((id, orca)).unwrap();
            }
        }

        marker_set
    }

    fn from_initial_set(data: DataFrame, opts: &Opts, initial_set: Vec<&str>) -> Result<Self> {
        let mut markers = MarkerSet::new(data, opts);
        // get indices of the initial set, checking that they are all present
        let initial_indices = initial_set
            .iter()
            .map(|id| {
                markers
                    .ids
                    .iter()
                    .position(|x| x == *id)
                    .ok_or_eyre("initial set marker not found")
            })
            .collect::<Result<Vec<_>>>()?;

        markers.cur = initial_indices.iter().copied().map(|x| x as u64).collect();
        markers.orcas = vec![0.0; initial_set.len()];
        markers.rest.retain(|i| !markers.cur.contains(i));

        Ok(markers)
    }

    fn current_ids(&self) -> Vec<String> {
        self.cur
            .iter()
            .map(|&i| self.ids[i as usize].clone())
            .collect()
    }

    // returns best orca score for the chosen set
    fn search_exhaustive<T: Orca + Sync>(
        &mut self,
        orca: T,
    ) -> impl Iterator<Item = OutRow> + use<'_, T> {
        let best = Arc::new(Mutex::new((0.0, vec![])));

        self.rest
            .clone()
            .into_iter()
            .combinations(self.r)
            .par_bridge()
            .for_each_init(
                || orca.init(),
                |orca, c| {
                    let score = orca.compute(self.markers.get_many(&c), &self.k_prior);
                    // check if score is better than the best so far
                    // if yes then update
                    let mut best = best.lock();
                    if score > best.0 {
                        *best = (score, c);
                    }
                },
            );

        // update the state with the best set
        let best = Arc::try_unwrap(best).unwrap().into_inner();
        self.cur = best.1;
        self.orcas = self.cur.iter().map(|_| best.0).collect();
        self.rest.retain(|i| !self.cur.contains(i));

        self.cur
            .iter()
            .zip(self.orcas.iter())
            .map(|(&idx, &orca)| OutRow {
                id: self.ids[idx as usize].clone(),
                orca,
            })
    }

    fn add_greedy<T: Orca + Sync>(&mut self, orca: T) -> OutRow {
        let best = Arc::new(Mutex::new((0.0, 0)));

        self.rest.clone().into_par_iter().for_each_init(
            || orca.init(),
            |orca, i| {
                let score = orca.compute(
                    self.markers
                        .get_many(self.cur.iter().chain(std::iter::once(&i))),
                    &self.k_prior,
                );
                let mut best = best.lock();
                if score > best.0 {
                    *best = (score, i);
                }
            },
        );

        let best = Arc::try_unwrap(best).unwrap().into_inner();
        self.cur.push(best.1);
        self.orcas.push(best.0);
        self.rest.retain(|i| *i != best.1);

        let id = self.ids[best.1 as usize].clone();
        debug!("greedily added marker {} to best set", id);
        OutRow { id, orca: best.0 }
    }

    fn evaluate(&self, orca: impl Orca + Sync) -> Vec<f64> {
        // evaluation happens after best markers were selected
        assert!(!self.cur.is_empty());

        // evaluation is essentially done by re-computing f_ORCA for the best set of markers
        (0..self.cur.len())
            .map(|i| {
                orca.init()
                    .compute(self.markers.get_many(&self.cur[..=i]), &self.k_prior)
            })
            .collect()
    }
}

// polymorphic code setup to add other ways to compute f_orca in the future
trait Orca {
    type State: OrcaState;
    fn init(&self) -> Self::State;
}

trait OrcaState {
    fn compute(&mut self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64;
}

#[derive(Debug, Copy, Clone)]
struct OrcaFull;

impl Orca for OrcaFull {
    type State = OrcaFullState;
    fn init(&self) -> Self::State {
        OrcaFullState
    }
}

struct OrcaFullState;
impl OrcaState for OrcaFullState {
    fn compute(&mut self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64 {
        (0..markers.len())
            .map(|_| (0..=1).map(|x| x as f64))
            .multi_cartesian_product()
            // map over all binary sequences, with b of length markers.len()
            .map(|b| {
                // map over all populations
                k_prior
                    .iter()
                    .enumerate()
                    .map(|(i, kp)| {
                        // in each population map over all markers
                        kp * b
                            .iter()
                            .zip(markers.iter())
                            .map(|(b, &m)| b * m[i] + (1.0 - b) * (1.0 - m[i]))
                            .product::<f64>()
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum()
    }
}

#[derive(Debug, Copy, Clone)]
struct OrcaSim {
    nsim: usize,
}

impl OrcaSim {
    fn new(opts: &Opts) -> Self {
        OrcaSim { nsim: opts.nsim }
    }
}

struct OrcaSimState {
    nsim: usize,
    rng: SmallRng,
}

impl Orca for OrcaSim {
    type State = OrcaSimState;
    fn init(&self) -> OrcaSimState {
        OrcaSimState {
            nsim: self.nsim,
            rng: SmallRng::from_os_rng(),
        }
    }
}

impl OrcaState for OrcaSimState {
    fn compute(&mut self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64 {
        // repeat `nsim` simulations
        (0..self.nsim)
            // generate the source population
            // and the genotype for each replicate
            .map(|_| {
                let pop = sim_with_proportions(k_prior, &mut self.rng);
                (
                    pop,
                    markers
                        .iter()
                        .map(|m| m[pop])
                        .map(|p| sim_coin(p, &mut self.rng))
                        .collect(),
                )
            })
            // try to assign the genotype to its own population...
            .map(|(pop, b): (_, Vec<_>)| {
                // ... by computing gamma for each population, p.1187 of Rosenberg (2005)
                let gammas = k_prior.iter().enumerate().map(|(i, kp)| {
                    // in each population map over all markers
                    kp * b
                        .iter()
                        .zip(markers.iter())
                        .map(|(b, m)| b * m[i] + (1.0 - b) * (1.0 - m[i]))
                        .product::<f64>()
                });

                // now two things must be done:
                // first we must find max of gamma,
                // then *count* how many max elements are there,
                //    and whether any of these is the correct population
                // if true, P(correct assignment) = 1/n_max,
                //    otherwise, it's zero

                // we will to this in one pass over `gammas`:
                let (_, n_max, correct_is_maximal): (_, usize, bool) = gammas.enumerate().fold(
                    (f64::NEG_INFINITY, 0, false),
                    |(current, n, correct), (i, x)| {
                        if x > current {
                            // found new maximum, restart
                            (x, 1, i == pop)
                        } else if (x - current).abs() < f64::EPSILON {
                            // found repeating maximum
                            (x, n + 1, correct | (i == pop))
                        } else {
                            // not a maximum, proceed
                            (current, n, correct)
                        }
                    },
                );

                // finally, decide on whether we managed to assign the population correctly
                if !correct_is_maximal {
                    // we failed, correct pop did not give maximal gamma
                    0.0
                } else {
                    // otherwise we only fail if multiple populations with same max exist
                    // then we randomly choose so our rate is
                    1.0 / (n_max as f64)
                }
            })
            // get mean of the outcomes across all simulations
            .sum::<f64>()
            / self.nsim as f64
    }
}

// math/computation helper functions
fn sim_with_proportions(p: &[f64], rng: &mut SmallRng) -> usize {
    let mut r = rng.random_range(0.0..1.0);
    for (i, &x) in p.iter().enumerate() {
        r -= x;
        if r < 0.0 {
            return i;
        }
    }
    p.len() - 1
}

fn sim_coin(p: f64, rng: &mut SmallRng) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if rng.random_bool(p) {
        1.0
    } else {
        0.0
    }
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

fn cluster_proportions(data: &DataFrame, k: usize) -> Vec<f64> {
    // sum freq{i} columns and divide by total
    let mut total = 0.0;
    let mut freqs = vec![0.0; k];
    for (i, f) in freqs.iter_mut().enumerate() {
        *f = data
            .column(&format!("freq{}", i + 1))
            .unwrap()
            .sum_reduce()
            .unwrap()
            .value()
            .try_extract::<f64>()
            .unwrap();
        total += *f;
    }

    freqs.into_iter().map(|x| x / total).collect()
}

// output infrastructure

struct OutputWriter {
    writer: csv::Writer<File>,
}

impl OutputWriter {
    fn new(output: &Path) -> Result<Self> {
        let writer = csv::Writer::from_path(output)?;

        Ok(OutputWriter { writer })
    }

    fn write_row(&mut self, row: &OutRow) -> Result<()> {
        self.writer
            .serialize(row)
            .wrap_err(format!("failed to write row {:?}", row))
    }
}

#[derive(Debug, serde::Serialize)]
struct OutRow {
    id: String,
    orca: f64,
}

// extras for convenience

// K is always going to be < 10 so this should work well
trait GetMany<T> {
    fn get_many<'a, I>(&'a self, indices: I) -> Vec<&'a T>
    where
        I: IntoIterator<Item = &'a u64>;
}

impl<T> GetMany<T> for IntMap<u64, T> {
    fn get_many<'a, I>(&'a self, indices: I) -> Vec<&'a T>
    // NOTE for future self: we have to allocate into a Vec here because
    // orca computations require multiple passes over the values
    where
        I: IntoIterator<Item = &'a u64>,
    {
        indices.into_iter().map(|&i| self.get(i).unwrap()).collect()
    }
}
