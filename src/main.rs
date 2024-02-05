use clap::Parser;
use color_eyre::eyre::{OptionExt, Result, WrapErr};
use intmap::IntMap;
use itertools::Itertools;
use jemallocator::Jemalloc;
use log::debug;
use parking_lot::Mutex;
use polars::prelude::*;
use rand::{prelude::*, rngs::SmallRng};
use rayon::prelude::*;
use std::marker::Sync;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about = "Compute f_ORCA for folklore markers", long_about = None)]
struct Opts {
    #[arg(short, long, help = "input csv file")]
    input: PathBuf,
    #[arg(short, long, help = "output csv file")]
    output: PathBuf,
    #[arg(short, help = "number of populations K")]
    k: usize,
    #[arg(
        short,
        default_value_t = 2,
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
        help = "compute non-uniform prior for the population distribution"
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
        help = "number of simulations for the simulation-based f_ORCA"
    )]
    nsim: usize,
    #[arg(long, help = "ids of initial set of markers to use, comma-separated")]
    initial_set: Option<String>,
    #[arg(
        long,
        default_value_t = 24,
        help = "maximal number of markers for full ORCA computation"
    )]
    full_orca_limit: usize,
}

fn main() -> Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    assert!(opts.nmarkers >= opts.r);

    // read input csv
    let data = read_input_csv(&opts.input, opts.k).wrap_err("failed to read input csv")?;
    let data = cut_data(data, &opts).wrap_err("failed to truncate the input dataset")?;

    // create marker set
    let mut markers = if let Some(initial_set_str) = opts.initial_set.as_ref() {
        debug!("using provided initial marker set {}", initial_set_str);
        let initial_set = initial_set_str.split(',').collect::<Vec<_>>();
        MarkerSet::from_initial_set(data, &opts, initial_set)
            .wrap_err("failed to use provided initial marker set")?
    } else {
        let mut markers = MarkerSet::new(data, &opts);
        debug!(
            "searching for {} best among {} markers in {} populations",
            opts.nmarkers,
            markers.markers.len(),
            opts.k
        );

        // first search exhaustive for first r
        debug!("searching for first {} markers exhaustively", opts.r);
        markers.search_exhaustive(OrcaFull);

        markers
    };

    // then add the rest of the markers greedily
    let orca_full = OrcaFull;
    let orca_sim = OrcaSim::new(&opts);

    let j = opts.nmarkers - opts.r;
    for i in 0..j {
        if i < opts.full_orca_limit {
            debug!(
                "adding marker {}/{} using exhaustive f_ORCA computation",
                markers.cur.len() + 1,
                opts.nmarkers
            );
            markers.add_greedy(orca_full);
        } else {
            debug!(
                "adding marker {}/{} using simulation f_ORCA computation",
                markers.cur.len() + 1,
                opts.nmarkers
            );
            markers.add_greedy(orca_sim);
        }
    }

    // save results
    write_output(&markers, &opts.output).wrap_err("failed to write output csv")?;

    Ok(())
}

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
    markers: IntMap<Vec<f64>>,
    // number of markers to consider for exhaustive searches
    r: usize,
    // prior probabilities of each population
    k_prior: Vec<f64>,
}

impl MarkerSet {
    fn new(data: DataFrame, opts: &Opts) -> Self {
        let k_prior = if opts.prior {
            debug!("computing non-uniform prior for the population distribution");
            cluster_proportions(&data, opts.k)
        } else {
            debug!("using uniform prior for the population distribution");
            vec![1.0 / opts.k as f64; opts.k]
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
        let mut markers: IntMap<Vec<f64>> = IntMap::with_capacity(data.height());
        for (j, row) in (0..data.height()).enumerate() {
            let freqs = (0..opts.k)
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

        MarkerSet {
            ids,
            cur: vec![],
            orcas: vec![],
            rest: markers.keys().copied().collect(),
            markers,
            r: opts.r,
            k_prior,
        }
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
    fn search_exhaustive(&mut self, orca: impl Orca + Sync) -> f64 {
        // NOTE: could be done with RwLock instead but i don't see the reason
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

        best.0
    }

    fn add_greedy(&mut self, orca: impl Orca + Sync) -> f64 {
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

        best.0
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
                            .map(|(b, m)| b * m[i] + (1.0 - b) * (1.0 - m[i]))
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
            rng: SmallRng::from_entropy(),
        }
    }
}

impl OrcaState for OrcaSimState {
    fn compute(&mut self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64 {
        (0..self.nsim)
            // generate the source population
            // and the genotype
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
            // map over all binary sequences, with b of length markers.len()
            .map(|(pop, b): (_, Vec<_>)| {
                // map over all populations
                k_prior
                    .iter()
                    .enumerate()
                    .map(|(i, kp)| {
                        // in each population map over all markers
                        kp * b
                            .iter()
                            .zip(markers.iter())
                            .map(|(b, m)| b * m[i] + (1.0 - b) * (1.0 - m[i]))
                            .product::<f64>()
                    })
                    // take argmax of this
                    // and return 1 if it's the same as the source population
                    .enumerate()
                    .map(|(i, x)| if i == pop { (1, x) } else { (0, x) })
                    // TODO: HOW TO CHOOSE RANDOMLY IN CASE OF TIE?
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            })
            // get mean of the binary outcomes
            .sum::<usize>() as f64
            / self.nsim as f64
    }
}

// math/computation helper function
fn sim_with_proportions(p: &[f64], rng: &mut SmallRng) -> usize {
    let mut r = rng.gen_range(0.0..1.0);
    for (i, &x) in p.iter().enumerate() {
        r -= x;
        if r < 0.0 {
            return i;
        }
    }
    p.len() - 1
}

fn sim_coin(p: f64, rng: &mut SmallRng) -> f64 {
    if rng.gen_bool(p) {
        1.0
    } else {
        0.0
    }
}

// IO functions

fn read_input_csv(input: &Path, k: usize) -> Result<DataFrame> {
    let reader = CsvReader::from_path(input)?
        .has_header(true)
        .infer_schema(Some(10))
        .finish()?;

    let mut cols = vec!["id".to_string(), "freq".to_string()];
    for i in 0..k {
        cols.push(format!("freq{}", i + 1));
    }

    let df = reader
        .select(cols)
        .wrap_err("couldn't find required columns in the input csv")?;

    Ok(df)
}

fn write_output(markers: &MarkerSet, output: &Path) -> Result<()> {
    #[derive(serde::Serialize)]
    struct OutRow {
        id: String,
        orca: f64,
    }

    let mut writer = csv::Writer::from_path(output)?;

    markers
        .current_ids()
        .iter()
        .zip(markers.orcas.iter())
        .map(|(id, orca)| OutRow {
            id: id.clone(),
            orca: *orca,
        })
        .map(|row| writer.serialize(row))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
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
        Ok(data.sort(["freq"], true, false)?.head(Some(cut)))
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
            .sum()
            .unwrap();
        total += *f;
    }

    freqs.into_iter().map(|x| x / total).collect()
}

// extras for convenience

// K is always going to be < 10 so this should work well
trait GetMany<T> {
    fn get_many<'a, I>(&'a self, indices: I) -> Vec<&'a T>
    where
        I: IntoIterator<Item = &'a u64>;
}

impl<T> GetMany<T> for IntMap<T> {
    fn get_many<'a, I>(&'a self, indices: I) -> Vec<&'a T>
    where
        I: IntoIterator<Item = &'a u64>,
    {
        indices.into_iter().map(|&i| self.get(i).unwrap()).collect()
    }
}
