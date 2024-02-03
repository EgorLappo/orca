use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use intmap::IntMap;
use itertools::Itertools;
use jemallocator::Jemalloc;
use log::debug;
use parking_lot::Mutex;
use polars::prelude::*;
use rayon::prelude::*;
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
}

fn main() -> Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    assert!(opts.nmarkers >= opts.r);

    // read input csv
    let data = read_input_csv(&opts.input, opts.k).wrap_err("failed to read input csv")?;

    let data = cut_data(data, &opts).wrap_err("failed to truncate the input dataset")?;

    // create marker set
    let mut markers = MarkerSet::new(data, &opts);
    debug!(
        "searching for {} best among {} markers in {} populations",
        opts.nmarkers,
        markers.markers.len(),
        opts.k
    );

    // first search exhaustive for first r
    debug!("searching for first {} markers exhaustively", opts.r);
    markers.search_exhaustive();

    // then add the rest of the markers greedily
    let j = opts.nmarkers - opts.r;
    for _ in 0..j {
        debug!("adding marker {}/{}", markers.cur.len() + 1, opts.nmarkers);
        markers.add_greedy();
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

    fn current_ids(&self) -> Vec<String> {
        self.cur
            .iter()
            .map(|&i| self.ids[i as usize].clone())
            .collect()
    }

    // returns best orca score for the chosen set
    fn search_exhaustive(&mut self) -> f64 {
        // NOTE: could be done with RwLock instead but i don't see the reason
        let best = Arc::new(Mutex::new((0.0, vec![])));

        self.rest
            .clone()
            .into_iter()
            .combinations(self.r)
            .par_bridge()
            .for_each(|c| {
                let score = OrcaFull.compute(self.markers.get_many(&c), &self.k_prior);
                // check if score is better than the best so far
                // if yes then update
                let mut best = best.lock();
                if score > best.0 {
                    *best = (score, c);
                }
            });

        // update the state with the best set
        let best = Arc::try_unwrap(best).unwrap().into_inner();
        self.cur = best.1;
        self.orcas = self.cur.iter().map(|_| best.0).collect();
        self.rest.retain(|i| !self.cur.contains(i));

        best.0
    }

    fn add_greedy(&mut self) -> f64 {
        let best = Arc::new(Mutex::new((0.0, 0)));

        self.rest.clone().into_par_iter().for_each(|i| {
            let score = OrcaFull.compute(
                self.markers
                    .get_many(self.cur.iter().chain(std::iter::once(&i))),
                &self.k_prior,
            );
            let mut best = best.lock();
            if score > best.0 {
                *best = (score, i);
            }
        });

        let best = Arc::try_unwrap(best).unwrap().into_inner();
        self.cur.push(best.1);
        self.orcas.push(best.0);
        self.rest.retain(|i| *i != best.1);

        best.0
    }
}

// polymorphic code setup to add other ways to compute f_orca in the future
trait Orca {
    fn compute(&self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64;
}

#[derive(Debug, Clone)]
struct OrcaFull;

impl Orca for OrcaFull {
    fn compute(&self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64 {
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
