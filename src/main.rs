use clap::Parser;
use color_eyre::eyre::{Result, WrapErr};
use intmap::IntMap;
use itertools::Itertools;
use polars::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
// use rand::{prelude::*, rngs::SmallRng};

use jemallocator::Jemalloc;

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output: PathBuf,
    #[arg(short)]
    k: usize,
    #[arg(short, default_value_t = 2)]
    r: usize,
    #[arg(short, long, default_value_t = 20)]
    nmarkers: usize,
    #[arg(short, long)]
    silent: bool,
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    // read input csv
    let data = read_input_csv(&opts.input, opts.k).wrap_err("failed to read input csv")?;

    // create marker set
    let mut markers = MarkerSet::new(data, &opts);
    if !opts.silent {
        println!(
            "searching for {} best among {} markers in {} populations",
            opts.nmarkers,
            markers.markers.len(),
            opts.k
        );
    }

    // first search exhaustive for first r
    if !opts.silent {
        println!("searching for first {} markers exhaustively", opts.r);
    }
    markers.search_exhaustive();

    // then add the rest of the markers greedily
    let j = opts.nmarkers - opts.r;
    for _ in 0..j {
        if !opts.silent {
            println!("adding marker {}/{}", markers.cur.len() + 1, opts.nmarkers);
        }
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
        let k_prior = cluster_proportions(&data, opts.k);

        // store string ids sepatarely from data
        let ids = data
            .column("id")
            .unwrap()
            .str()
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Vec<_>>();

        // convert data to HashMap<frequency vector"cargo>
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
        let mut best = (0.0, vec![]);

        self.rest
            .clone()
            .into_iter()
            .combinations(self.r)
            .par_bridge()
            .map(|c| {
                let score = OrcaFull.compute(self.markers.get_many(&c), &self.k_prior);
                (score, c)
            })
            .collect::<Vec<_>>()
            .iter()
            .for_each(|(score, c)| {
                if score > &best.0 {
                    best = (*score, c.clone());
                }
            });

        self.cur = best.1;
        self.orcas = self.cur.iter().map(|_| best.0).collect();
        self.rest.retain(|i| !self.cur.contains(i));

        best.0
    }

    fn add_greedy(&mut self) -> f64 {
        let mut best = (0.0, 0);

        self.rest
            .clone()
            .into_par_iter()
            .map(|i| {
                let score = OrcaFull.compute(
                    self.markers
                        .get_many(self.cur.iter().chain(std::iter::once(&i))),
                    &self.k_prior,
                );
                (score, i)
            })
            .collect::<Vec<_>>()
            .iter()
            .for_each(|(score, i)| {
                if score > &best.0 {
                    best = (*score, *i);
                }
            });

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

    let mut cols = vec!["id".to_string()];
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
