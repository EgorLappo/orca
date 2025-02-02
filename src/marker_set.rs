use color_eyre::eyre::{OptionExt, Result, WrapErr};
use intmap::IntMap;
use itertools::Itertools;
use log::debug;
use parking_lot::Mutex;
use polars::prelude::{DataFrame, DataType};
use rayon::prelude::*;
use std::sync::Arc;

use crate::{cli::Opts, orca::*};

#[derive(Debug, Clone)]
pub struct MarkerSet {
    // actual marker labels for printing/output
    pub ids: Vec<String>,
    // current best set of informative markers
    //   (in order of addition to the set)
    pub cur: Vec<u64>,
    // f_orca scores for each stage of the search
    //   (in the same order as cur)
    pub orcas: Vec<f64>,
    // remaining markers to search over
    rest: Vec<u64>,
    // number of markers
    pub n: usize,
    // marker frequencies in each of the populations
    markers: IntMap<u64, Vec<f64>>,
    // number of markers to consider for exhaustive searches
    r: usize,
    // number of populations
    pub k: usize,
    // prior probabilities of each population
    k_prior: Vec<f64>,
}

impl MarkerSet {
    pub fn new(data: DataFrame, opts: &Opts) -> Result<Self> {
        let k = data.shape().1 - 2;
        let k_prior = if opts.prior {
            debug!("computing non-uniform prior for the population distribution");
            cluster_proportions(&data, k)
        } else {
            debug!("using uniform prior for the population distribution");
            vec![1.0 / k as f64; k]
        };

        // here and below we kind of repeat data validation with unwrap
        // it all should work out if the dataframe was created by us in main

        // store string ids sepatarely from data
        let ids = data.column("id").wrap_err("failed to find column 'id'")?;
        let ids = ids
            .cast(&DataType::String)
            .wrap_err("failed to cast 'id' to string")?;
        let ids = ids
            .str()
            .wrap_err("error obtaining str array from column 'id'")?;

        let ids = ids
            .into_iter()
            .map(|x| x.unwrap().to_string())
            .collect::<Vec<_>>();

        // store marker frequencies in a map
        let mut markers: IntMap<u64, Vec<f64>> = IntMap::with_capacity(data.height());
        for (j, row) in (0..data.height()).enumerate() {
            let freqs = (0..k)
                .map(|i| -> Result<f64> {
                    let column = data
                        .column(&format!("freq{}", i + 1))
                        .wrap_err(format!("missing column 'freq{}'", i))?;
                    let column = column
                        .cast(&DataType::Float64)
                        .wrap_err(format!("column {:?} cannot be cast as f64", column))?;
                    let column = column.f64().wrap_err(format!(
                        "error obtaining f64 array from column {:?}",
                        column
                    ))?;
                    let row = column.get(row).ok_or_eyre(format!(
                        "failed to get row {:?} from column {:?}",
                        row, column
                    ))?;
                    Ok(row)
                })
                .collect::<Result<Vec<f64>>>()?;
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
            n: markers.len(),
            markers,
            r: opts.r,
            k,
            k_prior,
        };

        Ok(marker_set)
    }

    pub fn from_initial_set(data: DataFrame, opts: &Opts, initial_set: Vec<&str>) -> Result<Self> {
        let mut markers = MarkerSet::new(data, opts).wrap_err("failed to initialize MarkerSet")?;
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

    pub fn current_ids(&self) -> Vec<String> {
        self.cur
            .iter()
            .map(|&i| self.ids[i as usize].clone())
            .collect()
    }

    // returns best orca score for the chosen set
    pub fn search_exhaustive<T: Orca + Sync>(&mut self, orca: T) {
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
    }

    pub fn add_greedy<T: Orca + Sync>(&mut self, orca: T) -> (String, f64) {
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
        (id, best.0)
    }

    pub fn evaluate_sets(&self, orca: impl Orca + Sync) -> Vec<f64> {
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

    // outputs full and sim evaluation of the current best set
    pub fn evaluate_current(&self, orca: impl Orca + Sync) -> f64 {
        orca.init()
            .compute(self.markers.get_many(&self.cur), &self.k_prior)
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
