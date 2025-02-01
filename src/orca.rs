use crate::cli::Opts;
use itertools::Itertools;
use rand::{prelude::*, rngs::SmallRng};

// polymorphic code setup to add other ways to compute f_orca in the future
pub trait Orca {
    type State: OrcaState;
    fn init(&self) -> Self::State;
}

pub trait OrcaState {
    fn compute(&mut self, markers: Vec<&Vec<f64>>, k_prior: &[f64]) -> f64;
}

#[derive(Debug, Copy, Clone)]
pub struct OrcaFull;

impl Orca for OrcaFull {
    type State = OrcaFullState;
    fn init(&self) -> Self::State {
        OrcaFullState
    }
}

pub struct OrcaFullState;

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
pub struct OrcaSim {
    pub nsim: usize,
}

impl OrcaSim {
    pub fn new(opts: &Opts) -> Self {
        OrcaSim { nsim: opts.nsim }
    }
}

pub struct OrcaSimState {
    pub nsim: usize,
    pub rng: SmallRng,
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
