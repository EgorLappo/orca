use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Compute f_ORCA following Rosenberg (2005)", long_about = None)]
pub struct Opts {
    #[arg(short, long, value_name = "PATH", help = "input csv file")]
    pub input: PathBuf,
    #[arg(short, long, value_name = "PATH", help = "output file")]
    pub output: PathBuf,
    #[arg(
        short,
        default_value_t = 2,
        value_name = "R",
        help = "number of markers for exhaustive search"
    )]
    pub r: usize,
    #[arg(
        short,
        long,
        default_value_t = 20,
        help = "number of best markers to return"
    )]
    pub nmarkers: usize,
    #[arg(
        short,
        long,
        help = "compute population prior based on proportion of present markers"
    )]
    pub prior: bool,
    #[arg(
        short,
        long,
        help = "number of most frequent markers to keep from the input"
    )]
    pub cut: Option<usize>,
    #[arg(
        long,
        default_value_t = 10000,
        value_name = "NSIM",
        help = "number of simulations for the simulation-based f_ORCA"
    )]
    pub nsim: usize,
    #[arg(
        long,
        value_name = "SET",
        help = "ids of initial markers to use, comma-separated"
    )]
    pub initial_set: Option<String>,
    #[arg(
        long,
        default_value_t = 24,
        value_name = "LIMIT",
        help = "maximal number of markers for full ORCA computation"
    )]
    pub full_orca_limit: usize,
    #[arg(
        long,
        value_name = "PATH",
        help = "compute and save evaluation of each single marker to a provided file"
    )]
    pub single_marker_eval: Option<PathBuf>,
}
