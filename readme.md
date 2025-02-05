# ORCA

Help message of the program:

```
$> orca --help
Compute f_ORCA following Rosenberg (2005)

Usage: orca [OPTIONS] --input <PATH> --output <PATH>

Options:
  -i, --input <PATH>                    input csv file
  -o, --output <PATH>                   output file
  -r <R>                                number of markers for exhaustive search [default: 2]
  -n, --nmarkers <NMARKERS>             number of best markers to return [default: 20]
  -p, --prior                           compute population prior based on proportion of present markers
  -c, --cut <CUT>                       number of most frequent markers to keep from the input
      --nsim <NSIM>                     number of simulations for the simulation-based f_ORCA [default: 10000]
      --initial-set <SET>               ids of initial markers to use, comma-separated
  -s, --full-orca-search-limit <LIMIT>  max marker set size for full ORCA computation [default: 24]
  -e, --full-orca-eval-limit <LIMIT>    max marker set size for full ORCA computation during evaluation [default: 24]
      --single-marker-eval <PATH>       compute and save evaluation of each single marker to a provided file
  -h, --help                            Print help
  -V, --version                         Print version
  ```

## Usage notes



## Installation

Currently, all installation methods require Rust to be installed locally. The best way to do this is with [rustup](https://rustup.rs).

  1. **Use Cargo**. This method would produce a binary named `orca` on your `$PATH`.

  ```
  cargo install --git https://github.com/EgorLappo/orca.git
  ```

  2. **Build from scratch**. Clone the repository and run `cargo build --release` to build the program.
