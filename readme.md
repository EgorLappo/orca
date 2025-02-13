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
      --resume                          resume computation if output file already exists
  -h, --help                            Print help
  -V, --version                         Print version
```

## Usage notes

### Restrictions

Currently, `orca` is able to work only with **binary** markers.

### Input data format

The program expects a very specific input data format. The input csv table must have a single row per marker that records its frequencies in different populations. Mandatory columns are: `id` --- a string column containing marker id, `freq` --- global frequency of the marker across all populations (not really used in the main computation loop, so the requirement will be removed soon), and `freq{i}` --- frequencies of the marker in populations `i = 1, ..., n`. Read the errors given by the program to fix your input data.

### Algorithms

`orca` closely follows Rosenberg (2005) and implements two algorithms for marker selection and two algorithms for `f_ORCA` computation described in the paper.

Marker selection begins with an _exhaustive search_ in which all marker subsets of size `r` are evaluated to find the best starting set. Depending on the number of markers and population, exhaustive search is incredibly computationally expensive, growing as `O(n^r)` where `n` is the total number of markers. For data with the numbers of populations and markers on the order of `10^3`, I have found that it is possible to use `r=3` (with runtime of <=10 hours), while `r=4` is impractical. To save time in multiple repeat runs, initial set could be provided as an argument, avoiding exhaustive search.

After the exhaustive search, marker selection proceeds in sequence in a __greedy__ way.

The value of `f_ORCA` is the criterion that determines the performance of the marker set. The two ways of computing `f_ORCA` available here are the __full_ and __simulation-based__ methods. 

### Output format

### Parallel execution

Multiple nested loops of the marker selection procedure are trivially parallelizable, so `orca` by default would try to use all available cores to spread the computation. Parallel execution is achieved via the [`rayon`](https://docs.rs/rayon) crate, and hence the number of cores can be controlled by setting the `RAYON_NUM_THREADS` envronment variable.

### Checkpointing

Passing the `--resume` version to `orca` enables the program to resume previously interrupted search. Interruptions typically happen in HPC cluster environments due to job timeouts or preemptions. After restarting the job, `orca` will try to read the output file and continue adding new markers to the already computed initial set.

## Installation

Currently, all installation methods require Rust to be installed locally. The best way to do this is with [rustup](https://rustup.rs).

  1. **Use Cargo**. This method would produce a binary named `orca` on your `$PATH`.

  ```
  cargo install --git https://github.com/EgorLappo/orca.git
  ```

  2. **Build from scratch**. Clone the repository and run `cargo build --release` to build the program.
