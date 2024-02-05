```
Compute f_ORCA for folklore markers

Usage: folklore-orca [OPTIONS] --input <INPUT> --output <OUTPUT> -k <K>

Options:
  -i, --input <INPUT>
          input csv file
  -o, --output <OUTPUT>
          output csv file
  -k <K>
          number of populations K
  -r <R>
          number of markers for exhaustive search [default: 2]
  -n, --nmarkers <NMARKERS>
          number of best markers to return [default: 20]
  -p, --prior
          compute non-uniform prior for the population distribution
  -c, --cut <CUT>
          number of most frequent markers to keep from the input
      --nsim <NSIM>
          number of simulations for the simulation-based f_ORCA [default: 10000]
      --initial-set <INITIAL_SET>
          ids of initial set of markers to use, comma-separated
      --full-orca-limit <FULL_ORCA_LIMIT>
          maximal number of markers for full ORCA computation [default: 24]
  -h, --help
          Print help
  -V, --version
          Print version
```