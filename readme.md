# orca

```
Compute f_ORCA following Rosenberg (2005)

Usage: orca [OPTIONS] --input <PATH> --output <PATH>

Options:
  -i, --input <PATH>               input csv file
  -o, --output <PATH>              output file
  -r <R>                           number of markers for exhaustive search [default: 2]
  -n, --nmarkers <NMARKERS>        number of best markers to return [default: 20]
  -p, --prior                      compute population prior based on proportion of present markers
  -c, --cut <CUT>                  number of most frequent markers to keep from the input
      --nsim <NSIM>                number of simulations for the simulation-based f_ORCA [default: 10000]
      --initial-set <SET>          ids of initial markers to use, comma-separated
      --full-orca-limit <LIMIT>    maximal number of markers for full ORCA computation [default: 24]
      --single-marker-eval <PATH>  compute and save evaluation of each single marker to a provided file
  -h, --help                       Print help
  -V, --version                    Print version
```

