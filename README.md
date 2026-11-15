# Doubly Robust Estimation: A Reproduction of Bang & Robins (2005)

A simulation study reproducing the cross-sectional results of Bang, H. and Robins, J. M. (2005),
"Doubly Robust Estimation in Missing Data and Causal Inference Models," *Biometrics* 61, 962-973.
Two scenarios are studied, a missing-data problem with estimand `E(Y)` and a treatment-effect
problem with estimand `E{Y(1)} - E{Y(0)}`, each with eight estimators formed by crossing the
Horvitz-Thompson, outcome-regression and doubly robust estimators with correct and
deliberately misspecified nuisance models. The design is documented in [ADEMP.md](ADEMP.md)
and the critical discussion in [ANALYSIS.md](ANALYSIS.md).

For Unit 3 the study was optimized for speed and numerical stability. It now runs **24 times
faster** in serial and **57 times faster** on eight cores, and produces the same results.
[docs/BASELINE.md](docs/BASELINE.md) measures the Unit 2 code it started from and
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) sets out what was changed, what it bought and
what it cost. [docs/EXTENSIONS.md](docs/EXTENSIONS.md) goes on to ask how many replicates the
study actually needed, and finds it spends about a fifth of its simulation time and three
fifths of its analysis time on precision it does not report.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

The study:

```
make all        # simulate, summarise, and build all figures
make simulate   # raw estimates  -> results/raw/estimates.csv
make analyze    # summary table  -> results/raw/summary.csv
make figures    # figures        -> results/figures/
make parallel   # the simulation on every core
make test       # test suite
make clean      # remove generated files
```

The performance study. `TARGET` selects the implementation, `baseline` (the frozen Unit 2
code) or `optimized` (the current `src`):

```
make profile TARGET=baseline      # cProfile a representative run
make complexity                   # runtime against n, both implementations
make benchmark                    # baseline vs optimized, the ladder, core scaling
make stability-check TARGET=...   # warnings, convergence and weight diagnostics
make budget                       # replicates needed per scenario, and per n
make perf-figures                 # the performance figures
make performance                  # all of the above, in order
make regression                   # the evidence that the optimizations preserved results
```

Estimated runtime: about 3 seconds for `make all`, of which 0.9 seconds is the simulation;
`make test` takes a further 7 seconds. `make benchmark` takes several minutes, because it
reruns the Unit 2 code several times over. Rerunning `make all` from clean reproduces the
committed figures byte for byte.

## Key findings

All sixteen cells of Table 2 reproduce, with the largest gap from a published bias being 1.1
Monte Carlo standard errors. Reaching that agreement needs two choices the paper does not
state: its deliberately wrong propensity models must be fitted with an intercept, and its
treatment-effect results come from the two-covariate estimator of Scharfstein et al. rather
than the constrained variant the paper itself proposes, which turns out not to be doubly
robust.

## Layout

```
src/config.py        simulation parameters and model specifications
src/dgps.py          data-generating mechanisms
src/methods.py       the estimators, and the logistic solver underneath them
src/metrics.py       performance measures
src/simulation.py    orchestration; batching, parallelism, writes raw estimates
src/analyze.py       raw estimates -> summary table
src/figures.py       figures

src/bench_config.py  settings for the performance harness, kept apart from the design
src/profile_run.py   cProfile a run and attribute its time
src/complexity.py    runtime against n, with power-law and affine fits
src/benchmark.py     baseline vs optimized, the optimization ladder, core scaling
src/stability.py     warnings, convergence and inverse-probability weight diagnostics
src/budget.py        how many replicates each scenario needs, and what the rest cost
src/perf_figures.py  the performance figures

baseline/            the Unit 2 simulation code, frozen, as the reference to measure against
tests/               DGP, method, output and reproducibility checks
tests/test_regression.py   that the optimizations preserved the study's results
tests/test_budget.py       the arithmetic behind the replicate counts
docs/                the baseline measurement, the optimization record and the budget study
```

Outputs: `results/raw/estimates.csv` holds one row per estimate (20,000 rows),
`results/raw/summary.csv` one row per cell, `results/figures/` the figures,
`results/benchmarks/` and `results/profiles/` the performance measurements, and
`results/baseline/` the committed Unit 2 estimates that the regression tests compare against.

## AI assistance

Course policy allows AI tools with attribution. This records where they were used.

The following were drafted with Claude Code and then reviewed, corrected and run by me:

| File | Role |
| --- | --- |
| `src/config.py` | parameters and the model specifications transcribed from Table 1 |
| `src/dgps.py` | the two data-generating mechanisms |
| `src/methods.py` | the Horvitz-Thompson, outcome-regression and doubly robust estimators |
| `src/metrics.py` | performance measures and their Monte Carlo standard errors |
| `src/simulation.py`, `src/analyze.py` | the simulation loop and the summary table |
| `src/figures.py` | the three figures |
| `tests/` | the test suite |
| `Makefile`, `requirements.txt` | workflow and dependencies |
| `ADEMP.md`, `ANALYSIS.md`, `README.md` | documentation |

The choice of paper, the scope of the reproduction and the reporting decisions were mine.

The two discrepancies reported in ANALYSIS.md, the intercepts in the false propensity models and
the estimator behind Table 2, were found by running the code against the published values rather
than taken from any source. Each is covered by a test that fails if the decision is reverted, so
both can be checked rather than taken on trust.
