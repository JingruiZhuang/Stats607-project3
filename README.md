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
study actually needed: about a fifth of its runtime buys precision it never reports, and at
larger sample sizes the share reaches 94%.

## Reproducing everything

Clone the repository and run one command. Nothing else needs preparing: `python3` and `make` are
the only requirements, and `make reproduce` builds its own environment.

```
git clone https://github.com/JingruiZhuang/Stats607-project3.git
cd Stats607-project3
make reproduce
```

That target builds `.venv` from `requirements.txt`, deletes every generated file, reruns the
simulation, the summary table and the figures, and then runs the test suite and the regression
tests that show the optimizations preserved the study's results. About 20 seconds. No step needs
an activated shell: every stage runs against the interpreter in `.venv`, which the target creates
if it is missing.

```
make reproduce-full
```

adds the whole performance study on top of it -- profiling, complexity, the benchmarks, the
stability diagnostics, the budget and the performance figures. Several minutes, because the
benchmarks rerun the Unit 2 code repeatedly.

`requirements.txt` pins every package, direct and transitive, to the version the committed
results were produced with, and `.python-version` records the interpreter (3.14.7, on macOS).
`make env` says so if it is building against a different one. The pinned set is what the
byte-for-byte claim below was checked against; later versions may well work, but they are not
what was verified.

## Running individual stages

`make reproduce` is the only command needed. The stages below are the same work in pieces, useful
while developing; they call whatever `python3` is on the path, so activate the environment first
(`source .venv/bin/activate`) or pass the interpreter in (`make all PYTHON=.venv/bin/python`).

The study:

```
make env        # virtual environment in .venv/ from requirements.txt
make all        # simulate, summarise, and build all figures
make simulate   # raw estimates  -> results/raw/estimates.csv
make analyze    # summary table  -> results/raw/summary.csv
make figures    # figures        -> results/figures/
make parallel   # the simulation on every core
make test       # test suite
make clean      # remove generated files
make clean-env  # remove .venv/
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
make baseline-reference           # rebuild the committed Unit 2 estimates from baseline/
```

Estimated runtime: about 3 seconds for `make all`, of which 0.9 seconds is the simulation;
`make test` takes a further 7 seconds, and building the environment about 25 seconds on a warm
pip cache. `make benchmark` takes several minutes, because it reruns the Unit 2 code several
times over. Rerunning `make reproduce` from nothing reproduces the committed figures byte for
byte.

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

`baseline/` is the Unit 2 code exactly as it was measured, so later tidying stops short of it:
the docstring correction and the PEP257 and naming work in this repository are confined to `src/`
and `tests/`. `baseline/` therefore keeps the wording it was benchmarked with, including one
docstring that misstates the intercept decision and which `src/methods.py` now has right.

Outputs: `results/raw/estimates.csv` holds one row per estimate (20,000 rows),
`results/raw/summary.csv` one row per cell, `results/figures/` the figures,
`results/benchmarks/` and `results/profiles/` the performance measurements, and
`results/baseline/` the committed Unit 2 estimates that the regression tests compare against.

## AI assistance

Course policy allows AI tools with attribution.

I discussed the direction of this project with Claude and made the decisions myself: which
paper to reproduce, the scope of the reproduction, which optimizations to attempt for Unit 3,
and what to report. The code and documentation were drafted with Claude Code, then reviewed,
corrected and run by me.
