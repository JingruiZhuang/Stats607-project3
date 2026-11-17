# Baseline performance of the Unit 2 simulation study

Measured against `baseline/`, a frozen copy of the Unit 2 code. Apple M3
(8 cores), macOS 14.6, Python 3.14.7, NumPy 2.5.4, pandas 3.1.1, statsmodels
0.15.0. Every timing is the smallest of three runs after a discarded warm-up.
Reproduce with `make profile`, `make complexity` and `make stability-check`.

## 1. Total runtime

| stage | min of 3 | share |
|---|---|---|
| simulate | **17.56 s** | 89% |
| analyze | 1.07 s | 5% |
| figures | 0.75 s | 4% |
| **`make all`** | **19.7 s** | |

The simulation draws 1000 datasets of `n = 500` in each of two scenarios and
computes 20 estimator cells on them — 20,000 estimates. Scenario A takes 5.4 s,
scenario B 11.9 s; B costs twice as much because it runs twelve cells to A's
eight and evaluates its outcome model three times per estimator.

Almost all of the 1.07 s analysis stage is the bootstrap in `src/metrics.py`,
which resamples 2000 times per cell.

These figures time whole processes, so they include about 1.3 s of interpreter
start-up across the three stages — statsmodels alone costs 0.79 s to import.
`make benchmark` times the simulation in-process instead and reports 16.96 s;
that is the figure [OPTIMIZATION.md](OPTIMIZATION.md) compares against.

## 2. Profiler results

`make profile` runs 200 replicates per scenario under cProfile. The profiler
inflates the absolute seconds, so read the shares.

**Internal time by package** — a partition, every frame counted once:

| package | share |
|---|---|
| pandas | **41.3%** |
| numpy | 28.3% |
| statsmodels | 15.8% |
| Python stdlib and builtins | 10.9% |
| this study's own code | 3.7% |

The study spends 41% of its time in pandas and 1.5% in the linear algebra it
exists to perform. The numpy row is not that linear algebra either: it is
mostly the cost of building thousands of tiny arrays. The run makes 16.8
million function calls for 400 replicates — **42,078 calls per replicate**.

**Cumulative time by stage** — these nest, so they overlap rather than
partition:

| stage | share |
|---|---|
| `estimate`: all estimators | **90.6%** |
| — of which `fit_propensity` | **50.6%** |
| — — of which statsmodels IRLS | 34.8% |
| — of which the OLS solve | 1.5% |
| `generate`: draw one dataset | 9.0% |
| `design_matrix` (DGP + estimators) | **35.5%** |
| `with_treatment` (DataFrame copy) | 9.1% |

Three bottlenecks:

**(a) Every estimator refits the same propensity model.** A replicate performs
eight logistic regressions, but a scenario names only two distinct propensity
models. 75% of the fits recompute a result the replicate already holds.

**(b) Design matrices are rebuilt from strings, through pandas.** Per
replicate: 28 design matrices, 84 `eval` calls and 335 pandas column reads. The
namespace is rebuilt over all eleven columns on every call, whether the model
mentions them or not, and each term string is `eval`'d afresh.

**(c) `with_treatment` deep-copies the DataFrame** ten times per replicate, to
get the design at `D = 1` and `D = 0`. 9.1% of the run.

## 3. Computational complexity

**Theory.** Every stage of a replicate is linear in `n` with a fixed number of
columns `k ≤ 7`: the DGP draws `O(n)` normals and evaluates `O(nk)` terms; each
IRLS iteration forms `X'WX` in `O(nk²)` and solves it in `O(k³)`; each OLS
solve is `O(nk²)`. So a replicate should cost `t = c + a·n` — linear in `n`,
plus a fixed cost `c` of Python and pandas overhead that no sample size
amortises.

**Measurement.** `make complexity` times 25 replicates at eight sample sizes
from 125 to 16,000. Fitting `t = a·n^p`:

| fitted over | scenario A | scenario B |
|---|---|---|
| whole grid, n ∈ [125, 16000] | p = 0.48 (R² 0.91) | p = 0.45 (R² 0.92) |
| upper half, n ≥ 2000 | **p = 0.79** (R² 0.98) | **p = 0.71** (R² 1.00) |

No single exponent describes this code, and that is the finding. At large `n`
the exponent approaches the theoretical 1. At small `n` the curve is nearly
flat: four times the data, from `n = 125` to `n = 500`, costs 1.20 times the
time in scenario A and 1.13 times in scenario B.

The affine fit separates the two parts:

| scenario | fixed cost `c` | marginal cost `a` | **fixed share at n = 500** |
|---|---|---|---|
| A | 3.77 ms | 2.7 µs per observation | **74%** |
| B | 10.22 ms | 4.8 µs per observation | **81%** |

**This is the most important number here.** At the sample size the study runs
at, three quarters to four fifths of a replicate is fixed overhead with
nothing to do with the data. Faster arithmetic cannot touch it; only removing per-replicate Python
can. For the DGP alone the fixed share is 95% (A) and 93% (B).

## 4. Numerical warnings and convergence

`make stability-check` replays 200 replicates at each of `n ∈ {50, 100, 250,
500, 1000}` in both scenarios — 4,000 propensity fits — recording every
warning.

**Frequency: zero.** No overflow, no divide-by-zero, no `ConvergenceWarning`,
no `LinAlgWarning`, in any condition. Nor is anything hidden: IRLS converged in
all 4,000 fits (worst 23 iterations against a limit of 100, typical 4), no
design matrix was rank deficient, and the worst condition number was 10.2.

Two reasons, and only one is comfortable:

1. The design is well conditioned. Covariates are independent standard normals
   and the propensity models use bounded indicators `1(Vl > 0)`, so the true
   linear predictor is confined to `[-12.5, 12.5]` and `expit` is never asked
   to exponentiate anything large. The textbook overflow of `1/(1 + exp(-x))`
   is latent in the code but unreachable at this design.
2. **The stability comes from statsmodels, not from this code.** Its binomial
   link clips fitted probabilities into `[ε, 1-ε]`, which is why nothing
   divides by zero. And when IRLS exhausts its 100 iterations it sets
   `converged = False` and raises nothing — `methods.fit_propensity` discards
   that object, so a convergence failure would be invisible. It does not happen
   here. It would not be noticed if it did.

**The real exposure is weight concentration, which raises no warning at all.**
The estimators divide by `π` and `1 - π` with no trimming, so the risk shows up
as a few replicates carrying enormous weight. At `n = 500`:

| scenario | propensity model | median largest weight | worst largest weight | worst effective sample |
|---|---|---|---|---|
| A | correct | 4.5 | 7.5 | 120 of 500 (24%) |
| A | false | 3.9 | 5.6 | 127 of 500 (25%) |
| B | false | 3.5 | 5.8 | 429 of 500 (86%) |
| **B** | **correct** | **20.3** | **106.6** | **74 of 500 (15%)** |

One cell is unlike the others. Scenario B with the *correct* propensity model
puts a typical replicate's largest weight at 20 and its worst at 107, leaving
an effective sample of 74 out of 500. The cause is the design: B's true
coefficients `(-3, 2.5, 3, 1, -3)` push fitted probabilities as close as 0.009
to a bound, against 0.22 in scenario A. The correct model is the dangerous one
because it reproduces that extremity faithfully; the false model smooths it
away.

It gets worse at smaller `n`: at `n = 50` a fitted propensity comes within
`2.4 × 10⁻²⁰` of a bound, and 30% of replicates contain a propensity within
`10⁻³` of one. The weights stay finite only because a unit with `π ≈ 0` is
almost certainly untreated, so `1/π` is multiplied by `D = 0` and never enters.
The exposure is bounded by the same probability that creates it — but the study
as submitted reports no diagnostic that would reveal any of this.
