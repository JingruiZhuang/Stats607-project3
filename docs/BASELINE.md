# Baseline performance of the Unit 2 simulation study

Measured against `baseline/`, a frozen copy of the Unit 2 code. Apple M3
(8 cores), Python 3.14.7, NumPy 2.5.4, pandas 3.1.1, statsmodels 0.15.0.
Every timing is the smallest of three runs after a warm-up.

## 1. Total runtime

| stage | min of 3 | share |
|---|---|---|
| simulate | **17.56 s** | 89% |
| analyze | 1.07 s | 5% |
| figures | 0.75 s | 4% |
| **`make all`** | **19.7 s** | |

Scenario A takes 5.4 s, scenario B 11.9 s: B runs twelve estimator cells to
A's eight and evaluates its outcome model three times per estimator. Almost
all of the analyze stage is the 2000-resample bootstrap in `src/metrics.py`.

These time whole processes, including ~1.3 s of imports. `make benchmark`
times the simulation in-process and reports 16.96 s, which is what
[OPTIMIZATION.md](OPTIMIZATION.md) compares against.

## 2. Profiler results

`make profile`, 200 replicates per scenario under cProfile. The profiler
inflates absolute seconds, so read the shares.

**Internal time by package** (a partition — every frame counted once):

| package | share |
|---|---|
| pandas | **41.3%** |
| numpy | 28.3% |
| statsmodels | 15.8% |
| Python stdlib and builtins | 10.9% |
| this study's own code | 3.7% |

41% in pandas, 1.5% in the linear algebra the study exists to perform. The
numpy row is mostly building tiny arrays, not solving. **42,078 Python calls
per replicate** (16.8 million for 400 replicates).

**Cumulative time by stage** (these nest, so they overlap):

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

- **Redundant propensity fits.** Eight logistic regressions per replicate for
  only two distinct models: 75% are recomputations.
- **Design matrices rebuilt from strings.** Per replicate, 28 matrices, 84
  `eval` calls and 335 pandas column reads — the namespace is rebuilt over all
  eleven columns every call, whichever the model uses.
- **`with_treatment` deep-copies the DataFrame** ten times per replicate, to
  get the design at `D = 1` and `D = 0`.

## 3. Computational complexity

**Theory.** Every stage is linear in `n` at fixed `k ≤ 7` columns: the DGP
draws `O(n)` normals, each IRLS iteration forms `X'WX` in `O(nk²)` and solves
it in `O(k³)`, each OLS solve is `O(nk²)`. So a replicate should cost
`t = c + a·n`, with `c` the Python and pandas overhead no sample size
amortises.

**Measurement.** `make complexity`, 25 replicates at eight sizes from 125 to
16,000. Fitting `t = a·n^p`:

| fitted over | scenario A | scenario B |
|---|---|---|
| whole grid, n ∈ [125, 16000] | p = 0.48 (R² 0.91) | p = 0.45 (R² 0.92) |
| upper half, n ≥ 2000 | **p = 0.79** (R² 0.98) | **p = 0.71** (R² 1.00) |

No single exponent fits, and that is the finding: the exponent approaches the
theoretical 1 only at large `n`. At small `n` the curve is nearly flat — four
times the data, from `n = 125` to `n = 500`, costs 1.20x the time in A and
1.13x in B.

The affine fit separates the two parts:

| scenario | fixed `c` | marginal `a` | **fixed share at n = 500** |
|---|---|---|---|
| A | 3.77 ms | 2.7 µs/obs | **74%** |
| B | 10.22 ms | 4.8 µs/obs | **81%** |

**The most important number here.** At the study's own `n`, three quarters to
four fifths of a replicate is overhead unrelated to the data — 95% (A) and 93%
(B) for the DGP alone. Faster arithmetic cannot touch it; only removing
per-replicate Python can.

## 4. Numerical warnings and convergence

`make stability-check` replays 200 replicates at each of `n ∈ {50, 100, 250,
500, 1000}` in both scenarios — 4,000 propensity fits — recording every
warning.

**Frequency: zero**, in every condition. No overflow, no divide-by-zero, no
`ConvergenceWarning`, no `LinAlgWarning`. Nothing is hidden either: IRLS
converged in all 4,000 fits (worst 23 iterations against a limit of 100,
typical 4), no design matrix was rank deficient, worst condition number 10.2.

Two reasons, and only one is comfortable:

1. The design is well conditioned. Covariates are independent normals and the
   propensity models use bounded indicators `1(Vl > 0)`, so the linear
   predictor stays in `[-12.5, 12.5]` and `expit` never exponentiates anything
   large. The overflow of `1/(1 + exp(-x))` is latent but unreachable here.
2. **The stability comes from statsmodels, not from this code.** Its binomial
   link clips fitted probabilities into `[ε, 1-ε]`, so nothing divides by zero.
   And when IRLS exhausts its 100 iterations it sets `converged = False` and
   raises nothing — `methods.fit_propensity` discards that object, so a
   convergence failure would be invisible. It does not happen here; it would
   not be noticed if it did.

**The real exposure raises no warning at all.** The estimators divide by `π`
and `1 - π` untrimmed, so the risk appears as weight concentration. At
`n = 500`:

| scenario | propensity model | median largest weight | worst | worst effective sample |
|---|---|---|---|---|
| A | correct | 4.5 | 7.5 | 120 of 500 (24%) |
| A | false | 3.9 | 5.6 | 127 of 500 (25%) |
| B | false | 3.5 | 5.8 | 429 of 500 (86%) |
| **B** | **correct** | **20.3** | **106.6** | **74 of 500 (15%)** |

Scenario B with the *correct* model is the outlier: its true coefficients
`(-3, 2.5, 3, 1, -3)` push fitted probabilities as close as 0.009 to a bound,
against 0.22 in A. The correct model is the dangerous one because it
reproduces that extremity faithfully. At `n = 50` a propensity comes within
`2.4 × 10⁻²⁰` of a bound and 30% of replicates hold one within `10⁻³`.

Weights stay finite only because a unit with `π ≈ 0` is almost certainly
untreated, so `1/π` is multiplied by `D = 0` and never enters. The study as
submitted reports no diagnostic that would show any of this.
