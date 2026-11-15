# Baseline performance of the Unit 2 simulation study

Where the submitted study spends its time, how that time scales with the sample
size, and what numerical risk it carries. Nothing here changes the study: every
measurement is taken against `baseline/`, a frozen copy of the Unit 2 code, and
the study it reruns reproduces the submitted `results/raw/estimates.csv` byte
for byte.

The study itself is described in [ADEMP.md](../ADEMP.md) and its findings in
[ANALYSIS.md](../ANALYSIS.md). In one line: two scenarios from Bang & Robins
(2005), each drawing `N_SIM = 1000` datasets of `N = 500`, and computing eight
estimators on each (twelve in scenario B, which carries four extension cells) —
20,000 estimates in all.

## How to reproduce these numbers

```
make profile           # cProfile a representative run    -> results/profiles/
make complexity        # runtime against n                -> results/benchmarks/
make stability-check   # warnings and numerical exposure  -> results/benchmarks/
make perf-figures      # the figures below                -> results/figures/
```

Measured on an Apple M3 (8 cores), macOS 14.6, Python 3.14.7, NumPy 2.5.4 on
Accelerate, pandas 3.1.1, statsmodels 0.15.0. Every timing below is the
smallest of three runs after a discarded warm-up: a run can only be inflated by
other work on the machine, never deflated, so the minimum is the closest
estimate of the cost of the code itself.

## 1. Total runtime

| stage | command | min of 3 | share |
|---|---|---|---|
| simulate | `python -m src.simulation` | **17.56 s** | 89% |
| analyze | `python -m src.analyze` | 1.07 s | 5% |
| figures | `python -m src.figures` | 0.75 s | 4% |
| **`make all`, end to end** | | **19.7 s** | |

Within the simulation, scenario A takes 5.4 s for its 1000 replicates and
scenario B 11.9 s — 5.4 ms and 11.9 ms per replicate. Scenario B costs roughly
twice as much because it runs twelve estimator cells to scenario A's eight and
because its outcome model is evaluated three times per estimator rather than
once, at the observed treatment and at both counterfactual ones.

Two caveats on the headline figure. The *first* `make all` on a fresh machine
takes 33.6 s, the extra 14 s being one-off first-run costs, chiefly
matplotlib building its font cache; 19.7 s is the honest repeat cost. And `make all` runs three separate Python
processes, so it pays for importing its libraries three times over rather
than once: statsmodels costs 0.79 s to import, matplotlib 0.23 s and pandas
0.16 s, putting roughly 1.3 s of the 19.7 s into interpreter start-up. Small
enough to ignore, but not zero.

The analysis stage is worth a note even though it is only 5% of the run: almost
all of its 1.07 s is the bootstrap in `src/metrics.py`, which resamples 2000
times per cell to get Monte Carlo standard errors for the median and the IQR.
That is a genuinely quadratic-feeling cost that happens to be cheap here only
because there are 20 cells.

## 2. Where the time goes

`make profile` runs 200 replicates per scenario under cProfile. The profiler's
per-call overhead inflates the absolute seconds — 6.5 s for work that takes
3.5 s unprofiled — so the shares are what to read, not the totals. Full output
is in `results/profiles/baseline_profile.txt`. Shares move by a few tenths of
a point between runs; the tables below are one canonical run, and no claim here
turns on a difference that small.

**Cumulative time by stage.** These nest: `fit_propensity` contains the whole
statsmodels IRLS, and `design_matrix` is called from both the DGP and the
estimators, so the rows overlap rather than partition the run.

| stage | share of profiled run |
|---|---|
| `estimate`: all estimators | **90.7%** |
| — of which `fit_propensity` | **50.7%** |
| — — of which statsmodels IRLS | 34.9% |
| — of which the OLS solve | 1.5% |
| `generate`: draw one dataset | 8.9% |
| `design_matrix` (DGP + estimators) | **35.4%** |
| `with_treatment` (DataFrame copy) | 9.2% |

**Internal time by package.** This one *is* a partition — every frame counted
once — and it is the more alarming table.

| package | share |
|---|---|
| pandas | **41.1%** |
| numpy | 22.5% |
| Python stdlib and builtins | 16.8% |
| statsmodels | 15.9% |
| this study's own code | 3.7% |

The study spends 41% of its time in pandas and 1.5% in the linear algebra it
exists to perform. The run makes 16.8 million function calls for 400
replicates: **42,000 Python calls per replicate**, to produce eight numbers
from 500 rows of data.

### The three bottlenecks, named

**(a) Every estimator refits the same propensity model.** Each row of
`config.ESTIMATORS` names the propensity model it wants and `methods.estimate`
fits it from scratch, so a replicate performs eight logistic regressions. But a
scenario names only *two* distinct propensity models, `ps_true` and `ps_false`,
and every estimator that asks for one wants the identical fit on the identical
data. **75% of the propensity fits recompute a result the replicate already
holds**, and `fit_propensity` is 50.7% of the run.

**(b) Design matrices are rebuilt from strings, through pandas, 28 times per
replicate.** `dgps.design_matrix` is a nice piece of design — correct and
misspecified models differ only by a list of term strings in `config`, with no
duplicated code — and it is expensive in a way that is invisible from reading
it:

```python
namespace = {c: data[c].to_numpy(dtype=float) for c in data.columns}   # all 11 columns
...
value = eval(term, {"__builtins__": {}}, namespace)                    # per term
```

The namespace is rebuilt on every call over *all eleven* columns of the frame,
whether the specification mentions them or not, and each term string is
`eval`'d afresh. Per replicate that is **335 pandas column reads, 84 `eval`
calls and 28 design matrices** — for a handful of distinct matrices, over data
that never changes within the replicate.

**(c) `with_treatment` deep-copies the DataFrame, ten times per replicate.**
Scenario B needs the design matrix at `D = 1` and at `D = 0`; it gets them by
copying the whole frame and overwriting one column. 9.2% of the run.

## 3. Computational complexity

**Theory.** Every stage of a replicate is linear in `n` with a fixed number of
columns `k ≤ 7`: the DGP draws `O(n)` normals and evaluates `O(nk)` terms; each
IRLS iteration forms `X'WX` in `O(nk²)` and solves it in `O(k³)`, over a
number of iterations that depends on the data and not on `n`; each OLS solve is
`O(nk²)`. So the predicted cost of a replicate is `t = c + a·n` — linear in
`n`, plus a constant `c` of Python and pandas overhead that no sample size
amortises.

**Measurement.** `make complexity` times 25 replicates at each of eight sample
sizes from 125 to 16,000, splitting the DGP from the estimators (figure:
`results/figures/complexity.png`, data: `results/benchmarks/complexity.csv`).

Fitting `t = a·n^p`:

| fitted over | scenario A | scenario B |
|---|---|---|
| the whole grid, n ∈ [125, 16000] | p = 0.49 (R² 0.92) | p = 0.45 (R² 0.91) |
| the upper half, n ≥ 2000 | **p = 0.79** (R² 0.99) | **p = 0.75** (R² 0.99) |

A single exponent does not describe this code, and that is the finding. At
large `n` the measured exponent approaches the theoretical 1 — it falls a
little short because part of the per-replicate cost stays constant while the
rest grows. At small `n` the curve is almost flat: going from `n = 125` to
`n = 500`, four times the data, costs 1.26 times the time.

The affine model separates the two parts cleanly
(`results/benchmarks/complexity_overhead.csv`, figure:
`results/figures/overhead.png`):

| scenario | fixed cost `c` | marginal cost `a` | **fixed share at n = 500** |
|---|---|---|---|
| A | 3.93 ms | 2.7 µs per observation | **75%** |
| B | 9.50 ms | 5.3 µs per observation | **78%** |

**This is the single most important number in this document.** At the sample
size the study actually runs at, three quarters of a replicate is fixed
interpreter and pandas overhead that has nothing to do with the data. Faster
arithmetic cannot touch it. What can is removing the per-replicate Python —
doing the redundant work once instead of eight times, and performing the
remaining work across many replicates at a time rather than one.

The DGP tells the same story more sharply: its fixed share at `n = 500` is 95%
(A) and 93% (B). Drawing 500 normals is free; building a DataFrame and
evaluating term strings around them is not.

## 4. Numerical stability

`make stability-check` replays 200 replicates at each of `n ∈ {50, 100, 250,
500, 1000}` in both scenarios — 4,000 propensity fits — with every warning
recorded, and alongside them the quantities the estimators are actually exposed
to.

### Warnings: none

**Not one warning was raised in any condition.** No overflow, no divide-by-zero,
no `ConvergenceWarning`, no `LinAlgWarning`. Nor is anything hiding: across all
4,000 fits, IRLS converged every time (0% non-convergence against a limit of
100 iterations, with a worst observed count of 23 and a typical count of 4), no
design matrix was rank deficient, and the worst condition number seen was 10.2.

Two reasons for the silence, and only one of them is comfortable:

1. The design matrices really are well conditioned. The covariates are
   independent standard normals and the propensity models use bounded
   indicators `1(Vl > 0)`, so the true linear predictor takes finitely many
   values in `[-12.5, 12.5]` and `dgps.expit` is never asked to exponentiate
   anything large. The textbook instability of `1/(1 + exp(-x))` — overflow for
   `x < -709` — is latent in the code but unreachable at this design.

2. **The stability is inherited from statsmodels, not written here.** Its
   binomial link clips fitted probabilities into `[ε, 1-ε]`
   (`statsmodels/genmod/families/links.py:161`), which is why nothing ever
   divides by zero. And when IRLS exhausts its 100 iterations it sets
   `converged = False` on the results object and raises *nothing* —
   `methods.fit_propensity` returns `model.predict(X)` and discards that object,
   so a convergence failure would be silently invisible to the study. It does
   not happen here. It would not be noticed if it did.

This matters for Unit 3 rather than for Unit 2: any optimization that replaces
the statsmodels GLM with a hand-written IRLS inherits none of that protection
and must supply it deliberately.

### Exposure: real, and unreported

Warnings are the wrong instrument for this study. The estimators divide by `π`
and by `1 - π` with no trimming — the paper does not trim and neither does the
reproduction — so the risk shows up as weight concentration, not as an
exception. Figure: `results/figures/stability_weights.png`, data:
`results/benchmarks/stability_baseline_summary.csv`.

At the study's own `n = 500`:

| scenario | propensity model | median largest weight | worst largest weight | worst effective sample size |
|---|---|---|---|---|
| A | correct | 4.5 | 7.5 | 120 of 500 (24%) |
| A | false | 3.9 | 5.6 | 127 of 500 (25%) |
| B | false | 3.5 | 5.8 | 429 of 500 (86%) |
| **B** | **correct** | **20.3** | **106.6** | **74 of 500 (15%)** |

One cell is unlike the others. Scenario B with the *correctly specified*
propensity model puts a typical replicate's largest weight at 20 and its worst
at 107, and in the worst replicate the weighting leaves an effective sample of
74 observations out of 500. The cause is the design, not the code: scenario B's
true propensity coefficients `(-3, 2.5, 3, 1, -3)` put fitted probabilities as
close as 0.009 to a bound, and the median replicate's closest approach is 0.048
against 0.22 in scenario A. The correct model is the dangerous one precisely
because it reproduces that extremity faithfully, while the deliberately false
model smooths it away.

This is consistent with the published result — `DR.ofal` in scenario B has the
largest variance of any cell in Table 2 (0.289), and `ANALYSIS.md` already
observes that a false outcome model costs the doubly robust estimator
"occasional badly wrong estimates, not a wider spread" — but the study as
submitted reports no diagnostic that would reveal the mechanism.
At smaller `n` it becomes severe: at `n = 50` a fitted propensity comes within
`2.4 × 10⁻²⁰` of a bound, and 30% of replicates contain a propensity inside
`10⁻³` of one.

That these extremes do not produce astronomical weights is a property of the
design worth stating, since it is easy to assume the opposite: a unit whose
`π ≈ 0` is almost certainly untreated, so `1/π` is multiplied by `D = 0` and
never enters. The weight is large only when the realised treatment is the
improbable one, which is rare by construction. The exposure is real but it is
bounded by the same probability that creates it.

## 5. What this says about where to optimize

Ranked by the share of runtime each bottleneck controls, against the difficulty
of removing it:

1. **Cache the propensity fit within a replicate.** Removes 75% of the
   logistic regressions outright, against a stage that is 50.6% of the run.
   Pure redundancy elimination: the result is bit-identical by construction.
2. **Build each distinct design matrix once per replicate, from arrays rather
   than through a pandas namespace.** Attacks the 35.4% in `design_matrix`, the
   41.1% in pandas and the 335 column reads, and removes `with_treatment`'s
   copies with it.
3. **Batch replicates.** Only this reaches the 75% fixed cost that (1) and (2)
   reduce but cannot eliminate, since whatever remains of a replicate is still
   paid 1000 times. Running the DGP and the solves across replicates as stacked
   arrays converts per-replicate Python into per-study NumPy.
4. **Parallelise across replicates or scenarios.** Eight cores are available
   and `SeedSequence.spawn` already gives each replicate an independent,
   position-determined seed, so the work is embarrassingly parallel and
   reproducible without further care. Worth doing last: parallelising code that
   is 75% overhead buys eight copies of the overhead.
5. **Numerical stability is not currently a performance problem — it is a
   constraint on the optimizations above.** There is no instability to fix in
   the baseline. There is protection to preserve, in (1) and (3), if the
   statsmodels GLM is replaced; and there is a diagnostic gap worth closing on
   its own terms, since the study silently runs a cell whose effective sample
   size is 15% of nominal.

Expected results and the measured outcomes are in
[OPTIMIZATION.md](OPTIMIZATION.md).
