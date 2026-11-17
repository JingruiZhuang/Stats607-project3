# Optimizing the Unit 2 simulation study

Five optimizations, measured against `baseline/`, the frozen Unit 2 code.
Hardware, versions and timing method as in [BASELINE.md](BASELINE.md); every
timing is the smallest of three runs after a warm-up.

## 1. Cache the propensity fit within a replicate

**Problem.** Eight logistic regressions per replicate against only two distinct
propensity models. Each row of `config.ESTIMATORS` named the model it wanted
and `methods.estimate` fitted it from scratch, so `HT`, `DR`, `DR.ofal` and
`DR1` each refitted `ps_true` on identical data. `fit_propensity` was 50.6% of
the run, three quarters of it recomputation.

**Solution.** A `_Replicate` object owns the data and memoises each fit, keyed
on the model's *terms* rather than its name, since the terms determine the fit.

```python
# before: each estimator fits its own
def estimate_ht(data, scenario, ps_spec):
    propensity = fit_propensity(data, ps_spec)      # one GLM per estimator

# after: the replicate fits each distinct model at most once
def propensity(self, spec):
    key = tuple(spec["terms"])
    fitted = self._propensities.get(key)
    if fitted is None:
        fitted = self._propensities[key] = fit_logistic(
            self.design(spec["terms"]), self.treatment, self.report)
    return fitted
```

**Impact.** Logistic fits per replicate fall from 8 to 2: 4,000 fits where the
study performed 16,000.

**Trade-offs.** The estimators no longer own their fits, so reading
`doubly_robust` no longer tells you when the propensity model was fitted. The
cache is correct only because every estimator in a replicate sees the same
data — a property of the design the baseline also relied on. Bit-identical by
construction.

## 2. Build each design matrix once, from arrays

**Problem.** Per replicate: 28 design matrices, 84 `eval` calls, **335 pandas
column reads**. `design_matrix` rebuilt a namespace over all eleven columns on
every call and re-parsed each term string. `with_treatment` deep-copied the
whole DataFrame ten times per replicate to overwrite one column.

**Solution.** Compile each term string once and cache it; build the namespace
from only the names that term references; accept a plain `name -> array`
mapping so the hot path never touches pandas. `set_treatment` replaces
`with_treatment` and shares every array it does not change.

```python
# before: eleven columns out of pandas, and a fresh parse, on every call
namespace = {c: data[c].to_numpy(dtype=float) for c in data.columns}
value = eval(term, {"__builtins__": {}}, namespace)

# after: compiled once, and only the names this term mentions
code = _compile(term)                                  # cached by the string
namespace = {name: source[name] for name in _NAMES[term] if name in source}
value = eval(code, _EMPTY_GLOBALS, namespace)
```

**Impact.** Pandas falls from 41% of internal time to 2.9% and the DataFrame
copies disappear. With optimizations 1 and 3 this is the first rung of the
ladder: **9.9x**, 16.96 s to 1.72 s, before any batching.

**Trade-offs.** `design_matrix` now has two input forms, and the compile caches
are global mutable state — bounded, since the keys are the finitely many term
strings in `config`, but state nonetheless. The DataFrame API is kept, so the
Unit 2 tests still exercise the real code path.

## 3. Solve the logistic fit here, with the safeguards made explicit

**Problem.** Two. statsmodels' `GLM(...).fit()` builds a full results object
per call, and 34.8% of the run was inside its IRLS. And when that IRLS exhausts
its 100 iterations it sets `converged = False` and raises nothing;
`fit_propensity` returned `model.predict(X)` and discarded the object, so **a
convergence failure was invisible to the study.**

**Solution.** Fisher scoring, written to fit a stack of replicates at once:

```python
# before: a statsmodels GLM per estimator, results object discarded
model = sm.GLM(data["D"].to_numpy(dtype=float), X,
               family=sm.families.Binomial()).fit()
return model.predict(X)

# after: beta <- beta + (X'WX)^-1 X'(d - mu), over the whole batch
mu = np.clip(dgps.expit(_matvec(X, beta)), EPS, 1.0 - EPS)
gram, _ = _weighted_normal_equations(X, outcome, mu * (1.0 - mu))
step = np.where(active[..., None], _solve(gram, _matvec(X.swapaxes(-1, -2),
                                                        outcome - mu)), 0.0)
beta = beta + step
active = np.max(np.abs(step), axis=-1) > IRLS_TOLERANCE
```

Four safeguards replace protection statsmodels had supplied invisibly:

1. **No division by a weight.** Textbook IRLS regresses on a working response
   `eta + (d - mu) / W`, and `W = mu(1 - mu)` underflows to zero exactly when a
   fitted probability approaches a bound — which scenario B produces. Fisher
   scoring is the same iteration written as a coefficient update and never
   forms that quotient.
2. **Fitted probabilities are clipped** to `[eps, 1-eps]`, matching the
   statsmodels binomial link, so an inverse-probability weight stays finite.
3. **A singular system falls back** to a least-norm least-squares solve;
   Cholesky is used where it applies.
4. **Non-convergence is counted and reported.** `ConvergenceReport` travels
   through the run, merges back from worker processes, is logged, and raises a
   `RuntimeWarning` if any fit failed.

**Impact.** Part of the 9.9x first rung. The study now logs `4000 logistic
fits, 0 did not converge, worst 8 iterations, 0 fits clipped at the bound` —
the first time it has said anything about its own numerics. That reporting
immediately found something: at small `n` the two implementations disagree.

| scenario | n | not converged: baseline / optimized |
|---|---|---|
| A | 50 | 0% / **10.5%** |
| B | 50 | 0% / **9.5%** |
| B | 100, 250 | 0% / 0.5% |
| **A, B** | **500, 1000** | **0% / 0%, every weight identical** |

These fits are separated: the likelihood has no maximum and the coefficients
diverge. statsmodels stops on the change in *deviance*, which goes flat while
the coefficients run away; this solver stops on the change in the
*coefficients*, which does not. The baseline reported a converged fit for one
that had not converged. It does not touch this study — at `n = 500` both agree
to the last digit — but nothing in the baseline would ever have said so.

**Trade-offs.** The largest cost of the exercise, paid in trust rather than
time: statsmodels is tested by thousands of users, this solver by me, which is
why `tests/test_regression.py` checks it against `sm.GLM` directly on both
scenarios and both propensity models to `1e-8`.

*Precision — and why the results are not bit-identical.* This solver stops on
`max |step| <= 1e-10` against statsmodels' deviance test, so it lands **nearer**
the exact maximum likelihood estimate; it also solves by Cholesky rather than
the SVD inside `lstsq`, a different rounding path for the same answer, safe
because the worst design-matrix condition number is 10.2. Across all 20,000
estimates:

| | |
|---|---|
| worst absolute difference | `5.5e-10` |
| median absolute difference | `1.2e-14` |
| smallest Monte Carlo standard error reported | `1.9e-3` |
| worst difference, in units of that standard error | **`2.9e-7`** |

A difference three million times smaller than the study's own uncertainty
cannot change a conclusion. `tests/test_regression.py` expresses the tolerance
against that standard error rather than a convenient constant, checks each
cell's *distribution* with a two-sample Kolmogorov-Smirnov test (plus a
companion test proving it has power against a shift of a sixth of a standard
deviation), and confirms all sixteen Table 2 cells still reproduce with the
largest absolute z for the bias still 1.1. Four things remain exact: the DGP,
the output row order, reproducibility under any batch size or worker count, and
Unit 2's seed semantics.

## 4. Batch the replicates

**Problem.** At `n = 500`, 74–81% of a replicate was fixed cost independent of
the data. Optimizations 1–3 make a replicate cheaper, but whatever Python
remains is still paid 1,000 times.

**Solution.** A batch of 250 replicates held as arrays with a leading replicate
axis — `(250, 500)` per variable, `(250, 500, k)` per design matrix — with
every operation performed on the whole batch. `linalg.solve`, `linalg.cholesky`
and `matmul` are batch-aware over leading axes, so 250 small linear systems are
one call.

```python
# before: one replicate at a time
for replicate in tqdm(range(n_sim)):
    records.extend(run_replicate(scenario, replicate, seeds[replicate], n))

# after: a block at a time, as stacked arrays
generators = [np.random.default_rng(seed) for seed in seeds]
columns = dgps.generate_columns(scenario, n, generators, n_sim=len(seeds))
estimates = methods.estimate_batch(columns, scenario, cells(scenario), report)
```

Two details make it safe rather than merely fast. **The draws stay
per-replicate**: each replicate keeps its own generator and the three draws
happen in three passes, so every generator still sees covariates, treatment,
noise in Unit 2 order. **A converged replicate is frozen**: the batched IRLS
runs until its slowest member converges, and in the first version a replicate
that converged early kept taking steps of order the tolerance while it waited,
so its answer depended on which replicates shared its batch.
`np.where(active, step, 0.0)` fixes it.

**Impact.** A further **2.5x**, 1.72 s to 0.70 s, cumulative **24.1x**. The
fixed share of a replicate at `n = 500` falls from 74–81% to 6% in scenario A
and to indistinguishable from zero in B, and empirical complexity rises from
`n^0.71`–`n^0.79` to `n^1.04` (R² 1.000) — the code is finally doing work
proportional to the data.

**Trade-offs.** Where the code becomes harder to read: every array gains a
leading axis, every contraction is written with `...` leading axes, and
scenario A's regression on the observed units had to be rewritten as a
0/1-weighted regression on all of them, since subsetting would give each
replicate a different number of rows. Memory rises from one replicate's data to
250 — about 30 MB here, and `BATCH_SIZE` is the knob if it matters.

## 5. Parallelise across batches

**Problem.** Eight cores were idle.

**Solution.** `joblib` over batches, with `--jobs`. Seeds are derived from the
master seed by position, so which batch a replicate lands in and which process
runs it cannot affect its result. Each worker returns its own
`ConvergenceReport` for the parent to merge, since a warning raised inside a
worker would never reach the parent.

```python
# before: every batch run in this process, one after another
blocks = [run_batch(scenario, n, seeds, start, report)
          for scenario, n, seeds, start in work]

# after: batches handed to workers, each returning its own convergence counts
results = Parallel(n_jobs=jobs, backend="loky")(
    delayed(_batch_with_report)(*arguments) for arguments in work)
for block, counts in results:
    report.fits += counts.fits
    report.failures += counts.failures
```

**Impact.** A further **2.4x**, to 0.30 s, for **56.8x** overall — but the
scaling is poor: 2.6x on four workers and nothing beyond.

| workers | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| seconds | 0.72 | 0.41 | 0.27 | 0.28 | 0.28 |
| speedup | 1.0x | 1.7x | 2.6x | 2.6x | 2.5x |
| efficiency | 1.00 | 0.86 | 0.66 | 0.43 | 0.32 |

Two causes: the M3 has four performance and four efficiency cores, so workers
five to eight are not equivalent to one to four; and at 0.70 s the serial work
left to divide is comparable to process start-up and result pickling.

**Trade-offs.** The worst return on effort of the five (section 7), and the
only one that makes the output depend on something other than its seed — which
is why `test_parallel_matches_serial` exists.

## 6. Profiling evidence

| | baseline | optimized | |
|---|---|---|---|
| simulation, serial | 16.96 s | **0.70 s** | **24.1x** |
| simulation, 8 workers | — | **0.30 s** | **56.8x** |
| `make all`, end to end | 19.4 s | **2.6 s** | 7.3x |
| Python calls per replicate | 42,078 | **41** | 1,027x fewer |
| time in pandas | 41% | **2.9%** | |
| fixed runtime share at n = 500 | 74–81% | **0–6%** | |
| empirical complexity at large n | `n^0.71`–`n^0.79` | **`n^1.04`** | |

`make profile TARGET=optimized` re-runs the baseline's harness — 200 replicates
per scenario — giving the other half of the tables in BASELINE.md section 2.

**Internal time by package** (a partition):

| package | baseline | optimized |
|---|---|---|
| pandas | **41.3%** | **2.9%** |
| numpy | 28.3% | 12.2% |
| statsmodels | 15.8% | — (gone) |
| Python stdlib and builtins | 10.9% | 1.5% |
| this study's own code | 3.7% | **83.4%** |

This is the table that says the exercise worked: 41% in a data frame library
and 3.7% in its own code becomes 83.4% in its own code, with what remains in
numpy being the linear algebra that code exists to perform. The numpy share
*fell* while the arithmetic stayed the same, because the baseline's numpy time
was mostly building tiny arrays.

**Cumulative time by stage** (these nest):

| stage | baseline | optimized |
|---|---|---|
| all estimators | 90.6% | 79.8% |
| — of which the logistic fit | 50.6% | 45.4% |
| — of which the least-squares solve | 1.5% | 11.2% |
| draw the data | 9.0% | 12.8% |
| `design_matrix` (DGP + estimators) | 35.5% | 6.9% |

The *shape* barely moved — the logistic fit was half the run and still is. What
changed is the absolute cost. The solve rising from 1.5% to 11.2% is the
clearest sign the overhead around it is gone: the same arithmetic is now a
visible share of a run 24 times shorter. 16.8 million function calls for these
400 replicates are now 16,387.

| figure | what it shows |
|---|---|
| `benchmark.png` | runtime against simulation size, and the ladder of optimizations |
| `complexity.png` | runtime against n, log-log, both implementations, fitted exponents |
| `speedup.png` | parallel speedup against worker count, against the linear ideal |

## 7. Lessons learned

**Best return on investment.** Ranked by speedup per line changed — almost the
reverse of how interesting the techniques are:

1. **Caching the propensity fit.** Nine lines, three quarters of the logistic
   regressions gone, cannot change an answer. The first question to ask a
   profile is not "what is slow" but "what is computed twice".
2. **Getting pandas out of the inner loop.** The DataFrame was what the outer
   interface wanted, never what the inner loop wanted.
3. **Batching.** The largest idea and the largest cost in readability. Worth it
   because BASELINE.md had already proved three quarters of the runtime was
   fixed overhead; without that measurement it was a guess.
4. **Writing the logistic solver.** Bought speed, and bought back the
   convergence reporting the study lacked — but it is the piece I would most
   want someone else to review.
5. **Parallelism.** Last, and it earned its place there.

**What surprised me.**

*The study barely did any arithmetic.* `np.linalg.lstsq` was 1.5% of the run
and pandas was 41%. The cost was never the mathematics; it was the
eleven-column namespace rebuilt 28 times a replicate to reach it.

*`np.einsum` was two thirds of my first batched version.* einsum runs its own C
loop; `matmul` dispatches a stack to BLAS. Swapping them — same arithmetic,
four lines — gave **2.3x**. I had optimized by reasoning about what ought to be
fast, in the middle of a project about measuring.

*The next bottleneck is sometimes the thing you added.* With everything else
gone the profile pointed at my own overflow-safe `expit`, which
`exp(-logaddexp(0, -x))` replaced at three ufunc passes. That version
underflows by design, which first looked like a bug: telling the benign
floating-point event from the dangerous one took longer than writing the
function.

**Not worth the effort.** *Parallelism at this size* — 2.4x for a dependency, a
worker-only code path, cross-process report merging and a test to prove workers
change nothing, with all of that 2.4x available from four cores, against the
24x already won in serial. *Chasing the batch size* — runtime is flat between
about 50 and 1000, so the knob exists for memory, not speed.
