# Optimizing the Unit 2 simulation study

The study runs **24 times faster** in serial and **57 times faster** on eight
cores, and produces the same results: every one of the 20,000 estimates agrees
with its Unit 2 counterpart to better than `6e-10`, and all sixteen cells of
Table 2 still reproduce. [BASELINE.md](BASELINE.md) is the measurement this
work started from; this document is what was done about it and what it cost.

| | baseline (Unit 2) | optimized | |
|---|---|---|---|
| simulation, serial | 16.96 s | **0.70 s** | **24.1x** |
| simulation, 8 workers | — | **0.30 s** | **56.8x** |
| `make all`, end to end | 19.4 s | **2.6 s** | 7.3x |
| Python calls per replicate | 42,078 | **41** | 1,027x fewer |
| logistic fits per replicate | 8 | **2** | 4x fewer |
| time in pandas | 41% | **2.9%** | |
| runtime that does not scale with n, at n = 500 | 74–81% | **0–6%** | |
| empirical complexity at large n | `n^0.71`–`n^0.79` | **`n^1.04`** | |

The last two rows are the ones that matter most. The baseline spent three
quarters of its time on work that had nothing to do with the data; the
optimized code spends essentially all of its time on the data, which is why its
measured complexity has risen to the `O(n)` the algorithms always implied.
Going faster did not make the code scale better — it made the code's cost
*visible* as the cost of the arithmetic it performs.

## How to reproduce these numbers

```
make benchmark         # baseline vs optimized, the ladder, and core scaling
make complexity        # runtime against n, both implementations
make profile TARGET=optimized
make stability-check TARGET=optimized
make parallel          # the study on every core
make regression        # the correctness evidence in section 6
```

Hardware, versions and timing method are as in BASELINE.md: Apple M3 with 8
cores, and every figure the smallest of three runs after a discarded warm-up.

---

## 1. Cache the propensity fit within a replicate

**Problem.** The profile counted eight logistic regressions per replicate
against two distinct propensity models. Each row of `config.ESTIMATORS` names
the model it wants and `methods.estimate` fitted it from scratch, so `HT`,
`DR`, `DR.ofal` and `DR1` each refitted `ps_true` on identical data.
`fit_propensity` was 50.7% of the run and three quarters of it was recomputing
a result the replicate already held.

**Solution.** A `_Replicate` object owns the data and memoises each fit. The
cache is keyed on the model's *terms* rather than on the name it was asked
for, because the terms are what determine the fit.

```python
# before: methods.estimate dispatches, and each estimator fits its own
def estimate_ht(data, scenario, ps_spec):
    propensity = fit_propensity(data, ps_spec)      # one GLM per estimator
    ...

# after: the replicate fits each distinct model at most once
def propensity(self, spec):
    key = tuple(spec["terms"])
    fitted = self._propensities.get(key)
    if fitted is None:
        fitted = self._propensities[key] = fit_logistic(
            self.design(spec["terms"]), self.treatment, self.report)
    return fitted
```

**Impact.** Logistic fits per replicate fall from 8 to 2; the study performs
4,000 fits where it performed 16,000.

**Trade-offs.** The estimators no longer own their own fits, which is a real
loss of locality: reading `doubly_robust` no longer tells you when the
propensity model was fitted. The cache is correct only because every estimator
in a replicate sees the same data, which is a property of the design the
baseline also relied on. Bit-identical by construction.

## 2. Build each design matrix once, from arrays

**Problem.** 28 design matrices, 84 `eval` calls and **335 pandas column
reads** per replicate. `design_matrix` rebuilt a namespace over all eleven
columns of the frame on every call, whether a specification mentioned them or
not, and re-parsed each term string. `with_treatment` deep-copied the whole
DataFrame ten times per replicate to overwrite one column.

**Solution.** Three changes to the same function: compile each term string once
and cache it; build the namespace from only the names that term references; and
accept a plain `name -> array` mapping so the hot path never touches pandas.
`set_treatment` replaces `with_treatment` inside the batch and shares every
array it does not change.

```python
# before: eleven columns out of pandas, and a fresh parse, on every call
namespace = {c: data[c].to_numpy(dtype=float) for c in data.columns}
value = eval(term, {"__builtins__": {}}, namespace)

# after: compiled once, and only the names this term actually mentions
code = _compile(term)                                  # cached by the string
namespace = {name: source[name] for name in _NAMES[term] if name in source}
value = eval(code, _EMPTY_GLOBALS, namespace)
```

**Impact.** Pandas falls from 41% of internal time to 2.9%, and the DataFrame
copies disappear. Combined with optimizations 1 and 3 this is the first rung of
the ladder: **9.9x**, from 16.96 s to 1.72 s, before any batching at all.

**Trade-offs.** `design_matrix` now has two input forms, and the module-level
compile caches are global mutable state — bounded, since the keys are the
finitely many term strings in `config`, but state nonetheless. The public
DataFrame API is kept, so the Unit 2 tests still exercise the real code path.

## 3. Solve the logistic fit here, with the safeguards made explicit

**Problem.** Two, one of performance and one of correctness. statsmodels'
`GLM(...).fit()` builds a full results object per call — 16.8 million function
calls went somewhere, and 34.7% of the run was inside its IRLS. And, as
BASELINE.md documents, when that IRLS exhausts its 100 iterations it sets
`converged = False` and raises nothing; `fit_propensity` returned
`model.predict(X)` and discarded the object, so **a convergence failure was
invisible to the study.**

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

Four safeguards are deliberate, and replace protection that statsmodels had
been supplying invisibly:

1. **The update never divides by a weight.** Textbook IRLS regresses on a
   working response `eta + (d - mu) / W`, and `W = mu(1 - mu)` underflows to
   zero exactly when a fitted probability approaches a bound — which is the
   case scenario B actually produces. Fisher scoring is the same iteration
   written as an update to the coefficients, and never forms that quotient.
2. **Fitted probabilities are clipped** to `[eps, 1-eps]`, matching the
   clipping inside the statsmodels binomial link, so an inverse-probability
   weight is finite however extreme the fit.
3. **A singular system falls back** to a least-norm least-squares solve, which
   is what the baseline used throughout; Cholesky is used where it applies.
4. **Non-convergence is counted and reported.** `ConvergenceReport` travels
   through the run, is merged back from worker processes, is logged by every
   run, and raises a `RuntimeWarning` if any fit failed.

**Impact.** Part of the 9.9x first rung. The study now logs
`4000 logistic fits, 0 did not converge, worst 8 iterations, 0 fits clipped at
the bound` — the first time it has said anything at all about its own numerics.

**And it found something.** Running `make stability-check` against both
implementations over the conditions of BASELINE.md turns up a disagreement that
only the new reporting could have exposed:

| scenario | n | propensity model | baseline: not converged | optimized: not converged | largest weight, baseline / optimized |
|---|---|---|---|---|---|
| A | 50 | correct | 0% | **10.5%** | 36.5 / `4.5e15` |
| B | 50 | correct | 0% | **9.5%** | 64.9 / `4.5e15` |
| B | 100 | correct | 0% | 0.5% | 91.1 / `4.5e15` |
| B | 250 | correct | 0% | 0.5% | 103.1 / 103.1 |
| **A, B** | **500, 1000** | **both** | **0%** | **0%** | **identical** |

At small `n` these fits are separated or nearly so: the likelihood has no
maximum and the coefficients diverge. statsmodels stops on the *change in
deviance*, which goes flat while the coefficients are still running away, and
so declares convergence; this solver stops on the *change in the coefficients*,
which does not, and so reports that it failed. Neither answer is wrong about
the arithmetic, and the baseline's early stop is in effect a mild
regularisation — but the baseline reported a converged fit for a fit that had
not converged, and that is worth knowing.

It does not touch this study. At `n = 500` and `n = 1000` both implementations
converge on every replicate and agree on every weight to the last digit; the
divergence is confined to sample sizes a quarter of the design's or smaller.
The reason it is in this document rather than quietly fixed is that it is the
clearest demonstration of what the convergence reporting is for: the condition
existed in the baseline too, and nothing in the baseline would ever have said
so.

**Trade-offs.** This is the largest cost of the whole exercise, and it is paid
in trust rather than in time. statsmodels is tested by thousands of users and
this solver is tested by me. That is why `tests/test_regression.py` checks the
fit against `sm.GLM` directly on both scenarios and both propensity models, to
`1e-8`. It is also the reason the results are not bit-identical: see section 6.

## 4. Batch the replicates

**Problem.** The one BASELINE.md identified as the only route to the big win.
At `n = 500`, 75–78% of a replicate's runtime was fixed cost independent of the
data. Optimizations 1–3 make a replicate cheaper but it is still paid 1,000
times, and whatever Python remains is still paid 1,000 times.

**Solution.** A batch of 250 replicates is held as arrays with a leading
replicate axis — `(250, 500)` for each variable, `(250, 500, k)` for each
design matrix — and every operation is performed on the whole batch. NumPy's
`linalg.solve`, `linalg.cholesky` and `matmul` are all batch-aware over leading
axes, so a batch of 250 small linear systems is one call, not 250.

Two details make this safe rather than merely fast:

- **The draws stay per-replicate.** Each replicate keeps its own generator and
  the three draws happen in three passes over the batch, so every generator
  still sees covariates, then treatment, then noise, in the Unit 2 order. The
  DGP is bit-identical, which `test_dgp_is_bit_identical` asserts.
- **A converged replicate is frozen.** The batched IRLS iterates until its
  *slowest* member converges. In the first working version a replicate that
  converged early kept taking steps of order the tolerance while it waited, so
  its answer depended on which other replicates shared its batch — the study
  stopped being reproducible under a different batch size. `np.where(active,
  step, 0.0)` fixes it, and `test_batch_size_does_not_change_results` pins it.

```python
# before: one replicate at a time
for replicate in tqdm(range(n_sim)):
    records.extend(run_replicate(scenario, replicate, seeds[replicate], n))

# after: a block at a time, as stacked arrays
generators = [np.random.default_rng(seed) for seed in seeds]
columns = dgps.generate_columns(scenario, n, generators, n_sim=len(seeds))
estimates = methods.estimate_batch(columns, scenario, cells(scenario), report)
```

**Impact.** A further **2.5x** on top of the first rung, from 1.72 s to 0.70 s,
for a cumulative **24.1x**. The fixed share of a replicate's cost at `n = 500`
falls from 74–81% to 6% in scenario A and to a value indistinguishable from
zero in scenario B, and the empirical complexity rises from `n^0.71`–`n^0.79`
to `n^1.04` with an R² of 1.000 — the code is finally doing work proportional
to the data.

**Trade-offs.** This is where the code becomes harder to read. Every array has
an extra leading axis, every contraction is written with `...` leading axes,
and the scenario-A regression on the observed units alone had to be rewritten
as a 0/1-weighted regression on all of them, because subsetting would give each
replicate a different number of rows and break the rectangle. Memory rises
from one replicate's data to 250 of them; at the study's size that is about
30 MB, and `BATCH_SIZE` is the knob if it ever matters.

## 5. Parallelise across batches

**Problem.** Eight cores were idle.

**Solution.** `joblib` over batches, with `--jobs`. Seeds are derived from the
master seed by position, so which batch a replicate lands in and which process
runs it cannot affect its result. Each worker returns its own
`ConvergenceReport`, which the parent merges — a warning raised inside a worker
would never reach the parent, and a silent non-convergence is precisely what
this study was missing a way to see.

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

**Impact.** A further **2.4x**, to 0.30 s, for **56.8x** overall. But the
scaling is poor and the reason is interesting: 2.6x on four workers and
*nothing* beyond that.

| workers | seconds | speedup | efficiency |
|---|---|---|---|
| 1 | 0.72 | 1.0x | 1.00 |
| 2 | 0.41 | 1.7x | 0.86 |
| 4 | 0.27 | 2.6x | 0.66 |
| 6 | 0.28 | 2.6x | 0.43 |
| 8 | 0.28 | 2.5x | 0.32 |

Two causes. The M3 has four performance cores and four efficiency cores, so
workers five to eight are not equivalent to workers one to four. And the serial
work is now so small that process start-up and the pickling of results are a
large fraction of it — at 0.70 s there is not much left to divide.

**Trade-offs.** It is the worst return on effort of the five (section 7), and
it is the only one that makes the study's output depend on something other than
its seed — which is why `test_parallel_matches_serial` exists.

## 6. Why the results are not bit-identical

They differ, and the difference is a property of the solver rather than of the
study.

The optimized code solves the logistic fit itself and stops on a tighter
criterion than statsmodels does: `max |step| <= 1e-10` on the coefficients
against statsmodels' deviance-change test. It therefore lands **nearer** the
exact maximum likelihood estimate than the baseline did. The normal equations
are also solved by Cholesky rather than by the SVD inside `np.linalg.lstsq`,
which is a different rounding path for the same answer — safe here because
BASELINE.md measured the design matrices' worst condition number at 10.2, so
squaring it in the normal equations costs about two digits out of sixteen.

The measured disagreement across all 20,000 estimates:

| | |
|---|---|
| worst absolute difference | `5.5e-10` |
| median absolute difference | `1.2e-14` |
| smallest Monte Carlo standard error of any reported measure | `1.9e-3` |
| worst difference, in units of that standard error | **`2.9e-7`** |

A difference three million times smaller than the uncertainty the study
already reports cannot change a conclusion drawn from it. `tests/test_regression.py`
makes that argument an assertion rather than a claim: the tolerance is
expressed against the Monte Carlo standard error, not chosen for convenience.

Beyond the paired comparison it also checks, per cell, that the *distribution*
of estimates is unchanged — a two-sample Kolmogorov-Smirnov test, with a
companion test proving that test has power against a shift of a sixth of a
standard deviation, so passing it is evidence and not a formality. And it
re-runs the study's actual claim: all sixteen Table 2 cells reproduce, with the
largest absolute z for the bias still 1.1, exactly as in Unit 2.

Four things are still required to be exact, and are: the DGP, the row order of
the output file, reproducibility under any batch size or worker count, and
Unit 2's seed semantics — a scenario run alone draws what it draws in a full
run, and a short run reproduces the leading replicates of a long one.

One caveat on the scope of all this. The agreement above is measured at the
study's own `n = 500`, where the two implementations converge identically. At
`n` of 250 and below they genuinely diverge, for the reason set out at the end
of section 3: the baseline's stopping rule declares convergence on separated
fits that this one reports as failures. If this study were ever rerun at a
small sample size, the two would not agree, and the optimized answer — a fit
pushed to the edge of floating point, flagged as not converged — would be the
faithful one. The right response then would not be to restore the old stopping
rule but to decide what the study should do about separation, which is a
statistical question the paper does not address and this reproduction
inherited.

---

## 7. Lessons learned

**Which optimizations gave the best return on investment.**

Ranked by speedup per line of code changed, the order is almost the reverse of
how interesting the techniques are:

1. **Caching the propensity fit.** Nine lines. It removes three quarters of the
   logistic regressions and cannot change an answer, because it returns the
   same computation's result. If one thing in this project generalises, it is
   that the first question to ask a profile is not "what is slow" but "what is
   being computed more than once".
2. **Getting pandas out of the inner loop.** Pandas was 41% of a run that spent
   1.5% of its time in the linear algebra it exists to do. The DataFrame was
   never the data structure the inner loop wanted; it was the data structure
   the *outer* interface wanted, and the fix was to stop confusing the two.
3. **Batching.** The largest single idea and by far the largest cost in
   readability. Worth it here, and worth it mainly because BASELINE.md had
   already proved that 75% of the runtime was fixed overhead — without that
   measurement it would have been a guess.
4. **Writing the logistic solver.** Bought speed and bought back the
   convergence reporting the study was missing, but it is the piece I would
   most want someone else to review.
5. **Parallelism.** Last, and it earned its place there.

**What surprised me about where the time actually went.**

Three things, in increasing order of embarrassment.

*The study barely did any arithmetic.* I expected the logistic regressions to
dominate because they are the only iterative thing in the study. They did
dominate — but `np.linalg.lstsq` was 1.5% of the run and pandas was 41%. The
cost was never the mathematics; it was the eleven-column namespace rebuilt 28
times a replicate to get at the mathematics.

*`np.einsum` was two thirds of my first optimized version.* Having written the
batched code with `einsum` because it expresses a batched contraction so
cleanly, I profiled it and found 2.1 s of 3.2 s inside `c_einsum`. einsum
evaluates a contraction in its own C loop; `matmul` dispatches a stack of them
to BLAS. Swapping them — the same arithmetic, four lines — gave **2.3x** on top
of everything else. The lesson is the one the whole unit is about: I had
optimized by reasoning about what *ought* to be fast and skipped the
measurement, in the middle of a project about measuring.

*The stable logistic underflows on purpose, and that looks like a bug.* Running
the stability audit on the optimized code at first produced hundreds of
`underflow encountered in logaddexp` warnings that the baseline never raised.
They are the function working: an extreme linear predictor is supposed to
underflow to exactly 0 or 1, which is the correct limit, rather than overflow
to an infinity, which is what the naive form does. The audit now sets
`np.seterr(under="ignore")` deliberately and says why. Distinguishing the
benign floating-point event from the dangerous one took longer than writing
the function.

*My own `expit` then became the single largest line.* Having removed everything
else, the profile pointed at the overflow-safe logistic function I had written
with two boolean masks and four fancy-indexed passes. `exp(-logaddexp(0, -x))`
is three ufunc passes, equally overflow-free, and agrees with the masked
version to one unit in the last place. Optimization reveals the next
bottleneck, and the next bottleneck is sometimes the thing you added.

**Which optimizations were not worth the effort.**

*Parallelism, at this study's size.* It is 2.4x for a dependency, a code path
that only runs in worker processes, the machinery to merge convergence reports
across processes, and a test to prove workers change nothing. All of that 2.4x
is available from four cores; the other four cores contribute nothing at
all. Against the 24x already won in serial, it is the rung I would drop
first — and I would keep it only because it costs nothing to leave in and
because the study will be run at larger `n_sim` eventually, where the fixed
cost of starting workers is amortised and the picture changes.

*Chasing the batch size.* Runtime is flat for `BATCH_SIZE` anywhere between
about 50 and 1000; the knob exists for memory, not for speed.

**What I would do differently.** Measure first, which I did, and then keep
measuring, which I did not — the einsum mistake cost a 2.3x for about an hour
of work that a profile would have found in a minute. And the thing I am most
glad of is not a speedup at all: the baseline could have failed to converge and
said nothing, and now it cannot.

## 8. Profiling evidence, and figures

Re-profiling the optimized code under the same harness — `make profile
TARGET=optimized`, 200 replicates per scenario, full output in
`results/profiles/optimized_profile.txt` — gives the other half of the tables
in [BASELINE.md](BASELINE.md) section 2.

**Internal time by package**, a partition of the run with every frame counted
once:

| package | baseline | optimized |
|---|---|---|
| pandas | **41.3%** | **2.9%** |
| numpy | 28.3% | 12.2% |
| statsmodels | 15.8% | — (gone) |
| Python stdlib and builtins | 10.9% | 1.5% |
| this study's own code | 3.7% | **83.4%** |

This is the single table that says the exercise worked. The baseline spent 41%
of its time in a data frame library and 3.7% in its own code; the optimized
run spends 83.4% in its own code, and what remains in numpy is the linear
algebra that code exists to perform. The share in numpy *fell* while the
arithmetic stayed the same, because the baseline's numpy time was largely
thousands of tiny array constructions rather than the solves.

**Cumulative time by stage**, which nest rather than partition:

| stage | baseline | optimized |
|---|---|---|
| all estimators | 90.6% | 79.8% |
| — of which the logistic fit | 50.6% | 45.4% |
| — of which the least-squares solve | 1.5% | 11.2% |
| draw the data | 9.0% | 12.8% |
| `design_matrix` (DGP + estimators) | 35.5% | 6.9% |

The *shape* barely moved: the logistic fit was half the run before and is
nearly half the run now. What changed is what a replicate costs in absolute
terms, and the fact that the solve has risen from 1.5% to 11.2% is the
clearest sign that the overhead around it is gone — the same arithmetic is now
a visible share of a run twenty-four times shorter. `design_matrix` falling
from 35.5% to 6.9% is optimization 2, and the 16.8 million function calls for
these 400 replicates are now 16,387.

| figure | what it shows |
|---|---|
| `results/figures/benchmark.png` | runtime against simulation size, and the ladder of optimizations |
| `results/figures/complexity.png` | runtime against n, log-log, both implementations, with fitted exponents |
| `results/figures/overhead.png` | the fixed share of a replicate's cost, before and after |
| `results/figures/speedup.png` | parallel speedup against worker count, against the linear ideal |
| `results/figures/stability_weights.png` | the numerical exposure documented in BASELINE.md |
| `results/figures/budget.png` | the replicate counts of [EXTENSIONS.md](EXTENSIONS.md) |
