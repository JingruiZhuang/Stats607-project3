# Simulation budget: trading computational cost against statistical precision

*Optional extension: **Simulation Budget Optimization**.*

The study runs 1,000 replicates of every cell, at every sample size, whatever
the cell's variance. This document measures what that costs and what it should
have been.

The result in one line: **sampling variance falls like `1/n` while cost per
replicate rises like `n`, so once the replicate count is allowed to follow the
precision it buys, the cost of this study stops depending on `n` at all.**

| holding the precision the study reports at `n = 500` | uniform 1,000 | budgeted | |
|---|---|---|---|
| simulation at `n = 500` | 0.79 s | 0.61 s | 23% less |
| simulation at `n = 1000` | 1.46 s | 0.41 s | 72% less |
| simulation at `n = 2000` | 2.95 s | 0.37 s | 87% less |
| simulation at `n = 4000` | 5.78 s | **0.37 s** | **94% less** |

At the study's own `n = 500` the saving is smaller but real: scenario A needs
475 replicates, not 1,000, which is 19% off `make simulate` and `make analyze`
together for identical reported precision.

Measured by `make budget` (`src/budget.py`), which runs a 2,000-replicate pilot
at each of `n ∈ {125, 250, 500, 1000, 2000, 4000}` and times the same grid.
Hardware and timing method as in [BASELINE.md](BASELINE.md). Nothing in
`config.py` changes: the recommendation is measured and reported, not applied.

## 1. The statistical side: variance falls like 1/n

**Theory.** A root-`n` consistent estimator has sampling variance `∝ 1/n`, so
the Monte Carlo standard error of a mean over `n_sim` replicates is
`sd/√n_sim ∝ 1/√(n·n_sim)`. Holding that fixed means `n_sim ∝ 1/n`.

**Measurement.** Fitting `sd² = a·n^-q` per cell:

| | fitted `q` | R² |
|---|---|---|
| scenario A, all 8 cells | 0.973 – 1.036 | ≥ 0.998 |
| scenario B, 10 of its 12 | 1.002 – 1.064 | ≥ 0.996 |
| scenario B, `DR.ofal`, `DR1.ofal` | 1.18, 1.26 | ≥ 0.988 |

The theory holds for every cell but two, which fall *faster* than `1/n` — the
signature of section 4 fading out.

Variance also differs as much *across* cells at one `n` as across a four-fold
change in `n`: at `n = 500` the standard deviations run from 0.203 (A, `OR`) to
0.585 (B, `DR1.ofal`), a factor of 2.9 and so 8.3 in the replicates needed.

## 2. The computational side: cost rises like n

Seconds per replicate:

| `n` | scenario A | scenario B | B / A |
|---|---|---|---|
| 125 | 0.096 ms | 0.278 ms | 2.9 |
| 500 | 0.292 ms | 0.498 ms | 1.7 |
| 1000 | 0.530 ms | 0.932 ms | 1.8 |
| 4000 | 1.97 ms | 3.81 ms | 1.9 |

Linear in `n` (`n^1.04`, measured in [OPTIMIZATION.md](OPTIMIZATION.md)), and
scenario B costs about twice scenario A: twelve estimator cells to A's eight.
So the cells needing the most replicates sit in the scenario whose replicates
are the most expensive.

## 3. Putting them together

Replicates are shared within a scenario — every estimator sees the same
dataset, by design — so the budget has two numbers in it, not twenty, and each
scenario must satisfy its most demanding cell.

The target is the precision the uniform design actually delivers in its worst
cell, `0.0185` on the mean bias: the budgeted design is then required to be no
worse anywhere, which makes the two comparable.

**At `n = 500`,** each reported measure asks for a different count:

| measure | target | scenario A needs | scenario B needs |
|---|---|---|---|
| mean bias | 0.0185 | 368 | 1,000 |
| variance | 0.0153 | 137 | 1,000 |
| median bias | 0.0210 | 350 | 1,000 |
| interquartile range | 0.0298 | **475** | 1,000 |

All four come from the same replicates, so the design must serve the most
demanding: **475 for scenario A, 1,000 for scenario B**. Sizing on the mean
alone would have given 368 and left the interquartile range noisier than the
table implies. Scenario B is pinned at 1,000 by construction, since the target
was its worst cell's precision there.

Running it confirms the arithmetic: no cell exceeds its target by more than 2%
on any of the four measures (worst ratios 1.008, 1.017, 0.957, 0.968).

| at `n = 500` | uniform | budgeted | |
|---|---|---|---|
| `make simulate` | 0.710 s | 0.586 s | 17% less |
| `make analyze` | 0.877 s | 0.704 s | 20% less |
| together | 1.588 s | **1.290 s** | **19% less** |

**Across sample sizes,** holding that same target and letting the count follow:

| `n` | A | B | budgeted | uniform | saving |
|---|---|---|---|---|---|
| 125 | 1,544 | 2,350 | 0.80 s | 0.37 s | **−114%** |
| 250 | 660 | 1,310 | 0.82 s | 0.71 s | −16% |
| 500 | 368 | 1,000 | 0.61 s | 0.79 s | 23% |
| 1000 | 174 | 339 | 0.41 s | 1.46 s | 72% |
| 2000 | 80 | 149 | 0.37 s | 2.95 s | 87% |
| 4000 | 43 | 75 | 0.37 s | 5.78 s | 94% |

The budgeted column is flat. That is sections 1 and 2 cancelling: replicates
needed fall like `1/n`, each costs `n^1.04`, so the study costs `n^0.04`. The
uniform design instead pays in proportion to `n` for precision it never
reports.

The negative savings at small `n` are the same fact backwards: there the
estimators are *more* dispersed, so matching the precision the study reports at
`n = 500` takes more than 1,000 replicates, not fewer. A uniform 1,000 at
`n = 125` is not a bargain — it is a looser study.

## 4. Where this breaks down

Ten cell–sample-size points, over six cells, have no usable variance at all:
scenario A's `DR` and `DR.ofal` at `n = 125`, and scenario B's `DR`, `DR.ofal`,
`DR1`, `DR1.ofal` at `n = 125` and `n = 250`. Their sample standard deviations
reach `10^13`, with one replicate in 2,000 carrying **99.95%** of the total
squared deviation. The cause is the one in [BASELINE.md](BASELINE.md) section
4: at small `n` the false propensity model produces fitted probabilities near
zero and the inverse-probability weight explodes.

A replicate count derived from such a variance is meaningless, and the
derivation will happily produce one — before this guard, it reported precision
targets of order `10^12` and recommended a single replicate for scenario A.
`src/budget.py` excludes any cell where the largest squared deviation exceeds
5% of the total, and reports how many it excluded rather than dropping them
silently.

So at `n ≤ 250` the scenario-B doubly robust cells have no finite-sample
variance that 1,000 replicates can estimate, and only the median and
interquartile range are worth budgeting for. At the study's own `n = 500` the
worst such share across all twenty cells is 0.021.

## 5. What I did not change

`config.py` still says `N_SIM = 1000`. A study with 475 replicates in scenario
A is a different study with different numbers in it, and the correctness
argument of this unit rests on the optimized code computing the *same* study.
The recommendation is reported and left for the next one to take or leave;
`simulation.run` already accepts a per-call `n_sim`, so taking it needs no new
machinery — one call per scenario rather than one for both.

## 6. Figure

![budget](../results/figures/budget.png)

- **left** — each cell's sampling variance against `n`, log-log, against the
  `1/n` line. The carets are the section 4 cells, far off the scale.
- **middle** — the replicate count holding the study's precision as `n` grows,
  against `1/n`. Scenario B crosses the uniform 1,000 exactly at `n = 500`, by
  construction; everything right of that is over-provision.
- **right** — what that does to the cost. The uniform design climbs with `n`;
  the budgeted one is flat.
