"""Estimators: Horvitz-Thompson, outcome regression, and doubly robust.

The estimators themselves are unchanged from Unit 2.  What changed is how they
are computed: the propensity model is fitted once per replicate rather than
once per estimator that asks for it, the design matrices are built once and
reused, and every solve is performed for all replicates at once as a stack of
small systems.

The logistic fit is now solved here rather than by statsmodels, so the
safeguards statsmodels provided implicitly are supplied explicitly: fitted
probabilities are clipped away from the bounds, the update is written so that
it never divides by a weight that can underflow, a singular system falls back
to a least-norm solve, and -- unlike statsmodels, which records a failure to
converge and raises nothing -- a replicate that does not converge is counted
and reported.  See docs/OPTIMIZATION.md.
"""

import logging
import warnings

import numpy as np

from . import config, dgps

logger = logging.getLogger(__name__)

# Fitted probabilities are held this far from 0 and 1, matching the clipping
# statsmodels applies inside its binomial link.  It bounds an inverse-probability
# weight at 1/EPS rather than at infinity; it does not make such a weight
# meaningful, and `src/stability.py` reports the weights regardless.
EPS = float(np.finfo(float).eps)

# IRLS stops when no coefficient moves by more than this.  Tighter than the
# deviance-based test statsmodels stops on, so the fit lands nearer the exact
# maximum likelihood estimate than the baseline's did rather than merely near
# the baseline's answer.
IRLS_TOLERANCE = 1e-10
IRLS_MAX_ITERATIONS = 100


class ConvergenceReport:
    """How a batch of logistic fits behaved, for the caller to act on.

    The baseline threw this information away: statsmodels sets ``converged``
    on a results object that ``fit_propensity`` discarded, and raises nothing
    when IRLS exhausts its iterations.  Collecting it costs nothing and turns a
    silent failure into a reported one.
    """

    def __init__(self):
        self.fits = 0
        self.failures = 0
        self.max_iterations = 0
        self.clipped_fits = 0

    def record(self, converged, iterations, clipped):
        self.fits += int(np.size(converged))
        self.failures += int(np.size(converged) - np.count_nonzero(converged))
        self.max_iterations = max(self.max_iterations, int(iterations))
        self.clipped_fits += int(np.count_nonzero(clipped))

    def warn_if_needed(self):
        if self.failures:
            warnings.warn(
                f"{self.failures} of {self.fits} logistic fits did not converge "
                f"within {IRLS_MAX_ITERATIONS} iterations", RuntimeWarning,
                stacklevel=2)
        return self

    def __repr__(self):
        return (f"ConvergenceReport(fits={self.fits}, failures={self.failures}, "
                f"max_iterations={self.max_iterations}, "
                f"clipped_fits={self.clipped_fits})")


def _solve(a, b):
    """Solve a stack of symmetric positive-definite systems ``a x = b``.

    Cholesky is the right factorisation for the normal equations and for the
    Fisher information, both of which are symmetric and positive definite
    whenever the design has full rank and the weights are positive.  Where that
    fails -- a separated logistic fit drives its weights to zero and the
    information matrix towards singularity -- the offending systems fall back to
    a least-norm least-squares solve, which is what the baseline used
    throughout.  The fallback is per-system, so one bad replicate does not slow
    the rest.

    ``a`` is ``(..., k, k)`` and ``b`` is ``(..., k)``.
    """
    try:
        factor = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        factor = None

    if factor is not None:
        solved = np.linalg.solve(a, b[..., None])[..., 0]
        if np.all(np.isfinite(solved)):
            return solved

    # Reached only when a system is singular or produced a non-finite answer.
    flat_a = a.reshape(-1, a.shape[-2], a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    out = np.empty_like(flat_b)
    for i in range(flat_a.shape[0]):
        out[i] = np.linalg.lstsq(flat_a[i], flat_b[i], rcond=None)[0]
    return out.reshape(b.shape)


def _matvec(X, beta):
    """``X @ beta`` for a stack of designs and a stack of coefficient vectors.

    Written as a matmul rather than as ``np.einsum``: einsum evaluates a
    contraction in its own C loop, while matmul dispatches a stack of these to
    the batched BLAS routine underneath NumPy.  Profiling the first version of
    this module put two thirds of its time inside ``c_einsum``; the same
    arithmetic through matmul is several times faster and, being BLAS, is also
    the better-tested path.
    """
    return np.matmul(X, beta[..., None])[..., 0]


def _weighted_normal_equations(X, y, weights=None):
    """``(X' W X, X' W y)`` for a stack of designs, without forming ``W``.

    Weights enter as a vector multiplying the rows, so an ``n x n`` diagonal is
    never materialised.  A 0/1 weight vector fits on a subset of the rows,
    which is how scenario A regresses on the observed units alone while keeping
    every replicate the same rectangular shape -- a subset selection would give
    each replicate a different number of rows and so could not be batched.
    """
    weighted = X if weights is None else X * weights[..., None]
    transposed = np.swapaxes(weighted, -1, -2)
    return np.matmul(transposed, X), _matvec(transposed, y)


def solve_ols(X, y, weights=None):
    """Least-squares coefficients for a stack of regressions."""
    gram, moment = _weighted_normal_equations(X, y, weights)
    return _solve(gram, moment)


def fit_logistic(X, outcome, report=None):
    """Maximum-likelihood logistic fit for a stack of designs.

    Fisher scoring: ``beta <- beta + (X' W X)^-1 X' (d - mu)`` with
    ``W = mu (1 - mu)``.  This is the same iteration as iteratively reweighted
    least squares and reaches the same estimate, but it is written as an update
    to the coefficients rather than as a regression on a working response
    ``eta + (d - mu) / W``.  That matters numerically: the working response
    divides by a weight that underflows to zero exactly when a fitted
    probability approaches a bound, which is the case the study's scenario B
    actually produces.  This form never divides by ``W`` at all.

    ``X`` is ``(..., n, k)`` and ``outcome`` is ``(..., n)``; the leading axes
    are replicates and are fitted independently and simultaneously.
    """
    X = np.asarray(X, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    beta = np.zeros(X.shape[:-2] + (X.shape[-1],))

    # A replicate stops moving the moment it converges, even though the batch
    # iterates until its slowest member is done.  Without this, a replicate
    # that converged early would keep taking steps of order the tolerance while
    # it waited, and its answer would depend on which other replicates happened
    # to share its batch -- so a run would not reproduce under a different
    # batch size or a different number of workers.  Freezing them makes the
    # result a property of the replicate alone, as it was in Unit 2.
    active = np.ones(X.shape[:-2], dtype=bool)
    iterations = 0
    for iterations in range(1, IRLS_MAX_ITERATIONS + 1):
        mu = dgps.expit(_matvec(X, beta))
        mu = np.clip(mu, EPS, 1.0 - EPS)
        weights = mu * (1.0 - mu)

        gram, _ = _weighted_normal_equations(X, outcome, weights)
        score = _matvec(np.swapaxes(X, -1, -2), outcome - mu)
        step = np.where(active[..., None], _solve(gram, score), 0.0)

        beta = beta + step
        active = np.max(np.abs(step), axis=-1) > IRLS_TOLERANCE
        if not np.any(active):
            break
    converged = ~active

    fitted = np.clip(dgps.expit(_matvec(X, beta)), EPS, 1.0 - EPS)
    if report is not None:
        clipped = np.any((fitted <= EPS) | (fitted >= 1.0 - EPS), axis=-1)
        report.record(converged, iterations, clipped)
    return fitted


def fit_propensity(data, spec, report=None):
    """Fit a logistic model for ``D`` and return its fitted probabilities.

    The design matrix is built entirely from ``spec["terms"]``, so a model has
    an intercept exactly when its term list opens with ``"1"``.  Table 1 writes
    the two false propensity models without an intercept, but ``config`` gives
    them one: fitted literally they miss the published biases, and ``ADEMP.md``
    records why the intercept is kept.  Fitted probabilities are clipped only
    at the edge of floating point; the paper does not trim weights and neither
    does this.
    """
    columns = dgps._columns(data) if not isinstance(data, dict) else data
    X = dgps.design_matrix(None, spec["terms"], columns)
    return fit_logistic(X, columns["D"], report)


# ------------------------------------------------------------------ batched

class _Replicate:
    """The matrices one scenario needs, built once and shared by its estimators.

    The baseline rebuilt every design matrix inside every estimator, so a
    replicate built 28 of them and refitted its propensity model eight times.
    Each distinct matrix and each distinct propensity fit is computed here once
    and handed to whichever estimators ask for it.  Everything held is an array
    over all replicates at once.
    """

    def __init__(self, columns, scenario, report=None):
        self.columns = columns
        self.scenario = scenario
        self.report = report
        self.treatment = columns["D"]
        self.outcome = columns["Y"]
        self._designs = {}
        self._propensities = {}

        if scenario == "A":
            # Y is missing where D = 0.  The regressions use a 0/1 weight
            # rather than a subset so that every replicate keeps the same
            # shape; the NaNs are replaced by zeros, which those weights then
            # multiply out of both the gram matrix and the moment vector.
            self.observed = (self.treatment == 1.0).astype(float)
            self.response = np.where(self.treatment == 1.0, self.outcome, 0.0)
        else:
            self.observed = None
            self.response = self.outcome

    def design(self, terms, treatment=None):
        """The design matrix for ``terms``, optionally at a fixed treatment."""
        key = (tuple(terms), treatment)
        matrix = self._designs.get(key)
        if matrix is None:
            columns = (self.columns if treatment is None
                       else dgps.set_treatment(self.columns, treatment))
            matrix = self._designs[key] = dgps.design_matrix(None, terms, columns)
        return matrix

    def propensity(self, spec):
        """The fitted propensities for one model, fitted at most once.

        Cached on the model's own terms rather than on the name it was asked
        for, because the terms are what determine the fit: `HT`, `DR` and
        `DR.ofal` all name `ps_true` and all want the identical regression on
        the identical data.  This cache is where three quarters of the
        baseline's logistic fits go away.
        """
        key = tuple(spec["terms"])
        fitted = self._propensities.get(key)
        if fitted is None:
            fitted = self._propensities[key] = fit_logistic(
                self.design(spec["terms"]), self.treatment, self.report)
        return fitted

    # -- the three estimator families, each returning one value per replicate

    def ht(self, ps_spec):
        propensity = self.propensity(ps_spec)
        n = self.treatment.shape[-1]
        if self.scenario == "A":
            outcome = np.nan_to_num(self.outcome, nan=0.0)
            return np.sum(self.treatment * outcome / propensity, axis=-1) / n

        treated = np.sum(self.treatment * self.outcome / propensity, axis=-1)
        control = np.sum((1.0 - self.treatment) * self.outcome
                         / (1.0 - propensity), axis=-1)
        return (treated - control) / n

    def outcome_regression(self, or_spec):
        terms = or_spec["terms"]
        if self.scenario == "A":
            X = self.design(terms)
            beta = solve_ols(X, self.response, self.observed)
            return np.mean(_matvec(X, beta), axis=-1)

        beta = solve_ols(self.design(terms), self.response)
        treated = _matvec(self.design(terms, 1), beta)
        control = _matvec(self.design(terms, 0), beta)
        return np.mean(treated - control, axis=-1)

    def doubly_robust(self, ps_spec, or_spec, constrained=False):
        """Doubly robust estimate.

        The outcome-regression design is augmented with inverse-probability
        covariates and refitted by least squares; averaging the predictions
        then gives an estimate that stays consistent if either nuisance model
        is correct.

        Scenario A adds the single covariate 1 / pi(V).  Scenario B adds the
        two covariates D / pi(V) and (1 - D) / (1 - pi(V)) of Scharfstein et
        al. (1999), which is what reproduces Table 2.  With ``constrained``
        the two coefficients are forced equal, the variant proposed in Section
        2.2 of the paper; the normal equation for the single coefficient then
        sets the sum of the two weighted residual terms to zero rather than
        each of them, so the resulting estimator is not doubly robust for the
        treatment effect.

        The augmenting covariates depend on the treatment received, so the
        counterfactual predictions must use their values at the treatment being
        predicted rather than at the observed one.
        """
        propensity = self.propensity(ps_spec)
        terms = or_spec["terms"]

        if self.scenario == "A":
            X = np.concatenate([self.design(terms), (1.0 / propensity)[..., None]],
                               axis=-1)
            beta = solve_ols(X, self.response, self.observed)
            return np.mean(_matvec(X, beta), axis=-1)

        treatment = self.treatment
        zero = np.zeros_like(propensity)
        if constrained:
            received = (treatment * propensity
                        + (1.0 - treatment) * (1.0 - propensity))
            augment = [1.0 / received]
            augment_treated = [1.0 / propensity]
            augment_control = [1.0 / (1.0 - propensity)]
        else:
            augment = [treatment / propensity,
                       (1.0 - treatment) / (1.0 - propensity)]
            augment_treated = [1.0 / propensity, zero]
            augment_control = [zero, 1.0 / (1.0 - propensity)]

        def stack(base, extra):
            return np.concatenate([base] + [e[..., None] for e in extra], axis=-1)

        X = stack(self.design(terms), augment)
        beta = solve_ols(X, self.response)
        treated = _matvec(stack(self.design(terms, 1), augment_treated), beta)
        control = _matvec(stack(self.design(terms, 0), augment_control), beta)
        return np.mean(treated - control, axis=-1)

    def run(self, cells):
        """Every named estimator, as ``name -> array over replicates``."""
        spec = config.SCENARIOS[self.scenario]
        out = {}
        for name, family, ps_name, or_name in cells:
            ps_spec = spec[ps_name] if ps_name else None
            or_spec = spec[or_name] if or_name else None
            if family == "ht":
                out[name] = self.ht(ps_spec)
            elif family == "or":
                out[name] = self.outcome_regression(or_spec)
            elif family == "dr":
                out[name] = self.doubly_robust(ps_spec, or_spec)
            elif family == "dr1":
                out[name] = self.doubly_robust(ps_spec, or_spec, constrained=True)
            else:
                raise ValueError(f"unknown estimator family: {family}")
        return out


def estimate_batch(columns, scenario, cells, report=None):
    """All estimators for a batch of replicates held as stacked arrays."""
    return _Replicate(columns, scenario, report).run(cells)


# ------------------------------------------- per-replicate API, unchanged

def _single(data, scenario):
    """A one-replicate ``_Replicate``, for the per-replicate API below.

    The public functions of Unit 2 are kept so that the existing test suite
    tests the code that runs, and so that a single replicate can still be
    examined on its own.  They are the batched code with a leading axis of one,
    not a second implementation of the estimators.
    """
    columns = data if isinstance(data, dict) else dgps._columns(data)
    return _Replicate(columns, scenario)


def estimate_ht(data, scenario, ps_spec):
    """Horvitz-Thompson estimate of the scenario's estimand."""
    return float(_single(data, scenario).ht(ps_spec))


def estimate_or(data, scenario, or_spec):
    """Outcome-regression estimate: fit by least squares, then average the
    fitted values."""
    return float(_single(data, scenario).outcome_regression(or_spec))


def estimate_dr(data, scenario, ps_spec, or_spec, constrained=False):
    """Doubly robust estimate; see ``_Replicate.doubly_robust``."""
    return float(_single(data, scenario).doubly_robust(ps_spec, or_spec,
                                                       constrained))


def estimate(data, scenario, family, ps_name, or_name):
    """Dispatch to the estimator named by ``family``, resolving model names
    against the scenario's specifications in ``config``."""
    spec = config.SCENARIOS[scenario]
    ps_spec = spec[ps_name] if ps_name else None
    or_spec = spec[or_name] if or_name else None

    if family == "ht":
        return estimate_ht(data, scenario, ps_spec)
    if family == "or":
        return estimate_or(data, scenario, or_spec)
    if family == "dr":
        return estimate_dr(data, scenario, ps_spec, or_spec)
    if family == "dr1":
        return estimate_dr(data, scenario, ps_spec, or_spec, constrained=True)
    raise ValueError(f"unknown estimator family: {family}")
