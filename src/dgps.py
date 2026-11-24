"""Data-generating mechanisms for scenarios A and B.

The public surface is unchanged from Unit 2 -- ``generate`` still returns a
DataFrame and ``design_matrix`` still takes one -- but the work underneath is
done on plain arrays, and ``generate_batch`` draws many replicates at once so
the per-replicate Python cost the baseline profile exposed is paid once per
study rather than once per replicate.

Equivalence with the Unit 2 code is deliberate and load-bearing: the draws come
off each replicate's generator in the same order and the same shapes, so a
given seed yields the same dataset it did before.  ``tests/test_regression.py``
checks this against the committed Unit 2 estimates.
"""

import numpy as np
import pandas as pd

from . import config

COVARIATES = ["V1", "V2", "V3", "V4"]

# Columns a replicate carries, in the order `generate` built them in Unit 2.
COLUMNS = COVARIATES + ["I1", "I2", "I3", "I4", "D", "Y_full", "Y"]

# Compiled term strings, keyed by the string itself.  A specification's terms
# are fixed by `config` and reused across every replicate and every estimator,
# so compiling each one once removes the 84 `eval` calls per replicate that the
# baseline profile counted, along with the parse behind each of them.
_COMPILED = {}

# Names each term actually references.  The baseline rebuilt a namespace over
# all eleven columns of the frame on every call; a term mentioning `V1` needs
# one array, not eleven.
_NAMES = {}

_EMPTY_GLOBALS = {"__builtins__": {}}


def _compile(term):
    """Compile one term string, caching by the string."""
    code = _COMPILED.get(term)
    if code is None:
        code = _COMPILED[term] = compile(term, "<term>", "eval")
        _NAMES[term] = tuple(code.co_names)
    return code


def expit(x):
    """Logistic function, overflow-free for any finite input.

    ``1 / (1 + exp(-x))`` overflows for ``x`` below about -709 and returns a
    warning and an infinity rather than the zero the mathematics calls for.
    Writing it as ``exp(-log(1 + exp(-x)))`` and evaluating the inner part with
    ``logaddexp`` keeps every exponent negative, so nothing can overflow, and
    underflow is to the correct limits of 0 and 1.

    The DGP's own linear predictors are bounded well inside the range where the
    naive form is safe -- this matters for predictors fitted from data, which
    are not.  Over that safe range the two agree to one unit in the last place.

    Two earlier versions of this function were slower: the naive form needs a
    guard against its own overflow, and splitting the input into positive and
    negative branches costs two boolean masks and four fancy-indexed passes
    over the array.  This is three ufunc passes and no allocation beyond the
    result, which profiling found worth having -- it was the single largest
    line in the optimized study.
    """
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=float)))


def add_indicators(data):
    """Append ``Il`` = ``1(Vl > 0)`` for every covariate present."""
    out = data.copy()
    for v in COVARIATES:
        if v in out.columns:
            out["I" + v[1:]] = (out[v] > 0).astype(float)
    return out


def _columns(data):
    """Return a name -> array mapping for whatever ``data`` is.

    Accepts the DataFrame the public API promises and the plain dict of arrays
    the batched path uses, so one implementation of ``design_matrix`` serves
    both and the two cannot drift apart.
    """
    if isinstance(data, dict):
        return data
    return {c: data[c].to_numpy(dtype=float) for c in data.columns}


def design_matrix(data, terms, columns=None):
    """Build a design matrix from a list of term strings.

    Terms are evaluated against the columns of ``data``; ``"1"`` becomes an
    intercept column.  Keeping specifications as strings means the correct and
    false models differ only by a list in ``config``, with no duplicated code.

    ``columns`` lets a caller that already holds the arrays pass them straight
    in, which is how the batched path avoids touching pandas at all.  Each term
    is evaluated against only the names it mentions.

    The returned matrix has the terms as its last axis, so a batched caller can
    pass arrays of shape ``(n_sim, n)`` and get ``(n_sim, n, k)`` back from the
    same code that returns ``(n, k)`` for one replicate.
    """
    source = columns if columns is not None else _columns(data)
    shape = np.shape(next(iter(source.values())))
    built = []
    for term in terms:
        if term.strip() == "1":
            built.append(np.ones(shape))
            continue
        code = _compile(term)
        namespace = {name: source[name] for name in _NAMES[term] if name in source}
        value = eval(code, _EMPTY_GLOBALS, namespace)
        built.append(np.broadcast_to(np.asarray(value, dtype=float), shape))
    return np.stack(built, axis=-1)


def linear_predictor(data, spec, columns=None):
    """Evaluate a specification with known coefficients."""
    coefficients = spec.get("beta", spec.get("alpha"))
    return design_matrix(data, spec["terms"], columns) @ np.asarray(
        coefficients, dtype=float)


def with_treatment(data, value):
    """Copy of ``data`` with the treatment indicator fixed at ``value``.

    Used to read off potential outcomes when verifying the true estimand.
    """
    out = data.copy()
    out["D"] = float(value)
    return out


def set_treatment(columns, value):
    """``columns`` with ``D`` fixed at ``value``, sharing every other array.

    The array counterpart of ``with_treatment``.  The baseline copied the whole
    frame to overwrite one column, ten times per replicate; here the other
    arrays are not copied at all, because nothing downstream writes to them.
    """
    replaced = dict(columns)
    replaced["D"] = np.broadcast_to(np.asarray(float(value)),
                                    np.shape(columns["D"]))
    return replaced


def generate_columns(scenario, n, rng, n_sim=None):
    """Draw replicates as a ``name -> array`` mapping, without pandas.

    With ``n_sim`` the arrays are ``(n_sim, n)`` and ``rng`` must be a sequence
    of that many generators, one per replicate, so each replicate draws from
    its own stream exactly as it would on its own.  Without it they are ``(n,)``
    and ``rng`` is a single generator.

    The draw order per replicate -- four covariate vectors, then the treatment,
    then the outcome noise -- matches Unit 2 exactly, and drawing the four
    covariates as one ``(4, n)`` block is bit-identical to drawing them one at
    a time.  Only the draws themselves are per-replicate; the arithmetic
    between them is done for every replicate at once.
    """
    spec = config.SCENARIOS[scenario]
    batched = n_sim is not None
    generators = list(rng) if batched else [rng]
    rows = len(generators)

    # Three passes over the replicates rather than one, so that each
    # generator still sees covariates, then the treatment, then the outcome
    # noise, in that order -- the order Unit 2 drew them in.  Between the
    # passes the arithmetic is done for every replicate at once.
    covariates = np.empty((rows, len(COVARIATES), n))
    for i, generator in enumerate(generators):
        covariates[i] = generator.standard_normal((len(COVARIATES), n))

    columns = {name: covariates[:, j] for j, name in enumerate(COVARIATES)}
    columns.update({"I" + name[1:]: (columns[name] > 0).astype(float)
                    for name in COVARIATES})
    propensity = expit(linear_predictor(None, spec["ps_true"], columns))

    treatment = np.empty((rows, n))
    for i, generator in enumerate(generators):
        treatment[i] = generator.binomial(1, propensity[i])
    columns["D"] = treatment

    noise = np.empty((rows, n))
    for i, generator in enumerate(generators):
        noise[i] = generator.standard_normal(n)

    columns["Y_full"] = linear_predictor(None, spec["or_true"], columns) + noise
    columns["Y"] = (np.where(columns["D"] == 1.0, columns["Y_full"], np.nan)
                    if scenario == "A" else columns["Y_full"])

    if not batched:
        columns = {name: array[0] for name, array in columns.items()}
    return columns


def generate(scenario, n, rng):
    """Draw one dataset of size ``n`` for scenario ``"A"`` or ``"B"``.

    Returns ``V1..V4``, ``I1..I4``, ``D``, ``Y_full`` and ``Y``.  ``V4`` is a
    noise covariate used only by the false propensity model in scenario B.
    ``Y_full`` is the complete-data outcome, retained for DGP checks; the
    estimators see only ``Y``, which is missing where ``D == 0`` in scenario A.
    """
    columns = generate_columns(scenario, n, rng)
    return pd.DataFrame({name: columns[name] for name in COLUMNS})
