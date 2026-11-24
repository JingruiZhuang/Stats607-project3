"""Summarise raw estimates into the performance table."""

import argparse
import logging
import os

import pandas as pd

from . import config, metrics

logger = logging.getLogger(__name__)

MEASURES = ["bias_mean", "bias_median", "variance", "iqr"]


def load(raw_path):
    """Read the raw estimates, failing with a hint if they are not there yet.

    Args:
        raw_path: path to the long-format CSV written by ``simulation.run``.

    Returns:
        The estimates as a data frame, one row per estimate.

    Raises:
        SystemExit: if ``raw_path`` does not exist.
    """
    if not os.path.exists(raw_path):
        raise SystemExit(f"{raw_path} not found; run `make simulate` first")
    return pd.read_csv(raw_path)


def summarise(raw):
    """Collapse raw estimates into one row per cell, against the true values.

    Args:
        raw: long-format estimates, as written by ``simulation.run``.

    Returns:
        One row per scenario and estimator, with the published values joined on.
    """
    truths = {name: spec["truth"] for name, spec in config.SCENARIOS.items()}
    return metrics.summarise(raw, truths, config.PUBLISHED)


def comparison_table(summary):
    """Build the reproduction cells, each measure beside its published value.

    Args:
        summary: the frame returned by :func:`summarise`.

    Returns:
        The sixteen reproduction cells, formatted for printing.
    """
    core = summary[summary["published_bias_mean"].notna()].copy()
    table = core[["scenario", "estimator"]].copy()
    for measure in MEASURES:
        table[measure] = core[measure].map("{:.3f}".format)
        table["paper_" + measure] = core["published_" + measure].map("{:.2f}".format)
        table["z_" + measure] = core["z_" + measure].map("{:+.1f}".format)
    return table


def extension_table(summary):
    """Select the constrained-variant cells, which have no published counterpart.

    Args:
        summary: the frame returned by :func:`summarise`.

    Returns:
        The extension cells, which Table 2 does not report.
    """
    extra = summary[summary["published_bias_mean"].isna()]
    return extra[["scenario", "estimator"] + MEASURES]


def main():
    """Summarise the raw estimates and write the comparison table to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=config.RAW_PATH)
    parser.add_argument("--out", default=config.SUMMARY_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    summary = summarise(load(args.raw))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary.to_csv(args.out, index=False)
    logger.info("wrote %d cells to %s", len(summary), args.out)

    fmt = {"float_format": lambda x: f"{x: .3f}", "index": False, "na_rep": "-"}
    print("\nReproduction cells, against Table 2")
    print(comparison_table(summary).to_string(**fmt))

    extensions = extension_table(summary)
    if not extensions.empty:
        print("\nExtension cells: the phi1 = phi2 variant of Section 2.2")
        print(extensions.to_string(**fmt))

    worst = summary["z_bias_mean"].abs().max()
    print(f"\nLargest absolute z for the bias across reproduction cells: {worst:.1f}")


if __name__ == "__main__":
    main()
