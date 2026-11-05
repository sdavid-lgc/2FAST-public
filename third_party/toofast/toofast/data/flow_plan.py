"""
LHS flow-plan generator with run-status tracking.

Generates Latin Hypercube Sampled flow sets that sum to a fixed total flow
and exports them as a CSV file with a ``status`` column (``todo`` / ``done``).
The platform reads this CSV, runs the next ``todo`` row, and marks it ``done``.

Usage (CLI)::

    python -m toofast.data.flow_plan \\
        --pumps 5 --samples 20 --total-flow 8 \\
        --out flow_plan.csv

Usage (Python)::

    from toofast.data.flow_plan import generate_flow_plan, load_flow_plan, mark_done

    generate_flow_plan(n_pumps=5, n_samples=20, total_flow=8.0,
                       output_path="flow_plan.csv")
    df = load_flow_plan("flow_plan.csv")
    next_row = next_todo(df)            # first row with status == 'todo'
    flows = next_row.drop(['status']).values.astype(float)
    mark_done(df, next_row.name, "flow_plan.csv")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pyDOE import lhs

logger = logging.getLogger(__name__)


def generate_lhs_flows(
    n_pumps: int,
    n_samples: int,
    total_flow: float = 8.0,
    bounds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate LHS flow sets that sum to *total_flow*.

    Parameters
    ----------
    n_pumps : int
        Number of pumps (dimensions).
    n_samples : int
        Number of flow sets to generate.
    total_flow : float
        Target total flow rate (sum of each row).
    bounds : np.ndarray, optional
        Per-pump (min, max) bounds of shape (n_pumps, 2).
        If None, each pump is bounded [0, total_flow].

    Returns
    -------
    np.ndarray of shape (n_samples, n_pumps)
        Each row sums to *total_flow*.
    """
    samples = lhs(n_pumps, samples=n_samples)
    flows = np.zeros((n_samples, n_pumps))
    for i, sample in enumerate(samples):
        flows[i] = sample / sample.sum() * total_flow

    if bounds is not None:
        bounds = np.asarray(bounds)
        for i in range(n_samples):
            flows[i] = np.clip(flows[i], bounds[:, 0], bounds[:, 1])
            flows[i] = flows[i] / flows[i].sum() * total_flow

    return np.round(flows, 4)


def generate_flow_plan(
    n_pumps: int,
    n_samples: int,
    total_flow: float = 8.0,
    bounds: Optional[np.ndarray] = None,
    output_path: str = "flow_plan.csv",
) -> pd.DataFrame:
    """
    Generate an LHS flow plan CSV with a ``status`` column.

    Parameters
    ----------
    n_pumps, n_samples, total_flow, bounds
        Forwarded to :func:`generate_lhs_flows`.
    output_path : str
        Path where the CSV is written.

    Returns
    -------
    pd.DataFrame
        The generated plan (also saved to *output_path*).
    """
    flows = generate_lhs_flows(n_pumps, n_samples, total_flow, bounds)

    columns = [f"pump_{i+1}" for i in range(n_pumps)]
    df = pd.DataFrame(flows, columns=columns)
    df.insert(0, "status", "todo")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, sep=";")
    logger.info("Flow plan saved to %s (%d samples, %d pumps)", output_path, n_samples, n_pumps)
    return df


def load_flow_plan(path: str) -> pd.DataFrame:
    """Load a flow plan CSV."""
    return pd.read_csv(path, sep=";")


def next_todo(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the first row with status ``todo``, or None."""
    pending = df[df["status"] == "todo"]
    if pending.empty:
        return None
    return pending.iloc[0]


def mark_done(df: pd.DataFrame, idx: int, path: str) -> None:
    """Mark row *idx* as ``done`` and save."""
    df.loc[idx, "status"] = "done"
    df.to_csv(path, index=False, sep=";")
    logger.info("Marked row %d as done in %s", idx, path)


def main():
    parser = argparse.ArgumentParser(description="Generate LHS flow plan for 2FAST")
    parser.add_argument("--pumps", type=int, required=True, help="Number of pumps")
    parser.add_argument("--samples", type=int, required=True, help="Number of flow sets")
    parser.add_argument("--total-flow", type=float, default=8.0, help="Total flow rate (default: 8)")
    parser.add_argument("--out", type=str, default="flow_plan.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_flow_plan(
        n_pumps=args.pumps,
        n_samples=args.samples,
        total_flow=args.total_flow,
        output_path=args.out,
    )
    print(df.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
