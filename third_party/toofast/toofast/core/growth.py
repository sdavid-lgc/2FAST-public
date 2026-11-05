"""
Nanoparticle-growth termination logic (pure, hardware-free helpers).

Extracted from Experiment.np_growth_done so the decision can be unit-tested
and so the "minimum number of spectra" gate can be bypassed when growth has
*genuinely* stabilized before that minimum is reached.

Background
----------
``np_growth_done`` decides when a synthesis has reached steady state by
comparing the latest spectrum to one measured ~5 s earlier (Pearson
correlation + peak-amplitude change). Historically a hard gate also required
at least ``min_spectra`` (default 10) measurements before stopping — a safety
margin against transient early plateaus. For fast syntheses that finish in
fewer than ``min_spectra`` measurements this gate wastes time and reagent.

We keep the safety margin but make it bypassable: when ``allow_early_stop`` is
on, growth may be declared done before ``min_spectra`` *provided* the
convergence criteria have held for ``early_stop_confirmations`` consecutive
checks (guarding against a single early plateau). The hard floor remains
``minimum_line`` rows — the minimum needed to even compute the correlation.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import pearsonr


def minimum_line_for_rate(sampling_rate: float) -> int:
    """Rows spanning ~5 s of acquisition (lag used for the correlation check)."""
    return math.ceil(5.0 / float(sampling_rate)) + 1


def _parse_timestamp(value: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def growth_time_from_rows(rows: Sequence[Sequence[str]]) -> Optional[float]:
    """Elapsed seconds between the first and last spectrum timestamps.

    Each row is ``[timestamp_iso, a1, a2, ...]`` (the 2FAST measurement CSV
    format). Returns None if timestamps are missing/unparseable.
    """
    if not rows:
        return None
    first = _parse_timestamp(rows[0][0])
    last = _parse_timestamp(rows[-1][0])
    if first is None or last is None:
        return None
    return (last - first).total_seconds()


@dataclass
class GrowthAssessment:
    """Result of evaluating the growth-termination criteria on a CSV series."""

    line_count: int
    minimum_line: int
    correlation: Optional[float]
    amplitude_change: Optional[float]
    criteria_met: bool          # correlation + amplitude thresholds satisfied
    enough_spectra: bool        # line_count > min_spectra (historical gate)
    computable: bool            # enough rows to compute the correlation at all


def assess_growth(
    rows: Sequence[Sequence[str]],
    sampling_rate: float,
    correlation_threshold: float,
    amplitude_threshold: float,
    min_spectra: int,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
) -> GrowthAssessment:
    """Evaluate the convergence criteria over a measurement series.

    Pure function: takes the parsed CSV ``rows`` (incl. the header? no — data
    rows only, each ``[timestamp, *absorbances]``) and returns the metrics and
    boolean gates. Does not decide termination by itself (see
    :func:`should_terminate`).
    """
    line_count = len(rows)
    minimum_line = minimum_line_for_rate(sampling_rate)

    if line_count < minimum_line:
        return GrowthAssessment(
            line_count=line_count,
            minimum_line=minimum_line,
            correlation=None,
            amplitude_change=None,
            criteria_met=False,
            enough_spectra=line_count > min_spectra,
            computable=False,
        )

    last_row = np.array(list(map(float, rows[-1][1:])))
    previous_row = np.array(list(map(float, rows[-minimum_line][1:])))

    smoothed_last = savgol_filter(last_row, window_length=savgol_window, polyorder=savgol_polyorder)
    smoothed_previous = savgol_filter(
        previous_row, window_length=savgol_window, polyorder=savgol_polyorder
    )

    correlation, _ = pearsonr(smoothed_last, smoothed_previous)
    max_last = float(np.max(smoothed_last))
    max_previous = float(np.max(smoothed_previous))
    amplitude_change = abs(max_last - max_previous) / max_previous if max_previous else float("inf")

    criteria_met = (correlation > correlation_threshold) and (
        amplitude_change < amplitude_threshold
    )
    return GrowthAssessment(
        line_count=line_count,
        minimum_line=minimum_line,
        correlation=float(correlation),
        amplitude_change=float(amplitude_change),
        criteria_met=criteria_met,
        enough_spectra=line_count > min_spectra,
        computable=True,
    )


def should_terminate(
    assessment: GrowthAssessment,
    allow_early_stop: bool,
    confirmation_streak: int,
    early_stop_confirmations: int,
) -> bool:
    """Decide termination from an assessment + the current confirmation streak.

    - Always require the convergence criteria to be met and the correlation to
      be computable.
    - If enough spectra were collected (historical gate), terminate.
    - Otherwise, terminate only when early stop is allowed and the criteria
      have held for ``early_stop_confirmations`` consecutive checks.
    """
    if not assessment.computable or not assessment.criteria_met:
        return False
    if assessment.enough_spectra:
        return True
    if allow_early_stop and confirmation_streak >= early_stop_confirmations:
        return True
    return False
