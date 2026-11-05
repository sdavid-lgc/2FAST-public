"""
Data models for spectra and experiment results.

Provides Spectrum (moved from Spectrometer.py) and ExperimentResult
as proper data containers with clear interfaces.
"""

from __future__ import annotations

import csv
import datetime
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class Spectrum:
    """
    UV-Vis spectrum data container.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength values in nm.
    transmittance : np.ndarray
        Raw transmittance values.
    meas_type : str
        Type of measurement ('measurement', 'reference', 'target').
    absorbance : np.ndarray, optional
        Absorbance values. Computed from transmittance if not provided.
    """

    def __init__(
        self,
        wavelengths: np.ndarray,
        transmittance: np.ndarray,
        meas_type: str,
        absorbance: Optional[np.ndarray] = None,
    ):
        self.wavelengths = wavelengths
        self.transmittance = transmittance
        self.meas_type = meas_type
        self.absorbance = absorbance
        self.timestamp = datetime.datetime.now().isoformat()

        self.lamb_max: Optional[float] = None
        self.abs_max: Optional[float] = None
        self._calculate_spectral_properties()

    def _calculate_spectral_properties(self) -> None:
        if self.absorbance is not None and self.meas_type != "reference":
            filtered = savgol_filter(self.absorbance.copy(), 50, 3)
            lambda_max_idx = np.argmax(filtered)
            self.lamb_max = self.wavelengths[lambda_max_idx]
            self.abs_max = filtered[lambda_max_idx]

    def get_abs_at_wl(self, wavelength: float) -> float:
        """Return absorbance at the closest wavelength."""
        index = int(np.argmin(np.abs(self.wavelengths - wavelength)))
        return self.absorbance[index]

    def save_data(self, experiment_directory: str, sample_id: str = "") -> None:
        """Append spectrum data to CSV file."""
        os.makedirs(experiment_directory, exist_ok=True)

        if self.meas_type == "measurement":
            file_path = os.path.join(
                experiment_directory, f"measurement_spectra_{sample_id}.csv"
            )
        else:
            file_path = os.path.join(
                experiment_directory, f"{self.meas_type}_spectrum.csv"
            )

        with open(file_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            if os.path.getsize(file_path) == 0:
                writer.writerow(["timestamp"] + list(self.wavelengths))
            data = (
                self.absorbance
                if self.meas_type == "measurement"
                else self.transmittance
            )
            writer.writerow([self.timestamp] + list(data))


@dataclass
class ExperimentResult:
    """Container for completed experiment results."""

    flows: np.ndarray
    costs: np.ndarray
    spectra: Dict[str, Spectrum]
    optimal_flows: np.ndarray
    minimum_cost: float
    best_iteration: int
    success: bool
    gp_model: object = None

    @property
    def n_iterations(self) -> int:
        return len(self.costs)
