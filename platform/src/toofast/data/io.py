"""
Data I/O utilities for loading spectra and experiment data.

Extracted from tooFAST.py methods that handle CSV reading/writing.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_spectrum_from_csv(
    file_path: str,
    wavelengths: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Load a target spectrum from a CSV file.

    Expected CSV format: first row is wavelengths, second row is absorbance.
    Semicolon delimiter. Interpolates to match the spectrometer wavelengths.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    wavelengths : np.ndarray
        Spectrometer wavelength grid to interpolate onto.

    Returns
    -------
    np.ndarray or None
        Interpolated absorbance values, or None on error.
    """
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)

        if len(rows) < 2:
            logger.error(f"CSV file {file_path} has fewer than 2 rows")
            return None

        # Skip timestamp column if present
        csv_wavelengths = np.array([float(x) for x in rows[0] if _is_float(x)])
        csv_absorbance = np.array([float(x) for x in rows[-1] if _is_float(x)])

        if len(csv_wavelengths) != len(csv_absorbance):
            logger.error(
                f"Wavelength count ({len(csv_wavelengths)}) != "
                f"absorbance count ({len(csv_absorbance)})"
            )
            return None

        # Interpolate to match spectrometer wavelengths
        interpolated = np.interp(wavelengths, csv_wavelengths, csv_absorbance)
        logger.info(f"Loaded target spectrum from {file_path}")
        return interpolated

    except Exception as e:
        logger.error(f"Failed to load spectrum from {file_path}: {e}")
        return None


def load_previous_experiment(
    experiment_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """
    Load flows and costs from a previous experiment directory.

    Looks for flows.csv and cost.csv files.

    Parameters
    ----------
    experiment_path : str
        Path to the previous experiment directory.

    Returns
    -------
    flows : np.ndarray or None
    costs : np.ndarray or None
    metadata : dict
        Any additional metadata found (target_style, target_wavelength, etc.)
    """
    metadata = {}

    # Load flows
    flows_file = os.path.join(experiment_path, "flows.csv")
    if not os.path.exists(flows_file):
        logger.error(f"No flows.csv found in {experiment_path}")
        return None, None, metadata

    try:
        flows = np.loadtxt(flows_file, delimiter=";")
        if flows.ndim == 1:
            flows = flows.reshape(1, -1)
    except Exception as e:
        logger.error(f"Failed to load flows: {e}")
        return None, None, metadata

    # Load costs
    costs_file = os.path.join(experiment_path, "cost.csv")
    if not os.path.exists(costs_file):
        logger.error(f"No cost.csv found in {experiment_path}")
        return None, None, metadata

    try:
        costs = np.loadtxt(costs_file, delimiter=";")
    except Exception as e:
        logger.error(f"Failed to load costs: {e}")
        return None, None, metadata

    # Load params if available
    params_file = os.path.join(experiment_path, "params.txt")
    if os.path.exists(params_file):
        metadata = _parse_params_file(params_file)

    logger.info(
        f"Loaded previous experiment: {len(costs)} data points, "
        f"best cost: {costs.min():.6f}"
    )
    return flows, costs, metadata


def save_flows_csv(
    flows: np.ndarray, experiment_path: str, filename: str = "flows.csv"
) -> None:
    """Save flow array to CSV."""
    filepath = os.path.join(experiment_path, filename)
    np.savetxt(filepath, flows, delimiter=";", fmt="%.4f")
    logger.debug(f"Flows saved to {filepath}")


def save_costs_csv(
    costs: np.ndarray, experiment_path: str, filename: str = "cost.csv"
) -> None:
    """Save costs array to CSV."""
    filepath = os.path.join(experiment_path, filename)
    np.savetxt(filepath, costs, delimiter=";", fmt="%.8f")
    logger.debug(f"Costs saved to {filepath}")


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _parse_params_file(path: str) -> dict:
    """Parse a params.txt key-value file."""
    metadata = {}
    try:
        with open(path) as f:
            for line in f:
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()
    except Exception as e:
        logger.warning(f"Could not parse params file: {e}")
    return metadata
