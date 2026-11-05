"""Build idealized parametric LSPR target spectra from physical properties.

Lets a BO target be specified by nanoparticle physics rather than by picking one
of the synthesized spectra (which is circular — see the paper discussion):

    lambda_max (nm)  ->  particle SIZE / shape (LSPR resonance position)
    FWHM (nm)        ->  polydispersity (peak width)
    amplitude        ->  yield / concentration (peak absorbance)

The resulting spectrum is projected onto the frozen PCA basis at cost time, so
the BO performs genuine inverse design toward an independent specification.
"""

import csv
import datetime

import numpy as np


def build_parametric_target(wavelengths, lambda_max, fwhm,
                            amplitude=1.0, lineshape="lorentzian"):
    """Idealized absorbance peak over ``wavelengths``.

    Parameters
    ----------
    wavelengths : array-like (nm)
    lambda_max : float (nm)        peak position (~ particle size)
    fwhm : float (nm)              full width at half maximum (~ polydispersity)
    amplitude : float              peak absorbance (~ yield / concentration)
    lineshape : {'lorentzian', 'gaussian'}
        Lorentzian (default) is the natural LSPR resonance lineshape.

    Returns
    -------
    np.ndarray : absorbance on the same grid (peak = amplitude at lambda_max).
    """
    wl = np.asarray(wavelengths, dtype=float)
    fwhm = max(float(fwhm), 1e-6)
    if lineshape == "gaussian":
        shape = np.exp(-4.0 * np.log(2.0) * ((wl - lambda_max) / fwhm) ** 2)
    else:  # lorentzian
        shape = 1.0 / (1.0 + ((wl - lambda_max) / (fwhm / 2.0)) ** 2)
    return float(amplitude) * shape


def save_parametric_target_csv(path, wavelengths, lambda_max, fwhm,
                               amplitude=1.0, lineshape="lorentzian"):
    """Write a parametric target as a ``;``-delimited CSV (load_spectrum_from_csv
    format: header 'timestamp' + wavelengths, one row timestamp + absorbance)."""
    wl = np.asarray(wavelengths, dtype=float)
    ab = build_parametric_target(wl, lambda_max, fwhm, amplitude, lineshape)
    ts = datetime.datetime.now().isoformat()
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["timestamp"] + list(wl))
        w.writerow([ts] + list(ab))
    return path
