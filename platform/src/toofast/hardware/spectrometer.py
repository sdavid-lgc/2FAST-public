"""
UV-Vis spectrometer, independent of the vendor.

`Spectrometer` turns the raw detector counts of a `SpectrometerBackend` (the simulated one of this package, or a
laboratory driver for the real instrument — see ``toofast.hardware.drivers``) into reference / extinction spectra:
pixel window, acquisition settings, reference handling, absorbance = -log10(T / T_ref), "smart" averaging.

Original code: Sylvain David (2023), Avantes-specific; the vendor calls moved to the backend, the processing is
unchanged.
"""
import logging
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

from toofast import clock
from toofast.hardware.base import SpectrometerBackend

logger = logging.getLogger(__name__)


class Spectrometer:
    def __init__(self, logger, backend: SpectrometerBackend, is_simulation: bool = False):
        self.logger = logger
        self.backend = backend
        self.is_simulation = is_simulation
        self.initialized = False
        try:
            self.lamb = np.asarray(self.backend.open(), dtype=float)
            self.pixels = len(self.lamb)
            self.identity = dict(self.backend.identity() or {})
            self.initialized = True
            self.logger.info(f"Spectrometer initialized ({type(backend).__name__}, {self.pixels} pixels)")
        except Exception as e:
            self.logger.error(f"An error occurred during spectrometer initialization: {e}")
            raise

        self.wavelengths = np.array([self.lamb[pix] for pix in range(self.pixels)])
        self.start_wl = self.lamb[0]
        self.end_wl = self.lamb[self.pixels - 1]
        self.reference = None

        self.update_parameters()

    def update_parameters(self, integration_time=3, number_average=20, smoothing=1,
                          start_wl=None, end_wl=None):
        start_wl = start_wl or self.lamb[0]
        end_wl = end_wl or self.lamb[self.pixels - 1]

        start_pixel = self.find_index_for_wavelength(start_wl)
        end_pixel = self.find_index_for_wavelength(end_wl)

        new_wavelengths = np.array([self.lamb[pix] for pix in range(start_pixel, end_pixel + 1)])

        if not np.array_equal(new_wavelengths, self.wavelengths):
            self.reference = None
            self.wavelengths = new_wavelengths

        self.integration_time = integration_time
        self.number_average = number_average
        self.smoothing = smoothing

        self.backend.configure(start_pixel, end_pixel, self.integration_time, self.number_average, self.smoothing)

        self.logger.info("Spectrometer's parameters updated.")

    def measure_reference(self, scans=1):
        """
        Measure reference spectrum, returns a Spectrum object.
        """
        transmittance = self._measure(scans)
        self.logger.info("Reference spectrum measured.")
        self.reference = Spectrum(self.wavelengths, transmittance, "reference")
        return self.reference

    def smart_measure_reference(self, scans=1, period=1, max_iter=3, threshold=0.995, timeout_cycles=5):
        """
        Measures several reference spectra over a period, discards flat and noisy spectra due to bubbles/air,
        and averages the remaining spectra if their Pearson correlation is above a given threshold.
        Retries up to `timeout_cycles` if the correlation is too low, then raises an error.
        """
        transmittances = np.zeros((max_iter, len(self.wavelengths)))

        for cycle in range(timeout_cycles):
            self.logger.info(f"Cycle {cycle + 1}/{timeout_cycles}: Measuring reference spectra...")

            # Measure spectra and store transmittances
            for i in range(max_iter):
                transmittances[i] = self._measure(scans)  # Measure the spectrum
                clock.sleep(period)  # Wait for the specified period

            # Compute pairwise Pearson correlations between spectra
            correlations = [
                pearsonr(transmittances[i], transmittances[j])[0]
                for i in range(max_iter) for j in range(i + 1, max_iter)
            ]
            min_correlation = min(correlations)  # Get the minimum correlation between pairs

            self.logger.info(f"Minimum Pearson correlation between spectra: {min_correlation:.5f}")

            if min_correlation >= threshold:
                # Average the spectra and return the result if correlations are within the threshold
                self.logger.info("Spectra correlations are within the threshold. Averaging spectra...")
                averaged_spectrum = np.mean(transmittances, axis=0)
                return averaged_spectrum

            self.logger.warning("Spectra correlations are below the threshold. Retrying...")

        # Raise an error if we exceed the timeout_cycles without success
        self.logger.error("Failed to obtain a good reference spectrum after maximum retries.")
        raise RuntimeError("Reference spectrum measurement failed: correlation threshold not met.")

    def measure_spectrum(self, scans=1, meas_type="measurement"):
        if self.reference is None:
            raise ValueError('no reference spectrum.')
        else:
            transmittance = self._measure(scans)
        absorbance = - np.log10(transmittance / self.reference.transmittance)
        return Spectrum(self.wavelengths, transmittance, meas_type, absorbance)

    def smart_measure(self, scans=1, meas_type="measurement", period=1, max_iter=10, threshold=0.01):
        """
        Measures several spectra over a period, discards flat and noisy spectra due to bubbles/air,
        and averages the remaining spectra.
        """
        if self.reference is None:
            raise ValueError('No reference spectrum available.')

        sampling_rate = period / max_iter
        transmittances = np.zeros((max_iter, len(self.wavelengths)))

        # Measure spectra and store transmittances
        for i in range(max_iter):
            transmittances[i] = self._measure(scans)
            clock.sleep(sampling_rate)
        absorbances = -np.log10(transmittances / self.reference.transmittance)

        # Discard flat or noisy spectra
        valid_indices = np.std(absorbances, axis=1) > threshold
        absorbances = absorbances[valid_indices]
        transmittances = transmittances[valid_indices]

        # Calculate average absorbance and transmittance of remaining spectra
        absorbance = np.mean(absorbances, axis=0)
        transmittance = np.mean(transmittances, axis=0)

        return Spectrum(self.wavelengths, transmittance, meas_type, absorbance)

    def _measure(self, scans=1):
        counts = np.asarray(self.backend.read(scans), dtype=float)
        return np.array([counts[i] for i in range(len(self.wavelengths))])

    def find_index_for_wavelength(self, target_wavelength, domain=None):
        """
        Finds the index of the spectrometer's closest wavelength to the targeted
        wavelength.

        Parameters
        ----------
        target_wavelength : float
            Target wavelength for wich to find the corresponding index.

        Returns
        -------
        index : int
            Index corresponding to the closest wavelength found in the spectrometer's
            wavelengths.
        :param target_wavelength:
        :param domain:

        """
        if domain is None:
            domain = self.wavelengths

        wavelengths = np.array([self.lamb[pix] for pix in range(self.pixels)])
        index = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - target_wavelength))
        return index


class Spectrum:
    def __init__(self, wavelengths, transmittance, meas_type, absorbance=None):
        self.wavelengths = wavelengths
        self.transmittance = transmittance
        self.meas_type = meas_type            
        self.absorbance = absorbance
        self.timestamp = clock.now().isoformat()

        self._calculate_spectral_properties()
    def    _calculate_spectral_properties(self):
        if self.absorbance is not None and self.meas_type != "reference":
            # Smooth the spectrum and fix defective pixels
            self.absorbance_filtered = self.absorbance.copy()
            # remove pix and replace by value at start
            # self.absorbance_filtered[1200:1210] = self.absorbance_filtered[1199]
            # self.absorbance_filtered[1900:1910] = self.absorbance_filtered[1899]
            # self.absorbance_filtered[2462:2472] = self.absorbance_filtered[2461]

            # savgol filter
            self.absorbance_filtered = savgol_filter(self.absorbance_filtered, 50, 3)
            lambda_max_idx = np.argmax(self.absorbance_filtered)
            self.lamb_max = self.wavelengths[lambda_max_idx]
            self.abs_max = self.absorbance_filtered[lambda_max_idx]
        else:
            self.abs_max = None
            self.lamb_max = None

    def get_abs_at_wl(self, wavelength):
        """
        Absorbance at the specified wavelength
    
        Parameters
        ----------
        wavelength : float
            Target wavelength for which to find the corresponding absorbance.
    
        Returns
        -------
        float
            Corresponding absorbance.
    
        """
        index = min(range(len(self.wavelengths)),
                    key=lambda i: abs(self.wavelengths[i] - wavelength))
        return self.absorbance[index]

    def plot_spectrum(self, save_dir=None, title='spectrum'):
        """
        Plots (and saves) the spectrum.

        Parameters
        ----------
        save_dir : str, optional
            If not None, the plot is saved as a .png file. The default is None
        :param title:

        """
        try:
            # Use a dedicated figure with specific number to avoid conflicts
            fig = plt.figure(num='spectrum_plot', figsize=(10, 6))

            # Handle window close event properly
            def on_close(event):
                plt.close(fig)

            fig.canvas.mpl_connect('close_event', on_close)

            # Do not clear figure to allow spectrum overlay
            plt.plot(self.wavelengths, self.transmittance if self.meas_type == "reference" else self.absorbance,)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Absorbance' if self.meas_type != "reference" else 'Transmittance')
            plt.title(f'Spectrum')
            # plt.legend()  # Display legend to distinguish spectra
            plt.grid(True, alpha=0.3)  # Add grid for better readability

            if save_dir is not None:
                plt.savefig(f'{save_dir}/{title}.png', dpi=300, bbox_inches='tight')

            plt.show(block=False)

        except Exception as e:
            logger.error(f"Error displaying spectrum: {e}")
            plt.close('all')

    def save_data(self, experiment_directory, sample_id=""):
        # add overwriting logic
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)
        if self.meas_type == "measurement":
            file_path = f'{experiment_directory}/measurement_spectra_{sample_id}.csv'
        else:
            file_path = f'{experiment_directory}/{self.meas_type}_spectrum.csv'

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if os.path.getsize(file_path) == 0:
                writer.writerow(['timestamp'] + list(self.wavelengths))
            writer.writerow([self.timestamp] + list(self.absorbance if self.meas_type == 'measurement' else self.transmittance))
