# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:56:43 2023

@author: Sylvain David
"""
import logging
import os
import sys
import time
import csv
import datetime

import numpy as np
import matplotlib.pyplot as plt


from toofast.hardware import avaspec as avs
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

class Spectrometer:
    def __init__(self, logger, is_simulation=False):
        self.is_simulation = is_simulation
        self.logger = logger
        self.initialized = False
        if not self.is_simulation:
            try:
                avs.AVS_Init(0)  # open communication
                ret = avs.AVS_GetList()  # list available spectrometers
                self.handle = avs.AVS_Activate(ret[0])  # create spectrometer object
                self.pixels = avs.AVS_GetNumPixels(self.handle)  # int of 4094 (or 2048, depends on the model)
                self.lamb = avs.AVS_GetLambda(self.handle)  # avaspec_m.c_double_Array_4096
                self.initialized = True
                self.logger.info("AVS spectrometer has been initialized")
            except Exception as e:
                self.logger.error(f"An error occurred during AVS spectrometer initialization: {e}")
                sys.exit(1)
        else:
            self.pixels = 4094
            self.lamb = np.linspace(300, 800, self.pixels)
            self.initialized = True
            self.logger.info("Simulated AVS spectrometer has been initialized")

        self.wavelengths = np.array([self.lamb[pix] for pix in range(self.pixels)])
        self.start_wl = self.lamb[0]
        self.end_wl = self.lamb[self.pixels-1]
        self.reference = None
        
        self.update_parameters()

    def _generate_sample_transmittance(self, n_abs_bands=(1, 4)):
        """
        Generate a sample spectrum based on halogen spectrum
        with simulated absorption bands.
        n_abs_bands : tuple -> min and max number of absorption bands
        """
        # Base = halogen reference spectrum
        base_spectrum = self._generate_halogen_reference()

        # Number of absorption bands
        n_bands = np.random.randint(n_abs_bands[0], n_abs_bands[1] + 1)

        transmittance = base_spectrum.copy()

        # Add absorption dips
        for _ in range(n_bands):
            center = np.random.uniform(self.wavelengths.min(), self.wavelengths.max())
            width = np.random.uniform(5, 30)  # width in nm
            depth = np.random.uniform(0.05, 0.3)  # absorption depth
            transmittance -= depth * np.exp(-0.5 * ((self.wavelengths - center) / width) ** 2)

        # Add noise and drift
        noise = np.random.normal(0, 0.01, size=self.pixels)
        drift = 1 + 0.005 * np.sin(np.linspace(0, 2 * np.pi, self.pixels))
        transmittance = transmittance * drift + noise

        # Clip to [0, 1]
        return np.clip(transmittance, 0, 1)

    def _generate_halogen_reference(self, temperature=3000):
        """
        Simulate a halogen lamp spectrum (continuous, black body).
        temperature : Temperature in Kelvin (~3000 K for halogen)
        """
        # Physical constants
        h = 6.62607015e-34  # Planck (J.s)
        c = 2.99792458e8  # Speed of light (m/s)
        k = 1.380649e-23  # Boltzmann (J/K)

        wl_m = self.wavelengths * 1e-9  # nm -> m

        # Planck's law for spectral radiance
        intensity = (2 * h * c ** 2) / (wl_m ** 5) / (np.exp((h * c) / (wl_m * k * temperature)) - 1)

        # Normalize to [0, 1]
        intensity /= intensity.max()

        # Add realistic noise
        noise = np.random.normal(0, 0.005, size=self.pixels)
        reference_spectrum = np.clip(intensity + noise+ 1, 0, 1)

        return reference_spectrum
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
        
        if not self.is_simulation:
            avs.AVS_UseHighResAdc(self.handle, True)  # Sets the ADC range of the spectrometer readout
            self.measconfig = avs.MeasConfigType
            self.measconfig.m_StartPixel = start_pixel
            self.measconfig.m_StopPixel = end_pixel
            self.measconfig.m_IntegrationTime = self.integration_time  # integration_time #in milliseconds
            self.measconfig.m_IntegrationDelay = 0  # in FPGA clock cycles
            self.measconfig.m_NrAverages = self.number_average
            self.measconfig.m_CorDynDark_m_Enable = 0  # nesting of types does NOT work!!
            self.measconfig.m_CorDynDark_m_ForgetPercentage = 100
            self.measconfig.m_Smoothing_m_SmoothPix = self.smoothing
            self.measconfig.m_Smoothing_m_SmoothModel = 0
            self.measconfig.m_SaturationDetection = 0
            self.measconfig.m_Trigger_m_Mode = 0
            self.measconfig.m_Trigger_m_Source = 0
            self.measconfig.m_Trigger_m_SourceType = 0
            self.measconfig.m_Control_m_StrobeControl = 0
            self.measconfig.m_Control_m_LaserDelay = 0
            self.measconfig.m_Control_m_LaserWidth = 0
            self.measconfig.m_Control_m_LaserWaveLength = 0.0
            self.measconfig.m_Control_m_StoreToRam = 0
            avs.AVS_PrepareMeasure(self.handle, self.measconfig)
        
        self.logger.info("Spectrometer's parameters updated.")

    # def wait_for_steady_state(self):
    #     steady_state = False
    #     self.prepare_measure(integration_time=100, number_average=1)
    #     result = self._measure(1, None, lambda x: x)
    #     time.sleep(1)
    #     result2 = self._measure(1, None, lambda x: x)
    #     iter = 0
    #     while np.allclose(result, result2, rtol=0.05) == False or iter < 20:
    #         result = result2
    #         time.sleep(1)
    #         result2 = self._measure(1, None, lambda x: x)
    #         iter += 1
    #     self.logger.info("Steady state reached, iteration number: ", iter)
    #     return

    def measure_reference(self, scans=1):
        """
        Measure reference spectrum, returns a Spectrum object.
        """
        if self.is_simulation:
            transmittance = self._generate_halogen_reference()
            self.reference = Spectrum(
                wavelengths=self.wavelengths,
                transmittance=transmittance,
                meas_type="reference"
            )
            self.logger.info("Reference spectrum (halogen lamp) measured.")
        else:
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
        if self.is_simulation:
            self.logger.warning("Simulation mode: No measurement performed.")
            return None

        transmittances = np.zeros((max_iter, len(self.wavelengths)))

        for cycle in range(timeout_cycles):
            self.logger.info(f"Cycle {cycle + 1}/{timeout_cycles}: Measuring reference spectra...")

            # Measure spectra and store transmittances
            for i in range(max_iter):
                self.collector.open_lamp_shutter()  # Open the lamp shutter
                transmittances[i] = self._measure(scans)  # Measure the spectrum
                self.collector.close_lamp_shutter()  # Close the lamp shutter
                time.sleep(period)  # Wait for the specified period

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

        if not self.is_simulation:

            sampling_rate = period / max_iter
            transmittances = np.zeros((max_iter, len(self.wavelengths)))

            # Measure spectra and store transmittances
            for i in range(max_iter):
                transmittances[i] = self._measure(scans)
                time.sleep(sampling_rate)
            absorbances = -np.log10(transmittances / self.reference.transmittance)

            # Discard flat or noisy spectra
            valid_indices = np.std(absorbances, axis=1) > threshold
            absorbances = absorbances[valid_indices]
            transmittances = transmittances[valid_indices]

            # Calculate average absorbance and transmittance of remaining spectra
            absorbance = np.mean(absorbances, axis=0)
            transmittance = np.mean(transmittances, axis=0)
        else:
            # In simulation mode, set transmittance and absorbance to default values
            transmittance = np.ones_like(self.wavelengths)
            absorbance = -np.log10(transmittance / self.reference.transmittance)

        return Spectrum(self.wavelengths, transmittance, meas_type, absorbance)
    
    def _measure(self, scans=1):
        if not self.is_simulation:
            avs.AVS_Measure(self.handle, 0, scans)
            data_ready = False

            while not data_ready:
                data_ready = avs.AVS_PollScan(self.handle)
                time.sleep(self.measconfig.m_IntegrationTime / 100000)

            timestamp, scopedata = avs.AVS_GetScopeData(self.handle)

            transmittance = np.array([scopedata[i] for i in range(len(self.wavelengths))])
        else:
            base_transmittance = self._generate_sample_transmittance()
            # Spectrum variation at each measurement
            noise = np.random.normal(0, 0.02, size=self.pixels)
            drift = 1 + 0.01*np.sin(np.linspace(0, 2*np.pi, self.pixels))
            transmittance = base_transmittance * drift + noise
        return transmittance
    
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
        self.timestamp = datetime.datetime.now().isoformat()

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
