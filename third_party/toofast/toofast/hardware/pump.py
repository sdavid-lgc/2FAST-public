# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 14:27:53 2023

@author: Sylvain David
"""
import time
import os
import sys
import csv
import math
import numpy as np
import ctypes
import logging

from pyDOE import lhs

from qmixsdk import qmixbus
from qmixsdk import qmixpump


class MyPump(qmixpump.Pump):
    def __init__(self, logger, handle = ctypes.c_longlong(), inner_diameter_mm=10.301,
                 max_piston_stroke_mm=60, max_flow = 8, min_flow = 0, is_simulation=False):
        self.logger = logger
        self.is_simulation = is_simulation
        if not self.is_simulation:
            super().__init__(handle)
        self.inner_diameter_mm = inner_diameter_mm
        self.max_piston_stroke_mm = max_piston_stroke_mm
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.dye = ""
        self.color = None

    def initialization(self, config_file):
        while True:
            try:
                bus = qmixbus.Bus()
                bus.open(config_file, "")
                bus.start()
                self.logger.info(self.get_no_of_pumps())
                for i in range(self.get_no_of_pumps()):
                    self.lookup_by_device_index(i)
                    self.clear_fault()
                    self.enable(True)
                self.logger.info("Nemesys pumps have been initialized\n")
                break
            except Exception as e:
                error_message = f"An error occurred during pump initialization: {str(e)}"
                self.logger.error(error_message)
                proceed = input("Please check that the pump is plugged correctly. Try again? (y/n): ")
                if proceed == "y":
                    continue
                elif proceed == "n":             
                    sys.exit(1)
                    raise RuntimeError from e
                else:
                    self.logger.erro("Invalid input")
                    continue

    def set_parameters(self):
        if not self.is_simulation:
            self.set_volume_unit(qmixbus.UnitPrefix.micro, qmixpump.VolumeUnit.litres)
            self.set_flow_unit(qmixbus.UnitPrefix.micro, qmixpump.VolumeUnit.litres, qmixbus.TimeUnit.per_second)
            self.set_syringe_param(self.inner_diameter_mm, self.max_piston_stroke_mm)
        syringe_parameter = {
            'inner_diameter_mm': self.inner_diameter_mm,
            'max_piston_stroke_mm': self.max_piston_stroke_mm
        }
        self.volume = round((math.pi * math.pow(self.inner_diameter_mm / 2, 2) * self.max_piston_stroke_mm / 1000), 1)
        self.logger.info(f"Pump set with parameters: {syringe_parameter}, Syringe volume = {self.volume} mL")

    def calibration(self):
        if self.is_simulation:
            self.logger.info(f"{self} calibration simulated")
            return True
        else:
            self.logger.info(self.is_position_sensing_initialized())
            if self.is_position_sensing_initialized():
                self.logger.info(f"{self} is already calibrated")
            else:
                self.calibrate()
                timeout_timer = qmixbus.PollingTimer(30000)
                result = timeout_timer.wait_until(self.is_calibration_finished, True)
                if not result:
                    self.logger.info("Calibration timed out")
                else:
                    self.logger.info(f"{self} calibration complete")
            return self.is_position_sensing_initialized()

    def refill(self, flow=500, fill_level=None):
        if self.is_simulation:
            self.logger.info(f"Simulating refill with flow = {flow}")
            return True
        else:
            if fill_level is None:
                fill_level = self.get_volume_max()
            self.set_fill_level(fill_level, flow)
            timeout_timer = qmixbus.PollingTimer(30000)
            self.is_fill_complete = timeout_timer.wait_until(self.is_pumping, False)
            if self.is_fill_complete == False:
                self.stop_all_pumps()
                self.logger.info("Refill timed out, retry with a higher flow")
            else:
                self.logger.info(f"Refill of {self} complete")
            return self.is_fill_complete



class PumpSystem:
    def __init__(self, logger, number_of_pumps, inner_diameter_mm, max_piston_stroke_mm,
                 config_file, is_simulation, max_flow=8):
        self.logger = logger
        self.number_of_pumps = number_of_pumps
        self.inner_diameter_mm=inner_diameter_mm
        self.max_piston_stroke_mm=max_piston_stroke_mm
        self.max_flow = max_flow
        
        self.is_simulation=is_simulation
        self.pumps = [MyPump(self.logger, inner_diameter_mm=inner_diameter_mm, max_piston_stroke_mm=max_piston_stroke_mm,
                             is_simulation=self.is_simulation, max_flow=self.max_flow) for _ in range(self.number_of_pumps)]
        if not self.is_simulation:
            self.pumps[0].initialization(config_file)
            for i, pump in enumerate(self.pumps):
                pump.lookup_by_device_index(i)
            self.logger.info('')
        else:
            self.logger.info("Simulating pump initialization...")
            self.flows = np.zeros(self.number_of_pumps)
            
        self.stop_all_pumps()
        self.set_all_parameters()
        self.logger.info('Pump initialisation completed.')

    def get_flows(self):
        if not self.is_simulation:
            return np.array([pump.get_flow_is() for pump in self.pumps])
        else:
            return self.flows

    def get_fill_levels(self):
        if not self.is_simulation:
            return np.array([pump.get_fill_level() for pump in self.pumps])
        else:
            return np.array([0.5*self.pumps[0].volume*1000] * self.number_of_pumps)  # Simulated fill level

    def write_flows(self, experiment_directory, file_name="pump_flows.csv"):
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)
        file_path = os.path.join(experiment_directory, file_name)
        try:
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(self.get_flows())
        except Exception as e:
            self.logger.error(f"An error occurred while writing pump flows: {str(e)}")
        
    def generate_symmetric_flows(self):
        """
        Returns a set of symetric flows summing to the maximum flow.
        
        """
        return np.ones(self.number_of_pumps) * self.max_flow / self.number_of_pumps

    def generate_random_flows(self, bounds=None, total_var_flow=None, fixed_pumps=None, fixed_flows=None):
        """
        Returns a set of random flows summing to the maximum flow.
        
        """
        if bounds is None:
            flows = np.random.random(size=self.number_of_pumps)
            return flows / flows.sum() * self.max_flow
        else:
            # Generate random flow for the varying pumps
            nb_var = sum(~fixed_pumps)
            shuffle_indices = np.arange(nb_var)
            np.random.shuffle(shuffle_indices)

            var_bounds = bounds[~fixed_pumps]
            var_bounds = [var_bounds[i] for i in shuffle_indices]  # Shuffle the bounds to have uniform randomization
            shuffled_var_flows = np.zeros(nb_var)
            remain = total_var_flow
            for i in range(nb_var - 1):
                sum_inf = sum([var_bounds[j][0] for j in range(nb_var) if j > i])
                sum_sup = sum([var_bounds[j][1] for j in range(nb_var) if j > i])
                new_bounds = (max(remain - sum_sup, var_bounds[i][0]), min(remain - sum_inf, var_bounds[i][1]))
                shuffled_var_flows[i] = np.random.uniform(new_bounds[0], new_bounds[1])
                remain -= shuffled_var_flows[i]
            shuffled_var_flows[-1] = remain
            var_flows = np.zeros(nb_var)
            for i, index in enumerate(shuffle_indices):
                var_flows[index] = shuffled_var_flows[i]

            # Add the fixed flows
            j = 0
            flows = np.zeros(self.number_of_pumps)
            for i in range(self.number_of_pumps):
                if fixed_pumps[i]:
                    flows[i] = fixed_flows[i]
                else:
                    flows[i] = var_flows[j]
                    j += 1

            # Check if flows respect the bounds
            if not all([bounds[i][0] <= flow <= bounds[i][1] for i, flow in enumerate(flows)]):
                self.logger.error(f"Generated flows violate bounds: {flows}")
                raise ValueError('Problem with flow generation and pump bounds')

            return flows

    def generate_random_flows_sets(self, number_of_samples, bounds=None, total_var_flow=None, fixed_pumps=None, fixed_flows=None):
        """
        Returns sets of random flows summing to the maximum flow.
        
        """
        flows_sets = np.array([self.generate_random_flows(bounds, total_var_flow, fixed_pumps,
                                                          fixed_flows) for _ in range(number_of_samples)])
        return flows_sets

    def generate_lhs_flows_sets(self, number_of_samples, bounds=None,
                                total_var_flow=None, fixed_pumps=None,
                                fixed_flows=None):
        """
        Returns sets of flows summing to the maximum flow, sampled using the
        Latin Hypercube algorithm.  When *bounds* and *fixed_pumps* are given,
        the LHS samples respect per-pump bounds and the sum constraint.
        """
        if bounds is None:
            samples = lhs(self.number_of_pumps, samples=number_of_samples)
            flows_sets = np.zeros((number_of_samples, self.number_of_pumps))
            for i, sample in enumerate(samples):
                flows_sets[i] = sample / sample.sum() * self.max_flow
            return flows_sets

        nb_var = int(np.sum(~fixed_pumps))
        var_bounds = bounds[~fixed_pumps]

        samples = lhs(nb_var, samples=number_of_samples)
        flows_sets = np.zeros((number_of_samples, self.number_of_pumps))

        for s_idx, sample in enumerate(samples):
            scaled = np.array([
                lo + s * (hi - lo) for s, (lo, hi) in zip(sample, var_bounds)
            ])

            current_sum = scaled.sum()
            if current_sum > 0:
                scaled *= total_var_flow / current_sum

            scaled = np.clip(
                scaled,
                [b[0] for b in var_bounds],
                [b[1] for b in var_bounds],
            )

            remainder = total_var_flow - scaled.sum()
            if abs(remainder) > 1e-10:
                indices = np.argsort(-np.abs(scaled))
                for j in indices:
                    lo, hi = var_bounds[j]
                    room = (hi - scaled[j]) if remainder > 0 else (scaled[j] - lo)
                    if room > 0:
                        adjust = min(abs(remainder), room) * np.sign(remainder)
                        scaled[j] += adjust
                        remainder -= adjust
                        if abs(remainder) < 1e-10:
                            break

            j = 0
            for k in range(self.number_of_pumps):
                if fixed_pumps[k]:
                    flows_sets[s_idx, k] = fixed_flows[k]
                else:
                    flows_sets[s_idx, k] = scaled[j]
                    j += 1

        return flows_sets

    def prime_all_pumps(self):
        if not self.is_simulation:
            for pump in self.pumps:
                pump.generate_flow(pump.max_flow)
                time.sleep(5)
                self.stop_all_pumps()
        else:
            self.logger.info("Simulating pump priming")

    def check_for_faults(self):
        if not self.is_simulation:
            for pump in self.pumps:
                if pump.is_in_fault_state():
                    return True
            return False
        else:
            return False

    def clear_all_faults(self):
        if not self.is_simulation:
            for pump in self.pumps:
                pump.clear_fault()
        else:
            self.logger.info("Simulating fault clearing")

    def acid_wash(self, acid_pump=0):
        if not self.is_simulation:
            self.stop_all_pumps()
            self.pumps[acid_pump].generate_flow(self.pumps[acid_pump].max_flow)
            time.sleep(5)
            self.stop_all_pumps()
            self.logger.info("Acid wash complete")
        else:
            self.logger.info("Simulating acid wash")

    def set_all_parameters(self):
        for pump in self.pumps:
                pump.set_parameters()

    def calibration_all(self):
        for pump in self.pumps:
            pump.calibration()

    def refill_all(self, flow=500):
        self.logger.info(f'{self} is about to be refilled to max level')
        while True:
            proceed = input("Do you want to proceed with the filling? (y/n)")
            if proceed == "y":
                for pump in self.pumps:
                    pump.refill(flow)
                break
            else:
                self.logger.info("Refill aborted")
                break

    def set_flows(self, flows):
        """
        Sets the flow of all the pumps.

        Parameters
        ----------
        flows : array-like of size (nb_of_pumps).
            The flow rates at which to set the pumps.

        Raises
        ------
        ValueError
            if the input flows doesn't correspond to the number of pumps.

        """
        if len(flows) != len(self.pumps):
            raise ValueError("The length of the 'flows' list must be equal to the number of pumps.")
        if not self.is_simulation:
            for i in range(self.number_of_pumps):
                self.pumps[i].generate_flow(flows[i])
            time.sleep(0.35)
        else:
            self.logger.info(f"Simulating generating flows: {flows}")
            self.flows = flows

    def stop_all_pumps(self):
        if not self.is_simulation:
            self.pumps[0].stop_all_pumps()
            self.logger.info("Pumps stopped")
        else:
            self.logger.info("Simulating stopping all pumps")
            self.flows = np.zeros(self.number_of_pumps)