"""
Multi-syringe pump system, independent of the pump vendor.

`PumpSystem` holds one `SyringePump` per reagent (built by a `PumpBackend`: the simulated one of this package, or a
laboratory driver for the real pumps — see ``toofast.hardware.drivers``) and provides what the experiment needs:
set / read the flows, stop, prime, wash, syringe levels, and the generation of flow sets summing to the total flow (random, Latin hypercube on the simplex, projected LHS).

Original code: Sylvain David (2023), Nemesys-specific; the vendor calls moved to the backend. Random and
projected-LHS generators unchanged (the projected LHS was called "latinhypercube" in toofast 1.0).
"""
import csv
import os
from typing import List, Optional

import numpy as np
from pyDOE import lhs

from toofast import clock
from toofast.core.sampling import simplex_lhs
from toofast.hardware.base import PumpBackend, SyringePump


class PumpSystem:
    def __init__(self, logger, pumps: List[SyringePump], max_flow: float = 8,
                 backend: Optional[PumpBackend] = None):
        self.logger = logger
        self.pumps = list(pumps)
        self.number_of_pumps = len(self.pumps)
        self.max_flow = max_flow
        self.backend = backend
        self.is_simulation = all(p.is_simulation for p in self.pumps)

        self.stop_all_pumps()
        self.set_all_parameters()
        self.logger.info('Pump initialisation completed.')

    @property
    def inner_diameter_mm(self):
        return self.pumps[0].inner_diameter_mm

    @property
    def max_piston_stroke_mm(self):
        return self.pumps[0].max_piston_stroke_mm

    def get_flows(self):
        return np.array([pump.get_flow_is() for pump in self.pumps])

    def get_fill_levels(self):
        return np.array([pump.get_fill_level() for pump in self.pumps])

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
        Returns sets of flows summing to the maximum flow: Latin hypercube on the simplex of compositions
        (toofast.core.sampling.simplex_lhs) for the varying pumps, fixed pumps at their fixed flow.
        Without *bounds*, every pump varies in [0, max_flow] and the flows sum to max_flow.
        """
        if bounds is None:
            return simplex_lhs(number_of_samples, [(0.0, self.max_flow)] * self.number_of_pumps, self.max_flow)
        fixed_pumps = np.asarray(fixed_pumps, dtype=bool)
        var = simplex_lhs(number_of_samples, np.asarray(bounds, dtype=float)[~fixed_pumps], total_var_flow)
        flows_sets = np.zeros((number_of_samples, self.number_of_pumps))
        flows_sets[:, ~fixed_pumps] = var
        if fixed_pumps.any():
            flows_sets[:, fixed_pumps] = np.asarray(fixed_flows, dtype=float)[fixed_pumps]
        return flows_sets

    def generate_projected_lhs_flows_sets(self, number_of_samples, bounds=None,
                                          total_var_flow=None, fixed_pumps=None,
                                          fixed_flows=None):
        """
        Projected LHS (the initial design of the paper's runs, toofast 1.0 "latinhypercube"): a Latin hypercube of
        the flows rescaled to the total flow (then clipped to the bounds and the remainder redistributed). The
        rescaling breaks the Latin property; see generate_lhs_flows_sets for a Latin hypercube on the simplex.
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
        for pump in self.pumps:
            pump.generate_flow(pump.max_flow)
            clock.sleep(5)
            self.stop_all_pumps()

    def check_for_faults(self):
        return any(pump.is_in_fault_state() for pump in self.pumps)

    def clear_all_faults(self):
        for pump in self.pumps:
            pump.clear_fault()

    def acid_wash(self, acid_pump=0):
        self.stop_all_pumps()
        self.pumps[acid_pump].generate_flow(self.pumps[acid_pump].max_flow)
        clock.sleep(5)
        self.stop_all_pumps()
        self.logger.info("Acid wash complete")

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
        for i in range(self.number_of_pumps):
            self.pumps[i].generate_flow(flows[i])
        clock.sleep(0.35)

    def stop_all_pumps(self):
        if self.backend is not None:
            self.backend.stop_all(self.pumps)
        else:
            for pump in self.pumps:
                pump.stop()
        self.logger.info("Pumps stopped")
