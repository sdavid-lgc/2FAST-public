"""
Chip and UV-cell washes (C3000 acid / water syringes, back pressure) and the emergency stop.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
from typing import Optional

from toofast import clock


class WashMixin:
    """Chip and UV-cell washes (C3000 acid / water syringes, back pressure) and the emergency stop (part of `Experiment`)."""

    def flush_chip(self):
        """Flush the chip with nitric acid and water from c3000 pumps
        Reagent pumps are activated at low flow to maintain pressure in the syringes
        """
        ev = self._abort_event
        # Flush acid
        if self._check_abort(): return
        self.c3000_1.switch_valve_to_position("left")
        clock.sleep(0.2)
        if self._check_abort(): return
        self.c3000_2.switch_valve_to_position("right")
        clock.sleep(0.1)
        self.c3000_1.switch_valve_to_position("left")  # Remove when proper valve position check is implemented
        back_pressure_flow = float(self._gui_get(self.app.back_pressure_entry))
        flow_set = [back_pressure_flow]*self.number_of_pumps
        self.pump_syst.set_flows(flow_set)
        self.app.logger.info(f"set flow to {flow_set}")
        volume_acid = float(self._gui_get(self.app.nitric_acid_volume_entry))
        if self._check_abort(): return
        self.app.logger.info(f"dispense {volume_acid} µL of acid")
        self.c3000_2.dispense(volume=volume_acid, flow=10)
        self.c3000_2.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        clock.sleep(0.5)
        # Maintain pressure in reagent syringes
        self.c3000_1.switch_valve_to_position("right")
        clock.sleep(0.2)
        self.c3000_2.switch_valve_to_position("left")
        volume_eau = float(self._gui_get(self.app.water_volume_entry))
        if self._check_abort(): return
        self.app.logger.info(f"dispense {volume_eau} µL of water")
        self.c3000_1.dispense(volume=volume_eau, flow=11)
        self.c3000_1.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        self.pump_syst.stop_all_pumps()
        self.c3000_1.switch_valve_to_position("left")
        self.app.logger.info("Chip flush done.")

    def flush_chip_intensive(self):
        """Intensive chip flush: more acid, split water phase with mid-rinse back-pressure stop.

        Volumes and split point are read from the intensive-wash entries of the GUI.
        """
        ev = self._abort_event
        volume_acid = float(self._gui_get(self.app.intensive_acid_volume_entry))
        volume_water_1 = float(self._gui_get(self.app.intensive_water_bp_stop_entry))
        volume_water_total = float(self._gui_get(self.app.intensive_water_volume_entry))
        volume_water_2 = max(0.0, volume_water_total - volume_water_1)

        # Acid phase
        if self._check_abort(): return
        self.c3000_1.switch_valve_to_position("left")
        clock.sleep(0.2)
        if self._check_abort(): return
        self.c3000_2.switch_valve_to_position("right")
        clock.sleep(0.1)
        self.c3000_1.switch_valve_to_position("left")
        back_pressure_flow = float(self._gui_get(self.app.back_pressure_entry))
        flow_set = [back_pressure_flow] * self.number_of_pumps
        self.pump_syst.set_flows(flow_set)
        if self._check_abort(): return
        self.app.logger.info(f"[intensive] dispense {volume_acid} µL acid")
        self.c3000_2.dispense(volume=volume_acid, flow=10)
        self.c3000_2.wait_for_idle(stop_event=ev)
        if self._check_abort(): return
        clock.sleep(0.5)

        # Water phase 1: with back-pressure
        self.c3000_1.switch_valve_to_position("right")
        clock.sleep(0.2)
        self.c3000_2.switch_valve_to_position("left")
        if self._check_abort(): return
        self.app.logger.info(f"[intensive] dispense {volume_water_1} µL water (back-pressure on)")
        self.c3000_1.dispense(volume=volume_water_1, flow=11)
        self.c3000_1.wait_for_idle(stop_event=ev)
        if self._check_abort(): return

        # Stop back-pressure from reagent syringes
        self.pump_syst.stop_all_pumps()

        # Water phase 2: without back-pressure
        if volume_water_2 > 0:
            self.app.logger.info(f"[intensive] dispense {volume_water_2} µL water (back-pressure off)")
            self.c3000_1.dispense(volume=volume_water_2, flow=11)
            self.c3000_1.wait_for_idle(stop_event=ev)
            if self._check_abort(): return

        self.c3000_1.switch_valve_to_position("left")
        self.app.logger.info("Intensive chip flush done.")

    def _compute_ag_pvp_ratio(self, flows) -> Optional[float]:
        """Return molar Ag/PVP flow ratio for the given flow set, or None if not determinable."""
        ag_molar_flow = None
        pvp_molar_flow = None
        for i in range(self.number_of_pumps):
            reagent = self._gui_get(self.app.selected_reagents[i])
            try:
                conc = float(self._gui_get(self.app.reagent_concentrations[i]))
            except (ValueError, TypeError):
                continue
            flow = flows[i] if i < len(flows) else 0.0
            if reagent == "AgNO3":
                ag_molar_flow = flow * conc
            elif reagent == "PVP":
                pvp_molar_flow = flow * conc
        if ag_molar_flow is None or pvp_molar_flow is None or pvp_molar_flow == 0:
            return None
        return ag_molar_flow / pvp_molar_flow

    def _should_do_intensive_wash(self, flows) -> bool:
        """Return True if conditions require an intensive wash."""
        self.app.logger.debug(
            f"Wash counter before decision: {self._wash_sample_counter} "
            f"(samples since last intensive wash)")
        if self._gui_get(self.app.regular_wash_var):
            if self._wash_sample_counter >= 9:
                self.app.logger.info(
                    f"Regular intensive wash triggered (counter={self._wash_sample_counter})")
                return True
        if self._gui_get(self.app.pvp_threshold_wash_var):
            try:
                threshold = float(self._gui_get(self.app.pvp_threshold_entry))
            except (ValueError, TypeError):
                threshold = 3.0
            ratio = self._compute_ag_pvp_ratio(flows)
            if ratio is not None and ratio > threshold:
                self.app.logger.info(
                    f"PVP-threshold intensive wash triggered (Ag/PVP={ratio:.2f} > {threshold})")
                return True
        return False

    def trigger_manual_wash(self):
        """Run an intensive chip flush immediately (called from the manual wash button)."""
        self.app.logger.info("Manual intensive wash triggered.")
        self._wash_sample_counter = 0
        self._gui(self.app.update_wash_counter_display)
        self.flush_chip_intensive()

    def _emergency_hardware_stop(self):
        """Bring all hardware to a safe state.

        Called by abort/stop_all. Each step is wrapped in try/except so that one
        failure (e.g. unconnected MFC) does not prevent the others from running.
        Safe state:
          - Nemesys pumps: stopped
          - C3000 pumps: stopped, valves to "left" (refill)
          - MFC: flow set to 0
          - Collector: valve to trash, Z bar to 999
        Resets the wash counter as a new session boundary.
        """
        self.logger.warning("Emergency hardware stop initiated.")
        try:
            self.pump_syst.stop_all_pumps()
        except Exception as e:
            self.logger.error(f"Emergency stop: Nemesys stop failed: {e}")
        for pump, name in [(getattr(self, 'c3000_1', None), "C3000_1"),
                           (getattr(self, 'c3000_2', None), "C3000_2")]:
            if pump is None:
                continue
            try:
                pump.stop_pump()
            except Exception as e:
                self.logger.error(f"Emergency stop: {name} stop failed: {e}")
            try:
                pump.switch_valve_to_position("left")
            except Exception as e:
                self.logger.error(f"Emergency stop: {name} valve→left failed: {e}")
        try:
            if getattr(self, 'mfc', None) is not None:
                self.mfc.set_flow(0)
        except Exception as e:
            self.logger.error(f"Emergency stop: MFC set_flow(0) failed: {e}")
        try:
            if getattr(self, 'collector', None) is not None and self.collector_status:
                self.collector.switch_collection_valve(0)
                self.collector.move_z_to_position(999)
        except Exception as e:
            self.logger.error(f"Emergency stop: collector failed: {e}")

        self._wash_sample_counter = 0
        self._gui(self.app.update_wash_counter_display)
        self.logger.warning("Emergency hardware stop done.")

    def trigger_emergency_stop(self):
        """Public entry point for the Stop All button (always available).

        Sets the abort event so any cooperative loops exit, then performs the
        hardware stop on the calling thread (typically a daemon thread spawned
        by the GUI).
        """
        self._abort_event.set()
        # Brief pause to let in-flight serial transactions complete before we
        # touch the shared C3000 serial port from this thread.
        clock.sleep(1.0)
        self._emergency_hardware_stop()
        self._abort_event.clear()

    def perform_acid_wash(self):
        """Perform an acid wash for cleaning."""
        acid_pump_index = 0
        for i in range(self.number_of_pumps):
            if self._gui_get(self.app.selected_reagents[i]) == "Acetic Acid":
                acid_pump_index = i
                break

        self.pump_syst.acid_wash(acid_pump_index)
        self.logger.info("Acid wash done.")

    def perform_acid_wash_and_reference_measurement(self):
        """Perform an acid wash and measure the reference spectrum."""
        self.dispense_acid_and_flush()
        clock.sleep(5)
        while True:
            self.collector.open_lamp_shutter()
            self.measure_reference()
            self.collector.close_lamp_shutter()
            if self._gui_call(self.app.ask_for_spectrum_validation):
                break

    def dispense_acid_and_flush(self):
        """Dispense acid and flush the system."""
        # c3000_2 is used for nitric acid, c3000_1 for water
        self.c3000_1.switch_valve_to_position("left")
        self.c3000_2.switch_valve_to_position("center")
        clock.sleep(0.5)
        self.c3000_2.generate_flow(20)
        clock.sleep(5)
        self.c3000_2.stop_pump()
        clock.sleep(0.5)
        self.c3000_2.switch_valve_to_position("left")
        self.c3000_1.switch_valve_to_position("center")
        clock.sleep(0.5)
        self.c3000_1.generate_flow(100)
        clock.sleep(20)
        self.c3000_1.stop_pump()
        self.c3000_1.switch_valve_to_position("left")
        clock.sleep(0.5)

    def flush_nitric_acid(self):
        """Flush the system with nitric acid."""
        self.c3000_2.dispense(volume=25, flow=10)
        clock.sleep(0.5)
