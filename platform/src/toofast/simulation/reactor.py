"""
Virtual stop-flow reactor: what the simulated spectrometer sees in the UV cell.

A deliberately simple, documented toy model of seed-mediated silver nanoparticle growth, used only to run the
platform without hardware (GUI demo, headless BO campaigns, tests). It is NOT a physical model and is not used
by any result of the paper. It reproduces the qualitative behaviour the platform logic has to cope with:

- a plasmon band whose position, width and amplitude depend on the composition (flows × stock concentrations);
- no particles (flat spectrum) without reducer or without silver;
- a growth kinetics after the flow is stopped, so that the growth-termination rule converges after a
  composition-dependent time (a few seconds to a few minutes);
- measurement noise.

State machine, driven by the simulated hardware:
- reagent pumps flowing while the cell is in line (10-port valve at 0)  → the cell holds a fresh mixture of the
  current composition (age reset continuously);
- flow stopped, or cell isolated (10-port valve at 1)                   → the mixture ages (stop-flow growth);
- rinse (C3000 dispensing) while the cell is in line                   → the cell holds clean water.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from toofast import clock

# Stock concentrations (mM) of the paper's standard reagents (SI Table S2; PVP per chain, 40 kDa).
DEFAULT_STOCKS_mM: Dict[str, float] = {
    "AgNO3": 1.8, "PVP": 1.2, "Acetic Acid": 1256.0, "Ascorbic Acid": 1.8, "NaBH4": 0.036,
}
DEFAULT_REAGENTS = ("AgNO3", "PVP", "Acetic Acid", "Ascorbic Acid", "NaBH4")


@dataclass
class ToyAgNPModel:
    """Composition (final concentrations, mM) → final spectrum and growth time constant.

    All coefficients are arbitrary but fixed: the landscape is smooth, has a broad optimum for any λmax target in
    ~400–650 nm, and a "no particle" region (low AgNO3 or NaBH4), like the real chemistry.
    """

    ag_min_mM: float = 0.05            # below: no particles
    bh4_min_mM: float = 1.2e-3          # below: no nucleation
    lambda_seed_nm: float = 398.0       # small spherical seeds
    lambda_span_nm: float = 300.0       # max red shift (anisotropic growth)
    width_seed_nm: float = 32.0
    width_span_nm: float = 90.0
    amplitude_per_mM: float = 3.2       # extinction per mM of reduced Ag
    tau_min_s: float = 3.0
    tau_max_s: float = 180.0

    def describe(self, c: Dict[str, float]) -> Dict[str, float]:
        ag, bh4 = c.get("AgNO3", 0.0), c.get("NaBH4", 0.0)
        asc, pvp, acoh = c.get("Ascorbic Acid", 0.0), c.get("PVP", 0.0), c.get("Acetic Acid", 0.0)
        if ag < self.ag_min_mM or bh4 < self.bh4_min_mM:
            return dict(flat=True, lambda_max=0.0, width=0.0, amplitude=0.0, tau=self.tau_min_s)
        seeds = bh4 / ag                                   # nucleation strength (seeds per Ag)
        growth = asc / ag                                  # growth reducer per Ag
        capping = pvp / ag                                 # PVP limits anisotropic growth
        acid = acoh / 250.0                                # acid slows the reduction by ascorbate
        drive = growth / (1.0 + 40.0 * seeds) / (1.0 + 0.6 * capping) * (1.0 + 0.5 * acid)
        h = drive / (1.0 + drive)                          # 0 (spheres) → 1 (large anisotropic)
        reduced = ag * (0.35 + 0.65 * min(1.0, (bh4 / 0.004) + growth))
        tau = self.tau_min_s + (self.tau_max_s - self.tau_min_s) * h * (1.0 + acid) / (1.0 + acid + 4.0 * seeds)
        return dict(flat=False,
                    lambda_max=self.lambda_seed_nm + self.lambda_span_nm * h,
                    width=self.width_seed_nm + self.width_span_nm * h,
                    amplitude=min(2.5, self.amplitude_per_mM * reduced),
                    tau=float(np.clip(tau, self.tau_min_s, self.tau_max_s)))

    def absorbance(self, wl: np.ndarray, c: Dict[str, float], age_s: float) -> np.ndarray:
        """Extinction spectrum of the mixture `age_s` seconds after mixing."""
        d = self.describe(c)
        if d["flat"]:
            return np.zeros_like(wl, dtype=float)
        progress = 1.0 - np.exp(-max(age_s, 0.0) / d["tau"])           # 0 → 1
        lam = self.lambda_seed_nm + (d["lambda_max"] - self.lambda_seed_nm) * progress
        width = self.width_seed_nm + (d["width"] - self.width_seed_nm) * progress
        amp = d["amplitude"] * (0.15 + 0.85 * progress)
        band = amp * np.exp(-0.5 * ((wl - lam) / width) ** 2)
        interband = 0.18 * amp * np.exp(-(wl - 300.0) / 35.0)            # Ag interband edge below ~350 nm
        quadrupole = 0.12 * amp * progress * np.exp(-0.5 * ((wl - 340.0) / 12.0) ** 2) * (d["lambda_max"] > 520)
        return band + interband + quadrupole


@dataclass
class VirtualReactor:
    """Contents of the UV cell, updated by the simulated pumps / valves / rinse syringes."""

    reagents: Sequence[str] = DEFAULT_REAGENTS
    stocks_mM: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STOCKS_mM))
    model: ToyAgNPModel = field(default_factory=ToyAgNPModel)
    noise_au: float = 0.002
    seed: int = 0
    settle_s: float = 1.0      # flows must be steady this long to replace the cell contents

    def __post_init__(self):
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.seed)
        self._flows = np.zeros(len(self.reagents))
        self._cell_in_line = True
        self._composition: Optional[Dict[str, float]] = None   # None = clean water
        self._t_mix = clock.time()
        self._flows_changed_at = clock.time()

    # -- events from the simulated hardware ------------------------------------------------------------------
    def set_reagent_names(self, reagents: Sequence[str]) -> None:
        with self._lock:
            self.reagents = tuple(reagents)
            self._flows = np.zeros(len(self.reagents))
            self._t_mix = self._flows_changed_at = clock.time()

    def set_pump_flow(self, index: int, flow: float) -> None:
        with self._lock:
            self._update()           # a flowing cell holds a fresh mixture up to this instant
            if index >= len(self._flows):
                self._flows = np.concatenate([self._flows, np.zeros(index + 1 - len(self._flows))])
            if self._flows[index] != max(0.0, float(flow)):
                self._flows[index] = max(0.0, float(flow))
                self._flows_changed_at = clock.time()

    def set_cell_in_line(self, in_line: bool) -> None:
        with self._lock:
            self._update()           # contents before the switch
            self._cell_in_line = bool(in_line)
            self._update()

    def rinse(self) -> None:
        """Rinse syringe dispensing: cleans the cell if it is in the flow path."""
        with self._lock:
            if self._cell_in_line:
                self._composition = None

    # -- state -----------------------------------------------------------------------------------------------
    def _update(self) -> None:
        """While reagents flow steadily through the in-line cell, its contents are the current mixture, aged 0 s.

        Flow changes shorter than `settle_s` (pumps started / stopped one after the other) do not replace the
        contents: the cell keeps the mixture of the last steady flow."""
        total = float(self._flows.sum())
        steady = clock.time() - self._flows_changed_at >= self.settle_s
        if self._cell_in_line and total > 0 and steady:
            self._composition = self.concentrations(self._flows)
            self._t_mix = clock.time()

    def concentrations(self, flows: Sequence[float]) -> Dict[str, float]:
        flows = np.asarray(flows, dtype=float)
        total = float(flows.sum())
        if total <= 0:
            return {}
        return {r: self.stocks_mM.get(r, 0.0) * q / total for r, q in zip(self.reagents, flows)}

    def age(self) -> float:
        return clock.time() - self._t_mix

    def absorbance(self, wavelengths: np.ndarray) -> np.ndarray:
        """Noisy extinction spectrum currently in the cell."""
        with self._lock:
            self._update()
            comp, age = self._composition, self.age()
            noise = self._rng.normal(0.0, self.noise_au, size=len(wavelengths))
        clean = np.zeros(len(wavelengths)) if comp is None else self.model.absorbance(np.asarray(wavelengths), comp, age)
        return clean + noise
