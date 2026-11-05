"""
Flat (no-particle) spectra and physics ghost points at GP-fit time: the flat sentinel cost, the
remapping strategy of flat observations and the virtual 'no Ag' / 'no NaBH4' points.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
import numpy as np

from toofast.core.cost import huber_cost_nm


class FlatStrategyMixin:
    """Flat (no-particle) spectra and physics ghost points at GP-fit time: the flat sentinel cost, the (part of `Experiment`)."""

    def _flat_penalty_value(self) -> float:
        """Raw flat-spectrum penalty (max cost over the measurement domain).

        Single source of truth: used both by cost_function (what is stored)
        and as the sentinel to detect flat points in _apply_flat_strategy.
        Style-aware: the PCA target uses a PCA-score sentinel (see
        _pca_flat_sentinel) so the flat machinery and ghosts stay in the right
        units.
        """
        if self.target_style == 'pca':
            return self._pca_flat_sentinel()
        preset = getattr(self, "bo_preset_override", "default") or "default"
        use_huber = (preset == "robust")
        delta_nm = float(getattr(self, "huber_delta_nm_override", 30.0) or 30.0)
        divisor = self.tg_wl if self.rel_tol else 1
        min_wavelength, max_wavelength = 315, 750
        if use_huber:
            return max(
                huber_cost_nm(min_wavelength, self.tg_wl, delta_nm),
                huber_cost_nm(max_wavelength, self.tg_wl, delta_nm),
            ) / (divisor ** 2)
        return max(
            ((min_wavelength - self.tg_wl) / divisor) ** 2,
            ((max_wavelength - self.tg_wl) / divisor) ** 2,
        )

    def _pca_flat_sentinel(self) -> float:
        """Deterministic flat-spectrum cost for the PCA target.

        A no-particle spectrum is ~zero absorbance; its PCA cost vs the target
        is a reproducible, physically meaningful "no particle" penalty. Cached
        so the exact same value is both stored (cost_function) and matched as
        the sentinel in _apply_flat_strategy / carried by the ghost points.
        """
        cached = getattr(self, "_pca_flat_sentinel_value", None)
        if cached is not None:
            return cached
        wl = self.pca_model.wavelengths
        scores_flat = self.pca_model.transform(wl, np.zeros_like(wl, dtype=float))
        # Use the SAME metric as the live cost (sqdist/dist/cosine) so the flat
        # sentinel sits on the real cost scale and flat detection stays valid.
        val = float(self.pca_cost_fn._base_cost(scores_flat))
        # Stay consistent with PCASpectrumCost.__call__: a zero-amplitude
        # spectrum incurs the full amplitude term ((0 - A_t)/A_t)**2 = 1.
        if getattr(self.pca_cost_fn, "amp_weight", 0.0) > 0.0:
            val += float(self.pca_cost_fn.amp_weight)
        self._pca_flat_sentinel_value = val
        return val

    def _build_ghost_points_full(self):
        """Generate physics-based ghost flows in full pump space.

        For each varying pump carrying a reagent in ``ghost_reagent_keywords``
        (AgNO3, NaBH4), we sample ``n_per_face`` points on the hyperface
        ``flow_of_that_reagent = 0`` (sum of the remaining varying flows =
        total_var_flow, via Dirichlet). The ghost cost is the flat-spectrum
        sentinel so it gets remapped by the flat_strategy.

        Returns
        -------
        (ghost_full, ghost_costs) : (np.ndarray, np.ndarray) or (None, None)
            Ghost flows in full pump space and their sentinel costs.
        """
        if self.reagents is None or self.varying_pumps is None:
            return None, None

        keywords = tuple(k.lower() for k in self.ghost_reagent_keywords)
        var_global_idx = np.where(self.varying_pumps)[0]
        n_var = int(len(var_global_idx))
        if n_var < 2:
            # Need at least one other varying pump to absorb the flow.
            return None, None

        # Map: list of (zero_local_idx, zero_global_idx) for each ghost face.
        target_faces = []
        for local_i, global_i in enumerate(var_global_idx):
            r = str(self.reagents[global_i]).strip().lower()
            if any(k in r for k in keywords):
                target_faces.append((local_i, int(global_i)))

        if not target_faces:
            return None, None

        # Number of ghost points per face scales with the GP input dimension.
        # GP dim = n_var - 1 (sum constraint removes one DOF).
        # 2 var pumps (1D GP) → 3, 3 (2D) → 4, 4 (3D) → 6, 5 (4D) → 8.
        dim = max(1, n_var - 1)
        n_per_face = max(3, 2 * dim)

        # Resolve fixed-pump contributions to the full flow template.
        full_template = np.zeros(self.number_of_pumps, dtype=float)
        if self.fixed_pumps is not None and self.fixed_flows is not None:
            for k in range(self.number_of_pumps):
                if self.fixed_pumps[k] and self.fixed_flows[k] is not None:
                    full_template[k] = float(self.fixed_flows[k])

        rng = np.random.default_rng(seed=20251201)  # deterministic ghosts
        penalty = float(self._flat_penalty_value())
        ghost_full = []

        for zi_local, zi_global in target_faces:
            remaining_locals = [j for j in range(n_var) if j != zi_local]
            if not remaining_locals:
                continue

            if len(remaining_locals) == 1:
                # Only one absorber: the corner point (Ag=0, other=total_var_flow).
                full = full_template.copy()
                full[var_global_idx[remaining_locals[0]]] = self.total_var_flow
                ghost_full.append(full)
                continue

            for _ in range(n_per_face):
                alpha = np.ones(len(remaining_locals))
                sample = rng.dirichlet(alpha) * self.total_var_flow
                full = full_template.copy()
                for j, loc in enumerate(remaining_locals):
                    full[var_global_idx[loc]] = sample[j]
                # zi_global stays at 0
                ghost_full.append(full)

        if not ghost_full:
            return None, None

        ghost_full = np.asarray(ghost_full, dtype=float)
        ghost_costs = np.full(len(ghost_full), penalty, dtype=float)
        return ghost_full, ghost_costs

    def _augment_with_ghosts(self, X, costs):
        """Append ghost points (reduced to GP input space) to (X, costs).

        Ghosts are appended at the END so the last ``n_ghosts`` rows of the
        returned arrays are the ghost points (used to build the per-point
        alpha vector in _apply_flat_strategy).

        Returns
        -------
        (X, costs, n_ghosts) : the augmented arrays and the number of ghost
        rows appended (0 if ghosts are disabled, no relevant reagent varies,
        or the GP space cannot accept them).
        """
        if not getattr(self, "use_physics_ghosts", True):
            return X, costs, 0
        if getattr(self, "multi_objective", False):
            # Ghosts carry only a (flat) PCA cost, not a growth time, so they
            # don't fit the 2-objective scalarization cleanly. Disabled for the
            # ParEGO path in v1 (flats are still handled via the sentinel remap).
            return np.asarray(X, dtype=float), np.asarray(costs, dtype=float), 0
        ghost_full, ghost_costs = self._build_ghost_points_full()
        if ghost_full is None or len(ghost_full) == 0:
            return X, costs, 0
        # Apply the same reduction as the real flows: keep varying pumps,
        # drop the last column (its value is set by the sum constraint).
        ghost_reduced = ghost_full[:, self.varying_pumps][:, :-1]
        if ghost_reduced.shape[1] != X.shape[1]:
            # Shape mismatch (e.g. n_var=1 corner case) — skip rather than crash.
            self.logger.debug(
                "Ghost dim %d != X dim %d, skipping ghost injection",
                ghost_reduced.shape[1], X.shape[1])
            return X, costs, 0
        self.logger.debug(
            "Injecting %d physics ghost point(s) into GP training",
            len(ghost_reduced))
        return (np.concatenate([X, ghost_reduced], axis=0),
                np.concatenate([costs, ghost_costs], axis=0),
                len(ghost_reduced))

    def _apply_flat_strategy(self, X, costs):
        """Re-map flat-spectrum costs at GP-fit time per flat_strategy_override.

        Parameters
        ----------
        X : (m, d) array
            Reduced flows actually fed to the GP (flows_minus_one).
        costs : (m,) array
            Raw costs as stored (flat points hold the _flat_penalty_value).

        Returns
        -------
        (X_fit, costs_fit) : possibly row-filtered (for 'exclude'). Also
        stashes self._feas_X / self._feas_label (all points, reduced space)
        for the optional feasibility classifier.
        """
        X = np.asarray(X, dtype=float)
        costs = np.asarray(costs, dtype=float)

        # Inject physics-based ghost points before fitting. They carry the
        # flat-spectrum sentinel cost so they get remapped by 'nearest_high'
        # like any real flat measurement — no extreme outlier in the GP.
        X, costs, n_ghosts = self._augment_with_ghosts(X, costs)

        # Ghost rows are the last n_ghosts entries (see _augment_with_ghosts).
        ghost_mask = np.zeros(len(X), dtype=bool)
        if n_ghosts > 0:
            ghost_mask[-n_ghosts:] = True

        strategy = getattr(self, "flat_strategy_override", "nearest_high") or "nearest_high"
        if getattr(self, "multi_objective", False) and strategy == "exclude":
            # 'exclude' drops rows, breaking the row-alignment the ParEGO
            # scalarization relies on. Fall back to the row-preserving default.
            self.logger.info("multi_objective: 'exclude' flat_strategy -> nearest_high")
            strategy = "nearest_high"

        sentinel = float(self._flat_penalty_value())
        flat_mask = np.isclose(costs, sentinel, rtol=1e-9, atol=1e-9)

        # Per-point GP noise: ghosts get a larger diagonal alpha so the GP
        # treats them as soft constraints (see __init__ note on boundary_hit).
        # NB: the GP fits with normalize_y=True, so y is scaled to unit
        # variance before alpha is added to the kernel diagonal. The ghost
        # alpha is therefore expressed in NORMALIZED-target units (a fraction
        # of the signal variance) and is scale-invariant — a raw-cost value
        # like (0.2*std)^2 would be negligible for small costs. 'auto' uses
        # 0.05 (~ghosts carry ~22% relative uncertainty): low enough that the GP
        # mean actually RISES toward the ghost value (so the infeasible face
        # shows a soft slope, not just a variance dip), while still keeping the
        # length scale off its lower bound. Paired with ghost_penalty_lift.
        gnl = getattr(self, "ghost_noise_level", "auto")
        if gnl == "auto" or gnl is None:
            self._fit_ghost_alpha = 0.05
        else:
            self._fit_ghost_alpha = float(gnl)
        # Default alignment (full X order); overridden in the 'exclude' branch.
        self._fit_ghost_mask = ghost_mask

        # Soft warm-start: the leading _n_warmstart rows are historical points
        # (e.g. old .h5 data). They get a larger per-point alpha so the GP
        # treats them as a soft prior — robust to drift in stale experiments —
        # rather than trusting them as exact, fresh observations. Ghosts are
        # appended after, so warm-start rows keep their leading positions.
        n_ws = int(getattr(self, "_n_warmstart", 0))
        ws_mask = np.zeros(len(X), dtype=bool)
        if n_ws > 0:
            ws_mask[: min(n_ws, len(X))] = True
        wnl = getattr(self, "warmstart_noise_level", "auto")
        self._fit_warmstart_alpha = 0.15 if (wnl == "auto" or wnl is None) else float(wnl)
        self._fit_warmstart_mask = ws_mask

        # Stash labels for the (optional) feasibility classifier: 1 = feasible
        self._feas_X = X.copy()
        self._feas_label = (~flat_mask).astype(int)

        n_flat = int(flat_mask.sum())
        if n_flat == 0 or strategy == "penalty_max":
            self._gp_display_costs = costs.copy()
            return X, costs

        real_mask = ~flat_mask
        if real_mask.sum() == 0:
            self._gp_display_costs = costs.copy()
            return X, costs  # nothing to anchor on; leave as-is

        out = costs.copy()
        real_costs = costs[real_mask]

        if strategy == "exclude":
            # Flat points are not modelled; for display keep their true
            # measured cost (they will visibly sit off the GP surface,
            # which honestly reflects "excluded").
            self._gp_display_costs = costs.copy()
            # Ghosts carry the sentinel → they are in flat_mask → dropped here,
            # so realign the ghost/warm-start masks to the filtered rows.
            self._fit_ghost_mask = ghost_mask[real_mask]
            self._fit_warmstart_mask = ws_mask[real_mask]
            self.logger.info("flat_strategy=exclude -> %d flat point(s) dropped from GP fit", n_flat)
            return X[real_mask], costs[real_mask]

        if strategy == "penalty_moderate":
            factor = float(getattr(self, "flat_penalty_factor_override", 2.0) or 2.0)
            out[flat_mask] = factor * float(np.mean(real_costs))
            self._gp_display_costs = out.copy()
            self.logger.info(
                "flat_strategy=penalty_moderate -> %d flat cost(s) set to %.4g",
                n_flat, factor * float(np.mean(real_costs)))
            return X, out

        # nearest_high (default)
        margin = float(getattr(self, "flat_nearest_margin_override", 1.25) or 1.25)
        cap_mult = float(getattr(self, "flat_nearest_cap_override", 3.0) or 3.0)
        cap = cap_mult * float(np.max(real_costs))
        ri = np.where(real_mask)[0]
        fi = np.where(flat_mask)[0]
        Xr = X[ri]
        if len(X) >= 2:
            dm = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dm, np.inf)
            radius = 1.5 * float(np.median(dm.min(axis=1)))
        else:
            radius = np.inf
        for f in fi:
            dd = np.sqrt(((Xr - X[f]) ** 2).sum(-1))
            loc = costs[ri][dd <= radius]
            val = (margin * float(np.max(loc)) if loc.size
                   else float(np.percentile(real_costs, 90)))
            out[f] = min(val, cap)
        # Gentle ghost lift: ghosts (the infeasible Ag=0 / NaBH4=0 faces) sit a
        # bit ABOVE their nearest_high neighbour value so the GP mean rises into a
        # soft slope toward the face. Still bounded by the same cap, so no
        # discontinuity that would collapse the length scale. Applied to ghost
        # rows only (real flat measurements keep the plain nearest_high value).
        lift = float(getattr(self, "ghost_penalty_lift", 1.0) or 1.0)
        if n_ghosts > 0 and lift != 1.0:
            g = ghost_mask & flat_mask
            out[g] = np.minimum(out[g] * lift, cap)
        self._gp_display_costs = out.copy()
        self.logger.info(
            "flat_strategy=nearest_high -> %d flat cost(s) remapped "
            "(radius=%.3g, cap=%.4g)", n_flat, radius, cap)
        return X, out

    def _alpha_for_fit(self, base_alpha):
        """Per-point GP noise (alpha) aligned with the last _apply_flat_strategy
        output. Real points keep ``base_alpha``; ghost points get the (larger)
        ghost noise. Returns the scalar ``base_alpha`` when no ghosts are in
        the training set, so sklearn behaviour is unchanged in that case.

        Must be called AFTER _apply_flat_strategy and BEFORE the matching
        gaussian_process.fit(), passing the base alpha used at that fit site.
        """
        gmask = getattr(self, "_fit_ghost_mask", None)
        wmask = getattr(self, "_fit_warmstart_mask", None)
        has_g = gmask is not None and np.any(gmask)
        has_w = wmask is not None and np.any(wmask)
        if not has_g and not has_w:
            return base_alpha
        n = len(gmask) if gmask is not None else len(wmask)
        alpha = np.full(n, float(base_alpha), dtype=float)
        if has_g:
            gn = max(float(getattr(self, "_fit_ghost_alpha", 0.0)), float(base_alpha))
            alpha[gmask] = gn
        if has_w:
            wn = max(float(getattr(self, "_fit_warmstart_alpha", 0.0)), float(base_alpha))
            alpha[wmask] = np.maximum(alpha[wmask], wn)
        self.logger.debug(
            "Per-point alpha: %d ghost(s), %d warm-start, %d real at %.4g",
            int(np.sum(gmask)) if has_g else 0,
            int(np.sum(wmask)) if has_w else 0,
            int(n - (np.sum(gmask) if has_g else 0)), float(base_alpha))
        return alpha
