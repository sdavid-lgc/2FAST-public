"""
Choice of the next composition: acquisition function maximised by differential evolution on the
flow simplex, Thompson sampling candidates, exploration / random fall-backs.

Mixin of `toofast.core.experiment.Experiment` (methods moved verbatim from experiment.py; they use the
experiment's state through `self`).
"""
import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint

from toofast import clock
from toofast.acquisition import ExpectedImprovement, ExplorationWrapper, LowerConfidenceBound, FeasibilityWeightedAcquisition, fit_feasibility_classifier


class NextPointMixin:
    """Choice of the next composition: acquisition function maximised by differential evolution on the (part of `Experiment`)."""

    def _sample_candidates_ts(self, n: int = 2000) -> np.ndarray:
        """Return ``n`` feasible candidates in reduced GP space (n_var-1 dims).

        Each candidate satisfies the sum constraint
        ``sum(var_flows) == total_var_flow`` and stays within ``bounds``.
        Uses a Sobol sequence for uniform coverage, then projects onto the
        simplex face.
        """
        bounds_var = self.bounds[self.varying_pumps]   # (n_var, 2)
        lo = bounds_var[:-1, 0]
        hi = bounds_var[:-1, 1]
        dim = len(lo)                                  # n_var - 1

        # Sobol for low-discrepancy coverage, fall back to uniform if unavailable.
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=dim, scramble=True, seed=42)
            # next_power_of_2 >= n
            m = int(np.ceil(np.log2(max(n, 1))))
            raw = sampler.random_base2(m=m)[:n]        # (n, dim) in [0,1]
        except Exception:
            rng = np.random.default_rng(42)
            raw = rng.random(size=(n, dim))

        # Scale to individual bounds.
        cands = lo + raw * (hi - lo)                   # (n, dim)

        # Enforce sum constraint: last varying flow = total_var_flow - sum(rest).
        # Drop candidates where the last flow would be out of bounds.
        last = self.total_var_flow - cands.sum(axis=1)
        lo_last, hi_last = bounds_var[-1, 0], bounds_var[-1, 1]
        valid = (last >= lo_last - 1e-9) & (last <= hi_last + 1e-9)
        cands = cands[valid]

        if len(cands) == 0:
            # Degenerate: return the centroid as the only candidate.
            centre = np.full(dim, self.total_var_flow / (dim + 1))
            return centre.reshape(1, -1)

        return cands

    def _predict_next_ts(self, gaussian_process, rand_int: int) -> np.ndarray:
        """Thompson Sampling next point via a coherent GP posterior draw.

        Draws *one* function from the GP posterior using ``sample_y`` over a
        dense candidate grid (preserving inter-point correlations), then
        returns the full predicted flows for the candidate minimising that
        sample.  This is statistically correct and avoids passing a stochastic
        callable to ``differential_evolution``.

        Parameters
        ----------
        gaussian_process : GaussianProcessRegressor
            Fitted GP (already trained on the current data).
        rand_int : int
            Random seed — logged and stored in ``self._last_ts_seed`` for
            post-hoc traceability.

        Returns
        -------
        np.ndarray of shape (n_pumps,)
            Full predicted flows (varying + fixed).
        """
        candidates = self._sample_candidates_ts(n=2000)    # (m, dim)
        # Draw one coherent function from the GP posterior.
        sample = gaussian_process.sample_y(
            candidates, n_samples=1, random_state=rand_int
        ).ravel()                                           # (m,)
        best_idx = int(np.argmin(sample))
        x_reduced = candidates[best_idx]                   # (dim,)

        self.logger.info(
            f"TS: {len(candidates)} candidates | "
            f"sample min={sample.min():.4g} | "
            f"selected idx={best_idx} | seed={rand_int}"
        )

        # Reconstruct full flows (append last var flow + fixed flows).
        predicted_var_flows = np.append(x_reduced,
                                        self.total_var_flow - x_reduced.sum())
        predicted_flows = np.zeros(self.number_of_pumps)
        j = 0
        for i in range(self.number_of_pumps):
            if self.fixed_pumps[i]:
                predicted_flows[i] = self.fixed_flows[i]
            else:
                predicted_flows[i] = predicted_var_flows[j]
                j += 1
        return predicted_flows

    def predict_next(self, costs, gaussian_process, iteration=None):
        """
        Find the next flow set to evaluate by optimizing the acquisition function
        over the GP posterior, using differential evolution.

        If the GP fit is unreliable (too few points or length scale at bounds),
        returns a random exploration point instead.

        Parameters
        ----------
        costs : array-like of shape (n_samples)
            List of the known costs.
        gaussian_process : sk-learn GaussianProcessRegressor
            Regressor against which the expected improvement is calculated.
        iteration : int, optional
            Current BO iteration (post-init). Used for adaptive exploration.

        Returns
        -------
        array-like of shape (n_features-1)
            Flows optimizing the acquisition function.
        """
        if not self._is_gp_fit_reliable(gaussian_process):
            self.logger.info("GP fit unreliable — returning random exploration point")
            return self.pump_syst.generate_random_flows(
                self.bounds, self.total_var_flow, self.fixed_pumps, self.fixed_flows
            )

        # Signal scale (raw cost units) used to make xi dimensionless. Use the
        # costs the GP was actually trained on (_gp_display_costs: flats
        # remapped + ghosts), which matches gp.predict's denormalised output
        # scale and avoids the raw flat-sentinel inflating the std. Falls back
        # to the raw costs, then to 1.0 (legacy behaviour) if unusable.
        # gp.predict returns values in GP-target space (warped if cost_transform
        # == 'log'); best_y and y_scale must live in that same space.
        disp = getattr(self, "_gp_display_costs", None)
        scale_src = disp if (disp is not None and len(disp) > 1) else costs
        y_scale = float(np.std(self._cost_to_gp(scale_src)))
        if not np.isfinite(y_scale) or y_scale < 1e-12:
            y_scale = 1.0

        if self.acq_func == 'EI':
            best_y_gp = float(self._cost_to_gp(np.asarray(costs)).min())
            acq_func = ExpectedImprovement(gp_model=gaussian_process,
                                           best_y=best_y_gp, y_scale=y_scale)
        elif self.acq_func == 'LCB':
            acq_func = LowerConfidenceBound(gp_model=gaussian_process)
        elif self.acq_func == 'TS':
            # Thompson Sampling requires a coherent draw from the GP posterior.
            # Passing a stochastic callable to differential_evolution is
            # incorrect (non-deterministic objective → DE never converges).
            # Instead, use gp.sample_y over a dense candidate grid — see
            # _predict_next_ts / _sample_candidates_ts.
            rand_int = np.random.randint(0, 10_000)
            self._last_ts_seed = rand_int          # stored for traceability
            if iteration is not None:
                self.logger.info(
                    f"TS: exploration ratio N/A (coherent sample_y draw, "
                    f"seed={rand_int}, iter={iteration})"
                )
            return self._predict_next_ts(gaussian_process, rand_int)
        else:
            raise ValueError('Acquisition function not recognized.')

        if iteration is not None:
            acq_func = ExplorationWrapper(
                base_acq=acq_func,
                iteration=iteration,
            )
            self.logger.info(
                f"Exploration ratio: {acq_func.explore_ratio:.2f} "
                f"(iter {iteration})"
            )

        # Optional constrained-BO: multiply EI by P(feasible | x).
        if getattr(self, "feasibility_aware_override", False):
            if self.acq_func != 'EI':
                self.logger.info(
                    "Feasibility-aware requested but acquisition is %s; "
                    "the feasibility weight is only applied to EI — skipped.",
                    self.acq_func)
            else:
                feas_X = getattr(self, "_feas_X", None)
                feas_y = getattr(self, "_feas_label", None)
                if feas_X is not None and feas_y is not None:
                    clf = fit_feasibility_classifier(feas_X, feas_y, min_per_class=4)
                    if clf is not None:
                        acq_func = FeasibilityWeightedAcquisition(acq_func, clf)
                        self.logger.info("Feasibility-weighted EI active.")

        bounds = self.bounds[self.varying_pumps]
        constraints = LinearConstraint(np.ones(self.nb_of_var_pumps - 1), self.total_var_flow - bounds[-1][1],
                                       self.total_var_flow - bounds[-1][0])

        # Time the optimization
        start_time = clock.time()
        res = differential_evolution(acq_func,
                                     bounds=bounds[:-1],
                                     constraints=constraints,
                                     tol=self.de_tol,
                                     maxiter=self.de_maxiter,
                                     popsize=25,  # 25×(n_dims) individuals for better 4D exploration
                                     init='sobol',
                                     polish=True,  # Critical: local refinement with L-BFGS-B
                                     atol=0,  # Disable absolute tolerance to avoid premature stopping
                                     updating='deferred')
        elapsed_time = clock.time() - start_time

        # Detailed convergence diagnostics
        self.logger.info(f"Acquisition optimization (dim={self.nb_of_var_pumps-1}):")
        self.logger.info(f"  Time: {elapsed_time:.3f}s")
        self.logger.info(f"  Success: {res.success}")
        self.logger.info(f"  Iterations: {res.nit}/{self.de_maxiter}")
        self.logger.info(f"  Function evals: {res.nfev}")
        self.logger.info(f"  Acquisition value: {res.fun:.8e}")
        if res.nit >= self.de_maxiter * 0.9:
            self.logger.warning(f"  ⚠️ Nearly hit maxiter ({res.nit}/{self.de_maxiter})! Consider increasing de_maxiter")
        if not res.success:
            self.logger.warning(f"  ⚠️ Optimization did not converge: {res.message}")

        # Explore-vs-exploit diagnostic for the chosen point
        try:
            from scipy.stats import norm as _norm
            x_next_reduced = np.atleast_2d(res.x)
            mu_next, sigma_next = gaussian_process.predict(x_next_reduced, return_std=True)
            mu_next, sigma_next = float(mu_next[0]), float(sigma_next[0])

            # Reference sigma over the search domain (random sample of feasible points)
            n_ref = 200
            X_train = gaussian_process.X_train_
            lo = np.minimum(bounds[:-1, 0], X_train.min(axis=0))
            hi = np.maximum(bounds[:-1, 1], X_train.max(axis=0))
            X_ref = np.random.uniform(lo, hi, size=(n_ref, X_train.shape[1]))
            sigma_ref = gaussian_process.predict(X_ref, return_std=True)[1]
            sigma_pool_mean = float(np.mean(sigma_ref)) or 1e-12
            sigma_norm = sigma_next / sigma_pool_mean

            # EI decomposition (uses the same xi as the acquisition wrapper, if EI)
            base = getattr(acq_func, 'base_acq', acq_func)
            xi_used = float(getattr(base, 'xi', 0.0))
            # xi is dimensionless; scale it by y_scale (matching ExpectedImprovement).
            xi_scale = float(getattr(base, 'y_scale', 1.0))
            # mu_next is in GP-target space, so best_y must be too.
            best_y = float(self._cost_to_gp(np.asarray(costs)).min())
            improvement = best_y - mu_next - xi_used * xi_scale
            if sigma_next > 1e-12:
                z = improvement / sigma_next
                ei_mean = max(0.0, improvement) * _norm.cdf(z)
                ei_sigma = sigma_next * _norm.pdf(z)
            else:
                ei_mean = max(0.0, improvement)
                ei_sigma = 0.0
            ei_ratio = ei_sigma / (ei_mean + 1e-12)

            # Distance from x_next to nearest training point (reduced space)
            d_min = float(np.min(np.linalg.norm(X_train - x_next_reduced, axis=1)))

            tag = "EXPLORE" if (sigma_norm > 1.5 or ei_ratio > 1.0) else (
                  "EXPLOIT" if (sigma_norm < 0.5 and ei_ratio < 0.5) else "MIXED")
            self.logger.info(
                f"  Explore/exploit [{tag}]: "
                f"sigma_next={sigma_next:.3e} "
                f"sigma_norm={sigma_norm:.2f} "
                f"d_min={d_min:.3f} "
                f"EI_mean={ei_mean:.3e} EI_sigma={ei_sigma:.3e} ratio={ei_ratio:.2f} "
                f"xi={xi_used:.3f}"
            )
        except Exception as e:
            self.logger.debug(f"Explore/exploit diagnostic failed: {e}")

        # Add the last varying flow and fixed flows
        predicted_var_flows = np.append(res.x, self.total_var_flow - res.x.sum())
        j = 0
        predicted_flows = np.zeros(self.number_of_pumps)
        for i in range(self.number_of_pumps):
            if self.fixed_pumps[i]:
                predicted_flows[i] = self.fixed_flows[i]
            else:
                predicted_flows[i] = predicted_var_flows[j]
                j += 1
        return predicted_flows
