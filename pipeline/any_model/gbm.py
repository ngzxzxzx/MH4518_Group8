"""
gbm.py  —  Geometric Brownian Motion Model
==========================================
Concrete implementation of BaseSimulator for the standard GBM (Black-Scholes)
model.  Demonstrates how to plug in a new model without touching any other
stage.

Exposes:
  Simulator  : GBMSimulator(params, corr, tickers, r)

Model parameters per ticker
---------------------------
  sigma : annualised volatility (decimal, e.g. 0.30 = 30%)

Usage from main.py
------------------
  python main.py --model gbm

Usage (direct)
--------------
  from gbm import Simulator
  params = Simulator.calibrate(surface_df, tickers, r)
  sim    = Simulator(params, corr, tickers, r)
  paths  = sim.simulate(spots, T=1.5, n_sims=5000, n_steps=378)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from scipy.stats.qmc import Sobol
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False

from base_simulator import BaseSimulator


# ---------------------------------------------------------------------------
# GBMSimulator
# ---------------------------------------------------------------------------

class GBMSimulator(BaseSimulator):
    """
    Correlated multi-asset Geometric Brownian Motion.

    Each asset i follows:
        dS_i = r · S_i · dt  +  sigma_i · S_i · dW_i

    Cross-asset correlation  corr(dW_i, dW_j) = C_ij  is imposed via
    Cholesky of the n_assets × n_assets correlation matrix (inherited from
    BaseSimulator).

    Parameters per ticker
    ---------------------
    sigma : annualised log-vol (e.g. 0.30)

    Z_v is ignored (single-factor model — no vol-of-vol dimension).
    """

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _simulate_batch(self,
                        Z_S:     np.ndarray,
                        Z_v:     np.ndarray,
                        spots:   Dict[str, float],
                        params:  Dict[str, dict],
                        T:       float,
                        n_steps: int) -> np.ndarray:
        """
        Exact log-normal step (no discretisation error).

        S(t+dt) = S(t) · exp( (r - 0.5·σ²)·dt  +  σ·√dt·Z_corr )
        """
        n_sims = Z_S.shape[0]
        dt     = T / n_steps
        sdt    = np.sqrt(dt)

        # apply cross-asset correlation
        Z_corr = Z_S @ self.L.T   # (n_sims, n_steps, n_assets)

        paths = np.empty((n_sims, n_steps + 1, self.n_assets))

        for i, t in enumerate(self.tickers):
            paths[:, 0, i] = spots[t]

        for step in range(n_steps):
            for i, t in enumerate(self.tickers):
                sigma  = params[t]["sigma"]
                drift  = (self.r - 0.5 * sigma**2) * dt
                diff   = sigma * Z_corr[:, step, i] * sdt
                paths[:, step + 1, i] = paths[:, step, i] * np.exp(drift + diff)

        return paths

    def simulate(self,
                 spots:   Dict[str, float],
                 T:       float,
                 n_sims:  int,
                 n_steps: int,
                 method:  str = "crude",
                 params:  Optional[Dict[str, dict]] = None,
                 seed:    int = 42):
        """
        Simulate GBM paths with the chosen variance reduction method.

        method : "crude" | "antithetic" | "quasi" | "stratified" | "importance"

        Returns
        -------
        paths            for crude / antithetic / quasi / stratified
        (paths, weights) for importance

        Note: Z_v is passed as zeros (GBM has no second Brownian driver).
        """
        rng = np.random.default_rng(seed)
        p   = params if params is not None else self.params

        zero_v = lambda shape: np.zeros(shape)

        if method == "crude":
            Z_S = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, zero_v(Z_S.shape), spots, p, T, n_steps)

        elif method == "antithetic":
            half  = n_sims // 2
            Z_S_h = rng.standard_normal((half, n_steps, self.n_assets))
            z_v_h = zero_v(Z_S_h.shape)
            paths_pos = self._simulate_batch( Z_S_h, z_v_h, spots, p, T, n_steps)
            paths_neg = self._simulate_batch(-Z_S_h, z_v_h, spots, p, T, n_steps)
            return np.concatenate([paths_pos, paths_neg], axis=0)

        elif method == "quasi":
            if not SOBOL_AVAILABLE:
                print("  WARNING: scipy.stats.qmc unavailable; falling back to crude MC")
                return self.simulate(spots, T, n_sims, n_steps, "crude", p, seed)
            from scipy.stats import norm as _spnorm
            dim    = n_steps * self.n_assets
            engine = Sobol(d=dim, scramble=True, seed=seed)
            n_pow2 = int(2 ** np.ceil(np.log2(n_sims)))
            u      = engine.random(n_pow2)[:n_sims]
            Z_S    = _spnorm.ppf(np.clip(u, 1e-10, 1 - 1e-10))\
                             .reshape(n_sims, n_steps, self.n_assets)
            return self._simulate_batch(Z_S, zero_v(Z_S.shape), spots, p, T, n_steps)

        elif method == "stratified":
            Z_S = self._stratified_normals(n_sims, n_steps, self.n_assets, rng)
            return self._simulate_batch(Z_S, zero_v(Z_S.shape), spots, p, T, n_steps)

        elif method == "importance":
            Z_S, weights = self._importance_sampling_normals(
                n_sims, n_steps, self.n_assets, rng)
            paths = self._simulate_batch(Z_S, zero_v(Z_S.shape), spots, p, T, n_steps)
            return paths, weights

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Choose: crude | antithetic | quasi | "
                             "stratified | importance")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(cls,
                  surface_df: pd.DataFrame,
                  tickers:    list,
                  r:          float = 0.05) -> Dict[str, dict]:
        """
        Extract ATM implied vol from the surface for each ticker and
        return {ticker: {"sigma": atm_vol}}.

        Uses the median of all near-ATM points (moneyness in [0.95, 1.05])
        weighted by inverse distance to ATM, across the shortest available
        maturity band.

        Falls back to a flat 30% vol if the surface has no ATM data.
        """
        vc_name = "vol" if "vol" in surface_df.columns else "implied_vol"
        result  = {}

        for t in tickers:
            sub = surface_df[surface_df["ticker"] == t].copy()
            sub = sub.dropna(subset=[vc_name])
            sub = sub[(sub["moneyness"] >= 0.90) & (sub["moneyness"] <= 1.10)]

            if sub.empty:
                print(f"  [{t}] GBM: no ATM data — using σ=0.30")
                result[t] = {"sigma": 0.30}
                continue

            # weight by proximity to ATM
            weights = np.exp(-10.0 * (sub["moneyness"] - 1.0) ** 2)
            atm_vol = float(np.average(sub[vc_name].values, weights=weights.values))
            atm_vol = float(np.clip(atm_vol, 0.05, 1.50))

            print(f"  [{t}] GBM calibration:  σ (ATM vol) = {atm_vol:.4f}")
            result[t] = {"sigma": atm_vol}

        return result

    # ------------------------------------------------------------------
    # Parameter update for backtest
    # ------------------------------------------------------------------

    @staticmethod
    def update_params_for_vol(inception_params: dict,
                               atm_vol:          float) -> dict:
        """Replace sigma directly with the current ATM vol estimate."""
        p          = dict(**inception_params)
        p["sigma"] = float(np.clip(atm_vol, 0.05, 1.50))
        return p

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------

    @classmethod
    def model_name(cls) -> str:
        return "GBM"


# Canonical export name
Simulator = GBMSimulator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKERS = ["NFLX", "SPOT", "DIS"]

    dummy_params = {t: {"sigma": 0.30} for t in TICKERS}
    dummy_corr   = np.array([[1.0, 0.357, 0.352],
                              [0.357, 1.0, 0.489],
                              [0.352, 0.489, 1.0]])
    dummy_spots  = {"NFLX": 93.55, "SPOT": 565.41, "DIS": 97.88}

    sim   = GBMSimulator(dummy_params, dummy_corr, TICKERS)
    paths = sim.simulate(dummy_spots, T=1.5, n_sims=200, n_steps=100,
                         method="antithetic")
    print(f"Paths shape : {paths.shape}")
    print(f"NFLX mean   : {paths[:, -1, 0].mean():.2f}")
    print(f"Model name  : {sim.model_name()}")