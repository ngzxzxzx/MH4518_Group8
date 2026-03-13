"""
heston.py  —  Heston Stochastic Volatility Model
=================================================
Self-contained model module.  Exposes:

  Simulator   : HestonSimulator(params, corr, tickers, r)
                All simulation and calibration in one place.

  (public helpers used by stage1_data.py during data loading)
  calibrate          : fast inception-surface calibration
  calibrate_cf       : robust DE-based calibration for live chains
  heston_vol_surface : Heston approx formula (used by both calibrators)
  blend_terminal_params : Heston-aware blending (freezes kappa/sigma
                           when live chain is too sparse)

Usage from main.py
------------------
    from heston import Simulator
    params = Simulator.calibrate(surface_df, tickers, r)
    sim    = Simulator(params, corr, tickers, r)

Usage from stage1_data.py
--------------------------
    from heston import calibrate, calibrate_cf, blend_terminal_params
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as _spnorm
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
# Constants
# ---------------------------------------------------------------------------

KAPPA_UPPER_BOUND = 8.0
SIGMA_UPPER_BOUND = 1.5


# ---------------------------------------------------------------------------
# Heston vol surface approximation  (fast closed-form, used in calibration)
# ---------------------------------------------------------------------------

def heston_vol_surface(T: float, m: float, params: dict) -> float:
    """
    Approximate Heston implied vol at maturity T and moneyness m = K/S.

    Matches the formula used by heston_synthetic.py so that calibration
    to the synthetic inception surface recovers the true parameters.
    """
    v0, theta, kappa = params["v0"], params["theta"], params["kappa"]
    sigma, rho = params["sigma"], params["rho"]
    if T <= 1e-6:
        return float(np.sqrt(v0))
    decay  = (1 - np.exp(-kappa * T)) / (kappa * T)
    vol_T  = np.sqrt(abs(theta + (v0 - theta) * np.exp(-kappa * T)))
    skew   = rho * sigma * decay * np.log(m)
    convex = 0.1 * sigma * decay * np.log(m) ** 2
    return float(np.clip(vol_T + skew + convex, 0.15, 0.60))


# ---------------------------------------------------------------------------
# Helpers for calibration
# ---------------------------------------------------------------------------

def _vol_col(surface_df: pd.DataFrame) -> str:
    """Return the implied-vol column name ('vol' preferred over 'implied_vol')."""
    has_vol     = "vol"         in surface_df.columns
    has_imp_vol = "implied_vol" in surface_df.columns

    if has_vol and has_imp_vol:
        non_null_vol = surface_df["vol"].notna().sum()
        non_null_imp = surface_df["implied_vol"].notna().sum()
        print(f"  WARNING: surface has both 'vol' ({non_null_vol} non-NaN) and "
              f"'implied_vol' ({non_null_imp} non-NaN) columns. "
              f"Using 'vol'. Rows with vol=NaN will be dropped.")
        return "vol"
    if has_vol:
        return "vol"
    if has_imp_vol:
        return "implied_vol"
    raise KeyError(
        f"Surface DataFrame has no vol/implied_vol column. "
        f"Found: {list(surface_df.columns)}")


# ---------------------------------------------------------------------------
# Fast calibration  (inception surface — synthetic, clean data)
# ---------------------------------------------------------------------------

def calibrate(surface_df: pd.DataFrame, ticker: str,
              r: float = 0.05) -> dict:
    """
    Fit Heston parameters to the IV surface for one ticker.

    Uses multi-start L-BFGS-B with the fast heston_vol_surface() formula.
    Suitable for clean synthetic surfaces where the global minimum is easy
    to reach with gradient descent.
    """
    vc  = _vol_col(surface_df)
    sub = surface_df[surface_df["ticker"] == ticker].dropna(subset=[vc])
    if sub.empty:
        print(f"  [{ticker}] No surface data — using defaults")
        return dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)

    Ts  = sub["maturity_years"].values.astype(float)
    ms  = sub["moneyness"].values.astype(float)
    ivs = sub[vc].values.astype(float)

    weights = np.exp(-2 * (ms - 1.0) ** 2)
    weights /= weights.sum()

    def objective(x):
        v0, kappa, theta, sigma, rho = x
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-1 < rho < 1):
            return 1e8
        p    = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        pred = np.array([heston_vol_surface(T, m, p) for T, m in zip(Ts, ms)])
        return float(np.sum(weights * (pred - ivs) ** 2))

    bounds   = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (1e-4, 2.0), (-0.99, 0.99)]
    best_val = np.inf
    best_x   = [0.04, 2.0, 0.05, 0.3, -0.6]

    for v0_init in [0.02, 0.04, 0.08, 0.15]:
        for rho_init in [-0.7, -0.4, -0.2]:
            try:
                res = minimize(objective,
                               [v0_init, 2.0, 0.05, 0.3, rho_init],
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 1000, "ftol": 1e-12})
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x
            except Exception:
                pass

    v0, kappa, theta, sigma, rho = best_x
    rmse = np.sqrt(best_val)
    print(f"  [{ticker}] Heston fit  RMSE={rmse:.4f}  "
          f"v0={v0:.4f}  kappa={kappa:.3f}  theta={theta:.4f}  "
          f"sigma={sigma:.3f}  rho={rho:.3f}")
    return dict(v0=float(v0), kappa=float(kappa), theta=float(theta),
                sigma=float(sigma), rho=float(rho))


# ---------------------------------------------------------------------------
# Robust DE-based calibration  (live/historical chains — noisy data)
# ---------------------------------------------------------------------------

def calibrate_cf(surface_df: pd.DataFrame, ticker: str,
                 r: float = 0.05) -> dict:
    """
    Robust Heston calibration for real/noisy market option chains.

    Uses differential evolution (global search) followed by L-BFGS-B polish.
    Handles sparse chains by applying an ATM-proximity weighting scheme.
    Falls back to calibrate() if data is insufficient.
    """
    from scipy.optimize import differential_evolution

    sub = surface_df[surface_df["ticker"] == ticker].dropna(subset=[_vol_col(surface_df)])
    if sub.empty:
        print(f"  [{ticker}] No surface data — using defaults")
        return dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)

    vc  = _vol_col(sub)
    Ts  = sub["maturity_years"].values.astype(float)
    ms  = sub["moneyness"].values.astype(float)
    ivs = sub[vc].values.astype(float)

    valid = (ivs > 0.02) & (ivs < 2.5) & (Ts > 0.02)
    if valid.sum() < 3:
        return calibrate(surface_df, ticker, r)
    Ts, ms, ivs = Ts[valid], ms[valid], ivs[valid]

    weights = np.exp(-3.0 * (ms - 1.0) ** 2)
    weights /= weights.sum()

    def objective(x):
        v0, kappa, theta, sigma, rho = x
        params = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        feller_pen = max(0.0, sigma**2 - 2 * kappa * theta) * 5.0
        pred = np.array([heston_vol_surface(T, m, params) for T, m in zip(Ts, ms)])
        wmse = float(np.sum(weights * (pred - ivs) ** 2))
        return wmse + feller_pen

    bounds = [
        (1e-4, 0.8), (0.1, 8.0), (1e-4, 0.6), (0.05, 1.5), (-0.98, 0.10),
    ]

    de_result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=300, tol=1e-8,
        popsize=12, mutation=(0.5, 1.2), recombination=0.8,
        workers=1, polish=False,
    )

    best_x, best_val = de_result.x, de_result.fun
    for x0 in [de_result.x, list(calibrate(surface_df, ticker, r).values())]:
        x0 = list(x0)
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 800, "ftol": 1e-12})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x
        except Exception:
            pass

    v0, kappa, theta, sigma, rho = best_x
    params_best = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    rmse = float(np.sqrt(np.mean((
        np.array([heston_vol_surface(T, m, params_best) for T, m in zip(Ts, ms)]) - ivs) ** 2)))
    feller = 2 * kappa * theta / sigma**2
    print(f"  [{ticker}] Heston fit (DE)  RMSE={rmse:.4f}  "
          f"v0={v0:.4f}  kappa={kappa:.3f}  theta={theta:.4f}  "
          f"sigma={sigma:.3f}  rho={rho:.3f}  "
          f"Feller={feller:.2f}{'  OK' if feller >= 1 else '  warn<1'}")
    return dict(v0=float(v0), kappa=float(kappa), theta=float(theta),
                sigma=float(sigma), rho=float(rho))


# ---------------------------------------------------------------------------
# Terminal parameter blending  (Heston-specific: freeze kappa/sigma)
# ---------------------------------------------------------------------------

def blend_terminal_params(inception_params: dict,
                           terminal_params:  dict,
                           tickers:          list) -> dict:
    """
    Heston-specific blending: when the live chain is too short-dated to
    identify kappa reliably (it hits its upper bound), keep kappa/sigma/rho
    from inception and only update v0 and theta from the live chain.

    See base_simulator.BaseSimulator.blend_terminal_params() for the
    model-agnostic default (full replacement).
    """
    if tickers is None:
        tickers = list(inception_params.keys())

    blended  = {}
    kappa_tol = 0.05

    for t in tickers:
        inc  = inception_params.get(t, {})
        term = terminal_params.get(t, {})
        if not inc or not term:
            blended[t] = dict(inc or term)
            continue

        kappa_at_bound = term.get("kappa", 0) >= KAPPA_UPPER_BOUND * (1 - kappa_tol)

        if kappa_at_bound:
            blended[t] = {
                "v0":    term["v0"],
                "theta": term["theta"],
                "kappa": inc["kappa"],
                "sigma": inc["sigma"],
                "rho":   inc["rho"],
            }
            print(f"  [{t}] terminal kappa at bound ({term['kappa']:.3f}) — "
                  f"freezing kappa/sigma/rho from inception, "
                  f"updating v0={term['v0']:.4f}, theta={term['theta']:.4f}")
        else:
            blended[t] = dict(term)
            print(f"  [{t}] terminal kappa unconstrained ({term['kappa']:.3f}) — "
                  f"using full terminal params")

    return blended


# ---------------------------------------------------------------------------
# HestonSimulator  (concrete BaseSimulator implementation)
# ---------------------------------------------------------------------------

class HestonSimulator(BaseSimulator):
    """
    Correlated multi-asset Heston stochastic volatility simulator.

    Each asset i follows:
        dS_i = r · S_i · dt  +  sqrt(v_i) · S_i · dW_S_i
        dv_i = kappa_i·(theta_i - v_i)·dt  +  sigma_i·sqrt(v_i)·dW_v_i
        corr(dW_S_i, dW_v_i) = rho_i          (within-asset)

    Cross-asset correlation  corr(dW_S_i, dW_S_j) = C_ij  is imposed via
    Cholesky of the n_assets × n_assets spot correlation matrix.
    """

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _simulate_batch(self,
                        Z_S:    np.ndarray,
                        Z_v:    np.ndarray,
                        spots:  Dict[str, float],
                        params: Dict[str, dict],
                        T:      float,
                        n_steps: int) -> np.ndarray:
        """Euler-Maruyama loop for Heston with full-truncation scheme."""
        n_sims = Z_S.shape[0]
        dt     = T / n_steps
        sdt    = np.sqrt(dt)

        Z_corr = Z_S @ self.L.T   # (n_sims, n_steps, n_assets) — cross-asset corr

        paths = np.empty((n_sims, n_steps + 1, self.n_assets))
        v     = np.empty((n_sims, self.n_assets))

        for i, t in enumerate(self.tickers):
            paths[:, 0, i] = spots[t]
            v[:, i]        = params[t]["v0"]

        for step in range(n_steps):
            for i, t in enumerate(self.tickers):
                kappa = params[t]["kappa"]
                theta = params[t]["theta"]
                sigma = params[t]["sigma"]
                rho   = params[t]["rho"]

                v_pos = np.maximum(v[:, i], 0.0)
                sv    = np.sqrt(v_pos)

                dW_v = (rho * Z_corr[:, step, i]
                        + np.sqrt(1 - rho**2) * Z_v[:, step, i])

                v[:, i] = (v_pos
                           + kappa * (theta - v_pos) * dt
                           + sigma * sv * dW_v * sdt)
                v[:, i] = np.maximum(v[:, i], 0.0)

                drift  = (self.r - 0.5 * v_pos) * dt
                diff   = sv * Z_corr[:, step, i] * sdt
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
        Simulate Heston paths with the chosen variance reduction method.

        method : "crude" | "antithetic" | "quasi" | "stratified" | "importance"

        Returns
        -------
        paths            for crude / antithetic / quasi / stratified
        (paths, weights) for importance
        """
        rng = np.random.default_rng(seed)
        p   = params if params is not None else self.params

        if method == "crude":
            Z_S = rng.standard_normal((n_sims, n_steps, self.n_assets))
            Z_v = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        elif method == "antithetic":
            half      = n_sims // 2
            Z_S_h     = rng.standard_normal((half, n_steps, self.n_assets))
            Z_v_h     = rng.standard_normal((half, n_steps, self.n_assets))
            paths_pos = self._simulate_batch( Z_S_h,  Z_v_h, spots, p, T, n_steps)
            paths_neg = self._simulate_batch(-Z_S_h,  Z_v_h, spots, p, T, n_steps)
            return np.concatenate([paths_pos, paths_neg], axis=0)

        elif method == "quasi":
            if not SOBOL_AVAILABLE:
                print("  WARNING: scipy.stats.qmc unavailable; falling back to crude MC")
                return self.simulate(spots, T, n_sims, n_steps, "crude", p, seed)
            dim    = n_steps * self.n_assets * 2
            engine = Sobol(d=dim, scramble=True, seed=seed)
            n_pow2 = int(2 ** np.ceil(np.log2(n_sims)))
            u      = engine.random(n_pow2)[:n_sims]
            Z_flat = _spnorm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            half_d = n_steps * self.n_assets
            Z_S    = Z_flat[:, :half_d].reshape(n_sims, n_steps, self.n_assets)
            Z_v    = Z_flat[:, half_d:].reshape(n_sims, n_steps, self.n_assets)
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        elif method == "stratified":
            Z_S = self._stratified_normals(n_sims, n_steps, self.n_assets, rng)
            Z_v = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        elif method == "importance":
            Z_S, weights = self._importance_sampling_normals(
                n_sims, n_steps, self.n_assets, rng)
            Z_v   = rng.standard_normal((n_sims, n_steps, self.n_assets))
            paths = self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)
            return paths, weights

        else:
            raise ValueError(f"Unknown method '{method}'. "
                             "Choose: crude | antithetic | quasi | "
                             "stratified | importance")

    # ------------------------------------------------------------------
    # Calibration (class method — required by BaseSimulator interface)
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(cls,
                  surface_df: pd.DataFrame,
                  tickers:    list,
                  r:          float = 0.05) -> Dict[str, dict]:
        """
        Calibrate Heston parameters to a vol surface for all tickers.

        Selects fast (inception) vs robust DE (live chain) calibration
        based on whether the surface uses the 'vol' column (real data)
        or 'implied_vol' (synthetic).

        Returns {ticker: {v0, kappa, theta, sigma, rho}}.
        """
        use_cf = "vol" in surface_df.columns  # real historical chain → DE
        result = {}
        for t in tickers:
            if use_cf:
                result[t] = calibrate_cf(surface_df, t, r)
            else:
                result[t] = calibrate(surface_df, t, r)
        return result

    # ------------------------------------------------------------------
    # Parameter update for backtest  (required by BaseSimulator interface)
    # ------------------------------------------------------------------

    @staticmethod
    def update_params_for_vol(inception_params: dict,
                               atm_vol:          float) -> dict:
        """
        Scale v0 = atm_vol² while keeping kappa, theta, sigma, rho fixed.
        This is the Heston backtest vol-update rule.
        """
        p       = dict(**inception_params)
        p["v0"] = float(np.clip(atm_vol ** 2, 1e-4, 1.0))
        return p

    # ------------------------------------------------------------------
    # Terminal parameter blending  (Heston-aware override)
    # ------------------------------------------------------------------

    @staticmethod
    def blend_terminal_params(inception_params: Dict[str, dict],
                               terminal_params:  Dict[str, dict],
                               tickers:          list) -> Dict[str, dict]:
        """Freeze kappa/sigma/rho when live chain is too sparse."""
        return blend_terminal_params(inception_params, terminal_params, tickers)

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------

    @classmethod
    def model_name(cls) -> str:
        return "Heston"


# Canonical export name — main.py does:  from <model> import Simulator
Simulator = HestonSimulator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKERS = ["NFLX", "SPOT", "DIS"]

    dummy_params = {t: dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)
                    for t in TICKERS}
    dummy_corr   = np.array([[1.0, 0.357, 0.352],
                              [0.357, 1.0, 0.489],
                              [0.352, 0.489, 1.0]])
    dummy_spots  = {"NFLX": 850.0, "SPOT": 350.0, "DIS": 110.0}

    sim = HestonSimulator(dummy_params, dummy_corr, TICKERS)
    paths = sim.simulate(dummy_spots, T=1.5, n_sims=200, n_steps=100,
                         method="antithetic")
    print(f"Paths shape : {paths.shape}")
    print(f"NFLX mean   : {paths[:, -1, 0].mean():.2f}")
    print(f"Model name  : {sim.model_name()}")
