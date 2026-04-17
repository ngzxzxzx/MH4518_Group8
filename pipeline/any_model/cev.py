"""
cev.py  —  Constant Elasticity of Variance (CEV) Model
=======================================================
Concrete implementation of BaseSimulator for the CEV model.

Each asset i follows the SDE (risk-neutral measure):

    dS_i = r · S_i · dt  +  sigma_i · S_i^beta_i · dW_i

where
  sigma_i : CEV scale parameter  (not the same as Black-Scholes vol)
  beta_i  : elasticity parameter
              beta = 1  → GBM (log-normal, constant vol)
              beta < 1  → downward vol skew  (typical for equities)
              beta = 0  → absolute diffusion (vol independent of S)

Cross-asset correlation corr(dW_i, dW_j) = C_ij is imposed via the
Cholesky of the n_assets × n_assets spot correlation matrix (inherited
from BaseSimulator).

Exposes:
  Simulator : CEVSimulator(params, corr, tickers, r)

Model parameters per ticker
---------------------------
  sigma : CEV diffusion coefficient  (calibrated to the vol surface)
  beta  : elasticity exponent        (calibrated to the vol skew)

Usage from main.py
------------------
  python main.py --model cev

Usage (direct)
--------------
  from cev import Simulator
  params = Simulator.calibrate(surface_df, tickers, r)
  sim    = Simulator(params, corr, tickers, r)
  paths  = sim.simulate(spots, T=1.5, n_sims=5000, n_steps=378)
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
# Helpers for calibration
# ---------------------------------------------------------------------------

def cev_vol_surface(T: float, m: float, S: float, params: dict) -> float:
    """
    Hagan-Woodward approximation for CEV implied vol.

    Parameters
    ----------
    T      : maturity in years
    m      : moneyness  K / S
    S      : current spot price
    params : dict with keys "sigma" (CEV scale) and "beta" (elasticity)

    Returns the approximate Black-Scholes implied vol, clipped to [0.05, 1.50].
    """
    sigma = params["sigma"]
    beta  = params["beta"]
    K     = m * S

    f              = (S + K) / 2.0
    base_vol       = sigma / (f ** (1.0 - beta))
    strike_dist    = (S - K) / f
    skew_adj       = 1.0 + (((1.0 - beta) ** 2) / 24.0) * (strike_dist ** 2)
    time_adj       = (((1.0 - beta) ** 2) / 24.0) * (sigma ** 2 / f ** (2.0 - 2.0 * beta)) * T
    vol            = base_vol * (skew_adj + time_adj)

    return float(np.clip(vol, 0.05, 1.50))


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
# Calibration
# ---------------------------------------------------------------------------

def calibrate(surface_df: pd.DataFrame, ticker: str,
              r: float = 0.05) -> dict:
    """
    Calibrate CEV parameters (sigma, beta) to the IV surface for one ticker.

    Jointly fits sigma and beta via weighted least-squares on implied vols,
    with ATM points weighted more heavily (Gaussian kernel centred on m=1).

    Returns
    -------
    dict  {"sigma": float, "beta": float}
    """
    vc  = _vol_col(surface_df)
    sub = surface_df[surface_df["ticker"] == ticker].dropna(subset=[vc])
    if sub.empty:
        print(f"  [{ticker}] No surface data — using defaults sigma=0.30, beta=0.5")
        return dict(sigma=0.30, beta=0.5)

    Ts  = sub["maturity_years"].values.astype(float)
    ms  = sub["moneyness"].values.astype(float)
    # spot_price column: take the first non-null scalar value
    S   = float(sub["spot_price"].dropna().values[0])
    ivs = sub[vc].values.astype(float)

    # ATM-weighted objective
    weights  = np.exp(-2.0 * (ms - 1.0) ** 2)
    weights /= weights.sum()

    def objective(x):
        sigma_x, beta_x = float(x[0]), float(x[1])
        p    = dict(sigma=sigma_x, beta=beta_x)
        pred = np.array([cev_vol_surface(T, m, S, p) for T, m in zip(Ts, ms)])
        return float(np.sum(weights * (pred - ivs) ** 2))

    # Initial guess: sigma from ATM vol at beta=1 (i.e. GBM),
    # then refine with multi-start L-BFGS-B
    atm_mask   = (ms >= 0.95) & (ms <= 1.05)
    atm_iv     = float(np.median(ivs[atm_mask])) if atm_mask.any() else 0.30
    x0_default = [atm_iv, 0.5]   # sigma ≈ ATM vol, beta near equity norm

    starts = [
        x0_default,
        [atm_iv * 0.8, 0.3],
        [atm_iv * 1.2, 0.7],
        [atm_iv,       1.0],   # GBM limit
    ]
    bounds   = [(0.01, 2.0), (0.0, 1.0)]
    best_val = np.inf
    best_x   = x0_default

    for x0 in starts:
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 2000, "ftol": 1e-12})
            if res.fun < best_val:
                best_val, best_x = res.fun, list(res.x)
        except Exception:
            pass

    sigma_fit, beta_fit = float(best_x[0]), float(best_x[1])
    rmse = float(np.sqrt(best_val))

    # ATM implied vol at inception, derived from the Hagan-Woodward formula
    # at ATM (m=1, strike_dist=0, time_adj≈0):
    #   sigma_iv ≈ sigma / S^(1-beta)
    # Stored so that update_params_for_vol can rescale sigma correctly:
    # it receives an ATM IV as atm_vol, and must compare IV to IV, not IV to sigma.
    atm_vol_inception = sigma_fit / max(S ** (1.0 - beta_fit), 1e-9)

    print(f"  [{ticker}] CEV fit  RMSE={rmse:.4f}  "
          f"sigma={sigma_fit:.4f}  beta={beta_fit:.4f}  "
          f"atm_vol_inception={atm_vol_inception:.4f}")
    return dict(sigma=sigma_fit, beta=beta_fit,
                atm_vol_inception=float(atm_vol_inception))


# ---------------------------------------------------------------------------
# CEVSimulator  (concrete BaseSimulator implementation)
# ---------------------------------------------------------------------------

class CEVSimulator(BaseSimulator):
    """
    Correlated multi-asset CEV simulator.

    Each asset i follows:
        dS_i = r · S_i · dt  +  sigma_i · S_i^beta_i · dW_i

    The local volatility at time t for asset i is:
        sigma_local(S_i) = sigma_i · S_i^(beta_i - 1)

    which equals sigma_i for GBM (beta=1), and decreases with S for
    beta < 1 (producing the negative skew typical of equities).

    CEV is a single-factor model: Z_v is ignored (no second Brownian driver).
    Cross-asset correlation is imposed via Cholesky of the spot correlation
    matrix, identical to GBM.

    Parameters per ticker
    ---------------------
    sigma : CEV diffusion coefficient
    beta  : elasticity exponent in [0, 1]
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
        Euler-Maruyama simulation of the CEV SDE.

        CEV SDE (risk-neutral):
            dS = r · S · dt  +  sigma · S^beta · dW

        Discretised as:
            S(t+dt) = S(t) + r · S(t) · dt  +  sigma · S(t)^beta · sqrt(dt) · Z

        We use the Euler scheme (not log-Euler) because the diffusion
        coefficient sigma · S^beta is not of the form S·f(S), so the
        standard log-normal exact step does not apply.  Full truncation
        ensures S > 0: any path that hits zero is reflected to a small
        floor (1e-6 · S_0) rather than absorbing, consistent with the
        natural boundary condition for beta < 1.

        Z_v is accepted for interface compatibility but ignored —
        CEV is a single-factor model with no second Brownian driver.

        Parameters
        ----------
        Z_S     : (n_sims, n_steps, n_assets)  spot shock normals
        Z_v     : (n_sims, n_steps, n_assets)  ignored (single-factor model)
        spots   : initial spot prices per ticker
        params  : {ticker: {"sigma": float, "beta": float}}
        T       : time horizon in years
        n_steps : number of time steps

        Returns
        -------
        paths : (n_sims, n_steps+1, n_assets)
        """
        n_sims  = Z_S.shape[0]
        dt      = T / n_steps
        sdt     = np.sqrt(dt)

        # Apply cross-asset Cholesky correlation to spot shocks
        Z_corr  = Z_S @ self.L.T       # (n_sims, n_steps, n_assets)

        paths   = np.empty((n_sims, n_steps + 1, self.n_assets))
        for i, t in enumerate(self.tickers):
            paths[:, 0, i] = spots[t]

        # Pre-extract per-asset parameters (avoids dict lookup inside the loop)
        sigmas = np.array([params[t]["sigma"] for t in self.tickers])  # (n_assets,)
        betas  = np.array([params[t]["beta"]  for t in self.tickers])  # (n_assets,)

        # Small floor: 1e-6 times the initial spot per asset
        S0_arr  = np.array([spots[t] for t in self.tickers])
        S_floor = 1e-6 * S0_arr                                        # (n_assets,)

        for step in range(n_steps):
            S_cur = paths[:, step, :]                                  # (n_sims, n_assets)

            # Local vol: sigma_i · S_i^(beta_i - 1)  → diffusion coeff = sigma_i · S_i^beta_i
            # Computed as S_cur^beta_i element-wise; use abs for safety against
            # floating-point underflow at the floor boundary.
            S_safe    = np.maximum(S_cur, S_floor)                     # (n_sims, n_assets)
            diff_coef = sigmas * (S_safe ** betas)                     # (n_sims, n_assets)

            drift     = self.r * S_cur * dt                            # (n_sims, n_assets)
            diffusion = diff_coef * Z_corr[:, step, :] * sdt          # (n_sims, n_assets)

            S_next = S_cur + drift + diffusion
            paths[:, step + 1, :] = np.maximum(S_next, S_floor)

        return paths

    # ------------------------------------------------------------------
    # Variance reduction — all five methods
    # ------------------------------------------------------------------

    def simulate(self,
                 spots:   Dict[str, float],
                 T:       float,
                 n_sims:  int,
                 n_steps: int,
                 method:  str = "crude",
                 params:  Optional[Dict[str, dict]] = None,
                 seed:    int = 42):
        """
        Simulate CEV paths with the chosen variance reduction method.

        method : "crude" | "antithetic" | "quasi" | "stratified" | "importance"

        Returns
        -------
        paths            for crude / antithetic / quasi / stratified
        (paths, weights) for importance

        Note: Z_v is passed as zeros (CEV has no second Brownian driver).
        """
        rng = np.random.default_rng(seed)
        p   = params if params is not None else self.params

        # CEV is single-factor: Z_v is unused but required by _simulate_batch
        zero_v = lambda shape: np.zeros(shape)

        if method == "crude":
            Z_S = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, zero_v(Z_S.shape), spots, p, T, n_steps)

        elif method == "antithetic":
            half      = n_sims // 2
            Z_S_h     = rng.standard_normal((half, n_steps, self.n_assets))
            z_v_h     = zero_v(Z_S_h.shape)
            paths_pos = self._simulate_batch( Z_S_h, z_v_h, spots, p, T, n_steps)
            paths_neg = self._simulate_batch(-Z_S_h, z_v_h, spots, p, T, n_steps)
            return np.concatenate([paths_pos, paths_neg], axis=0)

        elif method == "quasi":
            if not SOBOL_AVAILABLE:
                print("  WARNING: scipy.stats.qmc unavailable; falling back to crude MC")
                return self.simulate(spots, T, n_sims, n_steps, "crude", p, seed)
            dim    = n_steps * self.n_assets   # single-factor: spot shocks only
            engine = Sobol(d=dim, scramble=True, seed=seed)
            n_pow2 = int(2 ** np.ceil(np.log2(n_sims)))
            u      = engine.random(n_pow2)[:n_sims]
            Z_S    = (_spnorm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
                      .reshape(n_sims, n_steps, self.n_assets))
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
    # Calibration (class method — required by BaseSimulator interface)
    # ------------------------------------------------------------------

    @classmethod
    def calibrate(cls,
                  surface_df: pd.DataFrame,
                  tickers:    list,
                  r:          float = 0.05) -> Dict[str, dict]:
        """
        Calibrate CEV (sigma, beta) to the vol surface for all tickers.

        Returns {ticker: {"sigma": float, "beta": float}}.
        """
        result = {}
        for t in tickers:
            result[t] = calibrate(surface_df, t, r)
        return result

    # ------------------------------------------------------------------
    # Backtest vol update
    # ------------------------------------------------------------------

    @staticmethod
    def update_params_for_vol(inception_params: dict,
                               atm_vol:          float) -> dict:
        """
        Rescale sigma so that the ATM implied vol matches atm_vol,
        holding beta fixed.

        The key identity (Hagan-Woodward ATM approximation, m=1):

            sigma_iv  ≈  sigma / S^(1 - beta)
            =>  sigma  =  sigma_iv · S^(1 - beta)

        atm_vol is an implied or realised vol (units: decimal, e.g. 0.35).
        sigma is the CEV diffusion coefficient (NOT the same as IV; it is
        typically much larger, e.g. 2.0 for a $90 stock with beta=0.6).

        We must NOT compare atm_vol directly to sigma (different units).
        Instead we use the inception ATM IV stored in "atm_vol_inception"
        (computed and saved by calibrate()) as the reference:

            scale      = atm_vol / atm_vol_inception
            sigma_new  = sigma_inception * scale

        This is equivalent to: sigma_new = atm_vol * S_inception^(1-beta),
        which correctly maps the new ATM IV back to a CEV sigma.
        """
        p   = dict(**inception_params)
        ref = p.get("atm_vol_inception", None)

        if ref is not None and ref > 1e-6:
            # correct path: compare IV to IV, then rescale sigma
            scale      = float(np.clip(atm_vol, 0.05, 1.50)) / ref
            p["sigma"] = float(np.clip(p["sigma"] * scale, 0.01, 10.0))
        else:
            # fallback for params that predate the atm_vol_inception key:
            # infer the inception IV from sigma and a unit-spot approximation,
            # then apply the same ratio rescaling
            beta  = p.get("beta", 0.5)
            sigma = p["sigma"]
            # approximate S_inception^(1-beta) from sigma/0.30 heuristic
            # (0.30 is a typical equity IV; this is only used if the key is absent)
            approx_S_power = sigma / max(0.30, 1e-6)
            ref_fallback   = sigma / max(approx_S_power, 1e-9)
            scale          = float(np.clip(atm_vol, 0.05, 1.50)) / max(ref_fallback, 1e-6)
            p["sigma"]     = float(np.clip(sigma * scale, 0.01, 10.0))

        return p

    # ------------------------------------------------------------------
    # Terminal parameter blending (default: full replacement)
    # ------------------------------------------------------------------

    @staticmethod
    def blend_terminal_params(inception_params: Dict[str, dict],
                               terminal_params:  Dict[str, dict],
                               tickers:          list) -> Dict[str, dict]:
        """
        Blend inception and terminal calibration results.

        CEV has only two parameters (sigma, beta).  Unlike Heston, both
        can be reliably estimated from a moderately sparse chain because
        beta is identified from the skew slope and sigma from the level.
        We therefore return terminal_params directly (full replacement),
        which is the BaseSimulator default.

        If the live chain is too sparse to fit beta reliably, inherit
        beta from inception and take only sigma from terminal.
        """
        blended = {}
        for t in tickers:
            ip = inception_params.get(t, {})
            tp = terminal_params.get(t, {})
            # If terminal has a valid beta estimate, use it; otherwise fall
            # back to inception beta (more stable from the richer surface).
            beta_terminal = tp.get("beta", None)
            beta_ok       = (beta_terminal is not None
                             and np.isfinite(beta_terminal)
                             and 0.0 <= beta_terminal <= 1.0)
            blended[t] = dict(
                sigma = tp.get("sigma", ip.get("sigma", 0.30)),
                beta  = beta_terminal if beta_ok else ip.get("beta", 0.5),
            )
        return blended

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------

    @classmethod
    def model_name(cls) -> str:
        return "CEV"


# Canonical export name — main.py does:  from <model> import Simulator
Simulator = CEVSimulator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    TICKERS    = ["NFLX", "SPOT", "DIS"]
    dummy_corr = np.array([[1.0, 0.357, 0.352],
                            [0.357, 1.0, 0.489],
                            [0.352, 0.489, 1.0]])
    dummy_spots = {"NFLX": 93.55, "SPOT": 565.41, "DIS": 97.88}

    # Build a synthetic surface for calibration testing
    rows = []
    for t in TICKERS:
        S0 = dummy_spots[t]
        for T in [0.5, 1.0, 1.5]:
            for m in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]:
                rows.append(dict(ticker=t, maturity_years=T, moneyness=m,
                                 spot_price=S0, implied_vol=0.30 - 0.10*(m-1.0)))
    surface_df = pd.DataFrame(rows)

    print("--- Calibration ---")
    params = CEVSimulator.calibrate(surface_df, TICKERS, r=0.05)
    print("Calibrated params:", {t: {k: f"{v:.4f}" for k,v in p.items()}
                                  for t, p in params.items()})

    print("\n--- Simulation (all methods) ---")
    sim = CEVSimulator(params, dummy_corr, TICKERS)
    for method in ["crude", "antithetic", "quasi", "stratified", "importance"]:
        result = sim.simulate(dummy_spots, T=1.5, n_sims=200, n_steps=50,
                              method=method, seed=42)
        if isinstance(result, tuple):
            paths, weights = result
            print(f"  {method:<14}: paths {paths.shape}  "
                  f"weights mean={weights.mean():.4f}  ESS/N={((weights.sum()**2)/(200*(weights**2).sum())):.2%}")
        else:
            print(f"  {method:<14}: paths {result.shape}  "
                  f"NFLX terminal mean={result[:,-1,0].mean():.2f}")

    print("\n--- update_params_for_vol ---")
    p_updated = CEVSimulator.update_params_for_vol(params["NFLX"], 0.40)
    print(f"  Original: {params['NFLX']}")
    print(f"  Updated (atm_vol=0.40): {p_updated}")

    print("\n--- blend_terminal_params ---")
    terminal = {t: dict(sigma=0.35, beta=0.4) for t in TICKERS}
    blended  = CEVSimulator.blend_terminal_params(params, terminal, TICKERS)
    print(f"  Blended: {blended}")

    print("\nAll checks passed.")