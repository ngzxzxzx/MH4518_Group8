"""
heston_v3.py  —  Heston with NMLE-CEKF Calibration
==================================================
Standalone Heston model built on BaseSimulator.

Calibration method follows Wang et al. (2018):
"Parameter estimates of Heston stochastic volatility model with MLE and
consistent EKF algorithm" (Sci China Inf Sci, 61:042202).

Key ideas implemented:
- CEKF state filtering for latent variance from transformed returns.
- NMLE parameter updates using Ito-transformed volatility dynamics.
- Iterative NMLE-CEKF loop for unknown volatility case.
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


KAPPA_UPPER_BOUND = 8.0
CEKF_LOOKBACK_DAYS = 252
VAR_FLOOR = 1e-8
PARAM_BOUNDS = {
    "mu": (-1.50, 1.50),
    "kappa": (0.05, 12.0),
    "theta": (1e-5, 0.60),
    "sigma": (0.05, 2.00),
    "rho": (-0.95, 0.95),
    "v0": (1e-5, 0.60),
}
STRUCTURAL_KEYS = {"mu", "kappa", "sigma", "rho"}


def heston_vol_surface(T: float, m: float, params: dict) -> float:
    """Approximate Heston implied vol at maturity T and moneyness m = K/S."""
    v0, theta, kappa = params["v0"], params["theta"], params["kappa"]
    sigma, rho = params["sigma"], params["rho"]
    if T <= 1e-6:
        return float(np.sqrt(v0))
    decay = (1 - np.exp(-kappa * T)) / (kappa * T)
    vol_T = np.sqrt(abs(theta + (v0 - theta) * np.exp(-kappa * T)))
    skew = rho * sigma * decay * np.log(m)
    convex = 0.1 * sigma * decay * np.log(m) ** 2
    return float(np.clip(vol_T + skew + convex, 0.15, 0.60))


def _vol_col(surface_df: pd.DataFrame) -> str:
    has_vol = "vol" in surface_df.columns
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
        f"Surface DataFrame has no vol/implied_vol column. Found: {list(surface_df.columns)}")


def calibrate(surface_df: pd.DataFrame, ticker: str, r: float = 0.05) -> dict:
    """Fast surface-only Heston calibration (same role as v2 baseline)."""
    vc = _vol_col(surface_df)
    sub = surface_df[surface_df["ticker"] == ticker].dropna(subset=[vc])
    if sub.empty:
        print(f"  [{ticker}] No surface data — using defaults")
        return dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)

    Ts = sub["maturity_years"].values.astype(float)
    ms = sub["moneyness"].values.astype(float)
    ivs = sub[vc].values.astype(float)

    weights = np.exp(-2 * (ms - 1.0) ** 2)
    weights /= weights.sum()

    def objective(x):
        v0, kappa, theta, sigma, rho = x
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-1 < rho < 1):
            return 1e8
        p = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        pred = np.array([heston_vol_surface(T, m, p) for T, m in zip(Ts, ms)])
        return float(np.sum(weights * (pred - ivs) ** 2))

    bounds = [(1e-4, 1.0), (0.1, 10.0), (1e-4, 1.0), (1e-4, 2.0), (-0.99, 0.99)]
    best_val = np.inf
    best_x = [0.04, 2.0, 0.05, 0.3, -0.6]

    for v0_init in [0.02, 0.04, 0.08, 0.15]:
        for rho_init in [-0.7, -0.4, -0.2]:
            try:
                res = minimize(
                    objective,
                    [v0_init, 2.0, 0.05, 0.3, rho_init],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-12},
                )
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


def _prepare_price_history(S) -> np.ndarray:
    if isinstance(S, pd.DataFrame):
        S = S.iloc[:, 0]

    price_arr = np.asarray(S, dtype=float).reshape(-1)
    price_arr = price_arr[np.isfinite(price_arr)]
    price_arr = price_arr[-(CEKF_LOOKBACK_DAYS + 1):]

    if len(price_arr) < 3:
        raise ValueError("Need at least 3 finite prices for NMLE-CEKF calibration")

    return price_arr


def _default_initial_guess(log_returns: np.ndarray, r: float) -> dict:
    if len(log_returns) < 2:
        realized_var = 0.04
    else:
        realized_var = float(np.clip(np.var(log_returns, ddof=1) * 252.0, 1e-4, 0.50))

    return {
        "mu": float(r),
        "kappa": 2.0,
        "theta": realized_var,
        "sigma": 0.30,
        "rho": -0.60,
        "v0": realized_var,
    }


def _apply_bounds(params: dict) -> dict:
    out = dict(params)
    for name, (lo, hi) in PARAM_BOUNDS.items():
        if name in out:
            out[name] = float(np.clip(out[name], lo, hi))
    return out


def _merge_with_fixed(params: dict, fixed_params: Optional[dict]) -> dict:
    out = _apply_bounds(params)
    if fixed_params is None:
        return out
    for name in STRUCTURAL_KEYS:
        if name in fixed_params and pd.notna(fixed_params[name]):
            lo, hi = PARAM_BOUNDS[name]
            out[name] = float(np.clip(fixed_params[name], lo, hi))
    return out


def _initial_variance_covariance(v0: float) -> float:
    return float(max(0.25 * v0 * v0, 1e-6))


def _cekf_filter(log_returns: np.ndarray,
                 params: dict,
                 dt: float,
                 r: float = 0.0):
    """
    CEKF filtering step for latent variance.

    Transformed observation:
        z_k = log(S_k / S_{k-1}) - r * dt
    """
    kappa = float(params["kappa"])
    theta = float(params["theta"])
    sigma = float(max(params["sigma"], 1e-6))
    rho = float(np.clip(params["rho"], -0.95, 0.95))

    v_hat = float(np.clip(params.get("v0", theta), VAR_FLOOR, 1.0))
    P = _initial_variance_covariance(v_hat)

    Q = np.eye(2)
    z = log_returns - float(r) * dt

    v_path = [v_hat]
    negloglik = 0.0

    for z_k in z:
        v_hat = float(np.clip(v_hat, VAR_FLOOR, 1.0))

        F = 1.0 - kappa * dt
        v_bar = float(np.clip(v_hat + kappa * theta * dt - kappa * v_hat * dt, VAR_FLOOR, 1.0))

        L = np.array([0.0, sigma * np.sqrt(max(v_hat * dt, 1e-12))], dtype=float)
        p_nominal = float(F * P * F + L @ Q @ L)

        p_upper = float(
            P * (abs(1.0 - kappa * dt) ** 2)
            + (dt ** 2) * (abs(kappa * theta) ** 2)
            + (abs(sigma) ** 2) * dt * v_hat * Q[1, 1]
        )
        delta_q = float(max(p_upper - p_nominal, 0.0))
        p_bar = float(max(p_nominal + delta_q, 1e-10))

        H = -0.5 * dt
        M = np.array([
            np.sqrt(max((1.0 - rho ** 2) * v_bar * dt, 1e-12)),
            rho * np.sqrt(max(v_bar * dt, 1e-12)),
        ], dtype=float)

        cross = float(H * (L @ Q @ M))
        innov_var = float(max(H * p_bar * H + M @ Q @ M + cross + cross, 1e-10))
        k_gain_num = float(p_bar * H + L @ Q @ M)
        k_gain = float(k_gain_num / innov_var)

        y_bar = float(-0.5 * v_bar * dt)
        innovation = float(z_k - y_bar)

        v_hat = float(np.clip(v_bar + k_gain * innovation, VAR_FLOOR, 1.0))

        ml_term = float(M @ Q @ L)
        delta_r = float(
            p_bar * (1.0 + k_gain * dt / 2.0) ** 2
            + 2.0 * (k_gain ** 2) * dt * v_bar * ((1.0 - rho ** 2) * Q[0, 0] + (rho ** 2) * Q[1, 1])
            - p_bar
            + k_gain * (H * p_bar + ml_term)
        )
        delta_r = float(max(delta_r, 0.0))

        P = float(max(p_bar - k_gain * (H * p_bar + ml_term) + delta_r, 1e-10))

        negloglik += 0.5 * (np.log(2.0 * np.pi * innov_var) + (innovation ** 2) / innov_var)
        v_path.append(v_hat)

    return float(negloglik), np.asarray(v_path, dtype=float)


def _nmle_from_vol_path(vol_path: np.ndarray,
                        log_returns: np.ndarray,
                        dt: float,
                        r: float) -> dict:
    """
    NMLE update from Ito-transformed volatility dynamics.

    Uses transformed volatility equation:
      sqrt(V_k)-sqrt(V_{k-1}) = (dt/(2 sqrt(V_{k-1}))) * P
                                - (dt/2) * sqrt(V_{k-1}) * kappa + eps_k,
    where P = kappa*theta - sigma^2/4.
    """
    v_prev = np.clip(vol_path[:-1], VAR_FLOOR, 1.0)
    v_next = np.clip(vol_path[1:], VAR_FLOOR, 1.0)

    if len(v_prev) < 2:
        raise ValueError("Insufficient filtered volatility points for NMLE update")

    s_prev = np.sqrt(v_prev)
    s_next = np.sqrt(v_next)
    y = s_next - s_prev

    X0 = dt / (2.0 * s_prev)
    X1 = -0.5 * dt * s_prev
    X = np.column_stack([X0, X1])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    P_hat = float(beta[0])
    kappa_hat = float(np.clip(beta[1], *PARAM_BOUNDS["kappa"]))

    resid = y - X @ beta
    sigma_hat = float(np.sqrt(max(4.0 * np.mean(resid ** 2) / max(dt, 1e-12), 1e-10)))
    sigma_hat = float(np.clip(sigma_hat, *PARAM_BOUNDS["sigma"]))

    theta_hat = float((P_hat + 0.25 * sigma_hat * sigma_hat) / max(kappa_hat, 1e-8))
    theta_hat = float(np.clip(theta_hat, *PARAM_BOUNDS["theta"]))

    z = log_returns - float(r) * dt
    eps1 = (z + 0.5 * v_prev * dt) / np.sqrt(np.maximum(v_prev * dt, 1e-12))
    eps2 = (v_next - v_prev - kappa_hat * (theta_hat - v_prev) * dt) / (
        sigma_hat * np.sqrt(np.maximum(v_prev * dt, 1e-12))
    )

    if np.std(eps1) < 1e-10 or np.std(eps2) < 1e-10:
        rho_hat = -0.5
    else:
        rho_hat = float(np.corrcoef(eps1, eps2)[0, 1])
    rho_hat = float(np.clip(np.nan_to_num(rho_hat, nan=-0.5), *PARAM_BOUNDS["rho"]))

    return {
        "mu": float(r),
        "kappa": kappa_hat,
        "theta": theta_hat,
        "sigma": sigma_hat,
        "rho": rho_hat,
        "v0": float(np.clip(v_next[-1], *PARAM_BOUNDS["v0"])),
    }


def heston_nmle_cekf_calibration(S,
                                 optimizer_budget: int = 10,
                                 dt: float = 1 / 252,
                                 initial_guess: Optional[dict] = None,
                                 fixed_params: Optional[dict] = None,
                                 r: float = 0.0) -> dict:
    """
    Estimate Heston parameters by iterative NMLE-CEKF.

    optimizer_budget controls CEKF-NMLE iteration count.
    If fixed_params is supplied, mu/kappa/sigma/rho are kept fixed while
    theta and v0 remain adaptive.
    """
    price_arr = _prepare_price_history(S)
    log_returns = np.diff(np.log(price_arr)).astype(float)

    params = _default_initial_guess(log_returns, r)
    if initial_guess:
        params.update(initial_guess)
    params = _merge_with_fixed(params, fixed_params)

    n_iter = max(2, int(optimizer_budget))
    best_params = dict(params)
    best_nll = np.inf

    for _ in range(n_iter):
        nll, v_path = _cekf_filter(log_returns, params, dt=dt, r=r)
        if nll < best_nll:
            best_nll = nll
            best_params = dict(params)
            best_params["v0"] = float(np.clip(v_path[-1], *PARAM_BOUNDS["v0"]))

        nmle_step = _nmle_from_vol_path(v_path, log_returns, dt=dt, r=r)
        params = _merge_with_fixed(nmle_step, fixed_params)

    return _apply_bounds(best_params)


def calibrate_cf(surface_df: pd.DataFrame,
                 ticker: str,
                 historical_price_df: pd.DataFrame = None,
                 r: float = 0.05) -> dict:
    if historical_price_df is None or historical_price_df.empty:
        raise ValueError(
            f"[{ticker}] NMLE-CEKF requires historical prices; "
            f"historical-only mode does not use option-surface fallback."
        )

    if isinstance(historical_price_df, pd.DataFrame):
        if ticker in historical_price_df.columns:
            hist_series = historical_price_df[ticker]
        elif historical_price_df.shape[1] == 1:
            hist_series = historical_price_df.iloc[:, 0]
        else:
            print(f"  [{ticker}] Historical prices missing ticker column — falling back to basic calibrate")
            return calibrate(surface_df, ticker, r)
    else:
        hist_series = historical_price_df

    nmle_results = heston_nmle_cekf_calibration(
        hist_series,
        optimizer_budget=10,
        dt=1 / 252,
        initial_guess=None,
        fixed_params=None,
        r=r,
    )
    v0 = nmle_results["v0"]
    mu = nmle_results["mu"]
    kappa = nmle_results["kappa"]
    theta = nmle_results["theta"]
    sigma = nmle_results["sigma"]
    rho = nmle_results["rho"]

    sub = pd.DataFrame()
    vc = "vol"

    if not sub.empty:
        Ts = sub["maturity_years"].values.astype(float)
        ms = sub["moneyness"].values.astype(float)
        ivs = sub[vc].values.astype(float)
        params_best = dict(v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
        rmse = float(np.sqrt(np.mean((
            np.array([heston_vol_surface(T, m, params_best) for T, m in zip(Ts, ms)]) - ivs) ** 2)))
    else:
        rmse = float("nan")

    feller = 2.0 * kappa * theta / max(sigma ** 2, 1e-10)
    print(f"  [{ticker}] Heston NMLE-CEKF  RMSE={rmse:.4f}  "
          f"v0={v0:.4f}  mu={mu:.4f}  kappa={kappa:.3f}  theta={theta:.4f}  "
          f"sigma={sigma:.3f}  rho={rho:.3f}  "
          f"Feller={feller:.2f}{'  OK' if feller >= 1 else '  warn<1'}")
    return dict(v0=float(v0), mu=float(mu), kappa=float(kappa), theta=float(theta),
                sigma=float(sigma), rho=float(rho))


def daily_heston_calibration(price_series: pd.Series,
                             window_days: int = 253,
                             step_days: int = 1,
                             optimizer_budget: int = 10,
                             use_previous_params: bool = True,
                             fixed_structural_params: Optional[dict] = None,
                             r: float = 0.0) -> pd.DataFrame:
    """Rolling NMLE-CEKF calibration."""
    window_days = CEKF_LOOKBACK_DAYS

    if len(price_series) < window_days + 1:
        raise ValueError(f"Price series too short ({len(price_series)} < {window_days + 1})")

    results = []
    previous_params = None
    all_end_idxs = list(range(window_days + 1, len(price_series) + 1, step_days))
    total_days = len(all_end_idxs)

    for cal_day, end_idx in enumerate(all_end_idxs, start=1):
        start_idx = end_idx - (window_days + 1)
        window_prices = price_series.iloc[start_idx:end_idx]
        date_label = price_series.index[end_idx - 1]
        print(f"  [Rolling calibration] Iteration {cal_day}/{total_days}  ({date_label.date()})")

        initial_guess = previous_params if use_previous_params else None

        try:
            params = heston_nmle_cekf_calibration(
                S=window_prices,
                optimizer_budget=optimizer_budget,
                dt=1 / 252,
                initial_guess=initial_guess,
                fixed_params=fixed_structural_params,
                r=r,
            )
            params["date"] = date_label
            results.append(params)
            previous_params = {
                "mu": params["mu"],
                "kappa": params["kappa"],
                "theta": params["theta"],
                "sigma": params["sigma"],
                "rho": params["rho"],
                "v0": params["v0"],
            }
        except Exception as exc:
            print(f"WARNING: Calibration failed at {date_label}: {exc}")
            continue

    return pd.DataFrame(results)


def calibrate_daily_for_backtest(realised_prices: pd.DataFrame,
                                 tickers: list,
                                 window_days: int = 253,
                                 optimizer_budget: int = 10,
                                 inception_params: Optional[Dict[str, dict]] = None,
                                 r: float = 0.0) -> Dict[str, pd.DataFrame]:
    window_days = CEKF_LOOKBACK_DAYS
    daily_params = {}

    for ticker in tickers:
        print(f"\n  Calibrating daily Heston NMLE-CEKF params for {ticker} "
              f"(window={window_days}d, budget={optimizer_budget}) ...")

        if ticker not in realised_prices.columns:
            print(f"WARNING: {ticker} not in price data")
            continue

        try:
            fixed_structural = None
            if inception_params is not None:
                fixed_structural = inception_params.get(ticker)

            daily_params[ticker] = daily_heston_calibration(
                price_series=realised_prices[ticker],
                window_days=window_days,
                step_days=1,
                optimizer_budget=optimizer_budget,
                use_previous_params=True,
                fixed_structural_params=fixed_structural,
                r=r,
            )
            print(f"Generated {len(daily_params[ticker])} calibrations")
        except Exception as exc:
            print(f"ERROR: {exc}")

    return daily_params


def blend_terminal_params(inception_params: dict,
                          terminal_params: dict,
                          tickers: list) -> dict:
    """
    Heston-specific blending: if terminal kappa is at the bound, keep
    structural params from inception and only update v0/theta.
    """
    if tickers is None:
        tickers = list(inception_params.keys())

    blended = {}
    kappa_tol = 0.05

    for t in tickers:
        inc = inception_params.get(t, {})
        term = terminal_params.get(t, {})
        if not inc or not term:
            blended[t] = dict(inc or term)
            continue

        kappa_at_bound = term.get("kappa", 0) >= KAPPA_UPPER_BOUND * (1 - kappa_tol)

        if kappa_at_bound:
            blended[t] = {
                "v0": term["v0"],
                "theta": term["theta"],
                "kappa": inc["kappa"],
                "sigma": inc["sigma"],
                "rho": inc["rho"],
            }
            print(f"  [{t}] terminal kappa at bound ({term['kappa']:.3f}) — "
                  f"freezing kappa/sigma/rho from inception, "
                  f"updating v0={term['v0']:.4f}, theta={term['theta']:.4f}")
        else:
            blended[t] = dict(term)
            print(f"  [{t}] terminal kappa unconstrained ({term['kappa']:.3f}) — "
                  f"using full terminal params")

    return blended


class HestonSimulator(BaseSimulator):
    """Correlated multi-asset Heston stochastic-volatility simulator."""

    def _simulate_batch(self,
                        Z_S: np.ndarray,
                        Z_v: np.ndarray,
                        spots: Dict[str, float],
                        params: Dict[str, dict],
                        T: float,
                        n_steps: int) -> np.ndarray:
        """Euler-Maruyama loop for Heston with full truncation."""
        n_sims = Z_S.shape[0]
        dt = T / n_steps
        sdt = np.sqrt(dt)

        Z_corr = Z_S @ self.L.T

        paths = np.empty((n_sims, n_steps + 1, self.n_assets))
        v = np.empty((n_sims, self.n_assets))

        for i, ticker in enumerate(self.tickers):
            paths[:, 0, i] = spots[ticker]
            v[:, i] = params[ticker]["v0"]

        for step in range(n_steps):
            for i, ticker in enumerate(self.tickers):
                kappa = params[ticker]["kappa"]
                theta = params[ticker]["theta"]
                sigma = params[ticker]["sigma"]
                rho = params[ticker]["rho"]

                v_pos = np.maximum(v[:, i], 0.0)
                sv = np.sqrt(v_pos)

                dW_v = rho * Z_corr[:, step, i] + np.sqrt(1 - rho ** 2) * Z_v[:, step, i]

                v[:, i] = v_pos + kappa * (theta - v_pos) * dt + sigma * sv * dW_v * sdt
                v[:, i] = np.maximum(v[:, i], 0.0)

                drift = (self.r - 0.5 * v_pos) * dt
                diff = sv * Z_corr[:, step, i] * sdt
                paths[:, step + 1, i] = paths[:, step, i] * np.exp(drift + diff)

        return paths

    def simulate(self,
                 spots: Dict[str, float],
                 T: float,
                 n_sims: int,
                 n_steps: int,
                 method: str = "crude",
                 params: Optional[Dict[str, dict]] = None,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        p = params if params is not None else self.params

        if method == "crude":
            Z_S = rng.standard_normal((n_sims, n_steps, self.n_assets))
            Z_v = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        if method == "antithetic":
            half = n_sims // 2
            Z_S_h = rng.standard_normal((half, n_steps, self.n_assets))
            Z_v_h = rng.standard_normal((half, n_steps, self.n_assets))
            paths_pos = self._simulate_batch(Z_S_h, Z_v_h, spots, p, T, n_steps)
            paths_neg = self._simulate_batch(-Z_S_h, Z_v_h, spots, p, T, n_steps)
            return np.concatenate([paths_pos, paths_neg], axis=0)

        if method == "quasi":
            if not SOBOL_AVAILABLE:
                print("  WARNING: scipy.stats.qmc unavailable; falling back to crude MC")
                return self.simulate(spots, T, n_sims, n_steps, "crude", p, seed)
            dim = n_steps * self.n_assets * 2
            engine = Sobol(d=dim, scramble=True, seed=seed)
            n_pow2 = int(2 ** np.ceil(np.log2(n_sims)))
            u = engine.random(n_pow2)[:n_sims]
            Z_flat = _spnorm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            half_d = n_steps * self.n_assets
            Z_S = Z_flat[:, :half_d].reshape(n_sims, n_steps, self.n_assets)
            Z_v = Z_flat[:, half_d:].reshape(n_sims, n_steps, self.n_assets)
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        if method == "stratified":
            Z_S = self._stratified_normals(n_sims, n_steps, self.n_assets, rng)
            Z_v = rng.standard_normal((n_sims, n_steps, self.n_assets))
            return self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)

        if method == "importance":
            Z_S, weights = self._importance_sampling_normals(n_sims, n_steps, self.n_assets, rng)
            Z_v = rng.standard_normal((n_sims, n_steps, self.n_assets))
            paths = self._simulate_batch(Z_S, Z_v, spots, p, T, n_steps)
            return paths, weights

        raise ValueError(f"Unknown method '{method}'. Choose: crude | antithetic | quasi | stratified | importance")

    @classmethod
    def calibrate(cls,
                  surface_df: pd.DataFrame,
                  tickers: list,
                  r: float = 0.05,
                  historical_price_df: pd.DataFrame = None,
                  is_inception: bool = False) -> Dict[str, dict]:
        if historical_price_df is None or historical_price_df.empty:
            raise ValueError(
                "Heston NMLE-CEKF historical-only mode requires historical_price_df."
            )

        result = {}
        for ticker in tickers:
            result[ticker] = calibrate_cf(surface_df, ticker, historical_price_df, r)
        return result

    @staticmethod
    def update_params_for_vol(inception_params: dict,
                              atm_vol: float) -> dict:
        p = dict(**inception_params)
        p["v0"] = float(np.clip(atm_vol ** 2, 1e-4, 1.0))
        p.setdefault("mu", 0.0)
        return p

    @classmethod
    def calibrate_daily_for_backtest(cls,
                                     realised_prices: pd.DataFrame,
                                     tickers: list,
                                     window_days: int = 253,
                                     optimizer_budget: int = 10,
                                     inception_params: Optional[Dict[str, dict]] = None) -> Dict[str, pd.DataFrame]:
        return calibrate_daily_for_backtest(
            realised_prices=realised_prices,
            tickers=tickers,
            window_days=window_days,
            optimizer_budget=optimizer_budget,
            inception_params=inception_params,
            r=0.0,
        )

    @staticmethod
    def blend_terminal_params(inception_params: Dict[str, dict],
                              terminal_params: Dict[str, dict],
                              tickers: list) -> Dict[str, dict]:
        return blend_terminal_params(inception_params, terminal_params, tickers)

    @classmethod
    def model_name(cls) -> str:
        return "Heston NMLE-CEKF"


Simulator = HestonSimulator
