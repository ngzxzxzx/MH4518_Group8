"""
heston_v2.py  —  Heston Stochastic Volatility Model
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
    from heston_v2 import calibrate, calibrate_cf, blend_terminal_params
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm as _spnorm
from tqdm import tqdm
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
GIBBS_LOOKBACK_DAYS = 252


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
def convertCDF(v, V_sort, W_sort, n_particles):

    W_sort = W_sort / np.sum(W_sort)
    N = len(V_sort)

    if v < V_sort[0]:
        return 0.0

    if v > V_sort[-1]:
        return 1.0

    j = np.searchsorted(V_sort, v) - 1
    j = max(0, min(j, N-2))

    vj = V_sort[j]
    vj1 = V_sort[j+1]

    if abs(vj1 - vj) < 1e-15:
        interp_factor = 0.0
    else:
        interp_factor = (v - vj) / (vj1 - vj)

    if j == 0:  # first interval
        weight = W_sort[0] + 0.5 * W_sort[1]
        return interp_factor * weight

    elif j == N-2:  # last interval
        base = np.sum(W_sort[:N-2]) + 0.5 * W_sort[N-2]
        weight = 0.5 * W_sort[N-2] + W_sort[N-1]
        return base + interp_factor * weight

    else:  # middle intervals
        base = np.sum(W_sort[:j]) + 0.5 * W_sort[j]
        weight = 0.5 * W_sort[j] + 0.5 * W_sort[j+1]
        return base + interp_factor * weight

def get_vol_estimator(cdf, n_particles, V_sort):

    U = np.random.rand(n_particles)
    particles = np.zeros(n_particles)

    N = len(V_sort)

    for i, u in enumerate(U):

        j = np.searchsorted(cdf, u)
        j = max(0, min(j, N-2))

        C_prev = 0 if j == 0 else cdf[j-1]
        C_curr = cdf[j]

        v1 = V_sort[j]
        v2 = V_sort[j+1]
        
        if C_curr - C_prev < 1e-15:
            # If the CDF is vertical here, just pick v1
            particles[i] = v1
        else:
            # Linear interpolation
            particles[i] = v1 + (v2 - v1) * (u - C_prev) / (C_curr - C_prev)

    return particles.mean()


def gibbs_sampler(m_b, L_b, a_s, b_s, s_i):

    # initial value
    sigma0 = s_i

    # sample beta | sigma
    cov_b = sigma0**2 * np.linalg.inv(L_b)
    beta_draw = np.random.multivariate_normal(m_b.flatten(), cov_b)
    beta = beta_draw.reshape(-1, 1)

    # sample sigma | beta
    sigma2 = 1 / np.random.gamma(shape=a_s, scale=1/b_s)
    sigma = np.sqrt(sigma2)

    return beta, sigma

def heston_gibbs_calibration(S, n_samples=5, dt=1/252, priors=None, initial_guesses=None, fixed_params=None):
    """
    fixed_params : dict with keys kappa, sigma, rho (and optionally mu).
                   When provided, those parameters are held constant and their
                   sampling steps are skipped.  Only v0 and theta are updated
                   each iteration, which is significantly faster.
    """
    # Enforce a fixed lookback window for Gibbs regardless of caller input.
    if isinstance(S, pd.DataFrame):
        S = S.iloc[:, 0]

    price_arr = np.asarray(S, dtype=float).reshape(-1)
    price_arr = price_arr[np.isfinite(price_arr)]
    price_arr = price_arr[-(GIBBS_LOOKBACK_DAYS + 1):]

    R = (price_arr[1:] / price_arr[:-1]).astype(float)
    n = len(R)
    n_particles = GIBBS_LOOKBACK_DAYS
    if priors is None:
        priors = {
            'lambda_prior': np.eye(2) * 1.0,
            'mu_prior': np.array([[0.0], [0.990]]),
            'tau_prior_eta': 0.01,
            'mu_prior_eta': 1.0,
            'a_prior_sigma': 2.0,
            'b_prior_sigma': 0.1,
            'tau_prior_phi': 0.01,
            'mu_prior_phi': -0.7,
            'a_prior_omega': 2.0,
            'b_prior_omega': 0.1
        }

    if initial_guesses is None:
        realized_vol = R.std() * np.sqrt(252) # annualized
        
        initial_guesses = {
            'mu': R.mean() * 252, 
            'kappa': 2.0,
            'theta': realized_vol**2, 
            'sigma': 0.3,
            'rho': -0.7
        }

    lambda_prior = np.atleast_2d(priors['lambda_prior']).reshape(2, 2)
    mu_prior = priors['mu_prior'].reshape(2, 1)
    tau_prior_eta = priors['tau_prior_eta']
    mu_prior_eta = priors['mu_prior_eta']
    a_prior_sigma = priors['a_prior_sigma']
    b_prior_sigma = priors['b_prior_sigma']
    tau_prior_phi = priors['tau_prior_phi']
    mu_prior_phi = priors['mu_prior_phi']
    a_prior_omega = priors['a_prior_omega']
    b_prior_omega = priors['b_prior_omega']

    mu_i, kappa_i, theta_i, sigma_i, rho_i = initial_guesses['mu'], initial_guesses['kappa'], initial_guesses['theta'], initial_guesses['sigma'], initial_guesses['rho']

    # Override with fixed structural params if provided
    if fixed_params is not None:
        kappa_i = float(fixed_params.get('kappa', kappa_i))
        sigma_i = float(fixed_params.get('sigma', sigma_i))
        rho_i   = float(fixed_params.get('rho',   rho_i))

    mu_chain, kappa_chain, theta_chain, sigma_chain, rho_chain = [], [], [], [], []
    all_v_estimates = []
    for i in range(n_samples):
        print(f"\n--- Gibbs Iteration {i+1}/{n_samples} ---")
        
        V = np.full(n_particles, float(np.atleast_1d(theta_i).item())) 
        v_estimates_list = [] 
        kappa_i, theta_i, sigma_i, mu_i, rho_i = [float(np.atleast_1d(x).item()) for x in [kappa_i, theta_i, sigma_i, mu_i, rho_i]]

        for k in tqdm(range(1, n)):
            R_k = R[k]
            V_prev = np.maximum(V.flatten(), 1e-7) # Floor to prevent sqrt(0)
            
            epsilon = np.random.normal(0, 1, n_particles)
            z = np.clip((R_k - mu_i * dt - 1) / (np.sqrt(dt) * np.sqrt(V_prev)), -4.0, 4.0) #Clip Z to prevent extreme returns from blowing up V
            w = rho_i * z + np.sqrt(1 - rho_i**2) * epsilon

            V = V_prev + kappa_i * (theta_i - V_prev) * dt + sigma_i * np.sqrt(dt) * np.sqrt(V_prev) * w
            V = np.clip(V.flatten(), 1e-8, 2.0)
            log_W = -0.5 * np.log(2 * np.pi * V * dt + 1e-12) - 0.5 * ((R_k - mu_i * dt - 1)**2) / (V * dt + 1e-12)
            W = np.exp(log_W - np.max(log_W))
            W /= (np.sum(W) + 1e-15)

            sort_idx = np.argsort(V)
            V_sorted, W_sorted = V[sort_idx], W[sort_idx]
            
            cdf = np.array([convertCDF(v, V_sorted, W_sorted, n_particles) for v in V_sorted])
            v_est = get_vol_estimator(cdf, n_particles, V_sorted)
            
            if not np.isfinite(v_est) or v_est <= 0:
                v_est = v_estimates_list[-1] if v_estimates_list else theta_i
            v_estimates_list.append(v_est)

        v_estimates_list.append(v_estimates_list[-1])
        v_arr = np.array(v_estimates_list).reshape(-1, 1)
        v_safe = np.maximum(v_arr, 1e-7)
        #all_v_estimates.append(np.mean(v_safe))

        x_mu = 1.0 / np.sqrt(v_safe * dt)
        y_mu = R.reshape(-1, 1) / np.sqrt(v_safe * dt)
        tau_post_mu = (x_mu.T @ x_mu).item() + tau_prior_eta
        eta_hat = np.linalg.inv(x_mu.T @ x_mu) * (x_mu.T @ y_mu)
        mu_post_eta = ((x_mu.T @ y_mu * eta_hat).item() + mu_prior_eta * tau_prior_eta) / tau_post_mu
        mu_i = (np.random.normal(mu_post_eta, 1/np.sqrt(tau_post_mu)) - 1) / dt
        mu_chain.append(mu_i)

        v_l, v_n = v_safe[:-1], v_safe[1:]

        if fixed_params is not None:
            phi_fixed  = float(np.clip(1.0 - kappa_i * dt, 0.01, 0.98))
            y_reg      = v_n / np.sqrt(v_l * dt)
            X0         = (1.0 / (v_l * np.sqrt(dt))).flatten()   # (n-1,)
            X1         = (np.sqrt(v_l) / np.sqrt(dt)).flatten()  # (n-1,)
            y_partial  = y_reg.flatten() - phi_fixed * X1
            # 1-D Normal-Normal posterior (sigma_i fixed as precision weight)
            L_b0    = float((X0 ** 2).sum()) + float(lambda_prior[0, 0])
            m_b0    = (float((X0 * y_partial).sum()) + float(lambda_prior[0, 0]) * float(mu_prior[0, 0])) / L_b0
            beta0_draw = float(np.random.normal(m_b0, sigma_i / np.sqrt(L_b0)))
            beta0_draw = np.clip(beta0_draw, 1e-7, 0.5)
            theta_i    = float(np.clip(beta0_draw / max(1.0 - phi_fixed, 1e-6), 1e-5, 0.4))
        
            kappa_chain.append(kappa_i)
            sigma_chain.append(sigma_i)
            theta_chain.append(theta_i)
        else:
            y_reg = v_n / np.sqrt(v_l * dt)
            X_reg = np.hstack((1.0 / (v_l * np.sqrt(dt)), np.sqrt(v_l) / np.sqrt(dt)))

            L_beta_post = X_reg.T @ X_reg + lambda_prior
            mu_beta_post = np.linalg.inv(L_beta_post) @ (lambda_prior @ mu_prior + X_reg.T @ y_reg)
            a_s_post = a_prior_sigma + n/2
            quad = (y_reg.T @ y_reg + mu_prior.T @ lambda_prior @ mu_prior - mu_beta_post.T @ L_beta_post @ mu_beta_post).item()
            b_s_post = b_prior_sigma + 0.5 * np.clip(quad, 0, 2.0)

            beta_draw, sigma_i = gibbs_sampler(mu_beta_post, L_beta_post, a_s_post, b_s_post, sigma_i)
            phi_draw = np.clip(beta_draw[1, 0], 0.01, 0.98)

            kappa_i = (1.0 - phi_draw) / dt

            beta0_draw = np.clip(beta_draw[0, 0], 1e-7, 0.5)

            theta_i = beta0_draw / (1.0 - phi_draw)
            sigma_i = np.clip(sigma_i, 0.01, 1.2)
            kappa_i = np.clip(kappa_i, 0.1, 50.0)
            theta_i = np.clip(theta_i, 1e-5, 0.4)

            sigma_chain.append(sigma_i)
            kappa_chain.append(kappa_i)
            theta_chain.append(theta_i)

        if fixed_params is not None:
            # rho is fixed — skip sampling, just record the fixed value.
            rho_chain.append(rho_i)
        else:
            e1 = (R[1:].reshape(-1,1) - mu_i*dt - 1) / (np.sqrt(dt * v_l))
            e2 = (v_n - v_l - kappa_i*(theta_i - v_l)*dt) / (sigma_i * np.sqrt(dt * v_l))
            A_rho = np.hstack((e1, e2)).T @ np.hstack((e1, e2))

            mu_phi_post = (A_rho[0, 1] + tau_prior_phi * mu_prior_phi) / (A_rho[0, 0] + tau_prior_phi)
            tau_phi_post = A_rho[0, 0] + tau_prior_phi

            a_omega_post = a_prior_omega + n/2
            b_omega_post = b_prior_omega + 0.5 * (A_rho[1, 1] - (A_rho[0, 1]**2) / np.maximum(A_rho[0, 0], 1e-12))

            omega_i = 1 / np.random.gamma(shape=a_omega_post, scale=1/b_omega_post)
            phi_i = np.random.normal(loc=mu_phi_post, scale=np.sqrt(omega_i)/np.sqrt(tau_phi_post))

            rho_i = np.clip(phi_i / np.sqrt(omega_i + phi_i**2), -0.98, 0.98)
            rho_chain.append(rho_i)


    return {
        "mu": np.mean(mu_chain),
        "v0": np.mean(v_estimates_list),
        "kappa": np.mean(kappa_chain),
        "theta": np.mean(theta_chain),
        "sigma": np.mean(sigma_chain),
        "rho": np.mean(rho_chain),
    }

def calibrate_cf(surface_df: pd.DataFrame, ticker: str, historical_price_df: pd.DataFrame = None,
                 r: float = 0.05) -> dict:
    """
    Robust Heston calibration for real/noisy market option chains.

    Uses differential evolution (global search) followed by L-BFGS-B polish.
    Handles sparse chains by applying an ATM-proximity weighting scheme.
    Falls back to calibrate() if data is insufficient.
    """
    if historical_price_df is None or historical_price_df.empty:
        print(f"  [{ticker}] No historical prices for Gibbs calibration — falling back to basic calibrate")
        return calibrate(surface_df, ticker, r)

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

    gibbs_results = heston_gibbs_calibration(hist_series, n_samples=10, dt=1/252)
    v0, mu, kappa, theta, sigma, rho = gibbs_results["v0"], gibbs_results["mu"], gibbs_results["kappa"], \
                                gibbs_results["theta"], gibbs_results["sigma"], gibbs_results["rho"]

    if surface_df is not None:
        sub = surface_df[surface_df["ticker"] == ticker]
        vc  = _vol_col(sub)
        atm_iv = sub.iloc[(sub['moneyness'] - 1.0).abs().argsort()[:1]][vc].values[0]
        v0 = float(atm_iv**2)
    else:
        v0 = gibbs_results['v0']
        vc  = "vol"
    Ts  = sub["maturity_years"].values.astype(float)
    ms  = sub["moneyness"].values.astype(float)
    ivs = sub[vc].values.astype(float)

    params_best = dict(v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    rmse = float(np.sqrt(np.mean((
        np.array([heston_vol_surface(T, m, params_best) for T, m in zip(Ts, ms)]) - ivs) ** 2)))
    feller = 2 * kappa * theta / sigma**2
    print(f"  [{ticker}] Heston fit (DE)  RMSE={rmse:.4f}  "
          f"v0={v0:.4f}  mu={mu:.4f}  kappa={kappa:.3f}  theta={theta:.4f}  "
          f"sigma={sigma:.3f}  rho={rho:.3f}  "
          f"Feller={feller:.2f}{'  OK' if feller >= 1 else '  warn<1'}")
    return dict(v0=float(v0), mu=float(mu), kappa=float(kappa), theta=float(theta),
                sigma=float(sigma), rho=float(rho))


# ---------------------------------------------------------------------------
# Daily calibration  (rolling window parameter estimation)
# ---------------------------------------------------------------------------

def daily_heston_calibration(price_series: pd.Series,
                            window_days: int = 253,
                            step_days: int = 5,
                            n_gibbs_samples: int = 10,
                            n_particles: int = 100,
                            use_previous_params: bool = True,
                            fixed_structural_params: Optional[dict] = None) -> pd.DataFrame:
    """
    Rolling-window Heston parameter estimation using Gibbs sampling.

    Performs daily (or periodic) calibration on simple returns, using previous
    day's parameters as warm-start for faster convergence.

    Args:
        price_series: pd.Series of closing prices (must have DatetimeIndex)
        window_days: Rolling window size in days (default 60 for ~3 months)
        step_days: Calibration frequency in days (1=daily, 5=weekly, etc.)
        n_gibbs_samples: MCMC iterations per calibration (lower=faster)
        n_particles: Particles per Gibbs iteration
        use_previous_params: Warm-start from previous day's estimates (faster)

    Returns:
        pd.DataFrame with columns [date, v0, kappa, theta, sigma, rho]

    Notes:
        If fixed_structural_params is provided, kappa/sigma/rho/mu are
        copied from it on every date, so only v0 and theta vary daily.
    """
    window_days = GIBBS_LOOKBACK_DAYS

    if len(price_series) < window_days + 1:
        raise ValueError(f"Price series too short ({len(price_series)} < {window_days + 1})")

    results = []
    previous_params = None

    # Rolling window loop
    all_end_idxs = list(range(window_days + 1, len(price_series) + 1, step_days))
    total_days = len(all_end_idxs)
    for cal_day, end_idx in enumerate(all_end_idxs, start=1):
        start_idx = end_idx - (window_days + 1)
        window_prices = price_series.iloc[start_idx:end_idx]

        date_label = price_series.index[end_idx - 1]
        print(f"  [Rolling calibration] Iteration {cal_day}/{total_days}  ({date_label.date()})")

        # Warm-start from previous day
        initial_guesses = previous_params if use_previous_params else None

        try:
            params = heston_gibbs_calibration(
                S=window_prices,
                n_samples=n_gibbs_samples,
                dt=1/252,
                priors=None,
                initial_guesses=initial_guesses,
                fixed_params=fixed_structural_params,  # skips kappa/sigma/rho sampling
            )

            params['date'] = date_label
            results.append(params)
            previous_params = {
                'mu':    params['mu'],
                'kappa': params['kappa'],
                'theta': params['theta'],
                'sigma': params['sigma'],
                'rho':   params['rho'],
            }
        except Exception as e:
            print(f"WARNING: Calibration failed at {date_label}: {e}")
            continue

    return pd.DataFrame(results)


def calibrate_daily_for_backtest(realised_prices: pd.DataFrame,
                                 tickers: list,
                                 window_days: int = 253,
                                 n_gibbs_samples: int = 10,
                                 n_particles: int = 100,
                                 inception_params: Optional[Dict[str, dict]] = None) -> Dict[str, pd.DataFrame]:
    """
    Daily calibration for all tickers (for backtest use).

    Args:
        realised_prices: DataFrame with DatetimeIndex, columns=tickers
        tickers: List of ticker symbols
        window_days: Rolling window size
        n_gibbs_samples: Gibbs iterations per day
        n_particles: Particles per iteration

    Returns:
        {ticker: daily_params_df}  where each df has [date, v0, kappa, theta, sigma, rho]
    """
    window_days = GIBBS_LOOKBACK_DAYS

    daily_params = {}

    for ticker in tickers:
        print(f"\n  Calibrating daily Heston params for {ticker} "
              f"(window={window_days}d, n_samples={n_gibbs_samples}) ...")
        
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
                step_days=5,
                n_gibbs_samples=n_gibbs_samples,
                n_particles=n_particles,
                use_previous_params=True,
                fixed_structural_params=fixed_structural,
            )
            print(f"Generated {len(daily_params[ticker])} calibrations")
        except Exception as e:
            print(f"ERROR: {e}")

    return daily_params


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
                  r:          float = 0.05,
                  historical_price_df: pd.DataFrame = None,
                  is_inception: bool = False) -> Dict[str, dict]:
        """
        Calibrate Heston parameters to a vol surface for all tickers.

        is_inception=True  : use Gibbs (calibrate_cf) when historical prices
                             are available — for inception pricing calibration.
        is_inception=False : always use fast L-BFGS-B calibrate() — for
                             terminal / live-chain calibration.

        Returns {ticker: {v0, kappa, theta, sigma, rho}}.
        """
        use_gibbs = (is_inception
                     and historical_price_df is not None
                     and not historical_price_df.empty)
        result = {}
        for t in tickers:
            if use_gibbs:
                result[t] = calibrate_cf(surface_df, t, historical_price_df, r)
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
        p.setdefault("mu", 0.0)
        return p

    # ------------------------------------------------------------------
    # Daily calibration for backtest  (class wrapper)
    # ------------------------------------------------------------------

    @classmethod
    def calibrate_daily_for_backtest(cls,
                                     realised_prices: pd.DataFrame,
                                     tickers: list,
                                     window_days: int = 253,
                                     n_gibbs_samples: int = 10,
                                     n_particles: int = 50,
                                     inception_params: Optional[Dict[str, dict]] = None) -> Dict[str, pd.DataFrame]:
        """module-level daily Gibbs calibration."""
        return calibrate_daily_for_backtest(
            realised_prices=realised_prices,
            tickers=tickers,
            window_days=window_days,
            n_gibbs_samples=n_gibbs_samples,
            n_particles=n_particles,
            inception_params=inception_params,
        )

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
        return "Heston_v2"


# Canonical export name — main.py does:  from <model> import Simulator
Simulator = HestonSimulator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKERS = ["NFLX", "SPOT", "DIS"]

    dummy_params = {t: dict(v0=0.04, mu=0.0, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)
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
