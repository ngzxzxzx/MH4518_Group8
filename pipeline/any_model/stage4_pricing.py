"""
Stage IV & V  —  BRC Pricing, Greeks, and Control Variate
==========================================================
Inputs  (from stages I-III)
  - paths                  : np.ndarray (n_sims, n_steps+1, n_assets)
  - initial_fixing_prices  : dict  {ticker: float}
  - BRC contract terms     : barrier, coupon, principal, payment_dates
  - simulator              : HestonSimulator  (for bump-reprice Greeks)

Outputs
  - price result dict  : price, std_error, CI, scenario probs
  - greeks dict        : delta×3, vega×3, rho, correlation sensitivity×3
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# BRC payoff engine
# ---------------------------------------------------------------------------

class BRCPayoff:
    """
    Encapsulates all BRC contract terms and evaluates payoffs given paths.

    Contract specification (from pricing_report.txt):
      Barrier   : 50% of initial fixing price, continuous monitoring
      Coupon    : 13.75% p.a., paid quarterly
      Principal : 100
      Maturity  : 2026-10-01  (18 months from Apr 2 2025)
    """

    # Exact coupon payment dates from UBS Termsheet (ISIN CH1431536452)
    TERMSHEET_COUPON_DATES = [
        "2025-07-10",   # i=1
        "2025-10-09",   # i=2
        "2026-01-12",   # i=3
        "2026-04-13",   # i=4
        "2026-07-10",   # i=5
        "2026-10-09",   # i=6  Expiration Date / Maturity Date
    ]

    def __init__(self,
                 tickers:               list,
                 initial_fixing_prices: Dict[str, float],
                 valuation_date:        str,
                 maturity_date:         str         = "2026-10-09",
                 barrier_level:         float       = 0.50,
                 coupon_rate:           float       = 0.1375,
                 principal:             float       = 100.0,
                 risk_free_rate:        float       = 0.05,
                 coupon_payment_dates:  list        = None,
                 rate_curve_fn                      = None):
        """
        Parameters
        ----------
        risk_free_rate  : flat fallback rate (decimal) used when rate_curve_fn
                          is None.  Also passed to the simulator as the drift
                          rate for the *remaining tenor* (see stage6_backtest).
        rate_curve_fn   : optional callable  r(tenor_years) -> float  that
                          returns the continuously compounded zero rate for a
                          given tenor.  When provided, each cash flow is
                          discounted at its own tenor-matched rate instead of
                          the flat risk_free_rate.  Build with:
                              from stage1_data import get_rate_curve_fn
                              r_fn = get_rate_curve_fn(ds.treasury_curve, date)
        coupon_payment_dates : explicit list of ISO date strings.
            Defaults to TERMSHEET_COUPON_DATES (UBS termsheet).
        """
        self.tickers               = tickers
        self.n_assets              = len(tickers)
        self.initial_fixing_prices = initial_fixing_prices
        self.valuation_date        = pd.Timestamp(valuation_date)
        self.maturity_date         = pd.Timestamp(maturity_date)
        self.barrier_level         = barrier_level
        self.coupon_rate           = coupon_rate
        self.principal             = principal
        self.r                     = risk_free_rate
        self.rate_curve_fn         = rate_curve_fn   # None → flat r

        self.T = (self.maturity_date - self.valuation_date).days / 365.0

        # barrier absolute levels (from termsheet fixing prices)
        self.barrier_prices = np.array([
            initial_fixing_prices[t] * barrier_level for t in tickers
        ])

        # payment dates: use explicit list if provided, else termsheet defaults
        raw_dates = coupon_payment_dates or self.TERMSHEET_COUPON_DATES
        self.payment_dates = [pd.Timestamp(d) for d in raw_dates
                              if pd.Timestamp(d) > self.valuation_date]
        # always include maturity as the last payment date
        if not self.payment_dates or self.payment_dates[-1] != self.maturity_date:
            self.payment_dates.append(self.maturity_date)

        self.payment_times = np.array([
            (d - self.valuation_date).days / 365.0
            for d in self.payment_dates
        ])

    def _discount(self, tenor_years: float) -> float:
        """
        Return exp(-r(t)·t) using the term-structure curve if available,
        otherwise the flat risk_free_rate.
        """
        if self.rate_curve_fn is not None:
            r = self.rate_curve_fn(tenor_years)
        else:
            r = self.r
        return float(np.exp(-r * tenor_years))

    # ------------------------------------------------------------------

    def _generate_payment_dates(self) -> List[pd.Timestamp]:
        # Retained for compatibility; logic now lives in __init__.
        return self.payment_dates

    # ------------------------------------------------------------------

    def evaluate(self, paths: np.ndarray,
                 already_breached: bool = False) -> np.ndarray:
        """
        Compute discounted payoff for every simulation path.

        Parameters
        ----------
        paths            : (n_sims, n_steps+1, n_assets)
        already_breached : True if the barrier was already knocked in
                           on a prior date (used in backtest conditioning)

        Returns
        -------
        payoffs : (n_sims,)  discounted total payoff per $100 principal
        """
        n_sims = paths.shape[0]

        # --- coupon cash flows (paid regardless of barrier) ---
        # Each coupon is discounted at the zero rate for ITS OWN tenor,
        # giving a term-structure-consistent present value.
        coupon_pv = 0.0
        quarterly_coupon = self.principal * self.coupon_rate * 0.25
        for t_pay in self.payment_times[:-1]:   # exclude maturity date entry
            coupon_pv += quarterly_coupon * self._discount(t_pay)

        # --- barrier detection ---
        if already_breached:
            barrier_touched = np.ones(n_sims, dtype=bool)
        else:
            # any asset below its barrier at any time step
            # paths[:, :, i]  vs  barrier_prices[i]
            min_prices      = paths.min(axis=1)           # (n_sims, n_assets)
            barrier_touched = np.any(
                min_prices <= self.barrier_prices[np.newaxis, :], axis=1
            )

        # --- principal at maturity ---
        final_prices = paths[:, -1, :]                 # (n_sims, n_assets)
        init_arr     = np.array([self.initial_fixing_prices[t]
                                 for t in self.tickers])

        # scenario flags
        no_breach       = ~barrier_touched
        performance     = final_prices / init_arr[np.newaxis, :]  # (n_sims, n_assets)
        all_above_init  = np.all(performance >= 1.0, axis=1)

        # principal values per simulation
        # Principal is discounted at the zero rate for the remaining tenor to maturity.
        principal_pv    = np.empty(n_sims)
        T_mat           = self.payment_times[-1]
        df_mat          = self._discount(T_mat)   # single discount factor for maturity

        # Scenario 1: no barrier touch  → full principal
        mask1 = no_breach
        principal_pv[mask1] = self.principal * df_mat

        # Scenario 2: barrier touched, all finals ≥ initial → full principal
        mask2 = barrier_touched & all_above_init
        principal_pv[mask2] = self.principal * df_mat

        # Scenario 3: barrier touched, worst-of final < initial → physical delivery
        mask3 = barrier_touched & ~all_above_init
        if mask3.any():
            worst_idx        = np.argmin(performance[mask3], axis=1)
            worst_perf       = performance[mask3][
                np.arange(mask3.sum()), worst_idx
            ]
            principal_pv[mask3] = (self.principal * worst_perf * df_mat)

        payoffs = coupon_pv + principal_pv

        # store for diagnostics
        self._last_barrier_touched = barrier_touched
        self._last_performance     = performance

        return payoffs

    # ------------------------------------------------------------------

    def result_summary(self, payoffs: np.ndarray) -> dict:
        bt  = self._last_barrier_touched
        n   = len(payoffs)
        perf = self._last_performance
        init_arr = np.array([self.initial_fixing_prices[t] for t in self.tickers])

        barrier_prob     = float(np.mean(bt))
        all_above        = np.all(perf >= 1.0, axis=1)
        full_after_touch = float(np.mean(bt & all_above))
        physical_del     = float(barrier_prob - full_after_touch)

        return dict(
            price                        = float(np.mean(payoffs)),
            std_error                    = float(np.std(payoffs) / np.sqrt(n)),
            ci_95_lower                  = float(np.percentile(payoffs, 2.5)),
            ci_95_upper                  = float(np.percentile(payoffs, 97.5)),
            barrier_touch_probability    = barrier_prob,
            prob_no_breach               = 1.0 - barrier_prob,
            prob_full_principal_after_touch = full_after_touch,
            prob_physical_delivery       = physical_del,
            n_simulations                = n,
        )


# ---------------------------------------------------------------------------
# Control variate: vanilla worst-of put (known BS price)
# ---------------------------------------------------------------------------

def _bs_worst_of_put_approx(spots: Dict[str, float],
                             initial_fixing: Dict[str, float],
                             tickers: list,
                             T: float, r: float, vols: Dict[str, float],
                             barrier: float) -> float:
    """
    Approximate analytical price for the dominant downside component:
    a portfolio of put options at the barrier strike, one per asset,
    taking the minimum (independence approximation for CV anchor).
    Used only as a rough control variate anchor.
    """
    pv = 0.0
    for t in tickers:
        S = spots[t]
        K = initial_fixing[t] * barrier
        sigma = vols.get(t, 0.30)
        if T <= 0 or sigma <= 0:
            pv += max(0.0, K - S)
            continue
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        from scipy.stats import norm
        put = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        pv += put / len(tickers)   # equal weight average
    return pv


def control_variate_price(simulator,
                           payoff_fn,
                           spots:         Dict[str, float],
                           initial_fixing: Dict[str, float],
                           tickers:       list,
                           T:             float,
                           n_sims:        int,
                           n_steps:       int,
                           r:             float,
                           vols:          Dict[str, float],
                           barrier:       float,
                           seed:          int = 42) -> Tuple[float, float]:
    """
    Control variate Monte Carlo using the average barrier put as CV.
    Returns (cv_price, cv_std_error).
    """
    paths = simulator.simulate(spots, T, n_sims, n_steps,
                               method="crude", seed=seed)

    # raw BRC payoffs
    Y = payoff_fn(paths)

    # control variate: sum of discounted barrier-strike put payoffs
    init_arr    = np.array([initial_fixing[t] for t in tickers])
    barrier_abs = init_arr * barrier
    final       = paths[:, -1, :]                # (n_sims, n_assets)
    put_payoffs = np.mean(
        np.maximum(barrier_abs[np.newaxis, :] - final, 0.0) * np.exp(-r * T),
        axis=1
    )   # (n_sims,)

    # analytical CV anchor
    cv_anchor = _bs_worst_of_put_approx(spots, initial_fixing, tickers,
                                         T, r, vols, barrier)

    # optimal coefficient beta
    cov_matrix = np.cov(Y, put_payoffs)
    beta       = -cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0

    Y_cv = Y + beta * (put_payoffs - cv_anchor)

    price  = float(np.mean(Y_cv))
    stderr = float(np.std(Y_cv) / np.sqrt(n_sims))
    return price, stderr


# ---------------------------------------------------------------------------
# Greeks via finite difference — pathwise noise reuse
# ---------------------------------------------------------------------------
#
# Speed problem with the old approach
# ------------------------------------
# The old BRCGreeks called simulate() (= full MC) twice per bump direction
# per Greek per ticker.  For 3 tickers that was 26 full simulations per
# Greek date, making the backtest 26× slower whenever Greeks were computed.
#
# Solution: pathwise finite difference
# -------------------------------------
# The Heston spot SDE  dS = r·S·dt + √v·S·dW_S  separates the noise dW_S
# from the starting level S₀.  With the same Brownian increments Z_S, Z_v
# the log-spot path is:
#
#   log S(t; S₀) = log S₀ + ∑ [ (r - ½v) dt  +  √v · Z_corr · √dt ]
#
# Bumping S₀ → S₀ ± h merely shifts every log-path by ±log(1 ± h/S₀).
# All variance paths v(t) are unchanged.  So delta and gamma can be priced
# from a single pair of noise draws at zero extra simulation cost.
#
# For vega, rho, and correlation the dynamics themselves change (v₀ or r or L
# enter the SDE), so separate simulations are unavoidable.  However these
# only need a smaller n_sims_greek (default 1000) because the signal-to-noise
# is higher for these smoother sensitivities.
#
# Total simulations:
#   Old: 26  (2 per bump × 13 bump dimensions)
#   New:  1 base + 3 asset × 2 vol bumps + 1 rho × 2 + 3 pairs × 2 = 1 + 6 + 2 + 6 = 15
#   But base and vol-bumped paths share the same Z_v for variance — the only
#   re-simulation is for params that change the variance SDE.
#   Effective cost ≈ 3–5× faster in practice, more at low n_sims_greek.

class BRCGreeks:
    """
    BRC Greeks by central finite difference with pathwise noise reuse.

    Delta / Gamma
    -------------
    Computed from a single set of (Z_S, Z_v) noise arrays.  The spot paths
    at S₀ ± h are reconstructed analytically by shifting the log-path,
    so no extra simulation is needed.  Cost = 1 simulation total.

    Vega
    ----
    Bumps v0 per ticker and re-simulates with the same Z_S, Z_v seed.
    Requires 2 × n_assets re-simulations but uses n_sims_greek ≤ n_sims.

    Rho
    ---
    Patches the payoff discount rate and re-evaluates on the BASE paths
    (rate change has no effect on the simulated paths in the risk-neutral
    measure — r only enters the drift and discount factor).  Cost = 0 extra
    simulations, just two payoff evaluations.

    Correlation sensitivity
    -----------------------
    Rebuilds the Cholesky with a bumped correlation matrix and re-simulates.
    2 simulations per pair × 3 pairs = 6 simulations.  Uses n_sims_greek.
    """

    # bump sizes
    SPOT_BUMP = 0.01      # 1%  relative spot bump
    VOL_BUMP  = 0.001     # bump to v0 (raw variance units)
    RATE_BUMP = 0.0001    # 1 basis point
    CORR_BUMP = 0.02      # 2 correlation points

    def __init__(self,
                 simulator,
                 payoff_obj:    BRCPayoff,
                 spots:         Dict[str, float],
                 T:             float,
                 n_sims:        int,
                 n_steps:       int,
                 n_sims_greek:  int  = None,
                 seed:          int  = 42):
        """
        Parameters
        ----------
        n_sims        : paths for base price (full quality)
        n_sims_greek  : paths for vega / corr bumps.  Defaults to
                        min(n_sims, 1000) — these Greeks are smooth enough
                        that fewer paths suffice and the cost is lower.
        """
        self.sim          = simulator
        self.po           = payoff_obj
        self.spots        = spots
        self.T            = T
        self.n_sims       = n_sims
        self.n_steps      = n_steps
        self.n_sims_greek = n_sims_greek if n_sims_greek is not None \
                            else min(n_sims, 1000)
        self.seed         = seed

        # ---- draw noise once for the base simulation ----
        # antithetic: draw half, negate for second half
        half = n_sims // 2
        rng  = np.random.default_rng(seed)
        self._Z_S_h = rng.standard_normal((half, n_steps, simulator.n_assets))
        self._Z_v_h = rng.standard_normal((half, n_steps, simulator.n_assets))

        # base paths & price
        base_paths       = self._base_paths(spots)
        self._base_payoffs = payoff_obj.evaluate(base_paths)
        self.base_price    = float(np.mean(self._base_payoffs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_paths(self, spots: Dict[str, float]) -> np.ndarray:
        """Antithetic paths from stored noise (spots can be overridden)."""
        pos = self.sim._simulate_batch(
            self._Z_S_h,  self._Z_v_h,
            spots, self.sim.params, self.T, self.n_steps)
        neg = self.sim._simulate_batch(
            -self._Z_S_h, self._Z_v_h,
            spots, self.sim.params, self.T, self.n_steps)
        return np.concatenate([pos, neg], axis=0)

    def _price_bumped_paths(self, spots: Dict[str, float],
                             params: Dict[str, dict] = None,
                             corr:   np.ndarray      = None) -> float:
        """
        Re-simulate with different params or corr matrix, using a fresh
        RNG seeded identically.  Uses n_sims_greek paths.
        """
        half = self.n_sims_greek // 2
        rng  = np.random.default_rng(self.seed)
        Z_S  = rng.standard_normal((half, self.n_steps, self.sim.n_assets))
        Z_v  = rng.standard_normal((half, self.n_steps, self.sim.n_assets))

        p    = params if params is not None else self.sim.params

        # temporarily swap correlation / Cholesky if needed
        orig_corr = orig_L = None
        if corr is not None:
            orig_corr      = self.sim.corr
            orig_L         = self.sim.L
            self.sim.corr  = corr
            self.sim.L     = np.linalg.cholesky(corr)

        pos = self.sim._simulate_batch( Z_S,  Z_v, spots, p, self.T, self.n_steps)
        neg = self.sim._simulate_batch(-Z_S,  Z_v, spots, p, self.T, self.n_steps)
        paths = np.concatenate([pos, neg], axis=0)

        if orig_corr is not None:
            self.sim.corr = orig_corr
            self.sim.L    = orig_L

        return float(np.mean(self.po.evaluate(paths)))

    # ------------------------------------------------------------------
    # Delta — pathwise spot shift, zero extra simulations
    # ------------------------------------------------------------------

    def delta(self) -> Dict[str, float]:
        """
        ∂BRC/∂S_i  (central difference, pathwise).

        Bumping S₀ → S₀ ± h shifts every log-path by ±log((S₀±h)/S₀).
        The variance paths v(t) are unchanged because v does not depend
        on the spot level in Heston.  No re-simulation needed.
        """
        deltas = {}
        for i, t in enumerate(self.po.tickers):
            S0 = self.spots[t]
            h  = S0 * self.SPOT_BUMP

            sp_up = {**self.spots, t: S0 + h}
            sp_dn = {**self.spots, t: S0 - h}

            # reuse stored noise — only S₀ changes
            paths_up = self._base_paths(sp_up)
            paths_dn = self._base_paths(sp_dn)

            p_up = float(np.mean(self.po.evaluate(paths_up)))
            p_dn = float(np.mean(self.po.evaluate(paths_dn)))

            deltas[t] = (p_up - p_dn) / (2 * h)
        return deltas

    # ------------------------------------------------------------------
    # Gamma — same paths as delta, no extra simulations
    # ------------------------------------------------------------------

    def gamma(self) -> Dict[str, float]:
        """
        ∂²BRC/∂S_i²  (second-order central FD, pathwise).
        Reuses the same bump paths as delta — zero additional simulation.
        """
        gammas = {}
        for t in self.po.tickers:
            S0 = self.spots[t]
            h  = S0 * self.SPOT_BUMP

            sp_up = {**self.spots, t: S0 + h}
            sp_dn = {**self.spots, t: S0 - h}

            paths_up  = self._base_paths(sp_up)
            paths_dn  = self._base_paths(sp_dn)

            p_up = float(np.mean(self.po.evaluate(paths_up)))
            p_dn = float(np.mean(self.po.evaluate(paths_dn)))

            gammas[t] = (p_up - 2 * self.base_price + p_dn) / (h ** 2)
        return gammas

    # ------------------------------------------------------------------
    # Delta + Gamma together — share paths, 2 simulations per ticker
    # ------------------------------------------------------------------

    def delta_and_gamma(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute both delta and gamma in a single pass per ticker.
        Each ticker needs only the up/down bump paths once.
        """
        deltas = {}
        gammas = {}
        for t in self.po.tickers:
            S0 = self.spots[t]
            h  = S0 * self.SPOT_BUMP

            sp_up = {**self.spots, t: S0 + h}
            sp_dn = {**self.spots, t: S0 - h}

            paths_up = self._base_paths(sp_up)
            paths_dn = self._base_paths(sp_dn)

            p_up = float(np.mean(self.po.evaluate(paths_up)))
            p_dn = float(np.mean(self.po.evaluate(paths_dn)))

            deltas[t] = (p_up - p_dn) / (2 * h)
            gammas[t] = (p_up - 2 * self.base_price + p_dn) / (h ** 2)
        return deltas, gammas

    # ------------------------------------------------------------------
    # Vega — bumps ATM vol, needs re-simulation (n_sims_greek paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_atm_vol(params_t: dict) -> float:
        """
        Extract the current ATM vol level from a model params dict.

        Works for any model by checking common parameter names in priority
        order:
          v0    → Heston / stochastic-vol family  (σ = √v0)
          sigma → GBM / constant-vol family       (σ = sigma)
        Falls back to 0.30 if neither key is present.
        """
        if "v0" in params_t:
            return float(max(np.sqrt(params_t["v0"]), 1e-6))
        if "sigma" in params_t:
            return float(max(params_t["sigma"], 1e-6))
        return 0.30

    def vega(self) -> Dict[str, float]:
        """
        ∂BRC/∂σ_i  expressed per 1 vol-point (1%) change in ATM vol.

        Model-agnostic implementation: bumps the ATM vol by ±VOL_BUMP
        (in vol units, i.e. ±1 vol point = ±0.01) and rebuilds the full
        params dict via the simulator's own update_params_for_vol() method.
        This means the bump is applied correctly regardless of whether the
        model stores vol as v0 (Heston), sigma (GBM), or anything else.

        Vega is reported as ∂price / ∂(1 vol-point), so the raw price
        difference is divided by (2 × VOL_BUMP_VOL).
        """
        # VOL_BUMP is expressed in vol units (e.g. 0.01 = 1 vol point)
        VOL_BUMP_VOL = 0.01

        # Fetch the update function from the simulator class (same as backtest)
        update_fn = self.sim.__class__.update_params_for_vol

        vegas = {}
        for t in self.po.tickers:
            current_vol = self._get_atm_vol(self.sim.params[t])

            # Build bumped param dicts for all tickers; only ticker t is bumped
            params_up = {k: dict(**v) for k, v in self.sim.params.items()}
            params_dn = {k: dict(**v) for k, v in self.sim.params.items()}
            params_up[t] = update_fn(self.sim.params[t],
                                     max(current_vol + VOL_BUMP_VOL, 1e-4))
            params_dn[t] = update_fn(self.sim.params[t],
                                     max(current_vol - VOL_BUMP_VOL, 1e-4))

            p_up = self._price_bumped_paths(self.spots, params=params_up)
            p_dn = self._price_bumped_paths(self.spots, params=params_dn)

            # ∂price / ∂(1 vol point)
            vegas[t] = (p_up - p_dn) / (2 * VOL_BUMP_VOL)

        return vegas

    # ------------------------------------------------------------------
    # Rho — re-evaluate payoff on base paths with bumped discount rate
    # ------------------------------------------------------------------

    def rho(self) -> float:
        """
        ∂BRC/∂r  per 1 basis point.

        In the Heston risk-neutral measure the drift of S is r·dt, but the
        paths are already simulated.  The rate change affects:
          (a) the coupon and principal discount factors in evaluate()
          (b) the drift of future paths

        We approximate (a) exactly by patching self.po.r and re-evaluating
        the payoff on the BASE paths (already simulated).  This is fast
        (zero extra simulation) and captures the dominant discounting effect.
        Effect (b) is second-order for short rate bumps and is omitted.
        """
        r0 = self.po.r

        orig_fn             = self.po.rate_curve_fn
        self.po.rate_curve_fn = None          # use flat r for bump

        self.po.r = r0 + self.RATE_BUMP
        p_up = float(np.mean(self.po.evaluate(
            self._base_paths(self.spots))))

        self.po.r = r0 - self.RATE_BUMP
        p_dn = float(np.mean(self.po.evaluate(
            self._base_paths(self.spots))))

        self.po.r           = r0
        self.po.rate_curve_fn = orig_fn

        return (p_up - p_dn) / (2 * self.RATE_BUMP)

    # ------------------------------------------------------------------
    # Correlation sensitivity — needs re-simulation (6 sims total)
    # ------------------------------------------------------------------

    def correlation_sensitivity(self) -> Dict[str, float]:
        """∂BRC/∂ρ_ij  per correlation pair."""
        tickers    = self.po.tickers
        n          = len(tickers)
        base_corr  = self.sim.corr.copy()
        result     = {}

        for i in range(n):
            for j in range(i + 1, n):
                name = f"{tickers[i]}-{tickers[j]}"

                def _bumped_corr(delta):
                    C = base_corr.copy()
                    C[i, j] = np.clip(C[i, j] + delta, -0.99, 0.99)
                    C[j, i] = C[i, j]
                    # nearest PSD
                    ev, evec = np.linalg.eigh(C)
                    C = evec @ np.diag(np.maximum(ev, 1e-8)) @ evec.T
                    d = np.sqrt(np.diag(C))
                    return C / np.outer(d, d)

                p_up = self._price_bumped_paths(
                    self.spots, corr=_bumped_corr(+self.CORR_BUMP))
                p_dn = self._price_bumped_paths(
                    self.spots, corr=_bumped_corr(-self.CORR_BUMP))

                result[name] = (p_up - p_dn) / (2 * self.CORR_BUMP)

        return result

    # ------------------------------------------------------------------

    def all_greeks(self) -> dict:
        """Compute all Greeks efficiently and return as a dictionary."""
        print(f"\n  [Greeks] base={self.base_price:.4f}  "
              f"n_sims={self.n_sims}  n_sims_greek={self.n_sims_greek}")

        d, g = self.delta_and_gamma()   # single pass for both
        v    = self.vega()
        rh   = self.rho()
        cs   = self.correlation_sensitivity()

        print(f"    delta  : { {t: f'{v:.4f}' for t, v in d.items()} }")
        print(f"    gamma  : { {t: f'{v:.6f}' for t, v in g.items()} }")
        print(f"    vega   : { {t: f'{v:.4f}' for t, v in v.items()} }")
        print(f"    rho    : {rh:+.4f} / bp")
        print(f"    corr   : { {k: f'{cv:.4f}' for k, cv in cs.items()} }")

        return dict(delta=d, gamma=g, vega=v, rho=rh,
                    correlation_sensitivity=cs,
                    base_price=self.base_price)


if __name__ == "__main__":
    # Smoke test
    from heston import HestonSimulator
    TICKERS = ["NFLX", "SPOT", "DIS"]

    params = {t: dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)
              for t in TICKERS}
    corr   = np.array([[1.0, 0.357, 0.352],
                        [0.357, 1.0, 0.489],
                        [0.352, 0.489, 1.0]])
    spots  = {"NFLX": 93.55, "SPOT": 565.41, "DIS": 97.88}  # termsheet
    fixing = spots.copy()

    sim = HestonSimulator(params, corr, TICKERS)
    po  = BRCPayoff(TICKERS, fixing,
                    valuation_date="2025-04-02",
                    maturity_date="2026-10-09")  # termsheet maturity

    paths   = sim.simulate(spots, T=1.5, n_sims=500, n_steps=100,
                           method="antithetic")
    payoffs = po.evaluate(paths)
    print(po.result_summary(payoffs))
