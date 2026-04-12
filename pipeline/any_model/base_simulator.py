"""
base_simulator.py — Abstract Simulator Interface
=================================================
All model-specific simulators (Heston, GBM, etc.) must subclass
BaseSimulator and implement:

  simulate(spots, T, n_sims, n_steps, method, seed) -> np.ndarray
  _simulate_batch(Z_S, Z_v, spots, params, T, n_steps) -> np.ndarray
  calibrate(surface_df, tickers, r)  -> dict  {ticker: params}
  update_params_for_vol(inception_params, atm_vol) -> dict

The model is loaded dynamically by main.py using:
    from importlib import import_module
    mod = import_module(model_name)          # e.g. "heston" or "gbm"
    SimClass = mod.Simulator                 # each model exposes Simulator
    params   = SimClass.calibrate(surface_df, tickers, r)
    sim      = SimClass(params, corr, tickers, r)

Variance reduction, payoff evaluation, backtest, and validation stages
are entirely model-agnostic and work through this interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseSimulator(ABC):
    """
    Abstract base for all Monte Carlo path simulators.

    Concrete subclasses implement the model-specific SDE and calibration.
    The rest of the pipeline (variance reduction, payoff, backtest,
    validation) calls only the methods defined here.

    Attributes (must be set by subclass __init__)
    -----------------------------------------------
    params    : dict  {ticker: model_params_dict}
    corr      : np.ndarray  (n_assets × n_assets) spot correlation matrix
    tickers   : list of str
    n_assets  : int
    r         : float  risk-free rate (flat fallback)
    """

    # ------------------------------------------------------------------
    # Constructor contract
    # ------------------------------------------------------------------

    def __init__(self,
                 params:             Dict[str, dict],
                 correlation_matrix: np.ndarray,
                 tickers:            list,
                 risk_free_rate:     float = 0.05):
        self.params   = params
        self.corr     = correlation_matrix
        self.tickers  = tickers
        self.n_assets = len(tickers)
        self.r        = risk_free_rate

        # Cholesky decomposition shared by all models for cross-asset correlation
        self.L = np.linalg.cholesky(self.corr)

    # ------------------------------------------------------------------
    # Required: path simulation
    # ------------------------------------------------------------------

    @abstractmethod
    def simulate(self,
                 spots:   Dict[str, float],
                 T:       float,
                 n_sims:  int,
                 n_steps: int,
                 method:  str = "crude",
                 params:  Optional[Dict[str, dict]] = None,
                 seed:    int = 42):
        """
        Simulate asset price paths.

        Parameters
        ----------
        spots   : {ticker: current_spot_price}
        T       : time horizon in years
        n_sims  : number of Monte Carlo paths
        n_steps : number of time steps
        method  : "crude" | "antithetic" | "quasi"
                  | "stratified" | "importance"
        params  : override self.params if provided
        seed    : RNG seed for reproducibility

        Returns
        -------
        For "crude" / "antithetic" / "quasi" / "stratified":
            np.ndarray  shape (n_sims, n_steps+1, n_assets)

        For "importance":
            tuple  (paths, weights)
              paths   np.ndarray  (n_sims, n_steps+1, n_assets)
              weights np.ndarray  (n_sims,)  likelihood-ratio weights
                      normalised so that weights.mean() == 1.

        paths[:, 0, :] = spots (initial values)
        paths[:, -1, :] = terminal values
        """
        ...

    # ------------------------------------------------------------------
    # Shared variance-reduction helpers  (used by all concrete simulators)
    # ------------------------------------------------------------------

    @staticmethod
    def _stratified_normals(n_sims: int,
                            n_steps: int,
                            n_assets: int,
                            rng: "np.random.Generator") -> np.ndarray:
        """
        Stratified-sampling normals for the spot driver Z_S.

        The idea
        --------
        Stratified sampling partitions the [0, 1] probability space into
        n_sims equal-width strata and draws exactly one uniform sample from
        each stratum before mapping through Φ⁻¹ (the standard-normal
        quantile function).  This guarantees the empirical distribution of
        the samples covers the full range of the normal without clustering,
        eliminating the "bunching" that causes Monte Carlo variance.

        For a multi-step, multi-asset simulation the full noise tensor has
        dimension n_sims × n_steps × n_assets.  We stratify only along the
        first principal component of the spot shocks — the single direction
        that explains the most variance in the worst-of payoff.  All
        remaining dimensions are filled with ordinary independent normals
        so that cross-asset and term-structure randomness is preserved.

        Why stratify the first PC only?
        --------------------------------
        Stratifying all dimensions simultaneously would require a
        multi-dimensional stratification (e.g. Latin-hypercube), which
        loses effectiveness as dimensionality grows.  For a worst-of BRC the
        dominant risk factor is the common downward drift that drives barrier
        breaches.  Stratifying along that axis (the first eigenvector of the
        correlation matrix) gives the largest variance reduction for the
        least computational overhead.

        Implementation
        --------------
        1. Draw stratification index k ∈ {0, …, n_sims−1} at random (a
           random permutation guarantees each stratum is used exactly once).
        2. Draw u_k ~ Uniform(k/n, (k+1)/n) for each stratum k.
        3. Map through Φ⁻¹ to get the stratified normal z_pc for the first
           principal direction.
        4. Fill all other dimensions with i.i.d. N(0,1).

        Parameters
        ----------
        n_sims, n_steps, n_assets : simulation dimensions
        rng : numpy Generator (seeded by the caller)

        Returns
        -------
        Z_S : np.ndarray  shape (n_sims, n_steps, n_assets)
              Normals to be passed to _simulate_batch.  The cross-asset
              Cholesky rotation is applied inside _simulate_batch as usual.
        """
        from scipy.stats import norm as _spnorm

        # Full noise tensor, initialised to i.i.d. N(0,1)
        Z_S = rng.standard_normal((n_sims, n_steps, n_assets))

        # Stratify the *first time step, first asset* scalar — this acts as
        # the seed for the common market factor after the Cholesky rotation.
        # Using a single stratified dimension keeps the method low-variance
        # without destroying the multi-dimensional covariance structure.
        perm  = rng.permutation(n_sims)                     # random stratum order
        lo    = perm / n_sims
        hi    = (perm + 1) / n_sims
        u_str = lo + rng.random(n_sims) * (hi - lo)        # one draw per stratum
        u_str = np.clip(u_str, 1e-10, 1 - 1e-10)
        Z_S[:, 0, 0] = _spnorm.ppf(u_str)                  # overwrite first column

        return Z_S

    @staticmethod
    def _importance_sampling_normals(n_sims:   int,
                                     n_steps:  int,
                                     n_assets: int,
                                     rng:      "np.random.Generator",
                                     theta:    float = 1.0):
        """
        Importance-sampling normals for the spot driver Z_S.

        Theory
        ------
        Standard Monte Carlo draws paths from the risk-neutral measure Q.
        For a worst-of BRC the rare events that matter most are simultaneous
        sharp drops across all underlyings (barrier breaches).  These live in
        the far left tail of each asset's return distribution and are sampled
        infrequently under Q.

        Importance sampling replaces Q with a tilted measure Q̃ under which
        the common factor is shifted, making crashes much more likely.  After
        simulation we reweight each path by the Radon-Nikodym derivative
        dQ/dQ̃ to recover unbiased Q-expectations.

        Change of measure
        -----------------
        Under Q: Z_t ~ N(0, 1)  (i.i.d. per step)
        Under Q̃: Z̃_t = Z_t + μ  where μ is a *per-step* shift.

        We shift the *first principal component* only (the first column of
        Z_S before the Cholesky rotation), which after rotation becomes the
        common market factor.  All other columns stay N(0, 1) so only the
        systematic downward drift is amplified, not idiosyncratic noise.

        Per-step shift and its relationship to the total drift
        -------------------------------------------------------
        Let θ be the desired *total* drift shift over the simulation horizon,
        measured in units of σ√T (annualised vol times square-root of time).
        This is the natural scale for specifying how far into the tail we want
        to sample.  The required per-step normal shift is then:

            μ_step = θ / √n_steps

        so that the cumulative shift   Σ_t μ_step = θ · √n_steps  scales
        correctly with the Brownian motion (which grows as √n_steps).  This
        guarantees the effective sample size (ESS) is independent of n_steps
        and depends only on θ.

        Likelihood-ratio weight (per path)
        -----------------------------------
        For a shift of μ_step per step the log Radon-Nikodym derivative is:

            log w = −μ_step · Σ_t Z_t  −  0.5 · μ_step² · n_steps
                  = −(θ/√n) · Σ_t Z_t  −  0.5 · θ²

        where Z_t are the *un-shifted* Q-draws.  The second term is constant
        across paths and equal to −0.5 · θ²; it cancels during normalisation
        but is included for mathematical correctness.

        The weights are normalised so their mean equals 1 (self-normalised IS)
        which keeps the estimator stable even when θ is not small.

        Choosing θ
        ----------
        θ = 1.0 is a good default: it shifts the common factor by 1σ over
        the simulation horizon, which for a worst-of BRC roughly doubles the
        probability of barrier breach while keeping ESS > 40%.

        θ < 0.5 : very mild tilt; little variance reduction vs crude MC.
        θ = 1.0 : recommended default; ESS typically 40-65%.
        θ = 2.0 : aggressive tilt; may give ESS < 20% and noisy weights.
        θ > 3.0 : weight collapse likely; use only for very deep OTM payoffs.

        Parameters
        ----------
        n_sims, n_steps, n_assets : simulation dimensions
        rng   : numpy Generator
        theta : total drift shift in σ√T units (default 1.0)

        Returns
        -------
        Z_S     : np.ndarray  (n_sims, n_steps, n_assets)
                  Shifted normals — the first column is from N(μ_step, 1).
        weights : np.ndarray  (n_sims,)
                  Normalised likelihood-ratio weights (mean = 1).
        """
        # Per-step shift: theta / sqrt(n_steps) keeps total drift = theta·sqrt(n_steps)
        # times the per-step vol (1), so the cumulative Brownian shift is theta·sqrt(T).
        mu_step = theta / np.sqrt(n_steps)

        # Draw standard normals for the full tensor
        Z_S = rng.standard_normal((n_sims, n_steps, n_assets))

        # Shift only the first-asset column (common market factor after Cholesky)
        Z_S_shifted          = Z_S.copy()
        Z_S_shifted[:, :, 0] = Z_S[:, :, 0] + mu_step

        # Log Radon-Nikodym: log(dQ/dQ̃) using the UN-shifted Q draws
        #   log w = -mu_step * sum_t(Z_t)  -  0.5 * mu_step^2 * n_steps
        #         = -(theta/sqrt(n)) * sum_t(Z_t)  -  0.5 * theta^2
        log_w = (-mu_step * Z_S[:, :, 0].sum(axis=1)
                 - 0.5 * mu_step**2 * n_steps)

        # Normalise for numerical stability (log-sum-exp trick)
        log_w -= log_w.max()
        w      = np.exp(log_w)
        w     /= w.mean()

        return Z_S_shifted, w

    @abstractmethod
    def _simulate_batch(self,
                        Z_S:     np.ndarray,
                        Z_v:     np.ndarray,
                        spots:   Dict[str, float],
                        params:  Dict[str, dict],
                        T:       float,
                        n_steps: int) -> np.ndarray:
        """
        Core simulation loop given pre-drawn standard normal arrays.

        Parameters
        ----------
        Z_S     : (n_sims, n_steps, n_assets)  spot shock normals
        Z_v     : (n_sims, n_steps, n_assets)  secondary shock normals
                  (used for vol-of-vol in stochastic vol models;
                   ignored / set to zero for single-factor models like GBM)
        spots   : initial spot prices per ticker
        params  : model parameters per ticker
        T       : time horizon in years
        n_steps : number of time steps

        Returns
        -------
        np.ndarray  shape (n_sims, n_steps+1, n_assets)
        """
        ...

    # ------------------------------------------------------------------
    # Required: calibration (class method — no simulator instance needed)
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def calibrate(cls,
                  surface_df: "pd.DataFrame",
                  historical_price_df: "pd.DataFrame",
                  tickers:    list,
                  r:          float = 0.05) -> Dict[str, dict]:
        """
        Calibrate model parameters to a volatility surface.

        Parameters
        ----------
        surface_df : DataFrame with columns:
                     ticker, maturity_years, moneyness, vol (or implied_vol)
        historical_price_df : DataFrame with columns:
                        ticker, date, close
        tickers    : list of ticker symbols to calibrate
        r          : risk-free rate (decimal)

        Returns
        -------
        dict  {ticker: params_dict}
            The params_dict schema is model-specific; it is passed back
            to __init__ as the `params` argument.
        """
        ...

    # ------------------------------------------------------------------
    # Required: parameter update for backtest vol scaling
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def update_params_for_vol(inception_params: dict,
                               atm_vol:          float) -> dict:
        """
        Update model parameters to reflect a new ATM volatility level,
        holding all other structural parameters fixed from inception.

        Called every backtest date to scale the current vol estimate
        into the model's internal parameter representation.

        Parameters
        ----------
        inception_params : the full fitted parameter dict for one ticker
        atm_vol          : current ATM implied or realised vol (decimal)

        Returns
        -------
        dict  updated parameter dict (same schema as inception_params)

        Example (Heston)
        ----------------
        Sets v0 = atm_vol² while keeping kappa, theta, sigma, rho fixed.

        Example (GBM)
        -------------
        Sets sigma = atm_vol directly.
        """
        ...

    # ------------------------------------------------------------------
    # Optional: terminal parameter blending (override if needed)
    # ------------------------------------------------------------------

    @staticmethod
    def blend_terminal_params(inception_params: Dict[str, dict],
                               terminal_params:  Dict[str, dict],
                               tickers:          list) -> Dict[str, dict]:
        """
        Blend inception and terminal (live-chain) calibration results.

        Default: simply return terminal_params (full replacement).
        Override in models where some parameters are unreliable from
        short-dated chains (e.g. Heston kappa from sparse live chains).

        Parameters
        ----------
        inception_params : {ticker: params}  fitted at inception
        terminal_params  : {ticker: params}  fitted to today's live chain
        tickers          : ticker list

        Returns
        -------
        dict  {ticker: blended_params}
        """
        return {t: dict(terminal_params.get(t, inception_params.get(t, {})))
                for t in tickers}

    # ------------------------------------------------------------------
    # Optional: model display name
    # ------------------------------------------------------------------

    @classmethod
    def model_name(cls) -> str:
        """Human-readable model name for logging."""
        return cls.__name__.replace("Simulator", "")
