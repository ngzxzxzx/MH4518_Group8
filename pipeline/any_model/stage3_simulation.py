"""
Stage III  —  Variance Reduction Comparison
============================================
Model-agnostic.  Accepts any object conforming to BaseSimulator.

Inputs  (injected by main.py)
  - simulator  : any BaseSimulator subclass instance
  - spots      : {ticker: float}
  - payoff_fn  : callable(paths: np.ndarray) -> np.ndarray (1-D payoffs)
  - T, n_steps : simulation parameters

Output
  - vr_df : pd.DataFrame  (method, n_sims, price, std_error, ci_lower,
                            ci_upper, elapsed_s, eff_n_sims)

Five variance reduction methods are supported.  The first three
(crude, antithetic, quasi) call simulator.simulate() and return plain
path arrays.  The last two return special objects handled here:

  stratified   simulator.simulate(..., method="stratified")
               Returns a plain paths array.  Z_S is drawn with stratified
               uniform samples on the first principal component so the
               empirical distribution of the common market factor exactly
               covers [0, 1] with no bunching.

  importance   simulator.simulate(..., method="importance")
               Returns (paths, weights).  Paths are drawn under a tilted
               measure that oversamples the barrier-breach region.  This
               module applies the likelihood-ratio weights when computing
               the price estimate and its standard error.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish's effective sample size for importance-sampling weights.

        ESS = (Σ w_i)² / Σ w_i²

    With normalised weights (mean = 1) this simplifies to:

        ESS = n² / Σ w_i²

    Returns the ESS as a fraction of the nominal n_sims so it can be
    compared across different simulation counts.
    """
    n = len(weights)
    return float((weights.sum() ** 2) / (n * (weights ** 2).sum()))


def _weighted_price_and_stderr(payoffs: np.ndarray,
                                weights: np.ndarray):
    """
    Compute the IS price estimate and its standard error.

    IS estimator:   mu_hat = (1/n) * sum(w_i * h_i)   (self-normalised IS)
    IS std error:   std(w_i * h_i) / sqrt(n)

    where w_i are normalised likelihood-ratio weights (mean = 1) and
    h_i are the (unweighted) payoffs evaluated on the IS paths.

    The self-normalisation step introduces negligible bias for n > 100 and
    keeps the estimator stable even when a handful of paths carry very
    large weights.
    """
    wh     = weights * payoffs
    price  = float(wh.mean())
    stderr = float(wh.std() / np.sqrt(len(payoffs)))
    return price, stderr


# ---------------------------------------------------------------------------
# Variance reduction comparison experiment
# ---------------------------------------------------------------------------

def variance_reduction_comparison(simulator,
                                  spots:      Dict[str, float],
                                  payoff_fn,
                                  T:          float,
                                  n_steps:    int,
                                  sim_counts: list = None,
                                  seed:       int  = 42) -> pd.DataFrame:
    """
    Run a controlled experiment across five VR methods at multiple
    simulation counts.

    Parameters
    ----------
    simulator  : any BaseSimulator instance (Heston, GBM, ...)
    spots      : current spot prices per ticker
    payoff_fn  : discounted payoff function, callable(paths) -> 1-D array
    T          : remaining time to maturity (years)
    n_steps    : number of simulation time steps
    sim_counts : list of N values to test (default [500,1000,2000,5000,10000])
    seed       : RNG seed for reproducibility

    Returns
    -------
    pd.DataFrame  columns: method, n_sims, price, std_error,
                           ci_lower, ci_upper, elapsed_s, eff_n_sims

      eff_n_sims : effective sample size
                   = n_sims for unweighted methods (crude, antithetic,
                     quasi, stratified)
                   = Kish ESS for importance sampling
    """
    if sim_counts is None:
        sim_counts = [500, 1000, 2000, 5000, 10000]

    methods = ["crude", "antithetic", "quasi", "stratified", "importance"]
    records = []

    model_name = getattr(simulator, "model_name",
                         lambda: type(simulator).__name__)()

    print("\n" + "="*72)
    print(f"VARIANCE REDUCTION COMPARISON  [{model_name}]")
    print("="*72)
    print(f"{'Method':<14} {'N':>7} {'Price':>9} {'Std Err':>9} "
          f"{'CI lower':>10} {'CI upper':>10} {'ESS':>7} {'Time(s)':>8}")
    print("-"*72)

    for method in methods:
        for n in sim_counts:
            t0     = time.perf_counter()
            result = simulator.simulate(spots, T, n, n_steps,
                                        method=method, seed=seed)
            elapsed = time.perf_counter() - t0

            # Unpack: IS returns (paths, weights); everything else returns paths
            if isinstance(result, tuple):
                paths, weights = result
            else:
                paths, weights = result, None

            payoffs = payoff_fn(paths)

            if weights is not None:
                # importance sampling: apply likelihood-ratio weights
                price, stderr = _weighted_price_and_stderr(payoffs, weights)
                ess = _effective_sample_size(weights) * n
            else:
                price  = float(np.mean(payoffs))
                stderr = float(np.std(payoffs) / np.sqrt(len(payoffs)))
                ess    = float(n)   # full effective sample for unweighted methods

            ci_lo = price - 1.96 * stderr
            ci_hi = price + 1.96 * stderr

            records.append(dict(method=method, n_sims=n, price=price,
                                std_error=stderr, ci_lower=ci_lo,
                                ci_upper=ci_hi, elapsed_s=elapsed,
                                eff_n_sims=ess))

            print(f"{method:<14} {n:>7,} {price:>9.4f} {stderr:>9.4f} "
                  f"{ci_lo:>10.4f} {ci_hi:>10.4f} "
                  f"{ess:>7.0f} {elapsed:>8.3f}")

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Summary: relative std-error reduction vs crude MC
    # ------------------------------------------------------------------
    print("\nStd-error relative to crude MC  (lower is better):")
    crude_se = df[df["method"] == "crude"].set_index("n_sims")["std_error"]

    for method in ["antithetic", "quasi", "stratified", "importance"]:
        sub   = df[df["method"] == method].set_index("n_sims")
        ratio = (sub["std_error"] / crude_se).mean()
        eff   = 1 / ratio**2
        print(f"  {method:<14}: avg std-error ratio = {ratio:.3f}  "
              f"(approx {eff:.1f}x equivalent crude paths)")

    # IS-specific ESS diagnostics
    is_rows = df[df["method"] == "importance"]
    if not is_rows.empty:
        print("\nImportance sampling diagnostics:")
        for _, row in is_rows.iterrows():
            ess_frac = row["eff_n_sims"] / row["n_sims"]
            flag = "[good]" if ess_frac >= 0.3 else "[weight collapse -- reduce mu]"
            print(f"  N={int(row['n_sims']):>6,}  "
                  f"ESS={row['eff_n_sims']:>7.0f}  "
                  f"ESS/N={ess_frac:.2%}  {flag}")

    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from stage1_data import TICKERS, RISK_FREE_RATE
    from heston import HestonSimulator

    dummy_params = {t: dict(v0=0.04, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.6)
                    for t in TICKERS}
    dummy_corr   = np.array([[1.0, 0.357, 0.352],
                              [0.357, 1.0, 0.489],
                              [0.352, 0.489, 1.0]])
    dummy_spots  = {"NFLX": 850.0, "SPOT": 350.0, "DIS": 110.0}

    sim = HestonSimulator(dummy_params, dummy_corr, TICKERS)

    def dummy_payoff(paths):
        return paths[:, -1, :].mean(axis=1)

    vr_df = variance_reduction_comparison(
        sim, dummy_spots, dummy_payoff,
        T=1.5, n_steps=50, sim_counts=[200, 500])
    print(vr_df)
