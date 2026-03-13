"""
main.py  —  BRC Model Pipeline Orchestrator
============================================
Model-agnostic.  Selects the simulation model via --model <name>.
The named module must exist in the same directory and expose:

    Simulator  — a class conforming to BaseSimulator

Built-in models:
  heston  (default) — Heston stochastic volatility
  gbm               — Geometric Brownian Motion (Black-Scholes)

Adding a new model
------------------
1. Create <model_name>.py in this directory.
2. Subclass BaseSimulator; implement simulate(), _simulate_batch(),
   calibrate(), and update_params_for_vol().
3. Add  Simulator = YourSimulatorClass  at module level.
4. Run:  python main.py --model <model_name>

Usage
-----
  python main.py                              # Heston, live data
  python main.py --model gbm                 # GBM
  python main.py --offline --no-greeks       # fast offline test
  python main.py --model heston --flat-rate --n-sims 300
"""

import argparse
import importlib
import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from stage1_data      import (build_data_store, load_surface_from_path,
                               load_historical_brc_prices,
                               TICKERS, RISK_FREE_RATE)
from stage3_simulation import variance_reduction_comparison
from stage4_pricing    import BRCPayoff, BRCGreeks, control_variate_price
from stage6_backtest   import BRCBacktester
from stage7_validation import (plot_backtest_price, plot_attribution,
                                plot_barrier_proximity, plot_vr_convergence,
                                plot_greeks, run_consistency_checks,
                                write_validation_report)


# ---------------------------------------------------------------------------
# BRC contract terms  (from UBS Final Termsheet, ISIN CH1431536452)
# ---------------------------------------------------------------------------

INITIAL_FIXING_PRICES = {
    "NFLX": 93.55,
    "SPOT": 565.41,
    "DIS":  97.88,
}

BARRIER_PRICES = {t: INITIAL_FIXING_PRICES[t] * 0.50 for t in TICKERS}

COUPON_PAYMENT_DATES = [
    "2025-07-10", "2025-10-09", "2026-01-12",
    "2026-04-13", "2026-07-10", "2026-10-09",
]

BRC_TERMS = dict(
    tickers        = TICKERS,
    valuation_date = "2025-04-02",
    maturity_date  = "2026-10-09",
    barrier_level  = 0.50,
    coupon_rate    = 0.1375,
    principal      = 100.0,
)

N_SIMS_MAIN    = 5000
N_SIMS_VR      = 10000
N_STEPS_PER_YR = 252


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_simulator_class(model_name: str):
    """
    Dynamically import <model_name>.py from the same directory as main.py
    and return its Simulator class.

    The module must expose:
        Simulator  —  a class conforming to BaseSimulator
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        mod = importlib.import_module(model_name)
    except ModuleNotFoundError:
        raise SystemExit(
            f"\n  ERROR: No model file '{model_name}.py' found in {module_dir}.\n"
            f"  Available models: heston, gbm (or any .py in the same directory\n"
            f"  that exposes a 'Simulator' class conforming to BaseSimulator).\n"
        )

    if not hasattr(mod, "Simulator"):
        raise SystemExit(
            f"\n  ERROR: '{model_name}.py' must expose a 'Simulator' class.\n"
            f"  Add:  Simulator = YourSimulatorClass  at module level.\n"
        )

    SimClass = mod.Simulator
    print(f"\n  Model loaded : {SimClass.model_name()}  (from {model_name}.py)")
    return SimClass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_name:          str  = "heston",
         use_live_data:        bool = True,
         compute_greeks:       bool = True,
         surface_type:         str  = "heston",
         surface_csv:          str  = None,
         data_root:            str  = None,
         backtest_frequency:   str  = "W",
         n_sims_main:          int  = N_SIMS_MAIN,
         use_live_rates:       bool = True):

    os.makedirs(f"./results/plots", exist_ok=True)

    # ── Load simulator class ─────────────────────────────────────────────
    SimClass = load_simulator_class(model_name)

    # ── STAGES I + II  —  Data & Calibration ────────────────────────────
    if surface_csv:
        from stage1_data import (DataStore, load_surface_from_path,
                                  load_log_returns, fetch_realised_prices,
                                  fetch_live_surface, estimate_correlation,
                                  compute_rolling_vol, resolve_paths)
        print(f"\n  Surface CSV (explicit): {surface_csv}")
        surf_df, inferred_type = load_surface_from_path(surface_csv)

        ds = DataStore()
        ds.surface_type     = inferred_type
        ds.inception_surface = surf_df
        paths_ = resolve_paths(data_root)

        try:
            ds.log_returns        = load_log_returns(paths_["log_returns"])
            ds.correlation_matrix = estimate_correlation(ds.log_returns)
        except Exception:
            print("  Using hardcoded correlation.")
            ds.correlation_matrix = np.array([
                [1.000, 0.357, 0.352],
                [0.357, 1.000, 0.489],
                [0.352, 0.489, 1.000],
            ])

        if use_live_data:
            try:
                ds.realised_spots        = fetch_realised_prices()
                ds.initial_fixing_prices = INITIAL_FIXING_PRICES
                ds.rolling_vol           = compute_rolling_vol(ds.realised_spots)
            except Exception as e:
                print(f"  WARNING: realised prices unavailable ({e})")
                ds.initial_fixing_prices = INITIAL_FIXING_PRICES
            try:
                ds.live_surface = fetch_live_surface()
            except Exception as e:
                print(f"  WARNING: live surface unavailable ({e})")
                ds.live_surface = surf_df.copy()
        else:
            ds.initial_fixing_prices = INITIAL_FIXING_PRICES
            ds.live_surface          = surf_df.copy()

        print(f"\n  Calibrating {SimClass.model_name()} on explicit surface ...")
        ds.model_params          = SimClass.calibrate(surf_df, TICKERS)
        ds.terminal_model_params = SimClass.calibrate(ds.live_surface, TICKERS)

        import os as _os
        hist_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "../../historical_BRC_prices.txt")
        ds.historical_brc_prices = load_historical_brc_prices(hist_path)

    else:
        print(f"\n  Surface type : {surface_type.upper()} (synthetic fallback)")
        ds = build_data_store(
            simulator_class = SimClass,
            surface_type    = surface_type,
            data_root       = data_root,
            use_live_data   = use_live_data,
        )

    # ── Initial fixing prices ────────────────────────────────────────────
    if ds.initial_fixing_prices:
        fixing = ds.initial_fixing_prices
    else:
        fixing = (
            {t: float(ds.realised_spots[t].iloc[0]) for t in TICKERS}
            if not ds.realised_spots.empty
            else INITIAL_FIXING_PRICES
        )
    print(f"\n  Initial fixing prices : {fixing}")

    # ── STAGE III  —  Simulator ──────────────────────────────────────────
    from stage1_data import get_rate_curve_fn, get_zero_rate

    if use_live_rates and not ds.treasury_curve.empty:
        print(f"\n  Rate mode : term-structure (treasury_curve.csv)")
        inception_r_fn = get_rate_curve_fn(ds.treasury_curve, BRC_TERMS["valuation_date"])
    else:
        print(f"\n  Rate mode : flat {RISK_FREE_RATE:.1%}")
        inception_r_fn = None

    T_full_yr  = (pd.Timestamp(BRC_TERMS["maturity_date"])
                  - pd.Timestamp(BRC_TERMS["valuation_date"])).days / 365.0
    inception_r = inception_r_fn(T_full_yr) if inception_r_fn else RISK_FREE_RATE

    # Instantiate the active simulator from calibrated params
    simulator = SimClass(
        params             = ds.model_params,
        correlation_matrix = ds.correlation_matrix,
        tickers            = TICKERS,
        risk_free_rate     = inception_r,
    )

    # ── Variance Reduction Comparison ───────────────────────────────────
    print("\n" + "="*60)
    print("STAGE III — VARIANCE REDUCTION COMPARISON")
    print("="*60)

    po_full = BRCPayoff(
        **BRC_TERMS,
        initial_fixing_prices = fixing,
        coupon_payment_dates  = COUPON_PAYMENT_DATES,
        risk_free_rate        = inception_r,
        rate_curve_fn         = inception_r_fn,
    )

    T_full      = po_full.T
    payoff_fn   = lambda paths: po_full.evaluate(paths)
    inception_spots = fixing.copy()

    vr_df = variance_reduction_comparison(
        simulator  = simulator,
        spots      = inception_spots,
        payoff_fn  = payoff_fn,
        T          = T_full,
        n_steps    = int(N_STEPS_PER_YR * T_full),
        sim_counts = [500, 1000, 2000, 5000, N_SIMS_VR],
        seed       = 42,
    )

    # control variate (uses simulator directly — model-agnostic)
    print("\n  Running control variate (CV) method ...")
    # ATM vol estimate: use v0 for Heston-family, sigma directly for GBM
    def _get_atm_vol(params_t: dict) -> float:
        if "v0" in params_t:
            return float(np.sqrt(params_t["v0"]))
        return float(params_t.get("sigma", 0.30))

    for n in [500, 1000, 2000, 5000, N_SIMS_VR]:
        t0 = time.perf_counter()
        atm_vols = {t: _get_atm_vol(ds.model_params[t]) for t in TICKERS}
        cv_price, cv_se = control_variate_price(
            simulator      = simulator,
            payoff_fn      = payoff_fn,
            spots          = inception_spots,
            initial_fixing = fixing,
            tickers        = TICKERS,
            T              = T_full,
            n_sims         = n,
            n_steps        = int(N_STEPS_PER_YR * T_full),
            r              = RISK_FREE_RATE,
            vols           = atm_vols,
            barrier        = BRC_TERMS["barrier_level"],
            seed           = 42,
        )
        elapsed = time.perf_counter() - t0
        vr_df = pd.concat([vr_df, pd.DataFrame([dict(
            method    = "control_variate",
            n_sims    = n,
            price     = cv_price,
            std_error = cv_se,
            ci_lower  = cv_price - 1.96 * cv_se,
            ci_upper  = cv_price + 1.96 * cv_se,
            elapsed_s = elapsed,
        )])], ignore_index=True)
        print(f"  CV  N={n:>6,}  price={cv_price:.4f}  se={cv_se:.4f}  t={elapsed:.2f}s")

    plot_vr_convergence(vr_df)

    # ── STAGE IV+V  —  Inception Pricing & Greeks ──────────────────────
    print("\n" + "="*60)
    print("STAGE IV — INCEPTION BRC PRICING")
    print("="*60)

    paths_inception   = simulator.simulate(
        inception_spots, T_full,
        n_sims  = n_sims_main,
        n_steps = int(N_STEPS_PER_YR * T_full),
        method  = "antithetic",
        seed    = 42,
    )
    payoffs_inception = po_full.evaluate(paths_inception)
    result_inception  = po_full.result_summary(payoffs_inception)

    print(f"\n  Inception BRC Price  : ${result_inception['price']:.4f}")
    print(f"  Std Error            : ±${result_inception['std_error']:.4f}")
    print(f"  95% CI               : [{result_inception['ci_95_lower']:.4f}, "
          f"{result_inception['ci_95_upper']:.4f}]")
    print(f"  Barrier probability  : {result_inception['barrier_touch_probability']:.2%}")

    if compute_greeks:
        greeks_obj = BRCGreeks(
            simulator      = simulator,
            payoff_obj     = po_full,
            spots          = inception_spots,
            T              = T_full,
            n_sims         = n_sims_main,
            n_steps        = int(N_STEPS_PER_YR * T_full),
            seed           = 42,
        )
        inception_greeks = greeks_obj.all_greeks()
    else:
        inception_greeks = None
        print("\n  Greeks skipped (--no-greeks).")

    # ── STAGE VI  —  Backtest ───────────────────────────────────────────
    backtester = BRCBacktester(
        data_store       = ds,
        simulator        = simulator,
        payoff_template  = po_full,
        n_sims           = n_sims_main,
        n_steps          = N_STEPS_PER_YR,
        frequency        = backtest_frequency,
        seed             = 42,
        compute_greeks   = compute_greeks,
        use_live_rates   = use_live_rates,
    )
    backtest_result = backtester.run()

    # ── Terminal live-chain price ────────────────────────────────────────
    terminal_price   = None
    blended_terminal = None

    if not ds.live_surface.empty and ds.terminal_model_params:
        print("\n  Computing terminal (live-chain) BRC price ...")

        # Blend terminal params using model-specific logic (override if provided)
        blend_fn = getattr(SimClass, "blend_terminal_params",
                           None)
        if blend_fn is not None:
            blended_terminal = blend_fn(ds.model_params,
                                         ds.terminal_model_params,
                                         TICKERS)
        else:
            blended_terminal = ds.terminal_model_params

        curr_spots = (
            {t: float(ds.realised_spots[t].iloc[-1]) for t in TICKERS}
            if not ds.realised_spots.empty
            else inception_spots
        )

        today_ts = pd.Timestamp(ds.today)
        T_rem    = max((po_full.maturity_date - today_ts).days / 365.0, 1/252)
        n_rem    = max(int(N_STEPS_PER_YR * T_rem), 10)

        if use_live_rates and not ds.treasury_curve.empty:
            today_r_fn = get_rate_curve_fn(ds.treasury_curve, today_ts)
        else:
            today_r_fn = None
        today_r = today_r_fn(T_rem) if today_r_fn else RISK_FREE_RATE

        term_sim = SimClass(
            params             = blended_terminal,
            correlation_matrix = ds.correlation_matrix,
            tickers            = TICKERS,
            risk_free_rate     = today_r,
        )

        po_term = BRCPayoff(
            tickers               = TICKERS,
            initial_fixing_prices = fixing,
            valuation_date        = ds.today,
            maturity_date         = BRC_TERMS["maturity_date"],
            barrier_level         = BRC_TERMS["barrier_level"],
            coupon_rate           = BRC_TERMS["coupon_rate"],
            principal             = BRC_TERMS["principal"],
            risk_free_rate        = today_r,
            rate_curve_fn         = today_r_fn,
            coupon_payment_dates  = COUPON_PAYMENT_DATES,
        )

        paths_term    = term_sim.simulate(curr_spots, T_rem, n_sims_main,
                                           n_rem, method="antithetic", seed=42)
        already_br    = backtester._barrier_breached_before(today_ts)
        payoffs_term  = po_term.evaluate(paths_term, already_breached=already_br)
        terminal_price = float(np.mean(payoffs_term))
        print(f"  Terminal (live-chain) BRC price: ${terminal_price:.4f}")

    # ── STAGE VII  —  Plots and Validation ──────────────────────────────
    print("\n" + "="*60)
    print("STAGE VII — VALIDATION AND VISUALISATION")
    print("="*60)

    plot_backtest_price(backtest_result,
                        realised_spots        = ds.realised_spots,
                        tickers               = TICKERS,
                        fixing                = fixing,
                        barrier_level         = BRC_TERMS["barrier_level"],
                        historical_brc_prices = ds.historical_brc_prices
                                                if not ds.historical_brc_prices.empty
                                                else None,
                        model_name            = SimClass.model_name())

    plot_attribution(backtest_result)
    plot_barrier_proximity(backtest_result, TICKERS)
    plot_greeks(backtest_result, TICKERS)

    checks = run_consistency_checks(
        backtest_result          = backtest_result,
        terminal_price_from_live = terminal_price,
        tickers                  = TICKERS,
        historical_brc_prices    = ds.historical_brc_prices
                                   if not ds.historical_brc_prices.empty else None,
    )

    write_validation_report(
        checks                = checks,
        backtest_result       = backtest_result,
        inception_params      = ds.model_params,
        terminal_params       = blended_terminal if blended_terminal else ds.terminal_model_params,
        tickers               = TICKERS,
        historical_brc_prices = ds.historical_brc_prices
                                if not ds.historical_brc_prices.empty else None,
        model_name            = SimClass.model_name(),
    )

    vr_path = f"./results/vr_comparison.csv"
    vr_df.to_csv(vr_path, index=False)
    print(f"\n  VR comparison table saved to {vr_path}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"  Model     : {SimClass.model_name()}")
    print(f"  Outputs in ./results/")

    return dict(
        data_store        = ds,
        simulator_class   = SimClass,
        inception_result  = result_inception,
        inception_greeks  = inception_greeks,
        backtest_result   = backtest_result,
        vr_comparison     = vr_df,
        terminal_price    = terminal_price,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BRC Monte Carlo Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model selection
  --model heston   Heston stochastic volatility (default)
  --model gbm      Geometric Brownian Motion (Black-Scholes)
  --model <name>   Any <name>.py in this directory that exposes Simulator

Examples:
  python main.py                              # Heston, full live run
  python main.py --model gbm                 # GBM model
  python main.py --offline --no-greeks       # fast offline test
  python main.py --model heston --flat-rate --n-sims 300
        """)

    parser.add_argument("--model",      type=str, default="heston",
                        help="Model name (module filename without .py, default: heston)")
    parser.add_argument("--offline",    action="store_true",
                        help="Disable all yfinance calls")
    parser.add_argument("--no-greeks",  action="store_true",
                        help="Skip Greek computation (faster)")
    parser.add_argument("--surface-type", type=str, default="heston",
                        choices=["heston", "bs"],
                        help="Synthetic fallback surface type (default: heston)")
    parser.add_argument("--surface",    type=str, default=None,
                        help="Explicit path to an inception surface CSV")
    parser.add_argument("--data-root",  type=str, default=None,
                        help="Path to final_dataset/ folder")
    parser.add_argument("--n-sims",     type=int, default=N_SIMS_MAIN,
                        help=f"Simulations per pricing date (default {N_SIMS_MAIN})")
    parser.add_argument("--freq",       type=str, default="W",
                        choices=["W", "ME"],
                        help="Fallback backtest frequency: W=weekly, ME=monthly")

    rate_grp = parser.add_mutually_exclusive_group()
    rate_grp.add_argument("--live-rates", dest="live_rates", action="store_true",
                          default=True, help="Use treasury_curve.csv (default)")
    rate_grp.add_argument("--flat-rate",  dest="live_rates", action="store_false",
                          help="Use flat 5%% risk-free rate throughout")

    args = parser.parse_args()

    main(
        model_name         = args.model,
        use_live_data      = not args.offline,
        compute_greeks     = not args.no_greeks,
        surface_type       = args.surface_type,
        surface_csv        = args.surface,
        data_root          = args.data_root,
        backtest_frequency = args.freq,
        n_sims_main        = args.n_sims,
        use_live_rates     = args.live_rates,
    )
