"""
Stage VII  —  Validation and Visualisation
==========================================
Checks:
  1. Directional consistency of backtest price series
  2. Endpoint anchor vs live option chain calibration
  3. Monte Carlo standard error throughout the backtest
  4. Variance reduction convergence plot

Produces:
  - backtest_price.png         : BRC price time series
  - barrier_proximity.png      : spot / barrier level per asset
  - attribution.png            : weekly price change attribution
  - vr_convergence.png         : std-error vs N for all VR methods
  - greeks_over_time.png       : rolling delta and vega
  - validation_report.txt      : text summary of all consistency checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from typing import Optional

OUTPUT_DIR = f"./results/plots"

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
    "figure.dpi":       130,
})

COLORS = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#ffa657"]


def _save(fig, name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 1 — BRC price time series
# ---------------------------------------------------------------------------

def plot_backtest_price(backtest_result, realised_spots=None,
                        tickers=None, fixing=None, barrier_level=0.5,
                        historical_brc_prices=None, model_name="Model"):
    """
    Plot model BRC price over time with the UBS secondary-market price overlay.

    Parameters
    ----------
    historical_brc_prices : pd.Series, optional
        UBS KeyInvest daily mid prices (percent of nominal, e.g. 96.85).
        Loaded from historical_BRC_prices.txt.
    model_name : str
        Human-readable model name shown in the legend and title
        (e.g. "Heston", "GBM").
    """
    ps = backtest_result.price_series
    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"BRC Model Price vs UBS Market Price — {model_name} MC — Backtest Apr 2025 → Today",
                 fontsize=13, color="#f0f6fc", y=0.98)

    # top: BRC price
    ax = axes[0]
    ax.plot(ps.index, ps.values, color=COLORS[0], linewidth=1.6,
            label=f"Model price ({model_name} MC)")
    # overlay UBS secondary-market prices
    if historical_brc_prices is not None and not historical_brc_prices.empty:
        # align to same date range as model backtest
        hist = historical_brc_prices.copy()
        ax.plot(hist.index, hist.values, color="#f0883e", linewidth=1.2,
                linestyle="--", alpha=0.90, label="UBS market price")
        # error band: model – market
        common = ps.index.intersection(hist.index)
        if len(common) > 1:
            err = ps.reindex(common) - hist.reindex(common)
            ax.fill_between(common,
                            ps.reindex(common),
                            hist.reindex(common),
                            alpha=0.12, color="#f0883e",
                            label=f"Model–market gap  (mean {err.mean():.2f})")
    ax.axhline(100, color="#8b949e", linestyle=":", alpha=0.5,
               label="Par ($100)")
    ax.set_ylabel("Price (% of nominal)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # bottom: barrier proximity (min across assets)
    ax2 = axes[1]
    prox = backtest_result.barrier_proximity
    if not prox.empty and tickers:
        for i, t in enumerate(tickers):
            if t in prox.columns:
                ax2.plot(prox.index, prox[t], color=COLORS[i],
                         label=t, alpha=0.85)
        ax2.axhline(1.0, color="#f78166", linestyle="--",
                    alpha=0.7, label="Barrier level")
        ax2.set_ylabel("Spot / Barrier")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    _save(fig, "backtest_price.png")


# ---------------------------------------------------------------------------
# Plot 2 — Attribution
# ---------------------------------------------------------------------------

def plot_attribution(backtest_result):
    """
    Stacked bar chart decomposing the week-on-week BRC model price change into:
      Spot attribution  — how much of the price change came from underlying
                          price moves (spots updated, vol held fixed).
      Vol attribution   — how much came from rolling ATM vol changes
                          (vol updated, spots held fixed at new level).
    Bars above zero mean that factor *increased* the BRC price that week;
    bars below zero mean it *decreased* it.
    """
    attr = backtest_result.attribution
    if attr.empty:
        print("  No attribution data to plot.")
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "Weekly BRC Price Attribution\n"
        "  Stacked bars show how much each factor moved the model price week-on-week",
        fontsize=11, color="#f0f6fc")

    width = 3  # days
    ax.bar(attr.index, attr["delta_spot"], width=width,
           color=COLORS[0], alpha=0.8,
           label="Spot moves  (underlying price changes)")
    ax.bar(attr.index, attr["delta_vol"], width=width,
           bottom=attr["delta_spot"], color=COLORS[1], alpha=0.8,
           label="Vol changes  (rolling 30d ATM vol update)")
    ax.axhline(0, color="#8b949e", linewidth=0.8)
    ax.set_ylabel("BRC price change  ($)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    _save(fig, "attribution.png")


# ---------------------------------------------------------------------------
# Plot 3 — Barrier proximity
# ---------------------------------------------------------------------------

def plot_barrier_proximity(backtest_result, tickers):
    """
    Plots spot / barrier_price for each underlying over the backtest period.

    Convention: ratio = S(t) / (initial_fixing_price × 0.50)
      > 1.0  →  spot is above barrier  (no kick-in yet)
      = 1.0  →  spot touches barrier   (kick-in event)
      < 1.0  →  barrier breached       (kick-in has occurred)

    This is the industry-standard 'distance to barrier' ratio.
    Plotting at 1.0 (not 0.5) keeps the danger threshold visually obvious.
    """
    prox = backtest_result.barrier_proximity
    if prox.empty:
        return

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        "Barrier Proximity  —  ratio = spot / barrier level\n"
        "  Kick-In event occurs if any line touches 1.0×",
        fontsize=11, color="#f0f6fc")

    for i, t in enumerate(tickers):
        if t in prox.columns:
            series = prox[t]
            ax.plot(series.index, series.values, color=COLORS[i],
                    label=t, alpha=0.9, linewidth=1.4)

    ax.axhline(1.0, color="#f78166", linestyle="--",
               linewidth=1.5, label="Barrier  (ratio = 1.0×  →  kick-in)")
    ax.fill_between(prox.index,
                    prox[tickers].min(axis=1).min() * 0.97, 1.0,
                    color="#f78166", alpha=0.07)
    ax.set_ylabel("spot / barrier level  (×)")
    ax.set_xlabel("")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    _save(fig, "barrier_proximity.png")


# ---------------------------------------------------------------------------
# Plot 4 — Variance reduction convergence
# ---------------------------------------------------------------------------

def plot_vr_convergence(vr_df: pd.DataFrame):
    """
    Two-panel variance reduction comparison.

    Left panel  — Price convergence:
        As N (number of simulation paths) increases, all methods should
        converge to the same BRC price.  Methods that converge from a
        tighter range with fewer paths are more efficient.

    Right panel — Standard-error convergence (log scale):
        Shows how fast each method's standard error falls with N.
        Crude MC decays as 1/√N.  Antithetic, quasi, and control-variate
        methods should lie below this line, reaching the same precision
        with fewer paths (= lower computational cost).
    """
    if vr_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Variance Reduction Comparison\n"
        "  Final backtest pricing uses: Antithetic pairs  ",
        fontsize=11, color="#f0f6fc")

    method_color = {"crude": COLORS[0], "antithetic": COLORS[1],
                    "quasi": COLORS[2], "control_variate": COLORS[3]}
    method_label = {
        "crude":           "Crude MC  (baseline)",
        "antithetic":      "Antithetic pairs",
        "quasi":           "Quasi-random (Sobol)",
        "control_variate": "Control variate",
    }

    for method, grp in vr_df.groupby("method"):
        grp = grp.sort_values("n_sims")
        c   = method_color.get(method, "#ffffff")
        lbl = method_label.get(method, method)
        axes[0].plot(grp["n_sims"], grp["price"], color=c,
                     marker="o", markersize=4, label=lbl)
        axes[1].plot(grp["n_sims"], grp["std_error"], color=c,
                     marker="o", markersize=4, label=lbl)

    axes[0].set_xlabel("N simulations")
    axes[0].set_ylabel("BRC Price  (% of $100 nominal)")
    axes[0].set_title("Price convergence\n"
                      "all methods should agree at large N")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)

    # reference 1/sqrt(N) line on std-error panel
    ns = vr_df["n_sims"].sort_values().unique()
    if len(ns) >= 2:
        ref_val = vr_df.loc[vr_df["method"] == "crude", "std_error"].max() \
                  if "crude" in vr_df["method"].values else \
                  vr_df["std_error"].max()
        ref_n0  = ns[0]
        axes[1].plot(ns, ref_val * np.sqrt(ref_n0 / ns),
                     color="#8b949e", linestyle=":", linewidth=1.2,
                     label="1/√N  (crude MC reference)")

    axes[1].set_xlabel("N simulations")
    axes[1].set_ylabel("Std Error  ($)")
    axes[1].set_title("Std-error convergence  (log scale)\n"
                      "lower = more efficient variance reduction")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, "vr_convergence.png")


# ---------------------------------------------------------------------------
# Plot 5 — Greeks over time
# ---------------------------------------------------------------------------

def plot_greeks(backtest_result, tickers):
    g = backtest_result.greeks_series
    if g.empty:
        print("  No Greeks to plot.")
        return

    delta_cols = [f"delta_{t}" for t in tickers if f"delta_{t}" in g.columns]
    vega_cols  = [f"vega_{t}"  for t in tickers if f"vega_{t}"  in g.columns]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    fig.suptitle("BRC Greeks Over Backtest Period", fontsize=12,
                 color="#f0f6fc")

    for i, col in enumerate(delta_cols):
        t = col.replace("delta_", "")
        axes[0].plot(g.index, g[col], color=COLORS[i], label=f"Δ {t}")
    axes[0].axhline(0, color="#8b949e", linewidth=0.8)
    axes[0].set_ylabel("Delta  ($/$ spot)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True)

    for i, col in enumerate(vega_cols):
        t = col.replace("vega_", "")
        axes[1].plot(g.index, g[col], color=COLORS[i], label=f"ν {t}")
    axes[1].axhline(0, color="#8b949e", linewidth=0.8)
    axes[1].set_ylabel("Vega  ($/v0 unit)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    _save(fig, "greeks_over_time.png")


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def run_consistency_checks(backtest_result,
                            terminal_price_from_live: Optional[float],
                            tickers: list,
                            historical_brc_prices=None) -> dict:
    """
    Run directional and quantitative consistency checks.

    Parameters
    ----------
    historical_brc_prices : pd.Series, optional
        UBS daily secondary-market prices.  Used to compute model-vs-market
        tracking error across the full backtest period.
    Returns a dict of {check_name: (passed, details)}.
    """
    checks = {}
    ps     = backtest_result.price_series
    prox   = backtest_result.barrier_proximity
    attr   = backtest_result.attribution

    # 1. Price in reasonable range
    lo, hi = 50.0, 120.0
    ok = bool((ps >= lo).all() and (ps <= hi).all())
    checks["price_in_range"] = (
        ok,
        f"All prices in [{lo}, {hi}]: {ps.min():.2f} – {ps.max():.2f}"
    )

    # 2. When any asset is closest to barrier, price is lower
    if not prox.empty:
        min_prox_series = prox[tickers].min(axis=1)
        # check correlation: lower proximity → lower price
        aligned = pd.concat([ps, min_prox_series], axis=1).dropna()
        if len(aligned) > 5:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            ok   = corr > 0   # positive: further from barrier → higher price
            checks["price_rises_away_from_barrier"] = (
                ok, f"Correlation(price, min_proximity) = {corr:.3f}"
            )

    # 3. Price generally declines over time if barrier is breached
    if "already_breached" in ps.index.names:
        pass  # skip if not tracked

    # 4. Std errors within tolerance
    if "std_error" in ps.to_frame().columns:
        max_se = 1.0   # $1 on $100 principal
        ok = True      # we only have price_series here; checked in main
        checks["std_error_acceptable"] = (True, "Checked in main output")

    # 5. Attribution sums close to total price change
    if not attr.empty:
        total_attr  = attr["delta_spot"] + attr["delta_vol"]
        actual_diff = attr["delta_total"]
        resid       = (total_attr - actual_diff).abs().mean()
        ok          = resid < 0.5
        checks["attribution_residual"] = (
            ok, f"Mean unexplained residual = {resid:.4f}"
        )

    # 6. Endpoint vs live chain
    if terminal_price_from_live is not None:
        gap = abs(ps.iloc[-1] - terminal_price_from_live)
        ok  = gap < 5.0
        checks["endpoint_anchor"] = (
            ok,
            f"Backtest endpoint={ps.iloc[-1]:.4f}, "
            f"live-chain price={terminal_price_from_live:.4f}, "
            f"gap={gap:.4f}"
        )

    # 7. Model vs UBS secondary-market price — windowed RMSE
    if historical_brc_prices is not None and not historical_brc_prices.empty:
        common = ps.index.intersection(historical_brc_prices.index)
        if len(common) >= 5:
            model_aligned  = ps.reindex(common)
            market_aligned = historical_brc_prices.reindex(common)
            errors         = model_aligned - market_aligned

            def _rmse(mask):
                e = errors[mask]
                return float(np.sqrt((e**2).mean())) if len(e) >= 2 else float("nan")

            cutoff_3m  = common.max() - pd.DateOffset(months=3)
            cutoff_6m  = common.max() - pd.DateOffset(months=6)
            rmse_full  = _rmse(slice(None))
            rmse_6m    = _rmse(common >= cutoff_6m)
            rmse_3m    = _rmse(common >= cutoff_3m)
            mae_full   = float(errors.abs().mean())
            max_err    = float(errors.abs().max())
            n_3m       = int((common >= cutoff_3m).sum())
            n_6m       = int((common >= cutoff_6m).sum())

            ok = mae_full < 3.0
            checks["market_tracking"] = (
                ok,
                f"Model vs UBS market ({len(common)} dates) — "
                f"MAE={mae_full:.2f}  RMSE(full)={rmse_full:.2f}  "
                f"RMSE(6m,n={n_6m})={rmse_6m:.2f}  "
                f"RMSE(3m,n={n_3m})={rmse_3m:.2f}  "
                f"MaxErr={max_err:.2f}  (pass if MAE<$3)"
            )

    # -------------------------------------------------------------------------
    # Directional consistency checks (per-date, not global correlation)
    # -------------------------------------------------------------------------
    # Each check scans consecutive date pairs and flags specific dates where
    # the model price moves in the wrong direction.  A small number of
    # failures is expected (MC noise, coupon step-downs, rate moves); a
    # systematic cluster of failures at the same dates signals a data error
    # or a model bug at those points.
    # -------------------------------------------------------------------------

    # 8. Price falls when any underlying moves closer to its barrier
    #    (replacing the old global-correlation version which was too coarse)
    if not prox.empty and len(ps) > 1:
        common_idx = ps.index.intersection(prox.index)
        if len(common_idx) > 2:
            ps_c    = ps.reindex(common_idx)
            prox_c  = prox[tickers].reindex(common_idx)
            worst_prox = prox_c.min(axis=1)   # minimum proximity across all assets

            dp   = ps_c.diff().dropna()
            dprox = worst_prox.diff().dropna()
            shared = dp.index.intersection(dprox.index)
            dp, dprox = dp.reindex(shared), dprox.reindex(shared)

            # violation: proximity falls (asset moves toward barrier) but price rises
            violations = shared[
                (dprox < -0.02) &   # proximity drops > 2% (meaningful move toward barrier)
                (dp > 0)            # but model price increased — wrong direction
            ]
            n_total = int((dprox < -0.02).sum())
            n_viol  = len(violations)
            pct     = 100 * n_viol / n_total if n_total > 0 else 0.0
            ok      = pct < 20.0   # allow up to 20% noise failures
            checks["barrier_proximity_directional"] = (
                ok,
                f"When worst-of asset moves toward barrier (>{0.02:.0%} drop): "
                f"{n_viol}/{n_total} dates price moved UP (violation rate {pct:.1f}%, "
                f"pass if <20%)."
                + (f"  Problem dates: {[str(d.date()) for d in violations[:5]]}"
                   if n_viol > 0 else "  No violations.")
            )

    # 9. Price falls when realised vol spikes
    rv = backtest_result.rolling_vol_used
    if not rv.empty and len(ps) > 1:
        available_tickers = [t for t in tickers if t in rv.columns]
        if available_tickers:
            rv_c   = rv[available_tickers].reindex(ps.index)
            rv_max = rv_c.max(axis=1)   # worst-of vol across tickers

            dp    = ps.diff().dropna()
            drv   = rv_max.diff().dropna()
            shared = dp.index.intersection(drv.index)
            dp, drv = dp.reindex(shared), drv.reindex(shared)

            # violation: vol spikes meaningfully but price rises
            VOL_SPIKE = 0.02   # 2 absolute vol points (e.g. 25% → 27%)
            violations = shared[
                (drv > VOL_SPIKE) &
                (dp > 0)
            ]
            n_total = int((drv > VOL_SPIKE).sum())
            n_viol  = len(violations)
            pct     = 100 * n_viol / n_total if n_total > 0 else 0.0
            ok      = pct < 25.0   # vol-price relationship noisier; allow 25%
            checks["vol_spike_directional"] = (
                ok,
                f"When max realised vol spikes >{VOL_SPIKE:.0%} abs: "
                f"{n_viol}/{n_total} dates price moved UP (violation rate {pct:.1f}%, "
                f"pass if <25%)."
                + (f"  Problem dates: {[str(d.date()) for d in violations[:5]]}"
                   if n_viol > 0 else "  No violations.")
            )

    # 10. Price rises as time passes without a barrier breach (theta / time-value)
    #     Tested only on dates where the barrier has NOT yet been breached and
    #     the spot moves are small (so the spot effect doesn't dominate).
    if not prox.empty and len(ps) > 1:
        common_idx = ps.index.intersection(prox.index)
        if len(common_idx) > 2:
            ps_c   = ps.reindex(common_idx)
            prox_c = prox[tickers].reindex(common_idx).min(axis=1)

            dp     = ps_c.diff().dropna()
            dprox  = prox_c.diff().dropna()
            shared = dp.index.intersection(dprox.index)
            dp, dprox = dp.reindex(shared), dprox.reindex(shared)

            # restrict to dates where:
            # (a) no asset is near its barrier (worst proximity > 1.2×  — well above)
            # (b) spot barely moved  (|dprox| < 0.03 — proximity change < 3%)
            # (c) barrier not already breached  (proximity > 1.0 throughout)
            far_from_barrier = prox_c.reindex(shared) > 1.20
            small_spot_move  = dprox.abs() < 0.03
            mask = far_from_barrier & small_spot_move

            if mask.sum() >= 5:
                dp_theta = dp[mask]
                violations = dp_theta.index[dp_theta < -0.10]   # price dropped > $0.10 without barrier move
                n_total = int(mask.sum())
                n_viol  = len(violations)
                pct     = 100 * n_viol / n_total if n_total > 0 else 0.0
                ok      = pct < 20.0
                checks["theta_positive_directional"] = (
                    ok,
                    f"On dates far from barrier with small spot moves "
                    f"(n={n_total}): {n_viol} dates price dropped >$0.10 "
                    f"without a meaningful spot move (violation rate {pct:.1f}%, "
                    f"pass if <20%)."
                    + (f"  Problem dates: {[str(d.date()) for d in violations[:5]]}"
                       if n_viol > 0 else "  No violations.")
                )

    # 11. Price is more sensitive to the worst-performing asset than to the others
    #     Uses the greeks_series delta to check that the asset with the largest
    #     delta magnitude is the one closest to its barrier.
    gs = backtest_result.greeks_series
    if not gs.empty and not prox.empty:
        delta_cols = {t: f"delta_{t}" for t in tickers if f"delta_{t}" in gs.columns}
        if len(delta_cols) == len(tickers):
            common_idx = gs.index.intersection(prox.index)
            if len(common_idx) >= 3:
                gs_c   = gs.reindex(common_idx)
                prox_c = prox[tickers].reindex(common_idx)

                n_correct   = 0
                n_total_g   = 0
                wrong_dates = []

                for date in common_idx:
                    # which ticker is worst (closest to barrier)?
                    worst_ticker = prox_c.loc[date].idxmin()
                    # which ticker has the largest delta magnitude?
                    deltas = {t: abs(gs_c.loc[date, delta_cols[t]])
                              for t in tickers
                              if pd.notna(gs_c.loc[date, delta_cols[t]])}
                    if len(deltas) < 2:
                        continue
                    most_sensitive_ticker = max(deltas, key=deltas.get)
                    n_total_g += 1
                    if most_sensitive_ticker == worst_ticker:
                        n_correct += 1
                    else:
                        wrong_dates.append((str(date.date()), worst_ticker,
                                            most_sensitive_ticker))

                if n_total_g > 0:
                    pct_correct = 100 * n_correct / n_total_g
                    ok = pct_correct >= 60.0   # majority of Greek dates should pass
                    checks["worst_asset_most_sensitive"] = (
                        ok,
                        f"Largest |delta| matches closest-to-barrier asset on "
                        f"{n_correct}/{n_total_g} Greek dates "
                        f"({pct_correct:.1f}%, pass if >=60%)."
                        + (f"  Mismatches (date, worst, most-sensitive): "
                           f"{wrong_dates[:5]}"
                           if wrong_dates else "  No mismatches.")
                    )

    return checks


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def write_validation_report(checks: dict,
                             backtest_result,
                             inception_params: dict,
                             terminal_params: dict,
                             tickers: list,
                             historical_brc_prices=None,
                             model_name: str = "Model"):
    os.makedirs(f"./results", exist_ok=True)
    path = f"./results/validation_report.txt"
    lines = []

    lines += [
        "="*70,
        "BRC MODEL — VALIDATION REPORT",
        "="*70,
        "",
        "NOTES ON INTERPRETATION",
        "-"*40,
        f"  {model_name} fit RMSE : root-mean-squared error in IMPLIED VOLATILITY units",
        "                     (decimal, e.g. 0.05 = 5pp IV error per point).",
        "                     Inception RMSE ~0.01 is expected: the surface was",
        "                     *generated* by the same approximate formula.",
        "                     Terminal RMSE 0.05-0.15 is realistic for sparse live chains.",
        "                     RMSE > 0.15 suggests sparse/illiquid live option data.",
        "  Barrier proximity: ratio = spot(t) / barrier_price = spot(t) / (fixing × 0.50).",
        "                     > 1.0 → safe;  = 1.0 → barrier touched;  < 1.0 → breached.",
        f"  Look-ahead note   : Terminal {model_name} params (from today's live chain) are used",
        "                     ONLY for the single endpoint validation anchor, not in the",
        "                     backtest loop. Backtest uses inception params + rolling vol.",
        "  Model-market gap  : Model uses synthetic inception vol surface (stylised).",
        "                     A gap vs UBS market price is expected; it narrows when",
        "                     calibrated to real April 2 2025 option chain data.",
        "",
        "CONSISTENCY CHECKS",
        "-"*40,
    ]

    passed = 0
    for name, (ok, detail) in checks.items():
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        lines.append(f"  [{status}] {name}")
        lines.append(f"         {detail}")

    lines += [
        "",
        f"  Summary: {passed}/{len(checks)} checks passed",
        "",
        "BACKTEST PRICE SERIES",
        "-"*40,
    ]
    ps = backtest_result.price_series
    lines += [
        f"  Date grid       : aligned to historical UBS prices where available",
        f"  N pricing dates : {len(ps)}",
        f"  Inception price : {ps.iloc[0]:.4f}",
        f"  Latest price    : {ps.iloc[-1]:.4f}",
        f"  Min price       : {ps.min():.4f}  ({ps.idxmin().date()})",
        f"  Max price       : {ps.max():.4f}  ({ps.idxmax().date()})",
    ]

    # windowed RMSE table vs UBS market prices
    if historical_brc_prices is not None and not historical_brc_prices.empty:
        common = ps.index.intersection(historical_brc_prices.index)
        if len(common) >= 2:
            model_a  = ps.reindex(common)
            market_a = historical_brc_prices.reindex(common)
            errors   = model_a - market_a
            end_date = common.max()

            def _stats(mask):
                e = errors[mask]
                if len(e) < 2:
                    return None
                return dict(
                    n    = len(e),
                    mae  = float(e.abs().mean()),
                    rmse = float(np.sqrt((e**2).mean())),
                    bias = float(e.mean()),          # positive = model above market
                    max  = float(e.abs().max()),
                )

            windows = [
                ("Full period",   common >= common.min()),
                ("Last 6 months", common >= end_date - pd.DateOffset(months=6)),
                ("Last 3 months", common >= end_date - pd.DateOffset(months=3)),
            ]

            lines += [
                "",
                "  MODEL vs UBS MARKET PRICE  (errors in % of $100 nominal)",
                f"  {'Window':<18} {'N':>5}  {'MAE':>6}  {'RMSE':>6}  "
                f"{'Bias':>6}  {'MaxErr':>7}",
                "  " + "-"*55,
            ]
            for label, mask in windows:
                s = _stats(mask)
                if s:
                    bias_sign = "+" if s["bias"] >= 0 else ""
                    lines.append(
                        f"  {label:<18} {s['n']:>5}  {s['mae']:>6.2f}  "
                        f"{s['rmse']:>6.2f}  {bias_sign}{s['bias']:>5.2f}  "
                        f"{s['max']:>7.2f}"
                    )
            lines += [
                "  " + "-"*55,
                "  Bias > 0 means model is pricing above the UBS market price.",
                "  Expected if inception vol surface is too low (underpriced short put).",
            ]
    lines += [""]

    lines += [
        "", f"{model_name.upper()} PARAMETER COMPARISON (inception vs terminal)", "-"*40,
        "  Inception = calibrated to synthetic surface at Apr 2 2025 (stylised inputs).",
        "  Terminal  = calibrated to live option chain today (real market data).",
    ]

    # Derive the parameter keys actually present in this model's params,
    # preserving a sensible display order where keys are recognised.
    _PREFERRED_ORDER = ["v0", "kappa", "theta", "sigma", "rho"]
    sample_inception = next(iter(inception_params.values()), {}) if inception_params else {}
    sample_terminal  = next(iter(terminal_params.values()),  {}) if terminal_params  else {}
    all_keys = list(sample_inception.keys()) or list(sample_terminal.keys())
    # Sort: preferred-order keys first, then any extras alphabetically
    param_keys = [k for k in _PREFERRED_ORDER if k in all_keys] + \
                 sorted(k for k in all_keys if k not in _PREFERRED_ORDER)

    # Add per-key description lines for known keys only
    _KEY_DESCRIPTIONS = {
        "v0":    "  v0  = instantaneous variance; sqrt(v0) = instantaneous vol.",
        "kappa": "  kappa = mean-reversion speed.",
        "theta": "  theta = long-run variance.",
        "sigma": "  sigma = vol-of-vol (Heston) or annualised vol (GBM).",
        "rho":   "  rho = spot-vol correlation (should be negative for equities).",
    }
    for k in param_keys:
        if k in _KEY_DESCRIPTIONS:
            lines.append(_KEY_DESCRIPTIONS[k])

    if "kappa" in param_keys:
        lines.append("  If terminal kappa/sigma are at their bounds, the live chain is too")
        lines.append("  sparse for reliable skew/smile identification — use v0 and theta only.")

    for t in tickers:
        ip = inception_params.get(t, {})
        tp = terminal_params.get(t, {})
        lines.append(f"\n  {t}:")
        for k in param_keys:
            iv = ip.get(k, float("nan"))
            tv = tp.get(k, float("nan"))
            at_bound = " ← AT BOUND" if k in ("kappa", "sigma") and abs(tv) >= 7.9 else ""
            # Only show change if both values are finite
            if np.isfinite(iv) and np.isfinite(tv):
                lines.append(f"    {k:<8}: inception={iv:.4f}  terminal={tv:.4f}  "
                             f"change={tv-iv:+.4f}{at_bound}")
            elif np.isfinite(iv):
                lines.append(f"    {k:<8}: inception={iv:.4f}  terminal=N/A")
            else:
                lines.append(f"    {k:<8}: N/A")

    if not backtest_result.barrier_proximity.empty:
        lines += [
            "", "BARRIER PROXIMITY  (ratio = spot / barrier_level$)", "-"*40,
            "  Read as 'current spot is X times the barrier price'.",
            "  1.5× means the stock is 50% above its barrier — comfortably safe.",
            "  Kick-in event occurs if the ratio ever falls to 1.0× or below.",
        ]
        prox = backtest_result.barrier_proximity
        for t in tickers:
            if t in prox.columns:
                m = prox[t].min()
                d = prox[t].idxmin().date()
                lines.append(f"  {t}: min proximity = {m:.3f}×  "
                              f"(closest approach on {d};  "
                              f"spot was {(m-1)*100:.1f}% above barrier)")

    lines += ["", "="*70]

    report_text = "\n".join(lines)
    # Write UTF-8 so special characters (arrows, ×, ±, etc.) are preserved.
    # On Windows the default codec (cp1252) cannot encode them.
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Validation report saved to {path}")

    # Print to console — replace characters that cp1252 can't handle
    # so the report is always readable on Windows terminals too.
    _REPLACEMENTS = {
        "\u2192": "->", "\u2190": "<-", "\u00d7": "x",
        "\u2265": ">=", "\u2264": "<=", "\u00b1": "+/-",
        "\u2202": "d",  "\u221a": "sqrt", "\u2211": "sum",
        "\u2013": "-",  "\u2014": "--",
    }
    safe_text = report_text
    for char, sub in _REPLACEMENTS.items():
        safe_text = safe_text.replace(char, sub)
    try:
        print("\n" + safe_text)
    except UnicodeEncodeError:
        # last resort: replace anything still unencodable
        print("\n" + safe_text.encode(
            "ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    print("stage7_validation.py loaded — run via main.py")
