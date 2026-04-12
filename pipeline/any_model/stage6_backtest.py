"""
Stage VI  —  Backtest Loop
==========================
Model-agnostic.  The only model-specific operation —
updating parameters to reflect a new ATM vol estimate —
is delegated to the simulator via:

    simulator.__class__.update_params_for_vol(inception_params, atm_vol)

Everything else (spot lookup, barrier tracking, payoff evaluation,
attribution, Greeks) works identically for any BaseSimulator subclass.

Outputs (BacktestResult)
  - price_series       : pd.Series  (date → BRC model price)
  - greeks_series      : pd.DataFrame  (date × greek)
  - barrier_proximity  : pd.DataFrame  (date × ticker, ratio price/barrier)
  - rolling_vol_used   : pd.DataFrame  (date × ticker)
  - attribution        : pd.DataFrame  (date × {Δspot, Δvol, Δtime, total})
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    price_series:      pd.Series    = field(default_factory=pd.Series)
    greeks_series:     pd.DataFrame = field(default_factory=pd.DataFrame)
    barrier_proximity: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_vol_used:  pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution:       pd.DataFrame = field(default_factory=pd.DataFrame)
    vr_comparison:     pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Main backtest engine
# ---------------------------------------------------------------------------

class BRCBacktester:

    def __init__(self,
                 data_store,              # DataStore from stage1_data
                 simulator,               # any BaseSimulator subclass instance
                 payoff_template,         # BRCPayoff from stage4_pricing
                 n_sims:    int  = 2000,
                 n_steps:   int  = 252,
                 frequency: str  = "W",
                 seed:      int  = 42,
                 compute_greeks:  bool = True,
                 use_live_rates:  bool = True):

        self.ds                  = data_store
        self.sim                 = simulator
        self.po_template         = payoff_template
        self.n_sims              = n_sims
        self.n_steps_pa          = n_steps
        self.frequency           = frequency
        self.seed                = seed
        self.compute_greeks_flag = compute_greeks
        self.use_live_rates      = use_live_rates

        self.tickers     = payoff_template.tickers
        self.maturity    = payoff_template.maturity_date
        self.fixing      = payoff_template.initial_fixing_prices
        self.barrier_lvl = payoff_template.barrier_level
        self.r           = payoff_template.r

        # The update function is fetched from the simulator's class so it
        # works regardless of which concrete model is in use.
        self._update_params = simulator.__class__.update_params_for_vol

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_realised_spots_at(self, date: pd.Timestamp) -> Optional[Dict[str, float]]:
        df = self.ds.realised_spots
        if df.empty:
            return None
        available = df[df.index <= date]
        if available.empty:
            return None
        row = available.iloc[-1]
        return {t: float(row[t]) for t in self.tickers}

    def _get_rolling_vol_at(self, date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        ATM vol per ticker, priority:
          1. historical_iv (ATM implied vol snapshots, ≤45 days stale)
          2. realised rolling vol (30-day backward-looking)
        Returns None if neither source has data.
        """
        STALENESS_LIMIT = pd.Timedelta(days=45)
        result = {}

        hiv = getattr(self.ds, "historical_iv", pd.DataFrame())
        if not hiv.empty:
            available = hiv[hiv.index <= date]
            if not available.empty:
                last_row  = available.iloc[-1]
                last_date = available.index[-1]
                stale     = (date - last_date) > STALENESS_LIMIT
                if not stale:
                    for t in self.tickers:
                        if t in last_row.index and pd.notna(last_row[t]):
                            result[t] = float(last_row[t])

        missing = [t for t in self.tickers if t not in result]
        if missing:
            df = self.ds.rolling_vol
            if not df.empty:
                avail_r = df[df.index <= date]
                if not avail_r.empty:
                    row = avail_r.iloc[-1]
                    for t in missing:
                        if t in row.index and pd.notna(row[t]):
                            result[t] = float(row[t])

        return result if result else None

    def _barrier_breached_before(self, date: pd.Timestamp) -> bool:
        df = self.ds.realised_spots
        if df.empty:
            return False
        hist = df[df.index <= date]
        for t in self.tickers:
            barrier_abs = self.fixing[t] * self.barrier_lvl
            if (hist[t] <= barrier_abs).any():
                return True
        return False

    def _barrier_proximity_at(self, date: pd.Timestamp) -> Dict[str, float]:
        spots = self._get_realised_spots_at(date) or self.fixing
        return {t: spots[t] / (self.fixing[t] * self.barrier_lvl)
                for t in self.tickers}

    # ------------------------------------------------------------------
    # Pricing at one date
    # ------------------------------------------------------------------

    def _price_at_date(self,
                       date:          pd.Timestamp,
                       spots:         Dict[str, float],
                       model_params:  Dict[str, dict],
                       already_breached: bool,
                       rate_curve_fn  = None) -> tuple:
        """
        Run Monte Carlo for the remaining life of the BRC as of `date`.
        Returns (price, std_error, payoffs_array).
        """
        from stage4_pricing import BRCPayoff

        T_rem   = max((self.maturity - date).days / 365.0, 1/252)
        n_steps = max(int(self.n_steps_pa * T_rem), 10)

        if rate_curve_fn is not None:
            from stage1_data import get_zero_rate
            r_drift = rate_curve_fn(T_rem)
        else:
            r_drift = self.r

        po = BRCPayoff(
            tickers               = self.tickers,
            initial_fixing_prices = self.fixing,
            valuation_date        = date.strftime("%Y-%m-%d"),
            maturity_date         = self.maturity.strftime("%Y-%m-%d"),
            barrier_level         = self.barrier_lvl,
            coupon_rate           = self.po_template.coupon_rate,
            principal             = self.po_template.principal,
            risk_free_rate        = r_drift,
            rate_curve_fn         = rate_curve_fn,
        )

        orig_params = self.sim.params
        orig_r      = self.sim.r
        self.sim.params = model_params
        self.sim.r      = r_drift

        paths   = self.sim.simulate(spots, T_rem, self.n_sims, n_steps,
                                    method="antithetic", seed=self.seed)
        payoffs = po.evaluate(paths, already_breached=already_breached)

        self.sim.params = orig_params
        self.sim.r      = orig_r

        price   = float(np.mean(payoffs))
        stderr  = float(np.std(payoffs) / np.sqrt(self.n_sims))
        return price, stderr, payoffs

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        from stage4_pricing import BRCGreeks

        result = BacktestResult()

        inception = pd.Timestamp(self.po_template.valuation_date)
        today     = pd.Timestamp(self.ds.today)

        hist = self.ds.historical_brc_prices
        if hist is not None and not hist.empty:
            dates = hist.index[
                (hist.index > inception) & (hist.index <= min(today, self.maturity))
            ]
            date_source = f"historical UBS prices  ({len(dates)} observations)"
        else:
            dates = pd.date_range(start=inception, end=today, freq=self.frequency)
            dates = dates[dates <= self.maturity]
            date_source = f"frequency grid  ({self.frequency})"

        print("\n" + "="*60)
        print("STAGE VI — BACKTEST")
        print("="*60)
        print(f"  Date source   : {date_source}")
        print(f"  Pricing dates : {dates[0].date()} → {dates[-1].date()}")
        print(f"  N dates       : {len(dates)}")
        print(f"  N sims/date   : {self.n_sims:,}")
        print("-"*60)

        price_records     = []
        greeks_records    = []
        proximity_records = []
        vol_records       = []
        attr_records      = []

        prev_price  = None
        prev_spots  = None
        prev_model  = None   # model_params dict from previous date

        for date in dates:
            spots = self._get_realised_spots_at(date)
            if spots is None:
                spots = dict(self.fixing)

            vols = self._get_rolling_vol_at(date) or {}

            # ── model-agnostic parameter update ──────────────────────────
            # self._update_params comes from the concrete simulator class
            # (e.g. HestonSimulator.update_params_for_vol or GBMSimulator…)
            # Each ticker falls back to the inception ATM vol if not in `vols`.
            inception_params = self.ds.model_params
            model_params = {}
            fallback_used = False

            for t in self.tickers:
                daily_df = self.ds.daily_params.get(t)
                # Use the most recent calibration available on/before this date.
                # Exact-date matching is brittle and can cause jumpy fallback behavior.
                row = None
                if (daily_df is not None and not daily_df.empty and 'date' in daily_df.columns):
                    cal_dates = pd.to_datetime(daily_df['date'], errors='coerce')
                    valid = cal_dates <= date
                    if valid.any():
                        row = daily_df.loc[valid].iloc[-1]

                if row is not None:
                    # Use full daily historical calibration when available.
                    daily_params = {}
                    for key in ["mu", "kappa", "theta", "sigma", "rho", "v0"]:
                        if key in row.index and pd.notna(row[key]):
                            daily_params[key] = float(row[key])

                    if daily_params:
                        model_params[t] = daily_params
                    else:
                        inception_param = inception_params[t]
                        vol = vols.get(t, float(np.sqrt(
                                    inception_param.get("v0", inception_param.get("sigma", 0.30) ** 2)
                            )))
                        model_params[t] = self._update_params(inception_param, vol)
                        fallback_used = True

                else:
                    # Fallback to original 
                    inception_param = inception_params[t]

                    vol = vols.get(t, float(np.sqrt(
                                inception_param.get("v0", inception_param.get("sigma", 0.30) ** 2)
                        )))
                    model_params[t] = self._update_params(inception_param, vol)
                    fallback_used = True

                # mu is always the risk-free rate (risk-neutral pricing).
                model_params[t]['mu'] = self.r


            already_breached = self._barrier_breached_before(date)

            from stage1_data import get_rate_curve_fn
            curve = getattr(self.ds, "treasury_curve", None)
            if (self.use_live_rates
                    and curve is not None
                    and not curve.empty):
                rate_curve_fn = get_rate_curve_fn(curve, date)
            else:
                rate_curve_fn = None

            price, stderr, payoffs = self._price_at_date(
                date, spots, model_params, already_breached,
                rate_curve_fn=rate_curve_fn)

            T_rem  = max((self.maturity - date).days / 365.0, 1/252)
            r_used = rate_curve_fn(T_rem) if rate_curve_fn else self.r
            print(f"  {date.date()}  price={price:7.4f}  se={stderr:.4f}  "
                  f"r={r_used:.3%}  "
                f"breached={'YES' if already_breached else 'no '}  "
                f"fallback_used={'YES' if fallback_used else 'NO'}")

            price_records.append(dict(date=date, price=price,
                                      std_error=stderr,
                                      already_breached=already_breached))

            prox = self._barrier_proximity_at(date)
            proximity_records.append({**{"date": date}, **prox})
            vol_records.append({**{"date": date}, **vols})

            # attribution
            if prev_price is not None and prev_spots is not None:
                T_rem_a = max((self.maturity - date).days / 365.0, 1/252)
                n_steps = max(int(self.n_steps_pa * T_rem_a), 10)
                old_params = prev_model if prev_model is not None else model_params

                from stage4_pricing import BRCPayoff
                po_tmp = BRCPayoff(
                    self.tickers, self.fixing,
                    valuation_date=date.strftime("%Y-%m-%d"),
                    maturity_date=self.maturity.strftime("%Y-%m-%d"),
                    barrier_level=self.barrier_lvl,
                    coupon_rate=self.po_template.coupon_rate,
                    principal=self.po_template.principal,
                    risk_free_rate=self.r,
                )
                self.sim.params = old_params
                paths_spot = self.sim.simulate(spots, T_rem_a, self.n_sims,
                                               n_steps, method="antithetic",
                                               seed=self.seed)
                price_spot_only = float(np.mean(
                    po_tmp.evaluate(paths_spot, already_breached)))
                self.sim.params = inception_params

                attr_records.append(dict(
                    date=date,
                    delta_total=price - prev_price,
                    delta_spot=price_spot_only - prev_price,
                    delta_vol=price - price_spot_only,
                ))

            # Greeks (every 8th date)
            if self.compute_greeks_flag and len(price_records) % 8 == 0:
                try:
                    g_obj = BRCGreeks(
                        self.sim, self.po_template, spots,
                        T=max((self.maturity - date).days / 365.0, 1/252),
                        n_sims=self.n_sims,
                        n_sims_greek=min(self.n_sims, 1000),
                        n_steps=max(int(self.n_steps_pa *
                                    (self.maturity - date).days / 365.0), 10),
                        seed=self.seed,
                    )
                    g = g_obj.all_greeks()
                    row = {"date": date}
                    for t in self.tickers:
                        row[f"delta_{t}"] = g["delta"].get(t, np.nan)
                        row[f"gamma_{t}"] = g["gamma"].get(t, np.nan)
                        row[f"vega_{t}"]  = g["vega"].get(t, np.nan)
                    row["rho"] = g["rho"]
                    for pair, val in g["correlation_sensitivity"].items():
                        row[f"corr_sens_{pair}"] = val
                    greeks_records.append(row)
                except Exception as e:
                    print(f"    WARNING: Greeks failed at {date.date()}: {e}")

            prev_price = price
            prev_spots = spots
            prev_model = model_params

        # assemble results
        price_df = pd.DataFrame(price_records).set_index("date")
        result.price_series = price_df["price"]

        if greeks_records:
            result.greeks_series = pd.DataFrame(greeks_records).set_index("date")

        result.barrier_proximity = pd.DataFrame(proximity_records).set_index("date")
        result.rolling_vol_used  = pd.DataFrame(vol_records).set_index("date")
        if attr_records:
            result.attribution = pd.DataFrame(attr_records).set_index("date")

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------

    def _print_summary(self, result: BacktestResult):
        ps = result.price_series
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        print(f"  Dates priced     : {len(ps)}")
        print(f"  Price range      : {ps.min():.4f} – {ps.max():.4f}")
        print(f"  Inception price  : {ps.iloc[0]:.4f}")
        print(f"  Latest price     : {ps.iloc[-1]:.4f}")
        print(f"  Price change     : {ps.iloc[-1] - ps.iloc[0]:+.4f}")

        prox = result.barrier_proximity
        if not prox.empty:
            print("\n  Minimum barrier proximity (spot/barrier level):")
            for t in self.tickers:
                if t in prox.columns:
                    min_prox = prox[t].min()
                    min_date = prox[t].idxmin()
                    print(f"    {t}: {min_prox:.3f}×  "
                          f"(closest on {min_date.date()})")

        if not result.attribution.empty:
            attr = result.attribution
            print("\n  Average weekly attribution:")
            print(f"    Spot moves   : {attr['delta_spot'].mean():+.4f}")
            print(f"    Vol changes  : {attr['delta_vol'].mean():+.4f}")


if __name__ == "__main__":
    print("stage6_backtest.py loaded — run via main.py")
