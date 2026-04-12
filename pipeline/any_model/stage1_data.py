"""
Stage I & II  —  Data Collection and Parameter Estimation
==========================================================
Model-agnostic.  Does NOT import or reference any specific simulator.

Calibration is delegated to the active model module, which is passed in
from main.py as a SimulatorClass.  The DataStore stores fitted params in
a generic  model_params  dict  {ticker: params_dict}  whose schema is
defined by the model.

Expected directory layout (relative to DATA_ROOT = final_dataset/):

  final_dataset/
    prices/
      fetch_prices_v1/
        processed/
          log_returns.csv
    synthetic_option/
      black_scholes/
        all_bs_surfaces.csv
      heston/
        all_heston_surfaces.csv
    historical_option/
      <TICKER>/
        <any>.csv

Outputs  (DataStore)
  correlation_matrix    : 3×3 numpy array
  log_returns           : DataFrame, pre-computed log returns
  realised_spots        : DataFrame, Apr 2025 – today
  inception_surface     : DataFrame, synthetic IV surface
  live_surface          : DataFrame, today's market IV surface
  rolling_vol           : DataFrame, 30-day rolling annualised vol
  model_params          : dict  {ticker: model-specific params}   ← generic
  terminal_model_params : dict  {ticker: model-specific params}   ← generic
  initial_fixing_prices : dict  {ticker: float}
  surface_type          : str
  historical_iv         : DataFrame  (ATM IV snapshots)
  treasury_curve        : DataFrame
  historical_brc_prices : pd.Series
"""

import os
import inspect
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Live data unavailable.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS            = ["NFLX", "SPOT", "DIS"]
INCEPTION_DATE     = "2025-04-02"
RISK_FREE_RATE     = 0.05
ROLLING_VOL_WINDOW = 30

# Official fixing prices from UBS Termsheet (ISIN CH1431536452)
INITIAL_FIXING_PRICES = {
    "NFLX": 93.55,
    "SPOT": 565.41,
    "DIS":  97.88,
}

_DEFAULT_DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "..", "final_dataset")
_LOG_RETURNS_RELPATH = os.path.join("prices", "fetch_prices_v1", "processed", "log_returns.csv")
_BS_SURFACE_RELPATH  = os.path.join("synthetic_option", "black_scholes", "all_bs_surfaces.csv")
_HST_SURFACE_RELPATH = os.path.join("synthetic_option", "heston", "all_heston_surfaces.csv")


def resolve_paths(data_root: str = None) -> dict:
    root = (data_root or os.environ.get("DATA_ROOT") or _DEFAULT_DATA_ROOT)
    root = os.path.abspath(root)
    return {
        "root":        root,
        "log_returns": os.path.join(root, _LOG_RETURNS_RELPATH),
        "bs_surface":  os.path.join(root, _BS_SURFACE_RELPATH),
        "hst_surface": os.path.join(root, _HST_SURFACE_RELPATH),
    }


# ---------------------------------------------------------------------------
# DataStore  (model-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class DataStore:
    correlation_matrix:       np.ndarray   = field(default_factory=lambda: np.eye(3))
    log_returns:              pd.DataFrame = field(default_factory=pd.DataFrame)
    realised_spots:           pd.DataFrame = field(default_factory=pd.DataFrame)
    inception_surface:        pd.DataFrame = field(default_factory=pd.DataFrame)
    live_surface:             pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_vol:              pd.DataFrame = field(default_factory=pd.DataFrame)
    model_params:             Dict         = field(default_factory=dict)   # generic
    daily_params:             Dict         = field(default_factory=dict)   # generic
    terminal_model_params:    Dict         = field(default_factory=dict)   # generic
    initial_fixing_prices:    Dict         = field(default_factory=dict)
    surface_type:             str          = "bs"
    today:                    str          = field(default_factory=lambda: date.today().isoformat())
    historical_brc_prices:    pd.Series    = field(default_factory=pd.Series)
    treasury_curve:           pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_iv:            pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------
    # Back-compat aliases for code that still uses the Heston-era names.
    # These delegate to the generic model_params / terminal_model_params dicts
    # so existing callers (stage6_backtest, main.py terminal section, etc.)
    # continue to work without changes.
    # ------------------------------------------------------------------

    @property
    def inception_heston_params(self) -> Dict:
        return self.model_params

    @inception_heston_params.setter
    def inception_heston_params(self, v):
        self.model_params = v

    @property
    def terminal_heston_params(self) -> Dict:
        return self.terminal_model_params

    @terminal_heston_params.setter
    def terminal_heston_params(self, v):
        self.terminal_model_params = v


# ---------------------------------------------------------------------------
# Black-Scholes helpers  (IV extraction only — not model-specific)
# ---------------------------------------------------------------------------

def _bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put(S, K, T, r, sigma):
    return _bs_call(S, K, T, r, sigma) - S + K * np.exp(-r * T)


def _implied_vol(market_price, S, K, T, r, option_type="call"):
    if T <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price < intrinsic - 1e-4:
        return np.nan
    fn = (lambda s: _bs_call(S, K, T, r, s) - market_price
          if option_type == "call"
          else _bs_put(S, K, T, r, s) - market_price)
    try:
        return brentq(fn, 1e-4, 5.0, xtol=1e-6, maxiter=200)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Historical BRC prices
# ---------------------------------------------------------------------------

def load_historical_brc_prices(path: str = "../../historical_BRC_prices.txt") -> pd.Series:
    import re
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path,
        os.path.join(script_dir, "../../historical_BRC_prices.txt"),
        os.path.join(script_dir, "../../../historical_BRC_prices.txt"),
        "/mnt/user-data/uploads/historical_BRC_prices.txt",
    ]
    raw = None
    for p in candidates:
        try:
            with open(p) as f:
                raw = f.read()
            print(f"  Loaded historical BRC prices from: {p}")
            break
        except FileNotFoundError:
            continue
    if raw is None:
        print(f"  WARNING: historical BRC prices not found")
        return pd.Series(dtype=float)

    pattern = r'\{\s*"date"\s*:\s*(\d+)\s*,\s*"value"\s*:\s*([\d.]+)\s*\}'
    matches = re.findall(pattern, raw)
    if not matches:
        print(f"  WARNING: no price records parsed from {path}")
        return pd.Series(dtype=float)

    records = [(pd.Timestamp(int(ts), unit="ms").normalize(), float(val))
               for ts, val in matches]
    s = pd.Series({dt: val for dt, val in records}, name="ubs_brc_price")
    s.index = pd.DatetimeIndex(s.index)
    s = s.sort_index()
    print(f"  Historical BRC prices: {len(s)} records "
          f"({s.index[0].date()} → {s.index[-1].date()})")
    return s


# ---------------------------------------------------------------------------
# Treasury yield curve
# ---------------------------------------------------------------------------

_TREASURY_TENORS = {
    "1_month": 1/12, "2_month": 2/12, "3_month": 3/12, "6_month": 6/12,
    "1_year": 1.0, "2_year": 2.0, "3_year": 3.0, "5_year": 5.0,
    "7_year": 7.0, "10_year": 10.0, "20_year": 20.0,
}
_TENOR_YEARS = list(_TREASURY_TENORS.values())
_TENOR_COLS  = list(_TREASURY_TENORS.keys())


def load_treasury_curve(path: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path,
        os.path.join(script_dir, "../../treasury_curve.csv"),
        os.path.join(script_dir, "../../../treasury_curve.csv"),
        "/mnt/user-data/uploads/treasury_curve.csv",
    ]
    raw = None
    used_path = None
    for p in candidates:
        try:
            raw = pd.read_csv(p)
            used_path = p
            break
        except FileNotFoundError:
            continue

    if raw is None:
        print("  WARNING: treasury curve CSV not found — falling back to flat rate.")
        return pd.DataFrame()

    date_col = next((c for c in raw.columns
                     if c.strip().lower() in ("date", "dates", "index")), None)
    if date_col is None:
        date_col = raw.columns[0]

    try:
        raw[date_col] = pd.to_datetime(raw[date_col], infer_datetime_format=True,
                                    errors="coerce")
    except TypeError:
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")

    raw = raw.dropna(subset=[date_col])
    df  = raw.set_index(date_col).copy()
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    df = df.sort_index()

    present = [c for c in _TENOR_COLS if c in df.columns]
    if not present:
        return pd.DataFrame()
    df = df[present].apply(pd.to_numeric, errors="coerce").copy()
    if df.max().max() > 1.0:
        df = df / 100.0

    print(f"  Treasury curve loaded: {used_path}  ({len(df)} rows)")
    return df


def load_historical_iv(path: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path,
        os.path.join(script_dir, "../../historical_iv.csv"),
        os.path.join(script_dir, "../../../historical_iv.csv"),
        "/mnt/user-data/uploads/historical_iv.csv",
    ]
    raw = None
    used_path = None
    for p in candidates:
        try:
            raw = pd.read_csv(p)
            used_path = p
            break
        except FileNotFoundError:
            continue

    if raw is None:
        print("  historical_iv.csv not found — will use realised vol only.")
        return pd.DataFrame()

    raw.columns = [c.strip().lower() for c in raw.columns]
    date_col   = next((c for c in raw.columns if "date" in c), None)
    ticker_col = next((c for c in raw.columns
                       if c in ("act_symbol", "symbol", "ticker")), None)
    vol_col    = next((c for c in raw.columns if "iv" in c or "vol" in c), None)

    if not all([date_col, ticker_col, vol_col]):
        print(f"  WARNING: historical_iv.csv missing expected columns. "
              f"Found: {list(raw.columns)}")
        return pd.DataFrame()

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw[vol_col]  = pd.to_numeric(raw[vol_col], errors="coerce")
    raw = raw[raw[vol_col] > 0].dropna(subset=[vol_col])
    if raw[vol_col].median() > 1.0:
        raw[vol_col] = raw[vol_col] / 100.0

    wide = (raw.pivot_table(index=date_col, columns=ticker_col,
                             values=vol_col, aggfunc="mean")
               .rename_axis(None, axis=1)
               .rename_axis("date"))
    wide.index = pd.DatetimeIndex(wide.index).normalize()
    wide = wide.sort_index()
    print(f"  Historical IV loaded: {used_path}  ({len(wide)} dates)")
    return wide


def get_zero_rate(curve: pd.DataFrame, as_of_date, tenor_years: float) -> float:
    if curve.empty:
        return RISK_FREE_RATE
    as_of = pd.Timestamp(as_of_date)
    available = curve[curve.index <= as_of]
    row = available.iloc[-1] if not available.empty else curve.iloc[0]

    cols_present   = [c for c in _TENOR_COLS if c in row.index and not np.isnan(row[c])]
    tenors_present = [_TREASURY_TENORS[c] for c in cols_present]
    rates_present  = [float(row[c]) for c in cols_present]

    if not tenors_present:
        return RISK_FREE_RATE

    tenor_clamped = float(np.clip(tenor_years, min(tenors_present), max(tenors_present)))
    r_par = float(np.interp(tenor_clamped, tenors_present, rates_present))
    return np.log(1.0 + r_par)


def get_rate_curve_fn(curve: pd.DataFrame, as_of_date):
    def _r(tenor_years: float) -> float:
        return get_zero_rate(curve, as_of_date, tenor_years)
    return _r


# ---------------------------------------------------------------------------
# Stage I — log returns
# ---------------------------------------------------------------------------

def load_log_returns(log_returns_path: str) -> pd.DataFrame:
    if not os.path.exists(log_returns_path):
        raise FileNotFoundError(f"Log returns file not found: {log_returns_path}")

    df = pd.read_csv(log_returns_path, index_col=0, parse_dates=True)
    available = [t for t in TICKERS if t in df.columns]
    if not available:
        raise ValueError(f"log_returns.csv has no recognised tickers")
    df = df[available].dropna()
    if df.abs().max().max() > 1.0:
        print("  Detected raw prices in log_returns.csv — computing log returns.")
        df = np.log(df / df.shift(1)).dropna()

    print(f"  Log returns: {len(df)} obs, tickers={available}")
    return df


# ---------------------------------------------------------------------------
# Stage I — inception surface
# ---------------------------------------------------------------------------

def load_inception_surface(surface_type: str = "heston",
                            data_root:    str = None) -> tuple:
    paths    = resolve_paths(data_root)
    path_map = {"bs": paths["bs_surface"], "heston": paths["hst_surface"]}
    csv_path = path_map[surface_type]

    if not os.path.exists(csv_path):
        other    = "bs" if surface_type == "heston" else "heston"
        fallback = path_map[other]
        if os.path.exists(fallback):
            print(f"  WARNING: {surface_type} surface not found; falling back to {other}.")
            csv_path     = fallback
            surface_type = other
        else:
            raise FileNotFoundError(f"No surface CSV found. Tried: {csv_path} and {fallback}")

    df = pd.read_csv(csv_path)
    required = {"ticker", "maturity_years", "moneyness"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface CSV missing columns: {missing}")
    if "vol" not in df.columns and "implied_vol" not in df.columns:
        raise ValueError("Surface CSV must have a 'vol' or 'implied_vol' column.")

    print(f"  Surface [{surface_type}]: {len(df)} rows, "
          f"{df['ticker'].nunique()} tickers")
    return df, surface_type


def load_surface_from_path(csv_path: str) -> tuple:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Surface CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"ticker", "maturity_years", "moneyness"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Surface CSV missing columns: {missing}")
    if "vol" not in df.columns and "implied_vol" not in df.columns:
        raise ValueError("Surface CSV must have a 'vol' or 'implied_vol' column.")
    surface_type = "heston" if "heston" in os.path.basename(csv_path).lower() else "bs"
    print(f"  Surface from explicit path: {csv_path}  type={surface_type}")
    return df, surface_type


# ---------------------------------------------------------------------------
# Stage I — realised prices and live surface
# ---------------------------------------------------------------------------

def fetch_realised_prices(tickers: list = TICKERS,
                           start:   str  = INCEPTION_DATE) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance required for realised prices")

    inception_ts = pd.Timestamp(start)
    lookback_start = (inception_ts - pd.Timedelta(days=365)).strftime("%Y-%m-%d") #require 1 year lookback for daily calibration

    today = date.today().isoformat()
    print(f"  Fetching realised prices {lookback_start} to {today} ...")
    raw = yf.download(tickers, start=lookback_start, end=today,
                      auto_adjust=True, progress=False)["Close"]
    raw = raw[tickers].dropna()
    print(f"  Realised prices: {len(raw)} trading days")
    return raw


def fetch_live_surface(tickers: list = TICKERS,
                        r: float = RISK_FREE_RATE) -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance required for live option chain")

    today_str   = date.today().isoformat()
    expiry_sets = []
    tickers_obj = {}

    for t in tickers:
        tk = yf.Ticker(t)
        tickers_obj[t] = tk
        exps = set(tk.options)
        expiry_sets.append(exps)
        print(f"  [{t}] {len(exps)} expiries available")

    common = sorted(expiry_sets[0].intersection(*expiry_sets[1:]))
    common = [e for e in common if e > today_str]
    if not common:
        raise RuntimeError("No common expiries found across tickers")

    rows = []
    for t in tickers:
        tk   = tickers_obj[t]
        hist = tk.history(period="1d")
        if hist.empty:
            raise ValueError(f"Cannot fetch spot for {t}")
        S0 = float(hist["Close"].iloc[-1])

        for expiry in common:
            days  = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
            T_yrs = days / 365.0
            if T_yrs <= 0:
                continue
            try:
                chain = tk.option_chain(expiry)
            except Exception:
                continue

            calls = chain.calls.copy()
            puts  = chain.puts.copy()

            def mid(df):
                b, a = df["bid"].fillna(0), df["ask"].fillna(0)
                return np.where((b > 0) & (a > 0), (b + a) / 2, df["lastPrice"])

            calls["mid"] = mid(calls)
            puts["mid"]  = mid(puts)

            for K in sorted(set(calls["strike"]) | set(puts["strike"])):
                m = K / S0
                if not (0.50 <= m <= 1.80):
                    continue
                otype = "call" if m >= 1.0 else "put"
                src   = calls if otype == "call" else puts
                row   = src[src["strike"] == K]
                if row.empty:
                    continue
                price = float(row["mid"].iloc[0])
                iv    = _implied_vol(price, S0, K, T_yrs, r, otype)
                if np.isnan(iv) or iv <= 0:
                    continue
                rows.append(dict(ticker=t, valuation_date=today_str,
                                 maturity_years=round(T_yrs, 6),
                                 maturity_days=days, strike=K,
                                 moneyness=round(m, 6), spot_price=S0,
                                 implied_vol=iv, risk_free_rate=r))

    df = pd.DataFrame(rows)
    print(f"  Live surface: {len(df)} IV points")
    return df


def _fetch_closing_price(ticker: str, date_str: str) -> float:
    if not YFINANCE_AVAILABLE:
        return None
    target = pd.Timestamp(date_str)
    start  = (target - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end    = (target + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hist = yf.download(ticker, start=start, end=end,
                           progress=False, auto_adjust=True)
        if hist.empty:
            return None
        hist  = hist[hist.index <= target]
        close = hist["Close"].iloc[-1]
        if hasattr(close, "item"):
            close = close.item()
        return float(close)
    except Exception as e:
        print(f"    WARNING: could not fetch {ticker} close on {date_str}: {e}")
        return None


# ---------------------------------------------------------------------------
# Column aliases for historical option CSVs
# ---------------------------------------------------------------------------

_COL_ALIASES = {
    "impliedvolatility":  "vol", "implied_volatility": "vol",
    "implied_vol":        "vol", "iv":                 "vol",
    "volatility":         "vol",
    "expiration":         "expiration", "expiration_date": "expiration",
    "expiry":             "expiration", "expiry_date":     "expiration",
    "expdate":            "expiration", "maturity":        "expiration",
    "maturity_date":      "expiration", "maturitydate":    "expiration",
    "strikeprice":        "strike",     "exercise_price":  "strike",
    "call_put":           "call_put",   "type":            "call_put",
    "putcall":            "call_put",   "put_call":        "call_put",
    "cp_flag":            "call_put",   "option_type":     "call_put",
    "underlyingprice":    "spot_price", "underlying_price": "spot_price",
    "spot":               "spot_price",
    "act_symbol":         "ticker_col", "symbol":          "ticker_col",
    "underlying":         "ticker_col",
}


def _normalise_option_df(df_raw: pd.DataFrame,
                          ticker: str,
                          valuation_date: str = "2025-04-01",
                          r: float = RISK_FREE_RATE) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.rename(columns=_COL_ALIASES, inplace=True)

    val_date = pd.Timestamp(valuation_date)

    if "vol" not in df.columns:
        print(f"  [{ticker}] No vol column found. Available: {list(df.columns)}")
        return pd.DataFrame()
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")
    if df["vol"].median() > 1.0:
        df["vol"] = df["vol"] / 100.0
    df = df[df["vol"] > 0].dropna(subset=["vol"])

    if "maturity_years" in df.columns:
        df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
        df["maturity_days"]  = (df["maturity_years"] * 365).round().astype(int)
    elif "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        df = df.dropna(subset=["expiration"])
        df["maturity_days"]  = (df["expiration"] - val_date).dt.days
        df["maturity_years"] = df["maturity_days"] / 365.0
    else:
        print(f"  [{ticker}] No expiration/maturity column.")
        return pd.DataFrame()
    df = df[df["maturity_years"] > 0]

    if "strike" not in df.columns:
        print(f"  [{ticker}] No strike column.")
        return pd.DataFrame()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])

    if "spot_price" in df.columns and not df["spot_price"].isna().all():
        df["spot_price"] = pd.to_numeric(df["spot_price"], errors="coerce")
    else:
        S0 = _fetch_closing_price(ticker, valuation_date)
        if S0 is None:
            fallback_spots = {"NFLX": 930.52, "SPOT": 565.41, "DIS": 97.88}
            S0 = fallback_spots.get(ticker, INITIAL_FIXING_PRICES.get(ticker))
            if S0 is None:
                return pd.DataFrame()
        df["spot_price"] = float(S0)

    # split-adjustment detection
    median_raw_moneyness = df["strike"].median() / df["spot_price"].iloc[0]
    if not (0.25 <= median_raw_moneyness <= 4.0):
        candidates = [2, 3, 4, 5, 8, 10, 20, 25, 0.5, 0.25, 0.1, 0.2]
        best_ratio  = 1.0
        best_dist   = abs(np.log(median_raw_moneyness))
        for ratio in candidates:
            adjusted = median_raw_moneyness / ratio
            dist = abs(np.log(adjusted)) if adjusted > 0 else 1e9
            if dist < best_dist:
                best_dist  = dist
                best_ratio = ratio
        if best_ratio != 1.0:
            print(f"  [{ticker}] Split-adjustment detected: dividing strikes by {best_ratio:.0f}.")
            df["strike"] = df["strike"] / best_ratio

    df["moneyness"] = df["strike"] / df["spot_price"]
    df = df[(df["moneyness"] >= 0.50) & (df["moneyness"] <= 1.80)]

    if "call_put" in df.columns:
        df["_is_call"] = df["call_put"].str.strip().str.lower().str.startswith("c")
        df = df[((df["moneyness"] >= 1.0) &  df["_is_call"]) |
                ((df["moneyness"] <  1.0) & ~df["_is_call"])]
        df = df.drop(columns=["_is_call"])

    df["ticker"]         = ticker
    df["valuation_date"] = valuation_date
    df["risk_free_rate"] = r

    canonical = ["ticker", "valuation_date", "maturity_years", "maturity_days",
                 "strike", "moneyness", "spot_price", "vol", "risk_free_rate"]
    for col in canonical:
        if col not in df.columns:
            df[col] = np.nan
    return df[canonical].reset_index(drop=True)


def load_historical_surface_from_csv(data_root: str = None,
                                      tickers: list  = None,
                                      valuation_date: str = "2025-04-01",
                                      r: float       = RISK_FREE_RATE) -> pd.DataFrame:
    if tickers is None:
        tickers = TICKERS

    root = (data_root or os.environ.get("DATA_ROOT")
            or os.path.join(os.path.dirname(__file__), "..", "..", "final_dataset"))
    root = os.path.abspath(root)

    all_rows = []
    for t in tickers:
        ticker_dir = os.path.join(root, "historical_option", t)
        candidates = []
        if os.path.isdir(ticker_dir):
            csvs = sorted(
                [os.path.join(ticker_dir, f)
                 for f in os.listdir(ticker_dir) if f.endswith(".csv")],
                key=os.path.getmtime, reverse=True)
            candidates = csvs
        candidates += [
            os.path.join(root, "historical_option", f"{t}.csv"),
            os.path.join(root, "historical_option", f"{t.lower()}.csv"),
        ]

        loaded = False
        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                df_raw = pd.read_csv(p)
                df_t   = _normalise_option_df(df_raw, t, valuation_date=valuation_date, r=r)
                if df_t.empty:
                    continue
                all_rows.append(df_t)
                print(f"  [{t}] Historical option chain: {len(df_t)} IV points  (from {p})")
                loaded = True
                break
            except Exception as e:
                print(f"  [{t}] Failed to load {p}: {e}")
                continue

        if not loaded:
            print(f"  [{t}] No historical_option CSV found — will use synthetic surface.")

    if not all_rows:
        return pd.DataFrame()
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"  Historical surface (combined): {len(combined)} IV points")
    return combined


# ---------------------------------------------------------------------------
# Stage II — correlation & rolling vol
# ---------------------------------------------------------------------------

def estimate_correlation(log_returns: pd.DataFrame) -> np.ndarray:
    available = [t for t in TICKERS if t in log_returns.columns]
    corr      = log_returns[available].corr().values

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals  = np.maximum(eigvals, 1e-8)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d        = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)

    print(f"  Correlation matrix from {len(log_returns)} observations")
    print(pd.DataFrame(corr_psd, index=available, columns=available).round(3).to_string())
    return corr_psd


def compute_rolling_vol(realised_spots: pd.DataFrame,
                         window: int = ROLLING_VOL_WINDOW) -> pd.DataFrame:
    log_ret = np.log(realised_spots / realised_spots.shift(1))
    rolling = log_ret.rolling(window).std() * np.sqrt(252)
    rolling = rolling.dropna()
    print(f"  Rolling vol: {len(rolling)} dates, window={window}d")
    return rolling


# ---------------------------------------------------------------------------
# build_data_store — model-agnostic orchestrator
# ---------------------------------------------------------------------------

def build_data_store(simulator_class,
                     surface_type:  str  = "heston",
                     data_root:     str  = None,
                     use_live_data: bool = True) -> DataStore:
    """
    Run all Stage I and Stage II steps.

    Parameters
    ----------
    simulator_class : a class conforming to BaseSimulator (e.g. HestonSimulator,
                      GBMSimulator).  Used only for calibration; the DataStore
                      returned is fully model-agnostic.
    surface_type    : "heston" or "bs" — which synthetic surface to load.
    data_root       : path to final_dataset/.
    use_live_data   : False skips all yfinance calls.
    """
    print("\n" + "="*60)
    print("STAGE I — DATA COLLECTION")
    print("="*60)

    ds = DataStore()
    ds.surface_type = surface_type
    paths = resolve_paths(data_root)

    # log returns
    print("\n[Log returns — correlation window]")
    try:
        ds.log_returns = load_log_returns(paths["log_returns"])
    except FileNotFoundError as e:
        print(f"  WARNING: {e}\n  Will use hardcoded correlation fallback.")
        ds.log_returns = pd.DataFrame()

    # inception surface
    print("\n[Inception surface]")
    hist_surf = load_historical_surface_from_csv(data_root=data_root)

    if not hist_surf.empty:
        loaded_tickers  = set(hist_surf["ticker"].unique()) if "ticker" in hist_surf.columns else set()
        missing_tickers = [t for t in TICKERS if t not in loaded_tickers]

        if missing_tickers:
            print(f"  Tickers missing from historical_option/: {missing_tickers}")
            synthetic_surf, _ = load_inception_surface(surface_type, data_root)
            synth_rows = synthetic_surf[synthetic_surf["ticker"].isin(missing_tickers)] \
                if "ticker" in synthetic_surf.columns else pd.DataFrame()

            if not synth_rows.empty:
                if "implied_vol" in synth_rows.columns and "vol" not in synth_rows.columns:
                    synth_rows = synth_rows.rename(columns={"implied_vol": "vol"})
                if "implied_vol" in hist_surf.columns and "vol" not in hist_surf.columns:
                    hist_surf = hist_surf.rename(columns={"implied_vol": "vol"})
                ds.inception_surface = pd.concat([hist_surf, synth_rows], ignore_index=True)
                ds.surface_type = "mixed (historical + synthetic)"
            else:
                ds.inception_surface = hist_surf
                ds.surface_type = "historical"
        else:
            ds.inception_surface = hist_surf
            ds.surface_type = "historical"
    else:
        print(f"  historical_option/ not found — using synthetic {surface_type.upper()} surface")
        ds.inception_surface, ds.surface_type = load_inception_surface(surface_type, data_root)

    if use_live_data:
        print("\n[Realised prices — BRC life]")
        ds.realised_spots        = fetch_realised_prices()
        ds.initial_fixing_prices = INITIAL_FIXING_PRICES
        print(f"  Initial fixing prices (termsheet): {ds.initial_fixing_prices}")

        print("\n[Live option chain — today]")
        try:
            ds.live_surface = fetch_live_surface()
        except Exception as e:
            print(f"  WARNING: Live surface unavailable ({e})")
            ds.live_surface = pd.DataFrame()
    else:
        print("  [offline mode] No yfinance calls.")
        ds.live_surface          = ds.inception_surface.copy()
        ds.initial_fixing_prices = INITIAL_FIXING_PRICES

    print("\n" + "="*60)
    print("STAGE II — PARAMETER ESTIMATION")
    print("="*60)

    print("\n[Correlation matrix]")
    if not ds.log_returns.empty:
        ds.correlation_matrix = estimate_correlation(ds.log_returns)
    else:
        print("  Using hardcoded correlation.")
        ds.correlation_matrix = np.array([
            [1.000, 0.357, 0.352],
            [0.357, 1.000, 0.489],
            [0.352, 0.489, 1.000],
        ])

    print("\n[Rolling realised vol]")
    if not ds.realised_spots.empty:
        ds.rolling_vol = compute_rolling_vol(ds.realised_spots)
    else:
        print("  Unavailable in offline mode.")

    print("\n[Historical UBS BRC prices]")
    hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "../../historical_BRC_prices.txt")
    ds.historical_brc_prices = load_historical_brc_prices(hist_path)

    print("\n[US Treasury yield curve]")
    tsy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../../treasury_curve.csv")
    ds.treasury_curve = load_treasury_curve(tsy_path)

    print("\n[Historical implied volatility]")
    iv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../../historical_iv.csv")
    ds.historical_iv = load_historical_iv(iv_path)

    # ── Model calibration  (delegated to the active simulator class) ──
    model_name = getattr(simulator_class, "model_name",
                         lambda: simulator_class.__name__)()
    use_cf = "historical" in ds.surface_type

    print(f"\n[{model_name} calibration — inception surface ({ds.surface_type})]")
    calibrate_sig = inspect.signature(simulator_class.calibrate)
    supports_inception_gibbs = (
        "historical_price_df" in calibrate_sig.parameters
        and "is_inception" in calibrate_sig.parameters
        and not ds.realised_spots.empty
    )

    if supports_inception_gibbs:
        ds.model_params = simulator_class.calibrate(
            ds.inception_surface,
            TICKERS,
            historical_price_df=ds.realised_spots,
            is_inception=True,
        )
    else:
        ds.model_params = simulator_class.calibrate(ds.inception_surface, TICKERS)

    print(f"\n[{model_name} calibration — terminal (yfinance live chain)]")
    if not ds.live_surface.empty:
        if supports_inception_gibbs:
            ds.terminal_model_params = simulator_class.calibrate(
                ds.live_surface,
                TICKERS,
                historical_price_df=ds.realised_spots,
                is_inception=False,
            )
        else:
            ds.terminal_model_params = simulator_class.calibrate(ds.live_surface, TICKERS)
    else:
        ds.terminal_model_params = {t: dict(**ds.model_params[t]) for t in TICKERS}
        print("  Terminal params set equal to inception params.")

    print("\n[Data store complete]")
    return ds


if __name__ == "__main__":
    # Minimal smoke test without a real simulator
    class _FakeSimulator:
        @classmethod
        def calibrate(cls, surface_df, tickers, r=0.05):
            return {t: {"sigma": 0.30} for t in tickers}
        @classmethod
        def model_name(cls): return "Fake"

    ds = build_data_store(_FakeSimulator, surface_type="bs", use_live_data=False)
    print("\nModel params:", ds.model_params)
