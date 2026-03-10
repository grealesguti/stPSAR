"""
run_scheduler.py
────────────────
Standalone process — runs the daily refresh + Telegram alerts
independently of the Streamlit browser session.

Usage (inside Docker):
    python run_scheduler.py

Or via Docker CMD / supervisord alongside Streamlit.
"""

import time
import sys
import os
from pathlib import Path
from datetime import datetime

# ── Make sure app modules are importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scheduler import refresh_all_tickers, _load_alert_state
import schedule


# ════════════════════════════════════════════════════════════════════════════
# CONFIG  — mirrors app.py defaults, edit here or load from env vars
# ════════════════════════════════════════════════════════════════════════════

DATA_DIR     = Path(os.getenv("DATA_DIR",     "data/investments"))
REFRESH_TIME = os.getenv("REFRESH_TIME",      "18:00")       # HH:MM 24h local

# Tickers — reads tickers.txt if it exists, otherwise falls back to env / default
TICKER_FILE  = DATA_DIR / "tickers.txt"
DEFAULT_TICKERS = ["IWDA.AS", "AAPL", "MSFT"]

# Telegram — read from environment variables (preferred) or .streamlit/secrets.toml
def _get_secret(key: str) -> str:
    # 1. Environment variable (ideal for Docker)
    val = os.getenv(key.upper(), "")
    if val:
        return val
    # 2. .streamlit/secrets.toml fallback
    secrets_file = Path(".streamlit/secrets.toml")
    if secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            if line.strip().startswith(key):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return parts[1].strip().strip('"').strip("'")
    return ""

TG_TOKEN   = _get_secret("telegram_token")
TG_CHAT_ID = _get_secret("telegram_chat_id")

# Indicator parameters — same defaults as app.py DEFAULTS dict
PARAMS = {
    "psar_acceleration": float(os.getenv("PSAR_ACCEL",  "0.02")),
    "psar_maximum":      float(os.getenv("PSAR_MAX",    "0.20")),
    "vwap_window":       int(  os.getenv("VWAP_WIN",    "20")),
    "sma_fast":          int(  os.getenv("SMA_FAST",    "20")),
    "sma_slow":          int(  os.getenv("SMA_SLOW",    "50")),
    "rsi_period":        int(  os.getenv("RSI_PERIOD",  "14")),
    "rsi_overbought":    int(  os.getenv("RSI_OB",      "70")),
    "rsi_oversold":      int(  os.getenv("RSI_OS",      "30")),
    "bb_period":         int(  os.getenv("BB_PERIOD",   "20")),
    "bb_std":            float(os.getenv("BB_STD",      "2.0")),
    "macd_fast":         int(  os.getenv("MACD_FAST",   "12")),
    "macd_slow":         int(  os.getenv("MACD_SLOW",   "26")),
    "macd_signal":       int(  os.getenv("MACD_SIG",    "9")),
    "atr_period":        int(  os.getenv("ATR_PERIOD",  "14")),
}

# Alert config — mirrors sidebar defaults
ALERT_CONFIG = {
    "on_psar_flip":     os.getenv("ALERT_PSAR_FLIP",     "true").lower()  == "true",
    "on_macd_cross":    os.getenv("ALERT_MACD_CROSS",    "false").lower() == "true",
    "on_rsi_extreme":   os.getenv("ALERT_RSI_EXTREME",   "false").lower() == "true",
    "on_vwap_cross":    os.getenv("ALERT_VWAP_CROSS",    "false").lower() == "true",
    "on_daily_summary": os.getenv("ALERT_DAILY",         "true").lower()  == "true",
    "rsi_overbought":   int(os.getenv("RSI_OB",          "70")),
    "rsi_oversold":     int(os.getenv("RSI_OS",          "30")),
    "inc_price":        True,
    "inc_psar":         True,
    "inc_streak":       True,
    "inc_rsi":          True,
    "inc_vwap":         True,
    "inc_macd":         False,
}

# Watchlist — comma-separated env var, or None = all tickers
_watched_env    = os.getenv("WATCHED_TICKERS", "")
WATCHED_TICKERS = [t.strip().upper() for t in _watched_env.split(",") if t.strip()] or None


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_tickers() -> list:
    if TICKER_FILE.exists():
        lines = [l.strip().upper() for l in TICKER_FILE.read_text().splitlines() if l.strip()]
        if lines:
            return lines
    return DEFAULT_TICKERS


def fetch_ticker(ticker: str, force_update: bool = False):
    """
    Re-uses the same fetch logic as app.py.
    Import here to avoid circular imports at module level.
    """
    # Inline the fetch logic so this script has no dependency on Streamlit
    import pandas as pd
    from pathlib import Path as P

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{ticker}_history.csv"

    if csv_path.exists() and not force_update:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return _normalize(df)
        except Exception:
            pass

    # Try yahooquery first, then yfinance
    df = _fetch_yahooquery(ticker) or _fetch_yfinance(ticker)
    if df is None or df.empty:
        print(f"[Scheduler] ❌ No data for {ticker}")
        return pd.DataFrame()

    df = _normalize(df)
    if not df.empty:
        df.reset_index().to_csv(csv_path, index=False)
        print(f"[Scheduler] ✅ {ticker} — {len(df)} rows saved")
    return df


def _fetch_yahooquery(ticker: str):
    try:
        from yahooquery import Ticker as YQT
        import pandas as pd
        df = YQT(ticker).history(period="max")
        if df is None or df.empty:
            return None
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.reset_index(inplace=True)
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker].copy()
        return df
    except Exception as e:
        print(f"[Scheduler] yahooquery failed for {ticker}: {e}")
        return None


def _fetch_yfinance(ticker: str):
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(ticker, period="max", progress=False)
        if df.empty:
            return None
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date", "index": "date"}, inplace=True)
        return df
    except Exception as e:
        print(f"[Scheduler] yfinance failed for {ticker}: {e}")
        return None


def _normalize(df):
    import pandas as pd
    import numpy as np

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        except Exception:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
        df.set_index("date", inplace=True)

    try:
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["close"]).sort_index()


def calculate(df, params: dict):
    """
    Thin wrapper — imports calculate_indicators from app.py.
    Falls back to a stub if Streamlit import fails in headless mode.
    """
    try:
        # app.py imports streamlit at module level; suppress any UI side-effects
        import importlib, types, unittest.mock as mock
        with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
            app = importlib.import_module("app")
        return app.calculate_indicators(df, params)
    except Exception as e:
        print(f"[Scheduler] ⚠️  Could not import calculate_indicators: {e}")
        return df   # return raw df — PSAR columns will be missing


# ════════════════════════════════════════════════════════════════════════════
# SCHEDULED JOB
# ════════════════════════════════════════════════════════════════════════════

def job():
    tickers = load_tickers()
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M}] Running scheduled refresh for: {tickers}")

    if not TG_TOKEN or not TG_CHAT_ID:
        print("[Scheduler] ⚠️  Telegram not configured — alerts will be skipped.")
        print("            Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID env vars.")

    refresh_all_tickers(
        tickers         = tickers,
        data_dir        = DATA_DIR,
        fetch_fn        = fetch_ticker,
        calculate_fn    = calculate,
        params          = PARAMS,
        tg_token        = TG_TOKEN,
        tg_chat_id      = TG_CHAT_ID,
        alert_config    = ALERT_CONFIG,
        watched_tickers = WATCHED_TICKERS,
    )


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PSAR Standalone Scheduler")
    print("=" * 60)
    print(f"  Data dir     : {DATA_DIR.resolve()}")
    print(f"  Refresh time : {REFRESH_TIME} daily")
    print(f"  Telegram     : {'✅ configured' if TG_TOKEN else '❌ NOT SET'}")
    print(f"  Daily summary: {ALERT_CONFIG['on_daily_summary']}")
    print(f"  PSAR flip    : {ALERT_CONFIG['on_psar_flip']}")
    print(f"  Tickers      : {load_tickers()}")
    print("=" * 60)

    # Run once immediately on startup so you get a message right away
    print("\n▶ Running initial refresh on startup …")
    job()

    # Then schedule daily
    schedule.every().day.at(REFRESH_TIME).do(job)
    print(f"\n⏰ Scheduled daily job at {REFRESH_TIME}. Waiting …\n")

    while True:
        schedule.run_pending()
        time.sleep(30)