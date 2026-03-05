"""
Parabolic SAR Multi-Ticker Streamlit App — fully parameterized indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import datetime as dt
import warnings
from telegram_alerts import check_and_send_alerts
from scheduler import start_scheduler, get_last_refresh_time
warnings.filterwarnings('ignore')

# ============================================================================
# DEFAULTS  (single source of truth — change here to alter app-wide defaults)
# ============================================================================

DEFAULT_TICKERS = ["IWDA.AS", "AAPL", "MSFT"]
DATA_DIR        = Path("data/investments")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TICKER_FILE     = DATA_DIR / "tickers.txt"

DEFAULTS = {
    # Parabolic SAR
    "psar_acceleration": 0.02,
    "psar_maximum":      0.20,
    # VWAP
    "vwap_window":       20,
    # SMAs
    "sma_fast":          20,
    "sma_slow":          50,
    # RSI
    "rsi_period":        14,
    "rsi_overbought":    70,
    "rsi_oversold":      30,
    # Bollinger Bands
    "bb_period":         20,
    "bb_std":            2.0,
    # MACD
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":       9,
    # ATR
    "atr_period":        14,
    # Display
    "days_to_show":      180,
    "show_sma":          True,
    "show_bb":           False,
    "show_vwap":         True,
    "show_rsi":          True,
    "show_macd":         False,
}


# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

def load_tickers_from_txt() -> list:
    if not TICKER_FILE.exists():
        return DEFAULT_TICKERS
    try:
        with open(TICKER_FILE, "r") as f:
            lines = [l.strip().upper() for l in f if l.strip()]
        return lines if lines else DEFAULT_TICKERS
    except Exception as e:
        st.warning(f"Could not read ticker file: {e}")
        return DEFAULT_TICKERS


def save_tickers_to_txt(tickers: list):
    try:
        with open(TICKER_FILE, "w") as f:
            f.writelines(t + "\n" for t in tickers)
        st.success("Tickers saved to tickers.txt")
    except Exception as e:
        st.error(f"Could not save tickers: {e}")


def fetch_from_yahooquery(ticker: str) -> pd.DataFrame:
    try:
        from yahooquery import Ticker as YQTicker
        df = YQTicker(ticker).history(period="max")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].copy()
        df.reset_index(inplace=True)
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker].copy()
        return df
    except Exception as e:
        st.warning(f"YahooQuery failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_from_yfinance(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, period="max", progress=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date", "index": "date"}, inplace=True)
        return df
    except Exception as e:
        st.warning(f"yfinance failed for {ticker}: {e}")
        return pd.DataFrame()


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    def _strip_tz(series):
        def _coerce(val):
            try:
                ts = pd.Timestamp(val)
                return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
            except Exception:
                return pd.NaT
        return series.map(_coerce)

    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        except Exception:
            df["date"] = _strip_tz(df["date"].astype(str))
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
        df.set_index("date", inplace=True)

    try:
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        df.index = pd.DatetimeIndex([_strip_tz(pd.Series([v])).iloc[0] for v in df.index])

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["close"]).sort_index()


def load_or_fetch_ticker(ticker: str, force_update: bool = False) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{ticker}_history.csv"
    if csv_path.exists() and not force_update:
        try:
            df = _normalize_df(pd.read_csv(csv_path))
            if not df.empty:
                return df
        except Exception as e:
            st.warning(f"Cache read failed for {ticker}: {e}. Re-fetching.")

    df = fetch_from_yahooquery(ticker)
    if df.empty:
        df = fetch_from_yfinance(ticker)
    if df.empty:
        st.error(f"Could not fetch data for {ticker}.")
        return pd.DataFrame()

    try:
        df = _normalize_df(df)
    except Exception as e:
        st.error(f"Normalize failed for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        st.error(f"No valid data for {ticker} after normalization.")
        return pd.DataFrame()

    try:
        df.reset_index().to_csv(csv_path, index=False)
        st.success(f"Saved {len(df)} rows for {ticker}")
    except Exception as e:
        st.warning(f"Could not save cache for {ticker}: {e}")

    return df


@st.cache_data(ttl=86400)
def fetch_ticker_name(ticker: str) -> str:
    try:
        from yahooquery import Ticker as YQTicker
        info = YQTicker(ticker).price
        if isinstance(info, dict) and ticker in info:
            name = info[ticker].get("longName") or info[ticker].get("shortName", "")
            if name:
                return name
    except Exception:
        pass
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName", "")
        if name:
            return name
    except Exception:
        pass
    return ticker


# ============================================================================
# INDICATORS  (all params passed explicitly via dict)
# ============================================================================

def calculate_psar(df: pd.DataFrame, acceleration: float, maximum: float) -> pd.DataFrame:
    df    = df.copy()
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    try:
        import talib
        df["psar"] = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
        return df
    except ImportError:
        pass

    n    = len(close)
    psar = np.full(n, np.nan)
    bull = True
    af   = acceleration
    hp   = high[0]
    lp   = low[0]

    for i in range(2, n):
        if bull:
            psar[i] = (psar[i-1] + af * (hp - psar[i-1])) if not np.isnan(psar[i-1]) else low[i-1]
            if low[i] < psar[i]:
                bull, psar[i], lp, af = False, hp, low[i], acceleration
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + acceleration, maximum)
        else:
            psar[i] = (psar[i-1] - af * (psar[i-1] - lp)) if not np.isnan(psar[i-1]) else high[i-1]
            if high[i] > psar[i]:
                bull, psar[i], hp, af = True, lp, high[i], acceleration
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + acceleration, maximum)

    psar[0] = psar[1] = low[0]
    df["psar"] = psar
    return df


def calculate_vwap(df: pd.DataFrame, window: int) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (tp * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()
    return df


def calculate_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Calculate all indicators using parameter dict `p`."""
    df    = calculate_psar(df, acceleration=p["psar_acceleration"], maximum=p["psar_maximum"])
    df    = calculate_vwap(df, window=p["vwap_window"])
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)

    try:
        import talib
        df["sma_fast"]  = talib.SMA(close, timeperiod=p["sma_fast"])
        df["sma_slow"]  = talib.SMA(close, timeperiod=p["sma_slow"])
        df["rsi"]       = talib.RSI(close, timeperiod=p["rsi_period"])
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(
            close, timeperiod=p["bb_period"], nbdevup=p["bb_std"], nbdevdn=p["bb_std"])
        df["macd"], df["macd_signal_line"], df["macd_hist"] = talib.MACD(
            close, fastperiod=p["macd_fast"], slowperiod=p["macd_slow"], signalperiod=p["macd_signal"])
        df["atr"] = talib.ATR(high, low, close, timeperiod=p["atr_period"])
    except ImportError:
        # ── pure-python fallbacks ─────────────────────────────────────────
        df["sma_fast"] = df["close"].rolling(p["sma_fast"]).mean()
        df["sma_slow"] = df["close"].rolling(p["sma_slow"]).mean()

        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(p["rsi_period"]).mean()
        loss  = (-delta.clip(upper=0)).rolling(p["rsi_period"]).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        roll               = df["close"].rolling(p["bb_period"])
        df["bb_middle"]    = roll.mean()
        df["bb_upper"]     = df["bb_middle"] + p["bb_std"] * roll.std()
        df["bb_lower"]     = df["bb_middle"] - p["bb_std"] * roll.std()

        ema_fast                 = df["close"].ewm(span=p["macd_fast"],   adjust=False).mean()
        ema_slow                 = df["close"].ewm(span=p["macd_slow"],   adjust=False).mean()
        df["macd"]               = ema_fast - ema_slow
        df["macd_signal_line"]   = df["macd"].ewm(span=p["macd_signal"], adjust=False).mean()
        df["macd_hist"]          = df["macd"] - df["macd_signal_line"]

        hl        = df["high"] - df["low"]
        hc        = (df["high"] - df["close"].shift()).abs()
        lc        = (df["low"]  - df["close"].shift()).abs()
        df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(p["atr_period"]).mean()

    return df


# ============================================================================
# SIGNALS
# ============================================================================

def get_psar_signal(df: pd.DataFrame) -> str:
    if df.empty or "psar" not in df.columns:
        return "N/A"
    last = df.iloc[-1]
    if pd.isna(last["psar"]):
        return "N/A"
    return "🟢 BULLISH" if last["psar"] < last["close"] else "🔴 BEARISH"


def get_signal_streak(df: pd.DataFrame) -> int:
    if df.empty or "psar" not in df.columns:
        return 0
    direction = df["psar"] < df["close"]
    current   = direction.iloc[-1]
    streak    = 0
    for val in reversed(direction.values):
        if val == current:
            streak += 1
        else:
            break
    return streak


# ============================================================================
# PLOTTING
# ============================================================================

def plot_psar_chart(
    df: pd.DataFrame,
    label: str,
    days_to_show: int,
    p: dict,
    show_sma:  bool,
    show_bb:   bool,
    show_vwap: bool,
    show_rsi:  bool,
    show_macd: bool,
) -> go.Figure:

    cutoff  = df.index.max() - pd.Timedelta(days=days_to_show)
    plot_df = df[df.index >= cutoff].copy()
    if plot_df.empty:
        return go.Figure()

    rows, row_heights, subplot_titles = 1, [0.55], [label]
    if show_rsi:
        rows += 1; row_heights.append(0.22); subplot_titles.append(f"RSI ({p['rsi_period']})")
    if show_macd:
        rows += 1; row_heights.append(0.23); subplot_titles.append("MACD")

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    # candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"],   close=plot_df["close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # PSAR
    bear = plot_df[plot_df["psar"] >= plot_df["close"]]
    bull = plot_df[plot_df["psar"] <  plot_df["close"]]
    fig.add_trace(go.Scatter(x=bear.index, y=bear["psar"], mode="markers",
        marker=dict(color="#ef5350", size=5), name="PSAR Bear"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bull.index, y=bull["psar"], mode="markers",
        marker=dict(color="#26a69a", size=5), name="PSAR Bull"), row=1, col=1)

    # SMAs
    if show_sma:
        if "sma_fast" in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_fast"],
                mode="lines", line=dict(color="#FF9800", width=1.2),
                name=f"SMA {p['sma_fast']}"), row=1, col=1)
        if "sma_slow" in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_slow"],
                mode="lines", line=dict(color="#2196F3", width=1.2),
                name=f"SMA {p['sma_slow']}"), row=1, col=1)

    # VWAP
    if show_vwap and "vwap" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap"],
            mode="lines", line=dict(color="#E91E63", width=1.5, dash="dashdot"),
            name=f"VWAP ({p['vwap_window']}d)"), row=1, col=1)

    # Bollinger Bands
    if show_bb and "bb_upper" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"],
            mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
            name=f"BB ({p['bb_period']}, {p['bb_std']}σ)", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"],
            mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.07)",
            name="BB Lower", showlegend=False), row=1, col=1)

    # RSI
    if show_rsi and "rsi" in plot_df.columns:
        rsi_row = 2
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"],
            mode="lines", line=dict(color="#9C27B0", width=1.5),
            name=f"RSI ({p['rsi_period']})"), row=rsi_row, col=1)
        fig.add_hline(y=p["rsi_overbought"], line=dict(color="#ef5350", dash="dash", width=1), row=rsi_row, col=1)
        fig.add_hline(y=p["rsi_oversold"],   line=dict(color="#26a69a", dash="dash", width=1), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    # MACD
    if show_macd and "macd" in plot_df.columns:
        macd_row = 3 if show_rsi else 2
        colors   = ["#26a69a" if v >= 0 else "#ef5350" for v in plot_df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["macd_hist"],
            marker_color=colors, name="MACD Hist", showlegend=False), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd"],
            mode="lines", line=dict(color="#2196F3", width=1.2),
            name=f"MACD ({p['macd_fast']}/{p['macd_slow']}/{p['macd_signal']})"), row=macd_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal_line"],
            mode="lines", line=dict(color="#FF9800", width=1.2),
            name="Signal"), row=macd_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600 if rows == 1 else 750,
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


# ============================================================================
# APP LAYOUT
# ============================================================================

st.set_page_config(page_title="📈 PSAR Analyzer", page_icon="📈", layout="wide")
st.title("📈 Parabolic SAR Multi-Ticker Analyzer")
st.markdown("Analyze PSAR signals across multiple tickers. All indicator parameters are configurable from the sidebar.")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── Tickers ──────────────────────────────────────────────────────────────
    st.subheader("📌 Tickers")
    if "loaded_tickers" not in st.session_state:
        st.session_state.loaded_tickers = load_tickers_from_txt()

    ticker_input = st.text_area("One ticker per line",
                                value="\n".join(st.session_state.loaded_tickers), height=120)
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]

    colA, colB = st.columns(2)
    with colA:
        if st.button("💾 Save"):
            save_tickers_to_txt(tickers)
            st.session_state.loaded_tickers = tickers
    with colB:
        if st.button("🔄 Reload"):
            st.session_state.loaded_tickers = load_tickers_from_txt()
            st.rerun()

    preset = st.selectbox("Load preset", ["Custom", "US Tech", "EU ETFs", "Crypto"])
    if preset == "US Tech":   tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    elif preset == "EU ETFs": tickers = ["IWDA.AS", "EMIM.AS", "VUSA.AS", "CSPX.AS", "EQQQ.AS"]
    elif preset == "Crypto":  tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

    st.divider()

    # ── Indicator Parameters ─────────────────────────────────────────────────
    st.subheader("🔬 Indicator Parameters")

    with st.expander("📐 Parabolic SAR", expanded=True):
        psar_acceleration = st.number_input(
            "Acceleration step", min_value=0.001, max_value=0.1, step=0.001,
            value=DEFAULTS["psar_acceleration"], format="%.3f",
            help="How quickly SAR catches up to price. Lower = slower/wider. Default: 0.02",
        )
        psar_maximum = st.number_input(
            "Maximum acceleration", min_value=0.05, max_value=0.5, step=0.01,
            value=DEFAULTS["psar_maximum"], format="%.2f",
            help="Cap on the acceleration factor. Default: 0.20",
        )
        if st.button("↩ Reset", key="reset_psar"):
            psar_acceleration = DEFAULTS["psar_acceleration"]
            psar_maximum      = DEFAULTS["psar_maximum"]

    with st.expander("📊 Moving Averages (SMA)", expanded=False):
        show_sma = st.checkbox("Show SMAs", value=DEFAULTS["show_sma"])
        sma_fast = st.slider("Fast period", 5, 100, DEFAULTS["sma_fast"],
                             help="Short-term SMA. Default: 20")
        sma_slow = st.slider("Slow period", 10, 300, DEFAULTS["sma_slow"],
                             help="Long-term SMA. Default: 50")
        if st.button("↩ Reset", key="reset_sma"):
            sma_fast = DEFAULTS["sma_fast"]
            sma_slow = DEFAULTS["sma_slow"]

    with st.expander("📉 VWAP", expanded=False):
        show_vwap   = st.checkbox("Show VWAP", value=DEFAULTS["show_vwap"])
        vwap_window = st.slider("Rolling window (days)", 5, 60, DEFAULTS["vwap_window"],
                                help="Days for rolling VWAP. Default: 20")
        if st.button("↩ Reset", key="reset_vwap"):
            vwap_window = DEFAULTS["vwap_window"]

    with st.expander("〽️ RSI", expanded=False):
        show_rsi       = st.checkbox("Show RSI", value=DEFAULTS["show_rsi"])
        rsi_period     = st.slider("Period", 2, 50, DEFAULTS["rsi_period"],
                                   help="RSI lookback. Default: 14")
        rsi_overbought = st.slider("Overbought level", 50, 95, DEFAULTS["rsi_overbought"],
                                   help="Upper threshold. Default: 70")
        rsi_oversold   = st.slider("Oversold level", 5, 50, DEFAULTS["rsi_oversold"],
                                   help="Lower threshold. Default: 30")
        if st.button("↩ Reset", key="reset_rsi"):
            rsi_period     = DEFAULTS["rsi_period"]
            rsi_overbought = DEFAULTS["rsi_overbought"]
            rsi_oversold   = DEFAULTS["rsi_oversold"]

    with st.expander("🎯 Bollinger Bands", expanded=False):
        show_bb   = st.checkbox("Show Bollinger Bands", value=DEFAULTS["show_bb"])
        bb_period = st.slider("Period", 5, 100, DEFAULTS["bb_period"],
                              help="Rolling window for mean/std. Default: 20")
        bb_std    = st.select_slider("Std deviations", options=[1.0, 1.5, 2.0, 2.5, 3.0],
                                     value=DEFAULTS["bb_std"],
                                     help="Band width in std devs. Default: 2.0")
        if st.button("↩ Reset", key="reset_bb"):
            bb_period = DEFAULTS["bb_period"]
            bb_std    = DEFAULTS["bb_std"]

    with st.expander("📈 MACD", expanded=False):
        show_macd     = st.checkbox("Show MACD", value=DEFAULTS["show_macd"])
        macd_fast     = st.slider("Fast EMA", 2, 50, DEFAULTS["macd_fast"],
                                  help="Fast EMA period. Default: 12")
        macd_slow     = st.slider("Slow EMA", 5, 200, DEFAULTS["macd_slow"],
                                  help="Slow EMA period. Default: 26")
        macd_signal_p = st.slider("Signal line", 2, 50, DEFAULTS["macd_signal"],
                                  help="Signal smoothing. Default: 9")
        if st.button("↩ Reset", key="reset_macd"):
            macd_fast     = DEFAULTS["macd_fast"]
            macd_slow     = DEFAULTS["macd_slow"]
            macd_signal_p = DEFAULTS["macd_signal"]

    with st.expander("📐 ATR", expanded=False):
        atr_period = st.slider("ATR period", 2, 50, DEFAULTS["atr_period"],
                               help="Average True Range lookback. Default: 14")
        if st.button("↩ Reset", key="reset_atr"):
            atr_period = DEFAULTS["atr_period"]

    st.divider()

    # ── Display ──────────────────────────────────────────────────────────────
    st.subheader("🖥️ Display")
    days_to_show = st.slider("Days to display", 30, 500, DEFAULTS["days_to_show"])

    st.divider()

    # ── Telegram ─────────────────────────────────────────────────────────────
    from telegram_alerts import (
        send_telegram_message, load_telegram_config, save_telegram_config,
    )

    st.subheader("🔔 Telegram Alerts")
    if "tg_loaded" not in st.session_state:
        cfg = load_telegram_config()
        st.session_state.update({
            "tg_token":   cfg["token"],
            "tg_chat_id": cfg["chat_id"],
            "tg_flip":    cfg["alert_on_flip"],
            "tg_daily":   cfg["alert_on_signal"],
            "tg_watched": ",".join(cfg["watched_tickers"]),
            "tg_loaded":  True,
        })

    tg_token        = st.text_input("Bot Token", type="password", value=st.session_state["tg_token"])
    tg_chat_id      = st.text_input("Chat ID",                    value=st.session_state["tg_chat_id"])
    alert_on_flip   = st.checkbox("🔁 Alert on signal flip",      value=st.session_state["tg_flip"])
    alert_on_signal = st.checkbox("📅 Daily signal summary",      value=st.session_state["tg_daily"])
    watched_input   = st.text_input("Watchlist (blank = all)",
                                    placeholder="e.g. AAPL, IWDA.AS",
                                    value=st.session_state["tg_watched"])
    watched_tickers = (
        [t.strip().upper() for t in watched_input.split(",") if t.strip()]
        if watched_input else None
    )
    st.session_state.update({
        "tg_token": tg_token, "tg_chat_id": tg_chat_id,
        "tg_flip": alert_on_flip, "tg_daily": alert_on_signal,
        "tg_watched": watched_input,
    })

    colT1, colT2 = st.columns(2)
    with colT1:
        if st.button("💾 Save config"):
            save_telegram_config(token=tg_token, chat_id=tg_chat_id,
                                 alert_on_flip=alert_on_flip, alert_on_signal=alert_on_signal,
                                 watched_tickers=watched_tickers or [])
            st.success("Saved!")
    with colT2:
        if st.button("🔄 Reload config"):
            cfg = load_telegram_config()
            st.session_state.update({
                "tg_token": cfg["token"], "tg_chat_id": cfg["chat_id"],
                "tg_flip": cfg["alert_on_flip"], "tg_daily": cfg["alert_on_signal"],
                "tg_watched": ",".join(cfg["watched_tickers"]),
            })
            st.rerun()

    if st.button("📨 Test connection"):
        if not tg_token or not tg_chat_id:
            st.error("⚠️ Enter Bot Token and Chat ID first.")
        else:
            with st.spinner("Sending..."):
                ok, err = send_telegram_message(tg_token, tg_chat_id, "✅ PSAR Analyzer connected!")
            st.success("✅ Sent!") if ok else st.error(f"❌ {err}")

    st.divider()

    # ── Data management ──────────────────────────────────────────────────────
    st.subheader("💾 Data")
    refresh_single = st.selectbox("Refresh single ticker", [""] + tickers, label_visibility="collapsed")
    if st.button("🔄 Refresh selected", disabled=not refresh_single):
        st.session_state["force_refresh_tickers"] = [refresh_single]
        st.rerun()
    if st.button("🔄 Refresh ALL"):
        st.session_state["force_refresh_tickers"] = tickers
        st.rerun()

    data_folder = st.text_input("Data folder", str(DATA_DIR))
    DATA_DIR    = Path(data_folder)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.button("🗑️ Clear cache"):
        for f in DATA_DIR.glob("*_history.csv"):
            f.unlink()
        st.success("Cache cleared!")

    st.divider()
    st.caption(f"Cache: `{DATA_DIR.resolve()}`")

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    st.subheader("⏰ Auto-Refresh")
    st.caption(f"Last auto-refresh: **{get_last_refresh_time(DATA_DIR)}**")
    refresh_time_input = st.time_input("Daily refresh time", value=dt.time(18, 0))
    now      = dt.datetime.now()
    next_run = now.replace(hour=refresh_time_input.hour, minute=refresh_time_input.minute, second=0, microsecond=0)
    if next_run < now:
        next_run += dt.timedelta(days=1)
    delta = next_run - now
    h, rem = divmod(int(delta.total_seconds()), 3600)
    st.info(f"⏳ Next refresh in: **{h}h {rem // 60}m**")


# ============================================================================
# BUILD PARAMETER DICT  (after all sidebar widgets are rendered)
# ============================================================================

params = {
    "psar_acceleration": psar_acceleration,
    "psar_maximum":      psar_maximum,
    "vwap_window":       vwap_window,
    "sma_fast":          sma_fast,
    "sma_slow":          sma_slow,
    "rsi_period":        rsi_period,
    "rsi_overbought":    rsi_overbought,
    "rsi_oversold":      rsi_oversold,
    "bb_period":         bb_period,
    "bb_std":            bb_std,
    "macd_fast":         macd_fast,
    "macd_slow":         macd_slow,
    "macd_signal":       macd_signal_p,
    "atr_period":        atr_period,
}

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not tickers:
    st.warning("Please enter at least one ticker symbol in the sidebar.")
    st.stop()

start_scheduler(tickers=tickers, data_dir=DATA_DIR,
                fetch_fn=load_or_fetch_ticker, refresh_time="18:00")

with st.spinner(f"Loading data for {len(tickers)} ticker(s)..."):
    data         = {}
    ticker_names = {}
    failed       = []
    force_refresh_list = st.session_state.pop("force_refresh_tickers", [])

    for ticker in tickers:
        should_force = ticker in force_refresh_list
        df = load_or_fetch_ticker(ticker, force_update=should_force)
        if not df.empty:
            df = calculate_indicators(df, params)
            data[ticker]         = df
            ticker_names[ticker] = fetch_ticker_name(ticker)
            if should_force:
                st.toast(f"✅ {ticker} refreshed", icon="🔄")
        else:
            failed.append(ticker)

st.session_state["data"] = data

if failed:
    st.error(f"Could not load: {', '.join(failed)}")
if not data:
    st.stop()

if st.session_state.get("tg_token") and st.session_state.get("tg_chat_id"):
    check_and_send_alerts(
        data=data, token=st.session_state["tg_token"],
        chat_id=st.session_state["tg_chat_id"],
        alert_on_flip=alert_on_flip, alert_on_signal=alert_on_signal,
        watched_tickers=watched_tickers,
    )

# ============================================================================
# SUMMARY TABLE
# ============================================================================

st.subheader("📋 PSAR Signal Summary")

summary_rows = []
for ticker, df in data.items():
    if df.empty:
        continue
    last      = df.iloc[-1]
    change_1d = ((last["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100) if len(df) > 1 else 0
    change_5d = ((last["close"] - df.iloc[-6]["close"]) / df.iloc[-6]["close"] * 100) if len(df) > 5 else 0

    summary_rows.append({
        "Ticker":         ticker,
        "Name":           ticker_names.get(ticker, ticker),
        "Last Close":     f"{last['close']:.2f}",
        "1D Change":      f"{change_1d:+.2f}%",
        "5D Change":      f"{change_5d:+.2f}%",
        "PSAR Signal":    get_psar_signal(df),
        "Streak (days)":  get_signal_streak(df),
        "RSI":            f"{last['rsi']:.1f}"  if "rsi"  in df.columns and pd.notna(last.get("rsi"))  else "N/A",
        "ATR":            f"{last['atr']:.2f}"  if "atr"  in df.columns and pd.notna(last.get("atr"))  else "N/A",
        "VWAP":           f"{last['vwap']:.2f}" if "vwap" in df.columns and pd.notna(last.get("vwap")) else "N/A",
        "vs VWAP":        (f"{((last['close']-last['vwap'])/last['vwap']*100):+.1f}%"
                           if "vwap" in df.columns and pd.notna(last.get("vwap")) else "N/A"),
        "Data Points":    len(df),
    })

if summary_rows:
    col_order = ["Ticker", "Name", "Last Close", "1D Change", "5D Change",
                 "PSAR Signal", "Streak (days)", "RSI", "ATR", "VWAP", "vs VWAP", "Data Points"]
    st.dataframe(pd.DataFrame(summary_rows)[col_order], use_container_width=True, hide_index=True)

st.divider()

# ── Active parameter summary ─────────────────────────────────────────────────
with st.expander("🔬 Active indicator parameters", expanded=False):
    pc = st.columns(4)
    pc[0].markdown(f"**PSAR**  \nAccel: `{params['psar_acceleration']}`  \nMax: `{params['psar_maximum']}`")
    pc[1].markdown(f"**SMA** fast/slow: `{params['sma_fast']}` / `{params['sma_slow']}`  \n"
                   f"**VWAP** window: `{params['vwap_window']}d`")
    pc[2].markdown(f"**RSI** period: `{params['rsi_period']}`  \n"
                   f"OB/OS: `{params['rsi_overbought']}` / `{params['rsi_oversold']}`  \n"
                   f"**ATR** period: `{params['atr_period']}`")
    pc[3].markdown(f"**BB** `{params['bb_period']}` · `{params['bb_std']}σ`  \n"
                   f"**MACD** `{params['macd_fast']}`/`{params['macd_slow']}`/`{params['macd_signal']}`")

st.divider()

# ============================================================================
# INDIVIDUAL CHARTS
# ============================================================================

st.subheader("📈 Parabolic SAR Charts")
view_mode   = st.radio("Chart layout", ["Single column", "Two columns"], horizontal=True)
ticker_list = list(data.keys())

chart_kwargs = dict(
    days_to_show=days_to_show, p=params,
    show_sma=show_sma, show_bb=show_bb,
    show_vwap=show_vwap, show_rsi=show_rsi, show_macd=show_macd,
)

if view_mode == "Two columns" and len(ticker_list) > 1:
    for i in range(0, len(ticker_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(ticker_list):
                t    = ticker_list[i + j]
                name = ticker_names.get(t, "")
                with col:
                    st.markdown(f"**{t}** · {name} — {get_psar_signal(data[t])}")
                    fig = plot_psar_chart(data[t], f"{t} · {name}", **chart_kwargs)
                    st.plotly_chart(fig, use_container_width=True)
else:
    tab_labels = [f"{t} · {ticker_names.get(t, t)}" for t in ticker_list]
    tabs       = st.tabs(tab_labels) if len(ticker_list) > 1 else [st.container()]

    for i, t in enumerate(ticker_list):
        df   = data[t]
        name = ticker_names.get(t, "")
        with (tabs[i] if len(ticker_list) > 1 else tabs[0]):
            last       = df.iloc[-1]
            prev       = df.iloc[-2] if len(df) > 1 else last
            change     = last["close"] - prev["close"]
            change_pct = (change / prev["close"]) * 100 if prev["close"] != 0 else 0

            st.caption(name)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Close",  f"{last['close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            m2.metric("PSAR Signal", get_psar_signal(df))
            m3.metric("Streak",      f"{get_signal_streak(df)} days")
            m4.metric("RSI",
                      f"{last['rsi']:.1f}" if "rsi" in df.columns and pd.notna(last.get("rsi")) else "N/A")

            fig = plot_psar_chart(df, f"{t} · {name}", **chart_kwargs)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"📄 Raw data — {t}"):
                show_cols = [c for c in
                             ["open", "high", "low", "close", "volume", "psar", "vwap",
                              "sma_fast", "sma_slow", "rsi", "bb_upper", "bb_lower", "macd", "atr"]
                             if c in df.columns]
                display_df       = df[show_cols].tail(100).copy()
                display_df.index = display_df.index.strftime("%Y-%m-%d")
                display_df       = display_df.rename(columns={
                    "sma_fast": f"sma_{params['sma_fast']}",
                    "sma_slow": f"sma_{params['sma_slow']}",
                })
                st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)
                csv = df.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(f"⬇️ Download {t} CSV", data=csv,
                                   file_name=f"{t}_psar_data.csv", mime="text/csv")

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=300_000, key="auto_ui_refresh")

st.divider()
st.caption("📁 Data cached in `data/investments/`. Use sidebar to force-refresh.")
st.caption("⚠️ For educational purposes only. Not financial advice.")