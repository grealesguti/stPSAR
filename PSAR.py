"""
Parabolic SAR Multi-Ticker Streamlit App
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
# CONFIGURATION
# ============================================================================

DAYS_BACK = 365
DEFAULT_TICKERS = ["IWDA.AS", "AAPL", "MSFT"]
DATA_DIR = Path("data/investments")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PSAR parameters
PSAR_ACCELERATION = 0.02
PSAR_MAXIMUM = 0.2

# ============================================================================
# DATA FETCHING & CACHING
# ============================================================================

TICKER_FILE = DATA_DIR / "tickers.txt"

def load_tickers_from_txt() -> list:
    if not TICKER_FILE.exists():
        return DEFAULT_TICKERS
    try:
        with open(TICKER_FILE, "r") as f:
            lines = [line.strip().upper() for line in f.readlines() if line.strip()]
        return lines if lines else DEFAULT_TICKERS
    except Exception as e:
        st.warning(f"Could not read ticker file: {e}")
        return DEFAULT_TICKERS


def save_tickers_to_txt(tickers: list):
    try:
        with open(TICKER_FILE, "w") as f:
            for t in tickers:
                f.write(t + "\n")
        st.success("Tickers saved to tickers.txt")
    except Exception as e:
        st.error(f"Could not save tickers: {e}")


def fetch_from_yahooquery(ticker: str) -> pd.DataFrame:
    try:
        from yahooquery import Ticker as YQTicker
        t_obj = YQTicker(ticker)
        df = t_obj.history(period="max")
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.reset_index(inplace=True)
        # Handle multi-level index from yahooquery
        if "symbol" in df.columns:
            df = df[df["symbol"] == ticker].copy()
        df.rename(columns={"date": "date"}, inplace=True)
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
def load_or_fetch_ticker(ticker: str, force_update: bool = False) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{ticker}_history.csv"

    if csv_path.exists() and not force_update:
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.dropna(subset=["date", "close"])
            df = df.sort_values("date")
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index, errors="coerce")  # ← ensure DatetimeIndex
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            st.warning(f"Could not read cache for {ticker}: {e}")

    df = fetch_from_yahooquery(ticker)
    if df.empty:
        df = fetch_from_yfinance(ticker)

    if df.empty:
        st.error(f"❌ Could not fetch data for {ticker}")
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index, errors="coerce")  # ← ensure DatetimeIndex
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.reset_index().to_csv(csv_path, index=False)
    st.success(f"✅ Saved {len(df)} rows for {ticker}")

    return df

# ============================================================================
# INDICATORS
# ============================================================================

def calculate_psar(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Parabolic SAR using talib if available, else pure Python fallback."""
    df = df.copy()
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    try:
        import talib
        df["psar"] = talib.SAR(high, low, acceleration=PSAR_ACCELERATION, maximum=PSAR_MAXIMUM)
        return df
    except ImportError:
        pass

    # Pure Python PSAR fallback
    n = len(close)
    psar = np.full(n, np.nan)
    bull = True
    af = PSAR_ACCELERATION
    ep = low[0]
    hp = high[0]
    lp = low[0]

    for i in range(2, n):
        if bull:
            psar[i] = psar[i-1] + af * (hp - psar[i-1]) if not np.isnan(psar[i-1]) else low[i-1]
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low[i]
                af = PSAR_ACCELERATION
                ep = low[i]
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + PSAR_ACCELERATION, PSAR_MAXIMUM)
                ep = hp
        else:
            psar[i] = psar[i-1] - af * (psar[i-1] - lp) if not np.isnan(psar[i-1]) else high[i-1]
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high[i]
                af = PSAR_ACCELERATION
                ep = high[i]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + PSAR_ACCELERATION, PSAR_MAXIMUM)
                ep = lp

    psar[0] = low[0]
    psar[1] = low[0]
    df["psar"] = psar
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add PSAR and optional indicators."""
    df = calculate_psar(df)

    # Try talib for extras
    try:
        import talib
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        df["sma20"] = talib.SMA(close, timeperiod=20)
        df["sma50"] = talib.SMA(close, timeperiod=50)
        df["rsi"] = talib.RSI(close, timeperiod=14)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(close, 12, 26, 9)
        df["atr"] = talib.ATR(high, low, close, timeperiod=14)
    except ImportError:
        # Minimal fallback: just rolling SMA
        df["sma20"] = df["close"].rolling(20).mean()
        df["sma50"] = df["close"].rolling(50).mean()

    return df


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def get_psar_signal(df: pd.DataFrame) -> str:
    """Return latest PSAR signal."""
    if df.empty or "psar" not in df.columns:
        return "N/A"
    last = df.iloc[-1]
    if pd.isna(last["psar"]):
        return "N/A"
    return "🟢 BULLISH" if last["psar"] < last["close"] else "🔴 BEARISH"


def get_signal_streak(df: pd.DataFrame) -> int:
    """How many consecutive days in current PSAR direction."""
    if df.empty or "psar" not in df.columns:
        return 0
    direction = df["psar"] < df["close"]
    streak = 0
    current = direction.iloc[-1]
    for val in reversed(direction.values):
        if val == current:
            streak += 1
        else:
            break
    return streak


# ============================================================================
# PLOTTING
# ============================================================================

def plot_psar_chart(df: pd.DataFrame, ticker: str, days_to_show: int = 180,
                    show_sma: bool = True, show_bb: bool = False,
                    show_rsi: bool = True, show_macd: bool = False) -> go.Figure:
    """Create a full PSAR chart with optional indicators."""

    # Filter to recent period
    cutoff = df.index.max() - pd.Timedelta(days=days_to_show)
    plot_df = df[df.index >= cutoff].copy()

    if plot_df.empty:
        return go.Figure()

    # Determine subplot rows
    rows = 1
    row_heights = [0.55]
    subplot_titles = [ticker]

    if show_rsi:
        rows += 1
        row_heights.append(0.22)
        subplot_titles.append("RSI (14)")

    if show_macd:
        rows += 1
        row_heights.append(0.23)
        subplot_titles.append("MACD")

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles
    )

    # --- Candlestick ---
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"],
        high=plot_df["high"],
        low=plot_df["low"],
        close=plot_df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        showlegend=True
    ), row=1, col=1)

    # --- PSAR dots ---
    psar_bear = plot_df[plot_df["psar"] >= plot_df["close"]]
    psar_bull = plot_df[plot_df["psar"] < plot_df["close"]]

    fig.add_trace(go.Scatter(
        x=psar_bear.index, y=psar_bear["psar"],
        mode="markers",
        marker=dict(color="#ef5350", size=5, symbol="circle"),
        name="PSAR Bearish"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=psar_bull.index, y=psar_bull["psar"],
        mode="markers",
        marker=dict(color="#26a69a", size=5, symbol="circle"),
        name="PSAR Bullish"
    ), row=1, col=1)

    # --- Optional SMAs ---
    if show_sma and "sma20" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["sma20"],
            mode="lines", line=dict(color="#FF9800", width=1.2),
            name="SMA 20"
        ), row=1, col=1)

    if show_sma and "sma50" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["sma50"],
            mode="lines", line=dict(color="#2196F3", width=1.2),
            name="SMA 50"
        ), row=1, col=1)

    # --- Bollinger Bands ---
    if show_bb and "bb_upper" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["bb_upper"],
            mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
            name="BB Upper", showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["bb_lower"],
            mode="lines", line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,150,0.07)",
            name="BB Lower", showlegend=False
        ), row=1, col=1)

    # --- RSI ---
    if show_rsi and "rsi" in plot_df.columns:
        rsi_row = 2
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["rsi"],
            mode="lines", line=dict(color="#9C27B0", width=1.5),
            name="RSI"
        ), row=rsi_row, col=1)

        fig.add_hline(y=70, line=dict(color="#ef5350", dash="dash", width=1), row=rsi_row, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", dash="dash", width=1), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    # --- MACD ---
    if show_macd and "macd" in plot_df.columns:
        macd_row = 3 if show_rsi else 2
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in plot_df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=plot_df.index, y=plot_df["macd_hist"],
            marker_color=colors, name="MACD Hist", showlegend=False
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["macd"],
            mode="lines", line=dict(color="#2196F3", width=1.2), name="MACD"
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["macd_signal"],
            mode="lines", line=dict(color="#FF9800", width=1.2), name="Signal"
        ), row=macd_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600 if rows == 1 else 750,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=20),
        xaxis_rangeslider_visible=False,
    )

    # Hide weekends
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )

    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="📈 PSAR Multi-Ticker Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Parabolic SAR Multi-Ticker Analyzer")
st.markdown("Analyze Parabolic SAR signals across multiple tickers. Historical data is cached locally in `data/investments/`.")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📌 Tickers")
    
    # Load tickers from file on first run
    if "loaded_tickers" not in st.session_state:
        st.session_state.loaded_tickers = load_tickers_from_txt()
    
    ticker_input = st.text_area(
        "Enter ticker symbols (one per line)",
        value="\n".join(st.session_state.loaded_tickers),
        height=120,
    )
    
    tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
    
    colA, colB = st.columns(2)
    
    with colA:
        if st.button("💾 Save tickers"):
            save_tickers_to_txt(tickers)
            st.session_state.loaded_tickers = tickers
    
    with colB:
        if st.button("🔄 Reload from file"):
            st.session_state.loaded_tickers = load_tickers_from_txt()
            st.experimental_rerun()

    # Quick-add preset
    preset = st.selectbox("Or load a preset", ["Custom", "US Tech", "EU ETFs", "Crypto"])
    if preset == "US Tech":
        tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    elif preset == "EU ETFs":
        tickers = ["IWDA.AS", "EMIM.AS", "VUSA.AS", "CSPX.AS", "EQQQ.AS"]
    elif preset == "Crypto":
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

    st.divider()

    # Chart options
    st.subheader("📊 Chart Options")
    days_to_show = st.slider("Days to display", 30, 500, 180)
    show_sma = st.checkbox("Show SMA (20/50)", value=True)
    show_bb = st.checkbox("Show Bollinger Bands", value=False)
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=False)


    st.divider()
    from telegram_alerts import (
        send_telegram_message, check_and_send_alerts,
        load_telegram_config, save_telegram_config
    )
    
    # --- inside the sidebar with block ---
    
    st.divider()
    st.subheader("🔔 Telegram Alerts")
    
    # Load from txt on first run only
    if "tg_loaded" not in st.session_state:
        cfg = load_telegram_config()
        st.session_state["tg_token"]          = cfg["token"]
        st.session_state["tg_chat_id"]        = cfg["chat_id"]
        st.session_state["tg_flip"]           = cfg["alert_on_flip"]
        st.session_state["tg_daily"]          = cfg["alert_on_signal"]
        st.session_state["tg_watched"]        = ",".join(cfg["watched_tickers"])
        st.session_state["tg_loaded"]         = True
    
    tg_token = st.text_input(
        "Bot Token", type="password",
        value=st.session_state["tg_token"]
    )
    tg_chat_id = st.text_input(
        "Chat ID",
        value=st.session_state["tg_chat_id"]
    )
    alert_on_flip   = st.checkbox("🔁 Alert on signal flip (Buy/Sell)",
                                   value=st.session_state["tg_flip"])
    alert_on_signal = st.checkbox("📅 Daily signal summary",
                                   value=st.session_state["tg_daily"])
    watched_input = st.text_input(
        "Watchlist (leave blank = all tickers)",
        placeholder="e.g. AAPL, IWDA.AS",
        value=st.session_state["tg_watched"]
    )
    watched_tickers = (
        [t.strip().upper() for t in watched_input.split(",") if t.strip()]
        if watched_input else None
    )
    
    # Persist to session state
    st.session_state["tg_token"]   = tg_token
    st.session_state["tg_chat_id"] = tg_chat_id
    st.session_state["tg_flip"]    = alert_on_flip
    st.session_state["tg_daily"]   = alert_on_signal
    st.session_state["tg_watched"] = watched_input
    
    colT1, colT2 = st.columns(2)
    
    with colT1:
        if st.button("💾 Save config"):
            save_telegram_config(
                token=tg_token,
                chat_id=tg_chat_id,
                alert_on_flip=alert_on_flip,
                alert_on_signal=alert_on_signal,
                watched_tickers=watched_tickers or []
            )
            st.success("Config saved!")
    
    with colT2:
        if st.button("🔄 Reload config"):
            cfg = load_telegram_config()
            st.session_state["tg_token"]   = cfg["token"]
            st.session_state["tg_chat_id"] = cfg["chat_id"]
            st.session_state["tg_flip"]    = cfg["alert_on_flip"]
            st.session_state["tg_daily"]   = cfg["alert_on_signal"]
            st.session_state["tg_watched"] = ",".join(cfg["watched_tickers"])
            st.rerun()
    
    if st.button("📨 Test Telegram connection"):
        if not tg_token or not tg_chat_id:
            st.error("⚠️ Enter both Bot Token and Chat ID first.")
        else:
            with st.spinner("Sending..."):
                ok, err = send_telegram_message(
                    tg_token, tg_chat_id, "✅ PSAR Analyzer connected!"
                )
            if ok:
                st.success("✅ Message sent! Check your Telegram.")
            else:
                st.error(f"❌ Failed: {err}")
    
    st.divider()


    
    # Data management
    st.subheader("💾 Data")
    
    # --- Per-ticker refresh ---
    st.caption("Refresh a single ticker:")
    refresh_single = st.selectbox("Select ticker to refresh", [""] + tickers, label_visibility="collapsed")
    if st.button("🔄 Refresh selected ticker", disabled=not refresh_single):
        if refresh_single:
            with st.spinner(f"Fetching fresh data for {refresh_single}..."):
                st.session_state["force_refresh_tickers"] = [refresh_single]
            st.rerun()
    
    # --- Refresh all ---
    if st.button("🔄 Refresh ALL tickers from web"):
        st.session_state["force_refresh_tickers"] = tickers
        st.rerun()
    
    # Resolve force_update from session state
    force_refresh_list = st.session_state.pop("force_refresh_tickers", [])
    force_update = False  # keep the variable alive for compatibility
    

    data_folder = st.text_input("Data folder", str(DATA_DIR))
    DATA_DIR = Path(data_folder)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.button("🗑️ Clear cached data"):
        for f in DATA_DIR.glob("*_history.csv"):
            f.unlink()
        st.success("Cache cleared!")

    st.divider()
    st.caption(f"Cache location: `{DATA_DIR.resolve()}`")

    st.divider()
    st.subheader("⏰ Auto-Refresh")
    
    last_refresh = get_last_refresh_time(DATA_DIR)
    st.caption(f"Last auto-refresh: **{last_refresh}**")
    
    refresh_time_input = st.time_input(
        "Daily refresh time (local)",
        value=dt.time(18, 0),
        help="Runs once daily after market close. Restart app to apply changes."
    )
    
    st.caption(f"Scheduled for: **{refresh_time_input.strftime('%H:%M')}** daily")
    
    # Live countdown to next refresh
    now = dt.datetime.now()
    next_run = now.replace(
        hour=refresh_time_input.hour,
        minute=refresh_time_input.minute,
        second=0, microsecond=0
    )
    if next_run < now:
        next_run += dt.timedelta(days=1)
    
    delta = next_run - now
    hours, remainder = divmod(int(delta.total_seconds()), 3600)
    minutes = remainder // 60
    st.info(f"⏳ Next refresh in: **{hours}h {minutes}m**")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not tickers:
    st.warning("Please enter at least one ticker symbol in the sidebar.")
    st.stop()

# ── Auto-scheduler (starts once per process) ──────────────────────────────
start_scheduler(
    tickers=tickers,
    data_dir=DATA_DIR,
    fetch_fn=load_or_fetch_ticker,
    refresh_time="18:00"   # ← change to your preferred time (after market close)
)

# --- Load data ---
with st.spinner(f"Loading data for {len(tickers)} ticker(s)..."):
    data = {}
    failed = []
    force_refresh_list = st.session_state.pop("force_refresh_tickers", [])

    for ticker in tickers:
        should_force = ticker in force_refresh_list
        df = load_or_fetch_ticker(ticker, force_update=should_force)
        if not df.empty:
            df = calculate_indicators(df)
            data[ticker] = df
            if should_force:
                st.toast(f"✅ {ticker} refreshed from web", icon="🔄")
        else:
            failed.append(ticker)

if failed:
    st.error(f"Could not load data for: {', '.join(failed)}")

if not data:
    st.stop()

# --- Telegram alerts ---
if st.session_state.get("tg_token") and st.session_state.get("tg_chat_id"):
    check_and_send_alerts(
        data=data,
        token=st.session_state["tg_token"],
        chat_id=st.session_state["tg_chat_id"],
        alert_on_flip=alert_on_flip,
        alert_on_signal=alert_on_signal,
        watched_tickers=watched_tickers
    )

# ============================================================================
# SUMMARY TABLE
# ============================================================================

st.subheader("📋 PSAR Signal Summary")

summary_rows = []
for ticker, df in data.items():
    if df.empty:
        continue
    last = df.iloc[-1]
    signal = get_psar_signal(df)
    streak = get_signal_streak(df)
    change_1d = ((last["close"] - df.iloc[-2]["close"]) / df.iloc[-2]["close"] * 100) if len(df) > 1 else 0
    change_5d = ((last["close"] - df.iloc[-6]["close"]) / df.iloc[-6]["close"] * 100) if len(df) > 5 else 0
    rsi_val = f"{last['rsi']:.1f}" if "rsi" in df.columns and pd.notna(last.get("rsi")) else "N/A"
    atr_val = f"{last['atr']:.2f}" if "atr" in df.columns and pd.notna(last.get("atr")) else "N/A"

    summary_rows.append({
        "Ticker": ticker,
        "Last Close": f"{last['close']:.2f}",
        "1D Change": f"{change_1d:+.2f}%",
        "5D Change": f"{change_5d:+.2f}%",
        "PSAR Signal": signal,
        "Streak (days)": streak,
        "RSI": rsi_val,
        "ATR": atr_val,
        "Data Points": len(df),
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# ============================================================================
# INDIVIDUAL CHARTS
# ============================================================================

st.subheader("📈 Parabolic SAR Charts")

# View mode
view_mode = st.radio("Chart layout", ["Single column", "Two columns"], horizontal=True)

ticker_list = list(data.keys())

if view_mode == "Two columns" and len(ticker_list) > 1:
    for i in range(0, len(ticker_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(ticker_list):
                t = ticker_list[i + j]
                df = data[t]
                with col:
                    st.markdown(f"**{t}** — {get_psar_signal(df)}")
                    fig = plot_psar_chart(
                        df, t, days_to_show,
                        show_sma=show_sma, show_bb=show_bb,
                        show_rsi=show_rsi, show_macd=show_macd
                    )
                    st.plotly_chart(fig, use_container_width=True)
else:
    tabs = st.tabs(ticker_list) if len(ticker_list) > 1 else [st.container()]

    for i, t in enumerate(ticker_list):
        df = data[t]
        container = tabs[i] if len(ticker_list) > 1 else tabs[0]

        with container:
            # Metric row
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            change = last["close"] - prev["close"]
            change_pct = (change / prev["close"]) * 100 if prev["close"] != 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Last Close", f"{last['close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
            m2.metric("PSAR Signal", get_psar_signal(df))
            m3.metric("Streak", f"{get_signal_streak(df)} days")
            if "rsi" in df.columns and pd.notna(last.get("rsi")):
                m4.metric("RSI", f"{last['rsi']:.1f}")
            else:
                m4.metric("RSI", "N/A")

            fig = plot_psar_chart(
                df, t, days_to_show,
                show_sma=show_sma, show_bb=show_bb,
                show_rsi=show_rsi, show_macd=show_macd
            )
            st.plotly_chart(fig, use_container_width=True)

            # Raw data expander
            with st.expander(f"📄 Raw data for {t}"):
                cols_to_show = ["open", "high", "low", "close", "volume", "psar"]
                if "sma20" in df.columns:
                    cols_to_show += ["sma20", "sma50"]
                if "rsi" in df.columns:
                    cols_to_show += ["rsi"]

                display_df = df[cols_to_show].tail(100).copy()
                display_df.index = display_df.index.strftime("%Y-%m-%d") if hasattr(display_df.index, 'strftime') else display_df.index
                st.dataframe(display_df.sort_index(ascending=False), use_container_width=True)

                # Download button
                csv = df.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇️ Download {t} data as CSV",
                    data=csv,
                    file_name=f"{t}_psar_data.csv",
                    mime="text/csv"
                )

# Auto-rerun the UI periodically so fresh data appears without manual refresh
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Recheck every 5 minutes (300,000 ms) — lightweight, just re-reads CSVs
st_autorefresh(interval=300_000, key="auto_ui_refresh")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("📁 Historical data cached in `data/investments/`. Use 'Force refresh' in sidebar to update.")
st.caption("⚠️ This tool is for educational purposes only. Not financial advice.")