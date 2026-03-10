# scheduler.py
#
# ════════════════════════════════════════════════════════════════════════════
# HOW DATA FLOWS FROM app.py INTO THE SCHEDULER
# ════════════════════════════════════════════════════════════════════════════
#
#  app.py calls start_scheduler() ONCE at startup (Streamlit reruns the
#  whole script on every user interaction, but the _scheduler_started
#  guard makes sure only the first call does anything).
#
#  The values passed at that moment — tg_token, tg_chat_id, alert_config,
#  watched_tickers — are CAPTURED by the closure inside run_scheduler() and
#  stay fixed for the lifetime of the process.  They are NOT re-read from
#  session_state on every tick.
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  app.py (Streamlit main thread)                                     │
#  │                                                                     │
#  │  sidebar widgets → alert_config dict                                │
#  │       │                                                             │
#  │       └─► start_scheduler(                                          │
#  │               tickers, DATA_DIR, load_or_fetch_ticker,              │
#  │               refresh_time   = "18:00",                             │
#  │               calculate_fn   = calculate_indicators,                │
#  │               params         = params,       ← indicator params     │
#  │               tg_token       = tg_token,                            │
#  │               tg_chat_id     = tg_chat_id,                          │
#  │               alert_config   = alert_config, ← all signal flags     │
#  │               watched_tickers= watched_tickers,                     │
#  │           )                                                          │
#  └─────────────────────────────────────────────────────────────────────┘
#           │
#           │  spawns daemon thread (runs forever, checks every 30 s)
#           ▼
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  Background thread                                                  │
#  │                                                                     │
#  │  Every day at refresh_time:                                         │
#  │    1. fetch fresh data for every ticker  (force_update=True)        │
#  │    2. run calculate_indicators(df, params) on each DataFrame        │
#  │    3. _run_alerts(data, alert_config, ...)                          │
#  │         ├─ load  alert_state.json  (persisted signal history)       │
#  │         ├─ for each ticker:                                         │
#  │         │    check PSAR flip    → queue flip message                │
#  │         │    check MACD cross   → queue cross message               │
#  │         │    check RSI extreme  → queue RSI message                 │
#  │         │    check VWAP cross   → queue VWAP message                │
#  │         ├─ send all queued messages via Telegram                    │
#  │         ├─ if on_daily_summary and not sent today:                  │
#  │         │    send full snapshot                                     │
#  │         └─ save alert_state.json                                    │
#  └─────────────────────────────────────────────────────────────────────┘
#
# WHEN IS TELEGRAM TRIGGERED?
# ────────────────────────────
#  Automatically — at refresh_time each day (default 18:00), regardless of
#  whether the browser tab is open.  The scheduler thread runs server-side.
#
#  Manually — the "📤 Send summary now" button in the sidebar sends
#  immediately using live session_state values (always up-to-date).
#
#  Signal checks are EDGE-TRIGGERED — each fires only on the bar where
#  the condition first becomes true, not every bar it stays true:
#
#    on_psar_flip     fires when PSAR crosses price direction vs last
#                     saved state in alert_state.json.
#    on_macd_cross    fires when MACD line crosses the signal line
#                     between prev bar and current bar.
#    on_rsi_extreme   fires when RSI enters OR exits overbought/oversold
#                     zone (4 sub-conditions, each edge-triggered).
#    on_vwap_cross    fires when close crosses VWAP between prev and
#                     current bar.
#    on_daily_summary fires once per calendar day, gated by last_daily
#                     date stored in alert_state.json.
#
# IMPORTANT LIMITATION
# ─────────────────────
#  start_scheduler captures alert_config at startup time.  Sidebar changes
#  made while the app is running only take effect after a full app restart.
#  The "📤 Send summary now" button always uses live session state and is
#  not affected by this limitation.
# ════════════════════════════════════════════════════════════════════════════

import json
import schedule
import threading
import time
from datetime import date, datetime
from pathlib import Path

_scheduler_started = False


# ─────────────────────────────────────────────────────────────────────────────
# Alert state persistence
#
# alert_state.json lives in DATA_DIR and stores per-ticker signal history
# so that edge-triggered alerts are not re-sent after an app restart.
#
# Keys written:
#   "{ticker}_psar"   → "BULL" | "BEAR"
#   "{ticker}_macd"   → "ABOVE" | "BELOW"   (macd vs signal line)
#   "{ticker}_rsi_ob" → True | False         (was overbought last bar)
#   "{ticker}_rsi_os" → True | False         (was oversold last bar)
#   "{ticker}_vwap"   → "ABOVE" | "BELOW"   (close vs vwap)
#   "last_daily"      → ISO date string      (date of last daily summary)
# ─────────────────────────────────────────────────────────────────────────────

def _state_file(data_dir: Path) -> Path:
    return data_dir / "alert_state.json"


def _load_alert_state(data_dir: Path) -> dict:
    f = _state_file(data_dir)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def _save_alert_state(data_dir: Path, state: dict) -> None:
    try:
        _state_file(data_dir).write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"[Scheduler] ⚠️  Could not save alert state: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers  (no numpy/pandas dependency at module level)
# ─────────────────────────────────────────────────────────────────────────────

def _notna(val) -> bool:
    """NaN-safe truthiness check — avoids importing numpy."""
    try:
        return val == val and val is not None
    except Exception:
        return False


def _streak(df) -> int:
    """Number of consecutive bars with the same PSAR direction."""
    if df.empty or "psar" not in df.columns or "close" not in df.columns:
        return 0
    direction = df["psar"] < df["close"]
    current   = direction.iloc[-1]
    count     = 0
    for v in reversed(direction.values):
        if v == current:
            count += 1
        else:
            break
    return count


def _detail_line(last, prev_row: dict, df, cfg: dict) -> str:
    """
    Build the optional context line appended to every alert body.
    Respects the inc_* toggles set by the user in the sidebar.

    Example output:
        💰 `98.14`  `+0.43%`  🟢 BULL  🔁 `12d`  〽️ RSI `61.2`  📉 VWAP `+1.1%`
    """
    parts = []

    if cfg.get("inc_price") and _notna(last.get("close")):
        close = last["close"]
        prev  = prev_row.get("close", close)
        chg   = (close - prev) / prev * 100 if prev else 0
        parts.append(f"💰 `{close:.2f}`  `{chg:+.2f}%`")

    if cfg.get("inc_psar") and "psar" in df.columns and _notna(last.get("psar")):
        sig = "🟢 BULL" if last["psar"] < last["close"] else "🔴 BEAR"
        parts.append(sig)

    if cfg.get("inc_streak") and "psar" in df.columns:
        parts.append(f"🔁 `{_streak(df)}d`")

    if cfg.get("inc_rsi") and "rsi" in df.columns and _notna(last.get("rsi")):
        rsi  = last["rsi"]
        ob   = cfg.get("rsi_overbought", 70)
        os_  = cfg.get("rsi_oversold",   30)
        flag = " 🔥OB" if rsi >= ob else (" 🧊OS" if rsi <= os_ else "")
        parts.append(f"〽️ RSI `{rsi:.1f}`{flag}")

    if cfg.get("inc_vwap") and "vwap" in df.columns and _notna(last.get("vwap")):
        vwap = last["vwap"]
        if vwap:
            parts.append(f"📉 VWAP `{(last['close'] - vwap) / vwap * 100:+.1f}%`")

    if cfg.get("inc_macd") and "macd_hist" in df.columns \
            and _notna(last.get("macd_hist")):
        h = last["macd_hist"]
        parts.append(f"📈 MACD `{h:.4f}` {'▲' if h > 0 else '▼'}")

    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Core refresh + alert job
# ─────────────────────────────────────────────────────────────────────────────

def refresh_all_tickers(
    tickers:         list,
    data_dir:        Path,
    fetch_fn,
    calculate_fn     = None,
    params:          dict = None,
    tg_token:        str  = "",
    tg_chat_id:      str  = "",
    alert_config:    dict = None,
    watched_tickers: list = None,
):
    """
    Called by the schedule library at the configured daily time.

    Phase 1 — Data refresh
        fetch_fn(ticker, force_update=True) is called for every ticker.
        If calculate_fn is provided it runs immediately after so that
        PSAR / MACD / RSI / VWAP columns exist before alert checks.

    Phase 2 — Alert evaluation
        Delegates to _run_alerts() which reads alert_config to decide
        which conditions to check and what to include in each message.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[Scheduler] Auto-refresh started at {timestamp}")

    data: dict = {}

    for ticker in tickers:
        try:
            df = fetch_fn(ticker, force_update=True)
            if df.empty:
                print(f"[Scheduler] ⚠️  No data for {ticker}")
                continue

            if calculate_fn is not None and params is not None:
                try:
                    df = calculate_fn(df, params)
                except Exception as e:
                    print(f"[Scheduler] ⚠️  Indicator calc failed {ticker}: {e}")

            data[ticker] = df
            print(f"[Scheduler] ✅ {ticker} ({len(df)} rows)")

        except Exception as e:
            print(f"[Scheduler] ❌ {ticker}: {e}")

    # Write refresh timestamp regardless of per-ticker success
    (data_dir / "last_auto_refresh.txt").write_text(datetime.now().isoformat())
    print(f"[Scheduler] Refresh complete at {datetime.now().strftime('%H:%M:%S')}")

    # ── Telegram ─────────────────────────────────────────────────────────────
    if not (tg_token and tg_chat_id):
        print("[Scheduler] Telegram not configured — skipping alerts.")
        return

    cfg         = alert_config or {}
    any_trigger = any(cfg.get(k) for k in (
        "on_psar_flip", "on_macd_cross",
        "on_rsi_extreme", "on_vwap_cross", "on_daily_summary",
    ))
    if not any_trigger:
        print("[Scheduler] All alert triggers disabled.")
        return

    if not data:
        print("[Scheduler] No data loaded — cannot check alerts.")
        return

    _run_alerts(
        data            = data,
        data_dir        = data_dir,
        tg_token        = tg_token,
        tg_chat_id      = tg_chat_id,
        alert_config    = cfg,
        watched_tickers = watched_tickers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alert evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _run_alerts(
    data:            dict,
    data_dir:        Path,
    tg_token:        str,
    tg_chat_id:      str,
    alert_config:    dict,
    watched_tickers: list | None,
) -> None:
    """
    Evaluate every enabled signal condition for every watched ticker.
    Queue a message for each condition that fires, then send them all.

    All checks are EDGE-TRIGGERED — a condition fires only on the bar
    where it first becomes true.  Previous state is read from / written
    to alert_state.json so the edge is not lost across runs.

    Message priority (lower = more urgent, sent first):
        1  PSAR flip
        2  MACD cross
        3  RSI extreme
        4  VWAP cross
        5  Daily summary rows  (batched into one message)
    """
    try:
        from telegram_alerts import send_telegram_message
    except ImportError:
        print("[Scheduler] ❌ telegram_alerts module not found.")
        return

    cfg     = alert_config
    state   = _load_alert_state(data_dir)
    today   = date.today().isoformat()
    changed = False

    # (priority, message_text)
    queued: list[tuple[int, str]] = []

    target = (
        [t for t in data if t in watched_tickers]
        if watched_tickers else list(data.keys())
    )

    for ticker in target:
        df = data.get(ticker)
        if df is None or df.empty:
            continue

        last     = df.iloc[-1]
        prev_row = df.iloc[-2].to_dict() if len(df) > 1 else last.to_dict()
        detail   = _detail_line(last, prev_row, df, cfg)

        # ── 1. PSAR flip ──────────────────────────────────────────────────────
        #
        #  Compares current PSAR direction against the saved baseline.
        #  The state key is updated after every run whether or not a flip
        #  occurred — this is essential so the next run has the right baseline.
        #
        if cfg.get("on_psar_flip") and "psar" in df.columns:
            if _notna(last.get("psar")) and _notna(last.get("close")):
                cur_sig  = "BULL" if last["psar"] < last["close"] else "BEAR"
                prev_sig = state.get(f"{ticker}_psar")

                if prev_sig and prev_sig != cur_sig:
                    arrow   = "📈 BUY signal"  if cur_sig == "BULL" else "📉 SELL signal"
                    sig_str = "🟢 BULLISH"      if cur_sig == "BULL" else "🔴 BEARISH"
                    body    = (
                        f"🔔 *PSAR Flip — {ticker}*\n"
                        f"{arrow}\n"
                        f"`{prev_sig}` → {sig_str}\n"
                    )
                    if detail:
                        body += detail
                    queued.append((1, body))
                    print(f"[Scheduler] PSAR flip: {ticker} {prev_sig}→{cur_sig}")

                state[f"{ticker}_psar"] = cur_sig   # always update baseline
                changed = True

        # ── 2. MACD line crosses signal line ─────────────────────────────────
        #
        #  Uses columns "macd" and "macd_signal_line" as produced by
        #  calculate_indicators() in app.py.
        #  No state file entry needed — fully edge-triggered from prev bar.
        #
        if cfg.get("on_macd_cross") and \
                all(c in df.columns for c in ["macd", "macd_signal_line"]):

            m_now  = last.get("macd");             m_prev  = prev_row.get("macd")
            sl_now = last.get("macd_signal_line"); sl_prev = prev_row.get("macd_signal_line")

            if all(_notna(v) for v in [m_now, m_prev, sl_now, sl_prev]):
                cross_bull = (m_prev <= sl_prev) and (m_now > sl_now)
                cross_bear = (m_prev >= sl_prev) and (m_now < sl_now)

                if cross_bull or cross_bear:
                    direction = "📈 Bullish crossover" if cross_bull else "📉 Bearish crossover"
                    body = (
                        f"〰️ *MACD Cross — {ticker}*\n"
                        f"{direction}\n"
                        f"MACD `{m_now:.4f}`  Signal `{sl_now:.4f}`\n"
                    )
                    if detail:
                        body += detail
                    queued.append((2, body))
                    print(f"[Scheduler] MACD cross: {ticker} "
                          f"{'bull' if cross_bull else 'bear'}")

        # ── 3. RSI extreme zones ──────────────────────────────────────────────
        #
        #  Fires when RSI enters OR exits overbought/oversold zones.
        #  Thresholds from cfg["rsi_overbought"] / cfg["rsi_oversold"].
        #  Four sub-conditions — all edge-triggered from prev vs current bar.
        #
        if cfg.get("on_rsi_extreme") and "rsi" in df.columns:
            rsi_now  = last.get("rsi")
            rsi_prev = prev_row.get("rsi")
            ob       = cfg.get("rsi_overbought", 70)
            os_      = cfg.get("rsi_oversold",   30)

            if _notna(rsi_now) and _notna(rsi_prev):
                for fired, label in [
                    ((rsi_prev < ob)   and (rsi_now >= ob),
                     f"🔥 Entered overbought ≥{ob}  RSI `{rsi_now:.1f}`"),
                    ((rsi_prev > os_)  and (rsi_now <= os_),
                     f"🧊 Entered oversold ≤{os_}  RSI `{rsi_now:.1f}`"),
                    ((rsi_prev >= ob)  and (rsi_now < ob),
                     f"RSI exited overbought zone  RSI `{rsi_now:.1f}`"),
                    ((rsi_prev <= os_) and (rsi_now > os_),
                     f"RSI exited oversold zone  RSI `{rsi_now:.1f}`"),
                ]:
                    if fired:
                        body = f"〽️ *RSI Alert — {ticker}*\n{label}\n"
                        if detail:
                            body += detail
                        queued.append((3, body))
                        print(f"[Scheduler] RSI alert: {ticker} {label[:40]}")

        # ── 4. Price crosses VWAP ─────────────────────────────────────────────
        #
        #  Edge-triggered from prev bar vs current bar.
        #  No state file entry needed (unlike PSAR which uses saved state
        #  across runs rather than consecutive bars).
        #
        if cfg.get("on_vwap_cross") and "vwap" in df.columns:
            c_now  = last.get("close");      v_now  = last.get("vwap")
            c_prev = prev_row.get("close");  v_prev = prev_row.get("vwap")

            if all(_notna(v) for v in [c_now, v_now, c_prev, v_prev]):
                cross_above = (c_prev <= v_prev) and (c_now > v_now)
                cross_below = (c_prev >= v_prev) and (c_now < v_now)

                if cross_above or cross_below:
                    direction = "📈 Crossed above VWAP" if cross_above \
                                else "📉 Crossed below VWAP"
                    body = (
                        f"📉 *VWAP Cross — {ticker}*\n"
                        f"{direction}\n"
                        f"Close `{c_now:.2f}`  VWAP `{v_now:.2f}`\n"
                    )
                    if detail:
                        body += detail
                    queued.append((4, body))
                    print(f"[Scheduler] VWAP cross: {ticker} "
                          f"{'above' if cross_above else 'below'}")

        # ── Accumulate daily summary row ──────────────────────────────────────
        #
        #  Always build a full detail line for the summary — overrides the
        #  user's inc_* toggles so the summary is always comprehensive.
        #
        if cfg.get("on_daily_summary"):
            full_cfg   = {**cfg,
                          "inc_price": True, "inc_psar": True, "inc_streak": True,
                          "inc_rsi":   True, "inc_vwap": True, "inc_macd":  True}
            row_detail = _detail_line(last, prev_row, df, full_cfg)
            psar_str   = ("🟢 BULL" if "psar" in df.columns
                          and _notna(last.get("psar"))
                          and last["psar"] < last["close"] else "🔴 BEAR")
            queued.append((5, f"*{ticker}*  {psar_str}\n{row_detail}"))

    # ── Send action alerts (priority 1–4) individually ────────────────────────
    action_msgs  = [(p, m) for p, m in sorted(queued) if p < 5]
    summary_rows = [m      for p, m in queued          if p == 5]

    ts_header = f"🗓 {datetime.now().strftime('%d %b %Y  %H:%M')}\n\n"

    for priority, text in action_msgs:
        ok, err = send_telegram_message(tg_token, tg_chat_id, ts_header + text)
        status  = "✅" if ok else "❌"
        print(f"[Scheduler] {status} Alert P{priority}: "
              f"{text[:50].strip()}{'…' if len(text) > 50 else ''}")
        if not ok:
            print(f"[Scheduler]    Error: {err}")
        time.sleep(0.35)   # stay well under Telegram's 30 msg/sec limit

    # ── Send daily summary (once per calendar day) ────────────────────────────
    if summary_rows and cfg.get("on_daily_summary") \
            and state.get("last_daily") != today:
        header = (
            f"📊 *Daily PSAR Summary*\n"
            f"🗓 {datetime.now().strftime('%d %b %Y  %H:%M')}\n\n"
        )
        ok, err = send_telegram_message(
            tg_token, tg_chat_id,
            header + "\n\n".join(summary_rows),
        )
        if ok:
            state["last_daily"] = today
            changed = True
            print(f"[Scheduler] ✅ Daily summary sent ({len(summary_rows)} tickers)")
        else:
            print(f"[Scheduler] ❌ Daily summary failed: {err}")


    if changed or action_msgs:
            _save_alert_state(data_dir, state)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start_scheduler(
    tickers:         list,
    data_dir:        Path,
    fetch_fn,
    refresh_time:    str  = "18:00",
    calculate_fn          = None,
    params:          dict = None,
    tg_token:        str  = "",
    tg_chat_id:      str  = "",
    alert_config:    dict = None,
    watched_tickers: list = None,
) -> None:
    """
    Start the background scheduler.  Safe to call on every Streamlit rerun —
    _scheduler_started ensures the daemon thread is created exactly once.

    Parameters
    ──────────
    tickers          List of ticker symbols (e.g. ["AAPL", "IWDA.AS"])
    data_dir         Path to the CSV cache directory
    fetch_fn         fetch_fn(ticker, force_update=True) → DataFrame
    refresh_time     "HH:MM" 24h local time.  Pick after market close.
    calculate_fn     calculate_fn(df, params) → DataFrame
                     Must produce columns: psar, close,
                     macd, macd_signal_line, macd_hist, rsi, vwap
    params           Passed as second argument to calculate_fn
    tg_token         Telegram bot token.  Empty string disables all alerts.
    tg_chat_id       Telegram chat / channel ID
    alert_config     Dict from the sidebar alert widgets:
                     {
                       # triggers — which signals fire a message
                       "on_psar_flip":     bool,
                       "on_macd_cross":    bool,
                       "on_rsi_extreme":   bool,
                       "on_vwap_cross":    bool,
                       "on_daily_summary": bool,
                       # thresholds
                       "rsi_overbought":   int,   # default 70
                       "rsi_oversold":     int,   # default 30
                       # message content toggles
                       "inc_price":        bool,
                       "inc_psar":         bool,
                       "inc_streak":       bool,
                       "inc_rsi":          bool,
                       "inc_vwap":         bool,
                       "inc_macd":         bool,
                     }
    watched_tickers  Restrict alerts to this subset.  None = all tickers.

    NOTE: alert_config is captured at startup.  Changes made in the sidebar
    while the app runs only take effect after a full restart.
    The "📤 Send summary now" button always uses live session_state.
    """
    global _scheduler_started
    if _scheduler_started:
        return

    cfg = alert_config or {}

    def run_scheduler():
        schedule.every().day.at(refresh_time).do(
            refresh_all_tickers,
            tickers         = tickers,
            data_dir        = data_dir,
            fetch_fn        = fetch_fn,
            calculate_fn    = calculate_fn,
            params          = params,
            tg_token        = tg_token,
            tg_chat_id      = tg_chat_id,
            alert_config    = cfg,
            watched_tickers = watched_tickers,
        )
        active = [k for k in (
            "on_psar_flip", "on_macd_cross",
            "on_rsi_extreme", "on_vwap_cross", "on_daily_summary",
        ) if cfg.get(k)]
        print(
            f"[Scheduler] Running.\n"
            f"[Scheduler]   Refresh time   : {refresh_time} daily\n"
            f"[Scheduler]   Active triggers: {active or ['none']}\n"
            f"[Scheduler]   Telegram       : {'configured' if tg_token else 'not set'}\n"
            f"[Scheduler]   Watchlist      : {watched_tickers or 'all tickers'}"
        )
        while True:
            schedule.run_pending()
            time.sleep(30)

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    _scheduler_started = True
    print("[Scheduler] Background thread started.")


def get_last_refresh_time(data_dir: Path) -> str:
    """Return formatted timestamp of the last successful auto-refresh."""
    f = data_dir / "last_auto_refresh.txt"
    if f.exists():
        try:
            return datetime.fromisoformat(f.read_text().strip()).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "Unknown"
    return "Never"


def get_alert_state(data_dir: Path) -> dict:
    """
    Return the raw persisted alert state dict.

    Useful for showing last-known signal per ticker in the UI:

        state = get_alert_state(DATA_DIR)
        st.caption(f"Last known PSAR: {state.get('AAPL_psar', 'unknown')}")
        st.caption(f"Last daily sent: {state.get('last_daily',  'never')}")
    """
    return _load_alert_state(data_dir)