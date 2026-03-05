# telegram_alerts.py

import requests
import pandas as pd
from pathlib import Path
import json
from datetime import date

ALERT_STATE_FILE = Path("data/investments/alert_state.json")
TELEGRAM_CONFIG_FILE = Path("data/investments/telegram_config.txt")

# ============================================================================
# CONFIG SAVE / LOAD
# ============================================================================

def save_telegram_config(token: str, chat_id: str, alert_on_flip: bool,
                          alert_on_signal: bool, watched_tickers: list):
    """Save Telegram config to a plain text file."""
    TELEGRAM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    watched_str = ",".join(watched_tickers) if watched_tickers else ""
    lines = [
        f"token={token}",
        f"chat_id={chat_id}",
        f"alert_on_flip={alert_on_flip}",
        f"alert_on_signal={alert_on_signal}",
        f"watched_tickers={watched_str}",
    ]
    TELEGRAM_CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")


def load_telegram_config() -> dict:
    """
    Load Telegram config from txt file.
    Returns a dict with keys: token, chat_id, alert_on_flip,
                               alert_on_signal, watched_tickers
    Returns empty defaults if file doesn't exist or is malformed.
    """
    defaults = {
        "token": "",
        "chat_id": "",
        "alert_on_flip": True,
        "alert_on_signal": False,
        "watched_tickers": [],
    }

    if not TELEGRAM_CONFIG_FILE.exists():
        return defaults

    try:
        for line in TELEGRAM_CONFIG_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()

            if key == "token":
                defaults["token"] = val
            elif key == "chat_id":
                defaults["chat_id"] = val
            elif key == "alert_on_flip":
                defaults["alert_on_flip"] = val.lower() == "true"
            elif key == "alert_on_signal":
                defaults["alert_on_signal"] = val.lower() == "true"
            elif key == "watched_tickers":
                defaults["watched_tickers"] = (
                    [t.strip().upper() for t in val.split(",") if t.strip()]
                    if val else []
                )
    except Exception as e:
        print(f"[Telegram] Could not load config: {e}")

    return defaults


# ============================================================================
# CORE SEND FUNCTION
# ============================================================================

def send_telegram_message(token: str, chat_id: str, message: str) -> tuple[bool, str]:
    """
    Send a message via Telegram Bot API.
    Returns (True, "") on success or (False, error_description) on failure.
    """
    if not token or not chat_id:
        return False, "Token or Chat ID is empty."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code == 200:
            return True, ""
        try:
            detail = resp.json().get("description", resp.text)
        except Exception:
            detail = resp.text
        return False, f"HTTP {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return False, "Connection error — check your internet connection."
    except requests.exceptions.Timeout:
        return False, "Request timed out — Telegram may be unreachable."
    except requests.exceptions.RequestException as e:
        return False, str(e)


# ============================================================================
# ALERT STATE — avoid duplicate alerts on every rerun
# ============================================================================

def load_alert_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            with open(ALERT_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_alert_state(state: dict):
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# SIGNAL DETECTION & ALERTING
# ============================================================================

def get_raw_signal(df: pd.DataFrame) -> str:
    if df.empty or "psar" not in df.columns:
        return "N/A"
    last = df.iloc[-1]
    if pd.isna(last["psar"]):
        return "N/A"
    return "BULLISH" if last["psar"] < last["close"] else "BEARISH"


def get_signal_streak(df: pd.DataFrame) -> int:
    if df.empty or "psar" not in df.columns:
        return 0
    direction = df["psar"] < df["close"]
    current = direction.iloc[-1]
    streak = 0
    for val in reversed(direction.values):
        if val == current:
            streak += 1
        else:
            break
    return streak


def check_and_send_alerts(
    data: dict,
    token: str,
    chat_id: str,
    alert_on_flip: bool = True,
    alert_on_signal: bool = False,
    watched_tickers: list = None
) -> list[str]:
    """
    Compare current PSAR signals against last known state.
    Returns a list of result strings for display in the UI.
    """
    if not token or not chat_id:
        return []

    state = load_alert_state()
    today = str(date.today())
    results = []
    changed = False

    for ticker, df in data.items():

        if watched_tickers and ticker not in watched_tickers:
            continue

        current_signal = get_raw_signal(df)
        if current_signal == "N/A":
            continue

        last   = df.iloc[-1]
        close  = last["close"]
        psar   = last["psar"]
        streak = get_signal_streak(df)

        prev        = state.get(ticker, {})
        prev_signal = prev.get("signal", None)
        prev_date   = prev.get("date", None)

        # --- Flip alert ---
        if alert_on_flip and prev_signal and prev_signal != current_signal:
            emoji  = "🟢" if current_signal == "BULLISH" else "🔴"
            action = "BUY signal 📈" if current_signal == "BULLISH" else "SELL signal 📉"
            msg = (
                f"{emoji} <b>PSAR Flip — {ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"Signal:  <b>{prev_signal} → {current_signal}</b>\n"
                f"Action:  <b>{action}</b>\n"
                f"Close:   <b>{close:.2f}</b>\n"
                f"PSAR:    {psar:.2f}\n"
                f"Streak:  {streak} day(s)\n"
                f"Date:    {today}"
            )
            ok, err = send_telegram_message(token, chat_id, msg)
            results.append(
                f"{'✅' if ok else '❌'} {ticker}: {prev_signal} → {current_signal}"
                + (f" ({err})" if not ok else "")
            )

        # --- Daily summary ---
        elif alert_on_signal and prev_date != today:
            emoji = "🟢" if current_signal == "BULLISH" else "🔴"
            msg = (
                f"{emoji} <b>Daily Signal — {ticker}</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"Signal:  <b>{current_signal}</b>\n"
                f"Close:   <b>{close:.2f}</b>\n"
                f"PSAR:    {psar:.2f}\n"
                f"Streak:  {streak} day(s)\n"
                f"Date:    {today}"
            )
            ok, err = send_telegram_message(token, chat_id, msg)
            results.append(
                f"{'✅' if ok else '❌'} {ticker}: daily summary"
                + (f" ({err})" if not ok else "")
            )

        state[ticker] = {"signal": current_signal, "date": today}
        changed = True

    if changed:
        save_alert_state(state)

    return results