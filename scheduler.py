# scheduler.py
import schedule
import time
import threading
import streamlit as st
from pathlib import Path
from datetime import datetime

_scheduler_started = False

def refresh_all_tickers(tickers: list, data_dir: Path, fetch_fn):
    """
    Called by the scheduler. Re-fetches all tickers and updates cache CSVs.
    Runs in a background thread so it doesn't block the UI.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"[Scheduler] Auto-refresh started at {timestamp}")

    for ticker in tickers:
        try:
            df = fetch_fn(ticker, force_update=True)
            if df.empty:
                print(f"[Scheduler] ⚠️  No data returned for {ticker}")
            else:
                print(f"[Scheduler] ✅ {ticker} updated ({len(df)} rows)")
        except Exception as e:
            print(f"[Scheduler] ❌ Failed for {ticker}: {e}")

    # Write a timestamp file so the UI knows when last refresh happened
    stamp_file = data_dir / "last_auto_refresh.txt"
    stamp_file.write_text(datetime.now().isoformat())
    print(f"[Scheduler] Auto-refresh complete at {datetime.now().strftime('%H:%M:%S')}")


def start_scheduler(tickers: list, data_dir: Path, fetch_fn, refresh_time: str = "18:00"):
    """
    Start the background scheduler (only once per process).
    refresh_time: "HH:MM" in 24h format — pick after market close, e.g. "18:00"
    """
    global _scheduler_started
    if _scheduler_started:
        return  # guard: only start once even across Streamlit reruns

    def run_scheduler():
        schedule.every().day.at(refresh_time).do(
            refresh_all_tickers, tickers=tickers, data_dir=data_dir, fetch_fn=fetch_fn
        )
        print(f"[Scheduler] Running. Next refresh scheduled at {refresh_time} daily.")
        while True:
            schedule.run_pending()
            time.sleep(30)  # check every 30 seconds

    thread = threading.Thread(target=run_scheduler, daemon=True)
    thread.start()
    _scheduler_started = True
    print("[Scheduler] Background thread started.")


def get_last_refresh_time(data_dir: Path) -> str:
    """Read the timestamp of the last successful auto-refresh."""
    stamp_file = data_dir / "last_auto_refresh.txt"
    if stamp_file.exists():
        try:
            ts = datetime.fromisoformat(stamp_file.read_text().strip())
            return ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "Unknown"
    return "Never"