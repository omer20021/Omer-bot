"""
Multi-Asset SMC Alert Bot — Advanced Engine
Scans XAU/USD, US30, NAS100, EUR/USD concurrently using identical
15m / 1H / 4H multi-timeframe logic.

7-layer SMC confluence (all must pass per asset):
  L1. 4H macro bias      (EMA20 + swing structure)
  L2. 1H intermediate    (EMA20 + swing structure)
  L3. Premium/Discount   (50% equilibrium of swing range)
  L4. Liquidity Sweep    (EQH / EQL wick-and-reject)
  L5. CHoCH / BOS        (structure break confirming direction)
  L6. Fair Value Gap     (institutional imbalance zone)
  L7. OB ∩ FVG           (Order Block confluent with FVG)

Telegram output: Pair · Entry · TP · SL  (minimal format)
"""

import os
import json
import time
import logging
import tempfile
import threading
import schedule
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from flask import Flask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Credentials ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Default timeframe stack (swing assets) ────────────────────────────────────
INTERVAL = "15m"   # Default execution TF
HTF1     = "1h"    # Default intermediate bias
HTF2     = "4h"    # Default macro bias

SCAN_EVERY_MINUTES = 15   # Swing asset scan cycle

# ── Asset Universe ─────────────────────────────────────────────────────────────
# Per-asset TF overrides:
#   interval      — execution (signal) timeframe
#   htf1          — intermediate bias TF
#   htf2          — macro bias TF  (both HTFs must agree)
#   exec_period   — yfinance history period for execution TF
#   htf1_period   — yfinance history period for htf1
#   htf2_period   — yfinance history period for htf2
#   scan_every    — minutes between scans (drives scheduler grouping)
# Fields not present fall back to the global defaults above.
ASSETS = [
    # ── Swing / session assets (15m exec, 1H+4H bias, scan every 15 min) ──────
    {
        "symbol":      "GC=F",
        "name":        "XAU/USD",
        "pip_size":    0.10,
        "min_atr":     0.50,
        "decimals":    2,
    },
    {
        "symbol":      "YM=F",
        "name":        "US30",
        "pip_size":    1.0,
        "min_atr":     20.0,
        "decimals":    0,
    },
    {
        "symbol":      "NQ=F",
        "name":        "NAS100",
        "pip_size":    1.0,
        "min_atr":     10.0,
        "decimals":    0,
    },
    {
        "symbol":      "EURUSD=X",
        "name":        "EUR/USD",
        "pip_size":    0.0001,
        "min_atr":     0.0003,
        "decimals":    5,
    },
    {
        "symbol":      "GBPUSD=X",
        "name":        "GBP/USD",
        "pip_size":    0.0001,
        "min_atr":     0.0003,
        "decimals":    5,
    },
    {
        "symbol":      "GBPJPY=X",
        "name":        "GBP/JPY",
        "pip_size":    0.01,
        "min_atr":     0.05,
        "decimals":    3,
    },
    # ── Scalp asset — BTC (1m exec, 5m bias, scan every 1 min, 24/7) ──────────
    {
        "symbol":      "BTC-USD",
        "name":        "BTC/USD",
        "pip_size":    1.0,
        "min_atr":     50.0,
        "decimals":    2,
        "interval":    "1m",
        "htf1":        "5m",
        "htf2":        "5m",
        "exec_period": "2d",
        "htf1_period": "5d",
        "htf2_period": "5d",
        "scan_every":  1,
    },
]

# ── Asset groups derived from ASSETS list ─────────────────────────────────────
SWING_ASSETS = [a for a in ASSETS if a.get("scan_every", SCAN_EVERY_MINUTES) == SCAN_EVERY_MINUTES]
SCALP_ASSETS = [a for a in ASSETS if a.get("scan_every", SCAN_EVERY_MINUTES) < SCAN_EVERY_MINUTES]

# ── ATR-based sizing (shared across all assets — scales with ATR) ─────────────
SL_ATR_MULT = 0.5   # SL buffer beyond OB edge
TP_ATR_MULT = 3.0   # Fallback TP multiplier when no opposing swing available

# ── Sweep / EQ detection ──────────────────────────────────────────────────────
EQ_TOLERANCE_MULT = 0.4   # Highs/lows within 0.4 × ATR → "equal"
SWEEP_WINDOW      = 6     # Recent candles to search for the sweep candle
EQ_MIN_GAP        = 4     # Minimum candle separation for a valid EQH/EQL pair

# ── Lookback windows ──────────────────────────────────────────────────────────
SWING_LOOKBACK = 5
FVG_LOOKBACK   = 40
OB_LOOKBACK    = 20
SWEEP_LOOKBACK = 40

# ── Trade journal ─────────────────────────────────────────────────────────────
TRADE_LOG_FILE = Path("scripts/smc_bot/trades.json")

# ── Per-asset open trade state ────────────────────────────────────────────────
# Keyed by asset symbol.  Access guarded by _trades_lock.
_open_trades: dict[str, dict] = {}
_trades_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════
def send_telegram(message: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        log.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        log.info("Telegram: sent.")
        return True
    except requests.RequestException as e:
        log.error(f"Telegram error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════
def fetch_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            log.warning(f"{symbol} [{interval}]: no data returned.")
            return None
        df = df[["Open", "High", "Low", "Close"]].dropna()
        log.info(f"{symbol} [{interval}]: {len(df)} candles")
        return df
    except Exception as e:
        log.error(f"{symbol} [{interval}] fetch error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  ATR
# ══════════════════════════════════════════════════════════════════════════════
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — MULTI-TIMEFRAME BIAS  (4H + 1H must both agree)
# ══════════════════════════════════════════════════════════════════════════════
def _bias_from_df(df: pd.DataFrame, label: str) -> str | None:
    """EMA20 direction + swing-structure confirmation on a single TF."""
    if df is None or len(df) < 25:
        log.info(f"{label}: insufficient candles.")
        return None

    closes   = df["Close"]
    ema20    = closes.ewm(span=20, adjust=False).mean()
    price    = closes.iloc[-1]
    ema_bias = "BUY" if price > ema20.iloc[-1] else "SELL"

    highs, lows = df["High"].values, df["Low"].values
    n = len(highs)
    sh_px, sl_px = [], []
    for i in range(3, n - 3):
        if highs[i] == max(highs[i - 3: i + 4]):
            sh_px.append(highs[i])
        if lows[i] == min(lows[i - 3: i + 4]):
            sl_px.append(lows[i])

    if len(sh_px) >= 2 and len(sl_px) >= 2:
        hh = sh_px[-1] > sh_px[-2];  hl = sl_px[-1] > sl_px[-2]
        lh = sh_px[-1] < sh_px[-2];  ll = sl_px[-1] < sl_px[-2]
        struct = "BUY" if (hh and hl) else ("SELL" if (lh and ll) else None)
        if struct and struct != ema_bias:
            log.info(f"{label}: EMA={ema_bias} vs structure={struct} — no bias.")
            return None

    log.info(f"{label}: {ema_bias}  price={price:.5g}  EMA={ema20.iloc[-1]:.5g}")
    return ema_bias


def get_htf_bias(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> str | None:
    b4 = _bias_from_df(df_4h, "4H")
    b1 = _bias_from_df(df_1h, "1H")
    if b4 is None or b1 is None:
        log.info("HTF: incomplete — one TF has no clear bias.")
        return None
    if b4 != b1:
        log.info(f"HTF: 4H={b4} conflicts with 1H={b1}.")
        return None
    log.info(f"HTF confirmed: {b4}")
    return b4


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — SWING STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
def find_swing_highs_lows(df: pd.DataFrame, strength: int = SWING_LOOKBACK):
    highs, lows, n = df["High"].values, df["Low"].values, len(df)
    sh_idx = sl_idx = None
    for i in range(strength, n - 1):
        if highs[i] == max(highs[max(0, i - strength): i + strength + 1]):
            sh_idx = i
        if lows[i]  == min(lows[max(0,  i - strength): i + strength + 1]):
            sl_idx = i
    return sh_idx, sl_idx


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — PREMIUM / DISCOUNT ZONE
# ══════════════════════════════════════════════════════════════════════════════
def get_pd_zone(df: pd.DataFrame, sh_idx, sl_idx) -> str | None:
    if sh_idx is None or sl_idx is None:
        return None
    swing_high = df["High"].iloc[sh_idx]
    swing_low  = df["Low"].iloc[sl_idx]
    if swing_high <= swing_low:
        return None
    mid   = (swing_high + swing_low) / 2
    price = df["Close"].iloc[-1]
    zone  = "discount" if price < mid else "premium"
    log.info(f"P/D: high={swing_high:.5g} low={swing_low:.5g} mid={mid:.5g} → {zone}")
    return zone


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — LIQUIDITY SWEEP  (EQH / EQL)
# ══════════════════════════════════════════════════════════════════════════════
def detect_sweep(df: pd.DataFrame, atr_val: float) -> tuple[str | None, float | None]:
    """Returns (direction, swept_level) or (None, None)."""
    n = len(df)
    if n < SWEEP_LOOKBACK + SWEEP_WINDOW + 2:
        return None, None

    tolerance = atr_val * EQ_TOLERANCE_MULT
    ref       = df.iloc[-(SWEEP_LOOKBACK + SWEEP_WINDOW): -SWEEP_WINDOW]
    ref_h, ref_l = ref["High"].values, ref["Low"].values
    recent    = df.iloc[-SWEEP_WINDOW:]

    eqh, eql = [], []
    m = len(ref_h)
    for i in range(m):
        for j in range(i + EQ_MIN_GAP, m):
            if abs(ref_h[i] - ref_h[j]) <= tolerance:
                eqh.append(max(ref_h[i], ref_h[j]))
            if abs(ref_l[i] - ref_l[j]) <= tolerance:
                eql.append(min(ref_l[i], ref_l[j]))

    if not eqh and not eql:
        log.info("Sweep: no EQH/EQL levels found.")
        return None, None

    for _, row in recent.iterrows():
        for lvl in eqh:
            if row["High"] > lvl and row["Close"] < lvl:
                log.info(f"Sweep: EQH swept @ {lvl:.5g} → SELL")
                return "SELL", lvl
        for lvl in eql:
            if row["Low"] < lvl and row["Close"] > lvl:
                log.info(f"Sweep: EQL swept @ {lvl:.5g} → BUY")
                return "BUY", lvl

    log.info("Sweep: no qualifying sweep in recent candles.")
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — CHoCH / BOS
# ══════════════════════════════════════════════════════════════════════════════
def detect_structure_break(
    df: pd.DataFrame, sh_idx, sl_idx
) -> tuple[str | None, float | None, str | None]:
    """Returns (direction, break_level, break_type) or (None, None, None)."""
    if len(df) < 3:
        return None, None, None
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]

    # CHoCH — crosses a pivot level
    if sh_idx is not None:
        sh = df["High"].iloc[sh_idx]
        if prev <= sh < last:
            log.info(f"Structure: CHoCH BUY (broke {sh:.5g})")
            return "BUY", sh, "CHoCH"
    if sl_idx is not None:
        sl = df["Low"].iloc[sl_idx]
        if prev >= sl > last:
            log.info(f"Structure: CHoCH SELL (broke {sl:.5g})")
            return "SELL", sl, "CHoCH"

    # BOS — closes at 20-candle extreme
    window = df["Close"].iloc[-21:-1]
    if len(window) >= 20:
        if last > window.max():
            log.info("Structure: BOS BUY")
            return "BUY", window.max(), "BOS"
        if last < window.min():
            log.info("Structure: BOS SELL")
            return "SELL", window.min(), "BOS"

    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 6 — FAIR VALUE GAP  (FVG)
# ══════════════════════════════════════════════════════════════════════════════
def find_fvg(df: pd.DataFrame, direction: str, atr_val: float) -> dict | None:
    price, n = df["Close"].iloc[-1], len(df)
    for i in range(n - 1, max(2, n - FVG_LOOKBACK - 1), -1):
        h2, l2 = df["High"].iloc[i - 2], df["Low"].iloc[i - 2]
        hi, li = df["High"].iloc[i],     df["Low"].iloc[i]
        if direction == "BUY" and li > h2:
            fvg = {"low": h2, "high": li, "mid": (h2 + li) / 2}
            if price <= fvg["high"] + atr_val:
                log.info(f"FVG BUY: [{fvg['low']:.5g}–{fvg['high']:.5g}]")
                return fvg
        elif direction == "SELL" and hi < l2:
            fvg = {"low": hi, "high": l2, "mid": (hi + l2) / 2}
            if price >= fvg["low"] - atr_val:
                log.info(f"FVG SELL: [{fvg['low']:.5g}–{fvg['high']:.5g}]")
                return fvg
    log.info(f"FVG: none for {direction}.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER 7 — ORDER BLOCK  +  OB ∩ FVG
# ══════════════════════════════════════════════════════════════════════════════
def find_order_block(df: pd.DataFrame, direction: str, atr_val: float) -> dict | None:
    price, n = df["Close"].iloc[-1], len(df)
    for i in range(n - 2, max(0, n - OB_LOOKBACK - 1), -1):
        o, c = df["Open"].iloc[i], df["Close"].iloc[i]
        h, l = df["High"].iloc[i], df["Low"].iloc[i]
        if direction == "BUY"  and c < o:
            ob = {"low": l, "high": h, "mid": (h + l) / 2}
            if ob["low"] - atr_val <= price <= ob["high"] + atr_val:
                log.info(f"OB BUY: [{ob['low']:.5g}–{ob['high']:.5g}]")
                return ob
        elif direction == "SELL" and c > o:
            ob = {"low": l, "high": h, "mid": (h + l) / 2}
            if ob["low"] - atr_val <= price <= ob["high"] + atr_val:
                log.info(f"OB SELL: [{ob['low']:.5g}–{ob['high']:.5g}]")
                return ob
    log.info(f"OB: none near price for {direction}.")
    return None


def ob_overlaps_fvg(ob: dict, fvg: dict, atr_val: float) -> bool:
    result = not (ob["high"] < fvg["low"] - atr_val or ob["low"] > fvg["high"] + atr_val)
    log.info(f"OB∩FVG: {result}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-CONFLUENCE SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def analyse(df: pd.DataFrame, htf_bias: str | None, min_atr: float) -> dict | None:
    atr_val = calc_atr(df).iloc[-1]
    if pd.isna(atr_val) or atr_val < min_atr:
        log.info(f"ATR {atr_val:.5g} < min {min_atr} — skip.")
        return None

    price = df["Close"].iloc[-1]

    if htf_bias is None:
        log.info("L1 FAIL: no HTF bias.");  return None

    sh_idx, sl_idx = find_swing_highs_lows(df)

    pd_zone = get_pd_zone(df, sh_idx, sl_idx)
    if pd_zone is None:
        log.info("L3 FAIL: P/D zone undefined.");  return None

    if   htf_bias == "BUY"  and pd_zone == "discount": direction = "BUY"
    elif htf_bias == "SELL" and pd_zone == "premium":  direction = "SELL"
    else:
        log.info(f"L3 FAIL: bias={htf_bias} zone={pd_zone}.");  return None

    sweep_dir, sweep_lvl = detect_sweep(df, atr_val)
    if sweep_dir != direction:
        log.info(f"L4 FAIL: sweep={sweep_dir} need {direction}.");  return None

    struct_dir, struct_lvl, struct_type = detect_structure_break(df, sh_idx, sl_idx)
    if struct_dir != direction:
        log.info(f"L5 FAIL: structure={struct_dir} need {direction}.");  return None

    fvg = find_fvg(df, direction, atr_val)
    if fvg is None:
        log.info("L6 FAIL: no FVG.");  return None

    ob = find_order_block(df, direction, atr_val)
    if ob is None:
        log.info("L7 FAIL: no OB.");  return None
    if not ob_overlaps_fvg(ob, fvg, atr_val):
        log.info("L7 FAIL: OB not confluent with FVG.");  return None

    log.info(f"✅ All 7 layers aligned! {direction} @ {price:.5g}")

    if direction == "BUY":
        sl = ob["low"] - atr_val * SL_ATR_MULT
        tp = (df["High"].iloc[sh_idx]
              if sh_idx is not None and df["High"].iloc[sh_idx] > price
              else price + TP_ATR_MULT * atr_val)
        if tp <= price:
            tp = price + TP_ATR_MULT * atr_val
    else:
        sl = ob["high"] + atr_val * SL_ATR_MULT
        tp = (df["Low"].iloc[sl_idx]
              if sl_idx is not None and df["Low"].iloc[sl_idx] < price
              else price - TP_ATR_MULT * atr_val)
        if tp >= price:
            tp = price - TP_ATR_MULT * atr_val

    rr = round(abs(tp - price) / abs(sl - price), 2) if abs(sl - price) > 0 else 0

    return {
        "direction":      direction,
        "price":          price,
        "sl":             sl,
        "tp":             tp,
        "rr":             rr,
        "atr":            atr_val,
        "ob":             ob,
        "fvg":            fvg,
        "sweep_level":    sweep_lvl,
        "struct_level":   struct_lvl,
        "struct_type":    struct_type or "CHoCH",
        "timestamp":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE FORMATTING
# ══════════════════════════════════════════════════════════════════════════════
def fmt_price(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"

def calc_pips(a: float, b: float, pip_size: float) -> int:
    return round(abs(a - b) / pip_size)


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════
def format_entry_alert(signal: dict, asset: dict) -> str:
    emoji = "🟢" if signal["direction"] == "BUY" else "🔴"
    d     = asset["decimals"]
    return "\n".join([
        f"{emoji} <b>{asset['name']} — {signal['direction']}</b>",
        "",
        f"<b>Entry:</b> {fmt_price(signal['price'], d)}",
        f"<b>Take Profit:</b> {fmt_price(signal['tp'], d)}",
        f"<b>Stop Loss:</b> {fmt_price(signal['sl'], d)}",
    ])

def format_tp_alert(trade: dict, asset: dict, close_price: float) -> str:
    d = asset["decimals"]
    return "\n".join([
        f"✅ <b>{asset['name']} — TARGET HIT</b>",
        "",
        f"<b>Entry:</b> {fmt_price(trade['entry'], d)}",
        f"<b>Take Profit:</b> {fmt_price(trade['tp'], d)}",
        f"<b>Stop Loss:</b> {fmt_price(trade['sl'], d)}",
    ])

def format_sl_alert(trade: dict, asset: dict, close_price: float) -> str:
    d = asset["decimals"]
    return "\n".join([
        f"❌ <b>{asset['name']} — STOP LOSS HIT</b>",
        "",
        f"<b>Entry:</b> {fmt_price(trade['entry'], d)}",
        f"<b>Take Profit:</b> {fmt_price(trade['tp'], d)}",
        f"<b>Stop Loss:</b> {fmt_price(trade['sl'], d)}",
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  PER-ASSET TRADE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def open_trade(asset: dict, signal: dict) -> None:
    sym = asset["symbol"]
    with _trades_lock:
        _open_trades[sym] = {
            "direction": signal["direction"],
            "entry":     signal["price"],
            "sl":        signal["sl"],
            "tp":        signal["tp"],
            "rr":        signal["rr"],
            "atr":       signal["atr"],
            "opened_at": signal["timestamp"],
        }
    log.info(
        f"[{asset['name']}] Trade opened: {signal['direction']} @ "
        f"{fmt_price(signal['price'], asset['decimals'])}  "
        f"SL={fmt_price(signal['sl'], asset['decimals'])}  "
        f"TP={fmt_price(signal['tp'], asset['decimals'])}"
    )

def close_trade(asset: dict) -> None:
    with _trades_lock:
        _open_trades.pop(asset["symbol"], None)
    log.info(f"[{asset['name']}] Trade closed.")


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE JOURNAL  —  persist every close to JSON
# ══════════════════════════════════════════════════════════════════════════════
_journal_lock = threading.Lock()

def record_closed_trade(asset: dict, trade: dict, result: str, exit_price: float) -> None:
    """Append one closed-trade record to the JSON journal."""
    direction  = trade["direction"]
    pip_size   = asset["pip_size"]
    signed_pips = round(
        (exit_price - trade["entry"]) / pip_size if direction == "BUY"
        else (trade["entry"] - exit_price) / pip_size
    )
    now_utc = datetime.now(timezone.utc)
    record = {
        "date":      now_utc.strftime("%Y-%m-%d"),
        "time":      now_utc.strftime("%H:%M UTC"),
        "asset":     asset["name"],
        "direction": direction,
        "entry":     trade["entry"],
        "exit":      exit_price,
        "pips":      signed_pips,
        "result":    result,          # "TP" or "SL"
    }
    with _journal_lock:
        records: list = []
        if TRADE_LOG_FILE.exists():
            try:
                records = json.loads(TRADE_LOG_FILE.read_text())
            except json.JSONDecodeError:
                records = []
        records.append(record)
        TRADE_LOG_FILE.write_text(json.dumps(records, indent=2))
    log.info(f"[{asset['name']}] Logged: {result}  {signed_pips:+d} pips")

def check_trade(asset: dict, current_price: float) -> bool:
    """Check SL/TP for open trade. Returns True if the trade was closed."""
    with _trades_lock:
        trade = _open_trades.get(asset["symbol"])
    if trade is None:
        return False

    direction, tp, sl = trade["direction"], trade["tp"], trade["sl"]
    tp_hit = (direction == "BUY"  and current_price >= tp) or \
             (direction == "SELL" and current_price <= tp)
    sl_hit = (direction == "BUY"  and current_price <= sl) or \
             (direction == "SELL" and current_price >= sl)

    if tp_hit:
        log.info(f"[{asset['name']}] TP hit @ {current_price:.5g}")
        record_closed_trade(asset, trade, "TP", current_price)
        send_telegram(format_tp_alert(trade, asset, current_price))
        close_trade(asset)
        return True
    if sl_hit:
        log.info(f"[{asset['name']}] SL hit @ {current_price:.5g}")
        record_closed_trade(asset, trade, "SL", current_price)
        send_telegram(format_sl_alert(trade, asset, current_price))
        close_trade(asset)
        return True

    log.info(
        f"[{asset['name']}] {direction} open @ "
        f"{fmt_price(trade['entry'], asset['decimals'])}  "
        f"live={fmt_price(current_price, asset['decimals'])}  "
        f"SL={fmt_price(sl, asset['decimals'])}  TP={fmt_price(tp, asset['decimals'])}"
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  CHART RENDERER
# ══════════════════════════════════════════════════════════════════════════════
_DARK_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit",
    ),
    gridstyle="--",
    gridcolor="#2d2d3a",
    facecolor="#131722",
    figcolor="#131722",
    y_on_right=True,
)


def render_chart(df: pd.DataFrame, signal: dict, asset: dict) -> Path:
    """
    Render the last 100 15m candles with all SMC overlays and return the
    path to a temporary PNG file (caller must delete it after sending).
    """
    direction   = signal["direction"]
    ob          = signal["ob"]
    fvg         = signal["fvg"]
    tp          = signal["tp"]
    sl          = signal["sl"]
    price       = signal["price"]
    atr_val     = signal["atr"]
    sweep_lvl   = signal.get("sweep_level")
    struct_lvl  = signal.get("struct_level")
    struct_type = signal.get("struct_type", "CHoCH")
    d           = asset["decimals"]

    df_chart = df.tail(100).copy()
    n        = len(df_chart)

    fig, axes = mpf.plot(
        df_chart,
        type="candle",
        style=_DARK_STYLE,
        returnfig=True,
        figsize=(14, 7),
        tight_layout=True,
        title=(
            f"\n{asset['name']}  ·  15m  ·  {direction}  "
            f"@  {fmt_price(price, d)}    [{signal['timestamp']}]"
        ),
    )
    ax = axes[0]
    ax.title.set_color("#e0e0e0")
    ax.title.set_fontsize(11)

    x0, x1 = -0.5, n - 0.5        # full-width x span for rectangles

    # ── Order Block (OB) ──────────────────────────────────────────────────────
    ob_clr = "#1976D2" if direction == "BUY" else "#F57C00"
    ax.add_patch(mpatches.Rectangle(
        (x0, ob["low"]), x1 - x0, ob["high"] - ob["low"],
        linewidth=1.2, edgecolor=ob_clr, facecolor=ob_clr, alpha=0.18, zorder=2,
    ))
    ax.annotate(
        "OB", xy=(0.99, (ob["low"] + ob["high"]) / 2),
        xycoords=("axes fraction", "data"),
        ha="right", va="center", fontsize=8, color=ob_clr, fontweight="bold",
    )

    # ── Fair Value Gap (FVG) ──────────────────────────────────────────────────
    fvg_clr = "#00897B" if direction == "BUY" else "#C62828"
    ax.add_patch(mpatches.Rectangle(
        (x0, fvg["low"]), x1 - x0, fvg["high"] - fvg["low"],
        linewidth=1.2, edgecolor=fvg_clr, facecolor=fvg_clr, alpha=0.22, zorder=2,
    ))
    ax.annotate(
        "FVG", xy=(0.99, (fvg["low"] + fvg["high"]) / 2),
        xycoords=("axes fraction", "data"),
        ha="right", va="center", fontsize=8, color=fvg_clr, fontweight="bold",
    )

    # ── Take Profit — solid green ──────────────────────────────────────────────
    ax.axhline(tp, color="#26a69a", linewidth=1.6, linestyle="-", zorder=3)
    ax.annotate(
        f"TP  {fmt_price(tp, d)}", xy=(0.01, tp),
        xycoords=("axes fraction", "data"),
        ha="left", va="bottom", fontsize=8, color="#26a69a", fontweight="bold",
    )

    # ── Stop Loss — solid red ─────────────────────────────────────────────────
    ax.axhline(sl, color="#ef5350", linewidth=1.6, linestyle="-", zorder=3)
    ax.annotate(
        f"SL  {fmt_price(sl, d)}", xy=(0.01, sl),
        xycoords=("axes fraction", "data"),
        ha="left", va="top", fontsize=8, color="#ef5350", fontweight="bold",
    )

    # ── Entry price — white dashed ─────────────────────────────────────────────
    ax.axhline(price, color="#ffffff", linewidth=0.9, linestyle="--",
               alpha=0.45, zorder=3)

    # ── CHoCH / BOS — yellow dotted ───────────────────────────────────────────
    if struct_lvl is not None:
        ax.axhline(struct_lvl, color="#FFD54F", linewidth=1.3,
                   linestyle=":", zorder=3)
        ax.annotate(
            struct_type, xy=(0.5, struct_lvl),
            xycoords=("axes fraction", "data"),
            ha="center", va="bottom", fontsize=8, color="#FFD54F", fontweight="bold",
        )

    # ── Liquidity Sweep — cyan arrow ──────────────────────────────────────────
    if sweep_lvl is not None:
        arrow_x   = max(0, n - SWEEP_WINDOW - 2)
        dy        = atr_val * 1.5
        if direction == "BUY":          # EQL swept → bullish arrow pointing up
            ax.annotate(
                "", xy=(arrow_x, sweep_lvl + dy),
                xytext=(arrow_x, sweep_lvl - dy * 0.4),
                arrowprops=dict(arrowstyle="->", color="#80DEEA", lw=1.8),
                zorder=4,
            )
            ax.annotate(
                "Sweep", xy=(arrow_x, sweep_lvl - dy * 0.5),
                ha="center", va="top", fontsize=7.5, color="#80DEEA",
            )
        else:                           # EQH swept → bearish arrow pointing down
            ax.annotate(
                "", xy=(arrow_x, sweep_lvl - dy),
                xytext=(arrow_x, sweep_lvl + dy * 0.4),
                arrowprops=dict(arrowstyle="->", color="#80DEEA", lw=1.8),
                zorder=4,
            )
            ax.annotate(
                "Sweep", xy=(arrow_x, sweep_lvl + dy * 0.5),
                ha="center", va="bottom", fontsize=7.5, color="#80DEEA",
            )

    tmp = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(tmp, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Chart saved → {tmp.name}")
    return tmp


# ══════════════════════════════════════════════════════════════════════════════
#  TELEGRAM PHOTO SENDER
# ══════════════════════════════════════════════════════════════════════════════
def send_telegram_photo(photo_path: Path, caption: str) -> bool:
    """Upload a chart PNG via sendPhoto with the alert text as caption."""
    if not BOT_TOKEN or not CHAT_ID:
        log.error("Missing Telegram credentials.")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as fh:
            r = requests.post(
                url,
                data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                files={"photo": fh},
                timeout=30,
            )
        r.raise_for_status()
        log.info("Telegram photo sent.")
        return True
    except requests.RequestException as e:
        log.error(f"Telegram photo error: {e}")
        # Fall back to plain text so the alert is never silently lost
        send_telegram(caption)
        return False
    finally:
        photo_path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-ASSET SCAN  (runs in its own thread)
# ══════════════════════════════════════════════════════════════════════════════
def scan_asset(asset: dict) -> None:
    sym  = asset["symbol"]
    name = asset["name"]

    # ── Per-asset TF config (falls back to global defaults) ───────────────────
    interval     = asset.get("interval",     INTERVAL)
    htf1         = asset.get("htf1",         HTF1)
    htf2         = asset.get("htf2",         HTF2)
    exec_period  = asset.get("exec_period",  "5d")
    htf1_period  = asset.get("htf1_period",  "10d")
    htf2_period  = asset.get("htf2_period",  "60d")

    log.info(f"[{name}] ── scan start ({interval} / {htf1} / {htf2}) ──")

    df_exec = fetch_ohlcv(sym, interval, exec_period)
    if df_exec is None:
        return

    current_price = df_exec["Close"].iloc[-1]

    # 1. Check open trade first
    with _trades_lock:
        has_trade = sym in _open_trades
    if has_trade:
        check_trade(asset, current_price)
        with _trades_lock:
            still_open = sym in _open_trades
        if still_open:
            return   # trade active — skip signal scan

    # 2. Fetch HTF data for bias
    df_h1  = fetch_ohlcv(sym, htf1, htf1_period)
    # Avoid double-fetching when both HTFs are the same interval (e.g. BTC 5m/5m)
    df_h2  = df_h1 if htf1 == htf2 else fetch_ohlcv(sym, htf2, htf2_period)
    htf_bias = get_htf_bias(df_h1, df_h2)

    # 3. Run 7-layer engine
    signal = analyse(df_exec, htf_bias, asset["min_atr"])
    if signal is None:
        log.info(f"[{name}] No confluence signal.")
        return

    log.info(f"[{name}] Signal: {signal['direction']} @ {signal['price']:.5g}")
    caption = format_entry_alert(signal, asset)
    try:
        chart_path = render_chart(df_exec, signal, asset)
        send_telegram_photo(chart_path, caption)
    except Exception as e:
        log.error(f"[{name}] Chart render failed: {e} — sending text only.")
        send_telegram(caption)
    open_trade(asset, signal)


# ══════════════════════════════════════════════════════════════════════════════
#  WEEKEND HIBERNATION + CONCURRENT SCAN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
# Forex/index sessions are closed from Friday 22:00 UTC to Sunday 21:30 UTC.
# The bot sleeps through this window to avoid processing data gaps / fake wicks.
_was_hibernating: bool = False


def is_weekend_session() -> bool:
    """
    Returns True when markets are closed:
      Friday  ≥ 22:00 UTC
      Saturday (all day)
      Sunday  < 21:30 UTC   (resume 30 min before the 22:00 open)
    """
    now  = datetime.now(timezone.utc)
    day  = now.weekday()          # 0 = Mon … 4 = Fri, 5 = Sat, 6 = Sun
    mins = now.hour * 60 + now.minute
    if day == 4 and mins >= 22 * 60:   return True   # Friday close
    if day == 5:                        return True   # Saturday
    if day == 6 and mins < 21 * 60 + 30: return True # Sunday pre-open
    return False


def run_all_scans() -> None:
    global _was_hibernating

    # ── Weekend gate ──────────────────────────────────────────────────────────
    if is_weekend_session():
        if not _was_hibernating:
            _was_hibernating = True
            log.info("Weekend — markets closed. Bot hibernating.")
            send_telegram(
                "🌙 <b>Weekend Hibernation</b>\n\n"
                "Markets are closed. The bot is pausing scans until\n"
                "30 minutes before Sunday market open (21:30 UTC).\n\n"
                "<i>SMC Bot will resume automatically.</i>"
            )
        return

    if _was_hibernating:
        _was_hibernating = False
        log.info("Markets reopening — resuming scans.")
        send_telegram(
            "☀️ <b>Markets Reopening</b>\n\n"
            "Weekend hibernation ended. Resuming live scans across\n"
            "all 6 assets. First scan running now…"
        )

    # ── Resilient concurrent scan (swing assets only — BTC has own cycle) ───────
    _run_pool(SWING_ASSETS, label="swing")


def _run_pool(assets: list[dict], label: str = "") -> None:
    """Submit a list of assets to the thread pool and collect results."""
    tag = f"[{label}] " if label else ""
    log.info(f"═══ {tag}Scanning {len(assets)} asset(s) ═══")
    try:
        with ThreadPoolExecutor(max_workers=len(assets)) as pool:
            futures = {pool.submit(scan_asset, asset): asset for asset in assets}
            for fut in as_completed(futures):
                asset = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    log.error(f"[{asset['name']}] Unhandled scan error: {e}")
    except Exception as e:
        log.error(f"{tag}Scan runner failed (will retry next cycle): {e}")
    log.info(f"═══ {tag}Scans complete ═══")


def run_btc_scans() -> None:
    """BTC/scalp scan — runs every minute, no weekend gate (crypto is 24/7)."""
    _run_pool(SCALP_ASSETS, label="scalp")


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def _load_journal() -> list[dict]:
    """Return all trade records from the JSON journal (safe — never raises)."""
    with _journal_lock:
        if not TRADE_LOG_FILE.exists():
            return []
        try:
            return json.loads(TRADE_LOG_FILE.read_text())
        except Exception:
            return []


def _pip_bar(pips: int) -> str:
    """Compact sparkline: ████ for positives, ▒▒▒▒ for negatives."""
    blocks = min(abs(pips) // 20, 10)      # 1 block per 20 pips, cap at 10
    if pips >= 0:
        return "█" * blocks if blocks else "▪"
    return "▒" * blocks if blocks else "▪"


def _fmt_pips(pips: int) -> str:
    sign = "+" if pips >= 0 else ""
    return f"{sign}{pips}"


def _stats(records: list[dict]) -> dict:
    total  = len(records)
    wins   = sum(1 for r in records if r["result"] == "TP")
    losses = total - wins
    net    = sum(r["pips"] for r in records)
    wr     = round(wins / total * 100, 1) if total else 0.0
    by_asset: dict[str, int] = {}
    for r in records:
        by_asset[r["asset"]] = by_asset.get(r["asset"], 0) + r["pips"]
    return {"total": total, "wins": wins, "losses": losses,
            "net": net, "wr": wr, "by_asset": by_asset}


def _asset_rows(by_asset: dict[str, int]) -> str:
    if not by_asset:
        return "  —  no trades"
    lines = []
    for name in [a["name"] for a in ASSETS]:   # respect fixed display order
        if name in by_asset:
            p = by_asset[name]
            bar = _pip_bar(p)
            lines.append(f"  {name:<10}  {_fmt_pips(p):>7} pips  {bar}")
    return "\n".join(lines)


def send_daily_report() -> None:
    """Send daily performance summary — called Mon–Thu at 22:00 UTC."""
    today     = datetime.now(timezone.utc).date()
    today_str = today.strftime("%Y-%m-%d")
    all_rec   = _load_journal()
    day_rec   = [r for r in all_rec if r["date"] == today_str]
    s         = _stats(day_rec)

    now_label = today.strftime("%A, %d %B %Y")
    wr_emoji  = "🟢" if s["wr"] >= 60 else ("🟡" if s["wr"] >= 40 else "🔴")
    net_emoji = "📈" if s["net"] >= 0 else "📉"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊  <b>DAILY PERFORMANCE REPORT</b>",
        f"🗓  {now_label}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📋  <b>SESSION SUMMARY</b>",
        f"  Total Trades   │  <b>{s['total']}</b>",
        f"  ✅ Targets Hit │  <b>{s['wins']}</b>",
        f"  ❌ Stops Hit   │  <b>{s['losses']}</b>",
        f"  {wr_emoji} Win Rate    │  <b>{s['wr']}%</b>",
        "",
        f"  {net_emoji} Net Pips    │  <b>{_fmt_pips(s['net'])}</b>",
        "",
        "💹  <b>PIPS BY ASSET</b>",
        _asset_rows(s["by_asset"]),
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🤖  <i>SMC Bot  ·  4 Assets  ·  15m / 1H / 4H</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    msg = "\n".join(lines)
    log.info("Sending daily report.")
    send_telegram(msg)


def send_weekly_report() -> None:
    """Send full weekly + MTD summary — called every Friday at 22:00 UTC."""
    today      = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())   # Monday
    month_str  = today.strftime("%Y-%m")

    all_rec    = _load_journal()
    week_rec   = [r for r in all_rec
                  if date.fromisoformat(r["date"]) >= week_start]
    month_rec  = [r for r in all_rec if r["date"].startswith(month_str)]

    ws = _stats(week_rec)
    ms = _stats(month_rec)

    week_label  = f"{week_start.strftime('%d %b')} – {today.strftime('%d %b %Y')}"
    month_label = today.strftime("%B %Y")
    wr_emoji_w  = "🟢" if ws["wr"] >= 60 else ("🟡" if ws["wr"] >= 40 else "🔴")
    wr_emoji_m  = "🟢" if ms["wr"] >= 60 else ("🟡" if ms["wr"] >= 40 else "🔴")
    net_emoji_w = "📈" if ws["net"] >= 0 else "📉"
    net_emoji_m = "📈" if ms["net"] >= 0 else "📉"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📋  <b>WEEKLY PERFORMANCE REPORT</b>",
        f"📅  Week of {week_label}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📊  <b>THIS WEEK</b>",
        f"  Total Trades   │  <b>{ws['total']}</b>",
        f"  ✅ Targets Hit │  <b>{ws['wins']}</b>",
        f"  ❌ Stops Hit   │  <b>{ws['losses']}</b>",
        f"  {wr_emoji_w} Win Rate    │  <b>{ws['wr']}%</b>",
        f"  {net_emoji_w} Net Pips   │  <b>{_fmt_pips(ws['net'])}</b>",
        "",
        "💹  <b>PIPS BY ASSET — WEEK</b>",
        _asset_rows(ws["by_asset"]),
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"🗓  <b>MONTH-TO-DATE  ({month_label})</b>",
        f"  Total Trades   │  <b>{ms['total']}</b>",
        f"  ✅ Targets Hit │  <b>{ms['wins']}</b>",
        f"  ❌ Stops Hit   │  <b>{ms['losses']}</b>",
        f"  {wr_emoji_m} Win Rate    │  <b>{ms['wr']}%</b>",
        f"  {net_emoji_m} Cumul. Pips│  <b>{_fmt_pips(ms['net'])}</b>",
        "",
        "💹  <b>PIPS BY ASSET — MTD</b>",
        _asset_rows(ms["by_asset"]),
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🤖  <i>SMC Bot  ·  4 Assets  ·  15m / 1H / 4H</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    msg = "\n".join(lines)
    log.info("Sending weekly report.")
    send_telegram(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  KEEP-ALIVE WEB SERVER
# ══════════════════════════════════════════════════════════════════════════════
_flask_app = Flask(__name__)

@_flask_app.route("/")
def health():
    return "OK", 200

def start_web_server() -> None:
    log.info("Keep-alive server on port 5000")
    _flask_app.run(host="0.0.0.0", port=5000, use_reloader=False)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        log.error("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in Secrets.")
        return

    log.info(
        f"Multi-asset SMC Bot starting — "
        f"{len(SWING_ASSETS)} swing asset(s) @ {SCAN_EVERY_MINUTES}m, "
        f"{len(SCALP_ASSETS)} scalp asset(s) @ 1m."
    )

    threading.Thread(target=start_web_server, daemon=True).start()

    swing_lines = "\n".join(f"  • {a['name']}  (15m / 1H / 4H)" for a in SWING_ASSETS)
    scalp_lines = "\n".join(
        f"  • {a['name']}  ({a.get('interval','15m')} / {a.get('htf1','1h')} / {a.get('htf2','4h')})"
        for a in SCALP_ASSETS
    )
    send_telegram(
        "🤖 <b>Multi-Asset SMC Bot — Advanced Engine</b>\n\n"
        f"📊 <b>Swing assets</b> (scan every 15 min):\n{swing_lines}\n\n"
        f"⚡ <b>Scalp assets</b> (scan every 1 min):\n{scalp_lines}\n\n"
        "🔬 <b>7 SMC confluence layers per asset</b>\n"
        "Signals fire only when ALL layers align on that asset's TF stack."
    )

    # ── Immediate scans on startup ─────────────────────────────────────────────
    log.info("Running immediate BTC scalp test scan…")
    run_btc_scans()
    run_all_scans()

    # ── Schedulers ────────────────────────────────────────────────────────────
    schedule.every(SCAN_EVERY_MINUTES).minutes.do(run_all_scans)
    schedule.every(1).minutes.do(run_btc_scans)

    # ── Performance reports  (22:00 UTC = 5 PM EST / US session close) ────────
    schedule.every().monday.at("22:00").do(send_daily_report)
    schedule.every().tuesday.at("22:00").do(send_daily_report)
    schedule.every().wednesday.at("22:00").do(send_daily_report)
    schedule.every().thursday.at("22:00").do(send_daily_report)
    schedule.every().friday.at("22:00").do(send_weekly_report)   # weekly on Fri

    log.info(
        f"Schedulers active — "
        f"swing: every {SCAN_EVERY_MINUTES}m | scalp: every 1m | "
        f"reports: Mon–Thu 22:00 UTC | weekly: Fri 22:00 UTC."
    )
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            log.error(f"Scheduler tick error (continuing): {e}")
        time.sleep(15)


if __name__ == "__main__":
    main()
