#!/usr/bin/env python3
"""
BTC + XAUUSD Multi Mini-Bot Analyzer  (rule-based, NO AI)
----------------------------------------------------------
Mini bots:  Trend | Momentum | Volatility | Levels | Volume | HigherTimeframe
Boss bot :  MasterBot  -> combines votes -> BUY / SELL / WAIT
Helper   :  RiskBot    -> Entry, Stop Loss, TP1, TP2, position size

Install :  pip install requests pandas numpy
Run once:  python trading_bot.py
Watch   :  python trading_bot.py --watch 300        (every 5 minutes)
Options :  --balance 1000 --risk 1                 (account $ and risk % per trade)

This bot ONLY ANALYSES. It does not place orders. Not financial advice.
"""
import argparse, json, os, sys, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def get_btc(limit=1000):
    errors = []
    # 1) Bitstamp (works from US cloud servers such as GitHub)
    try:
        r = requests.get("https://www.bitstamp.net/api/v2/ohlc/btcusd/",
                         params={"step": 3600, "limit": limit}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        d = pd.DataFrame(r.json()["data"]["ohlc"])
        d.index = pd.to_datetime(d["timestamp"].astype(int), unit="s")
        return d[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    except Exception as e:
        errors.append(f"bitstamp: {e}")
    # 2) Binance public-data mirror, then normal Binance
    for host in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            r = requests.get(host + "/api/v3/klines",
                             params={"symbol": "BTCUSDT", "interval": "1h", "limit": limit},
                             headers=HEADERS, timeout=20)
            r.raise_for_status()
            df = pd.DataFrame(r.json(), columns=["t", "open", "high", "low", "close", "volume",
                                                 "ct", "qv", "n", "tb", "tq", "i"])
            df.index = pd.to_datetime(df["t"], unit="ms")
            return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception as e:
            errors.append(f"{host}: {e}")
    raise RuntimeError("Could not download BTC data: " + " | ".join(errors))


def get_gold():
    for sym in ("XAUUSD=X", "GC=F"):  # spot first, futures as backup
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                             params={"interval": "1h", "range": "60d"},
                             headers=HEADERS, timeout=15)
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame(q, index=pd.to_datetime(res["timestamp"], unit="s"))
            df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            df["volume"] = df["volume"].fillna(0)
            if len(df) > 250:
                return df
        except Exception:
            continue
    raise RuntimeError("Could not download gold data")


def resample(df, rule="4h"):
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()

# ----------------------------------------------------------------------------
# INDICATORS
# ----------------------------------------------------------------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def macd_hist(s):
    m = ema(s, 12) - ema(s, 26)
    return m - ema(m, 9)

def adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()

# ----------------------------------------------------------------------------
# MINI BOTS  (each returns score from -1 (strong sell) to +1 (strong buy))
# ----------------------------------------------------------------------------
@dataclass
class Signal:
    bot: str
    score: float
    note: str


def clamp(x): return max(-1.0, min(1.0, float(x)))


def trend_bot(df):
    c = df["close"]; e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    p = c.iloc[-1]; s = 0.0
    s += 0.4 if p > e20.iloc[-1] else -0.4
    s += 0.3 if e20.iloc[-1] > e50.iloc[-1] else -0.3
    s += 0.3 if e50.iloc[-1] > e200.iloc[-1] else -0.3
    slope = (e50.iloc[-1] - e50.iloc[-6]) / e50.iloc[-6] * 100
    txt = "Uptrend" if s > 0.5 else "Downtrend" if s < -0.5 else "Mixed trend"
    return Signal("TrendBot", clamp(s), f"{txt} (EMA50 slope {slope:+.2f}%)")


def momentum_bot(df):
    c = df["close"]; r = rsi(c).iloc[-1]; h = macd_hist(c)
    s = 0.5 if h.iloc[-1] > 0 else -0.5
    s += 0.2 if h.iloc[-1] > h.iloc[-2] else -0.2          # momentum growing?
    if r > 70: s -= 0.4; note = f"RSI {r:.0f} overbought"
    elif r < 30: s += 0.4; note = f"RSI {r:.0f} oversold"
    elif r > 55: s += 0.2; note = f"RSI {r:.0f} bullish zone"
    elif r < 45: s -= 0.2; note = f"RSI {r:.0f} bearish zone"
    else: note = f"RSI {r:.0f} neutral"
    return Signal("MomentumBot", clamp(s), f"{note}, MACD {'up' if h.iloc[-1] > 0 else 'down'}")


def volatility_bot(df):
    c = df["close"]; ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    up, lo = ma + 2 * sd, ma - 2 * sd
    pos = (c.iloc[-1] - lo.iloc[-1]) / (up.iloc[-1] - lo.iloc[-1] + 1e-9)   # 0..1
    width = (up - lo) / ma
    squeeze = width.iloc[-1] <= width.tail(100).quantile(0.2)
    s = clamp((0.5 - pos) * 1.4)                 # near top = sell bias, near bottom = buy bias
    note = f"Price at {pos*100:.0f}% of Bollinger band" + (" | SQUEEZE: big move coming" if squeeze else "")
    return Signal("VolatilityBot", s * (0.5 if squeeze else 1), note)


def find_levels(df, look=60):
    d = df.tail(300); hi, lo = d["high"].values, d["low"].values
    sup, res = [], []
    for i in range(3, len(d) - 3):
        if hi[i] == hi[i - 3:i + 4].max(): res.append(hi[i])
        if lo[i] == lo[i - 3:i + 4].min(): sup.append(lo[i])
    p = df["close"].iloc[-1]
    s_below = max([x for x in sup if x < p], default=d["low"].min())
    r_above = min([x for x in res if x > p], default=d["high"].max())
    return s_below, r_above


def level_bot(df):
    p = df["close"].iloc[-1]; sup, res = find_levels(df); a = atr(df).iloc[-1]
    ds, dr = (p - sup) / a, (res - p) / a
    s = 0.0
    if ds < 1.0: s = 0.7; note = "Price sitting on support"
    elif dr < 1.0: s = -0.7; note = "Price hitting resistance"
    else: s = clamp((dr - ds) / (dr + ds) * 0.6); note = "Between levels"
    return Signal("LevelBot", clamp(s), f"{note} (S {sup:,.2f} | R {res:,.2f})")


def volume_bot(df):
    if df["volume"].tail(50).sum() == 0:
        return Signal("VolumeBot", 0.0, "No volume data")
    v = df["volume"]; ratio = v.iloc[-1] / v.tail(30).mean()
    direction = 1 if df["close"].iloc[-1] > df["open"].iloc[-1] else -1
    obv = (np.sign(df["close"].diff()).fillna(0) * v).cumsum()
    obv_up = obv.iloc[-1] > obv.iloc[-20]
    s = 0.4 * (1 if obv_up else -1) + (0.3 * direction if ratio > 1.3 else 0)
    return Signal("VolumeBot", clamp(s), f"Volume x{ratio:.1f} avg, OBV {'rising' if obv_up else 'falling'}")


def htf_bot(df4h):
    c = df4h["close"]; e50, e200 = ema(c, 50), ema(c, 200)
    s = (0.5 if c.iloc[-1] > e50.iloc[-1] else -0.5) + (0.5 if e50.iloc[-1] > e200.iloc[-1] else -0.5)
    return Signal("HigherTF(4H)Bot", clamp(s), "4H trend " + ("UP" if s > 0 else "DOWN" if s < 0 else "flat"))

# ----------------------------------------------------------------------------
# RISK BOT
# ----------------------------------------------------------------------------
def risk_bot(name, side, price, a, balance, risk_pct, sup, res):
    dist = 1.5 * a
    if side == "BUY":
        sl = min(price - dist, sup - 0.2 * a); sl_d = price - sl
        tp1, tp2 = price + 1.5 * sl_d, price + 3.0 * sl_d
    else:
        sl = max(price + dist, res + 0.2 * a); sl_d = sl - price
        tp1, tp2 = price - 1.5 * sl_d, price - 3.0 * sl_d
    risk_usd = balance * risk_pct / 100
    if name == "XAUUSD":
        size = risk_usd / (sl_d * 100); unit = "lots (1 lot = 100 oz)"
    else:
        size = risk_usd / sl_d; unit = "BTC"
    return dict(entry=price, sl=sl, tp1=tp1, tp2=tp2, size=size, unit=unit, risk_usd=risk_usd)

# ----------------------------------------------------------------------------
# SNIPER BOT  -> perfect TIME + perfect ENTRY (+ checks its own past results)
# ----------------------------------------------------------------------------
from datetime import datetime, timezone


def session_quality(name, now):
    """Returns (quality 0..1, text, hours_to_best_window). All times are UTC."""
    h, wd = now.hour + now.minute / 60, now.weekday()
    if name == "XAUUSD":
        if wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 22):
            return 0.0, "Gold market CLOSED (weekend)", None
        wins = [(12, 16, 1.0, "London + New York overlap (BEST time)"),
                (7, 12, 0.7, "London session (good)"),
                (16, 20, 0.6, "New York afternoon (okay)")]
        low, best = "Asia / late session: low volume, fake moves likely", (12, 16)
    else:
        wins = [(13, 21, 1.0, "US session (BEST time for BTC)"),
                (7, 13, 0.7, "Europe session (good)")]
        low, best = "Quiet hours: low volume, fake moves likely", (13, 21)
    q, label = 0.3, low
    for s, e, qq, t in wins:
        if s <= h < e: q, label = qq, t
    if name != "XAUUSD" and wd >= 5:
        q, label = q * 0.7, label + " | weekend = thinner market"
    to_best = 0 if best[0] <= h < best[1] else (best[0] - h) % 24
    return q, label, to_best


def news_danger(now):
    """Only knows the monthly US jobs report (first Friday, ~12:30-14:30 UTC). Always check a calendar too."""
    return now.weekday() == 4 and now.day <= 7 and 11.5 <= now.hour + now.minute / 60 < 15


def candle_dir(o, h, l, c, po, pc):
    """+1 bullish reversal/confirmation candle, -1 bearish, 0 nothing."""
    body, rng = abs(c - o), h - l + 1e-9
    lower, upper = min(o, c) - l, h - max(o, c)
    if (c > o and pc < po and c >= po and o <= pc) or (lower > 2 * body and upper < body and c > o and body / rng > 0.05):
        return 1
    if (c < o and pc > po and c <= po and o >= pc) or (upper > 2 * body and lower < body and c < o and body / rng > 0.05):
        return -1
    return 0


def backtest_setup(df, lookback=900, horizon=30):
    """Replays the sniper rule on past candles: trend + pullback to EMA + confirmation candle.
    Entry next candle open, SL 1.5 ATR, TP1 = 1.5R. Returns (trades, win%, avg R)."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    e20, e50, e200 = (ema(df["close"], n).values for n in (20, 50, 200))
    a, r = atr(df).values, rsi(df["close"]).values
    R, i = [], max(210, len(df) - lookback)
    while i < len(df) - horizon - 1:
        side = 0
        if e20[i] > e50[i] > e200[i] and l[i] <= e20[i] + 0.3 * a[i] and c[i] > e50[i] and 40 < r[i] < 68:
            side = 1
        elif e20[i] < e50[i] < e200[i] and h[i] >= e20[i] - 0.3 * a[i] and c[i] < e50[i] and 32 < r[i] < 60:
            side = -1
        if side and candle_dir(o[i], h[i], l[i], c[i], o[i - 1], c[i - 1]) == side:
            ent, risk = o[i + 1], 1.5 * a[i]
            sl, tp = ent - side * risk, ent + side * 1.5 * risk
            out = 0.0
            for j in range(i + 1, i + 1 + horizon):
                if (side == 1 and l[j] <= sl) or (side == -1 and h[j] >= sl): out = -1.0; break
                if (side == 1 and h[j] >= tp) or (side == -1 and l[j] <= tp): out = 1.5; break
            R.append(out); i = j
        i += 1
    if not R: return 0, 0.0, 0.0
    return len(R), 100 * sum(x > 0 for x in R) / len(R), float(np.mean(R))


def sniper_bot(name, df, side, now=None):
    now = now or datetime.now(timezone.utc)
    q, s_txt, to_best = session_quality(name, now)
    news = news_danger(now)
    n, wr, avg_r = backtest_setup(df)
    out = dict(session=s_txt, quality=q, to_best=to_best, news=news, bt=(n, wr, avg_r),
               status="NO SETUP", entry=None, trigger=None, why=[])
    if side == "WAIT":
        return out
    sd = 1 if side == "BUY" else -1
    c = df["close"]; price = c.iloc[-1]; a = atr(df).iloc[-1]
    e20, e50, r = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1], rsi(c).iloc[-1]
    last, prev = df.iloc[-2], df.iloc[-3]               # last CLOSED candle + the one before
    cd = candle_dir(last["open"], last["high"], last["low"], last["close"], prev["open"], prev["close"])
    dist = (price - e20) / a * sd                       # + = stretched away from EMA20 in trade direction
    deep = (price - e50) / a * sd < -0.5                # pullback went through EMA50 = trend may be failing
    ok_zone, ok_candle = -0.5 <= dist <= 0.6, cd == sd
    ok_rsi = (r < 68) if sd == 1 else (r > 32)
    out["trigger"] = last["high"] if sd == 1 else last["low"]
    out["why"] = [f"Pullback zone (near EMA20): {'YES' if ok_zone else 'NO'}",
                  f"Confirmation candle: {'YES' if ok_candle else 'NO'}",
                  f"RSI not exhausted ({r:.0f}): {'YES' if ok_rsi else 'NO'}",
                  f"Good session: {'YES' if q >= 0.6 else 'NO'}",
                  f"No news danger: {'YES' if not news else 'NO'}"]
    if q < 0.6 or news:
        out["status"] = "WAIT FOR BETTER TIME"
    elif deep:
        out["status"] = "NO ENTRY - pullback too deep, trend may be failing"
    elif dist > 0.6:
        out["status"] = "TOO LATE - DON'T CHASE. Place LIMIT order"; out["entry"] = e20
    elif ok_zone and ok_candle and ok_rsi:
        out["status"] = "ENTER NOW"; out["entry"] = price
    elif ok_zone and ok_rsi:
        out["status"] = "ALMOST - enter only when price breaks the trigger level"; out["entry"] = out["trigger"]
    else:
        out["status"] = "WAIT"
    return out

# ----------------------------------------------------------------------------
# MASTER BOT
# ----------------------------------------------------------------------------
WEIGHTS = {"TrendBot": 0.22, "MomentumBot": 0.18, "VolatilityBot": 0.10,
           "LevelBot": 0.15, "VolumeBot": 0.10, "HigherTF(4H)Bot": 0.25}


def market_state(df):
    a = adx(df).iloc[-1]
    atr_pct = atr(df).iloc[-1] / df["close"].iloc[-1] * 100
    kind = "STRONG TREND" if a > 25 else "WEAK TREND" if a > 18 else "RANGING / SIDEWAYS"
    return kind, a, atr_pct


def analyze(name, df, balance, risk_pct):
    df4h = resample(df)
    sigs = [trend_bot(df), momentum_bot(df), volatility_bot(df),
            level_bot(df), volume_bot(df), htf_bot(df4h)]
    total = sum(WEIGHTS[s.bot] * s.score for s in sigs)
    state, a_val, atr_pct = market_state(df)

    # Master rules
    threshold = 0.30 if "TREND" in state else 0.45        # need stronger proof in sideways market
    htf = sigs[-1].score
    side = "BUY" if total >= threshold else "SELL" if total <= -threshold else "WAIT"
    if side == "BUY" and htf < 0: side = "WAIT"            # never fight the 4H trend
    if side == "SELL" and htf > 0: side = "WAIT"
    agree = sum(1 for s in sigs if s.score * total > 0)
    confidence = min(100, int(abs(total) * 120 + agree * 4))

    price = df["close"].iloc[-1]; a = atr(df).iloc[-1]
    sup, res = find_levels(df)

    print("\n" + "=" * 66)
    print(f" {name}   price {price:,.2f}   {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 66)
    print(f" Market state : {state}  (ADX {a_val:.0f}, volatility {atr_pct:.2f}% per candle)")
    print(" ---- Mini bot votes ----")
    for s in sigs:
        arrow = "BUY " if s.score > 0.15 else "SELL" if s.score < -0.15 else "FLAT"
        print(f"  {s.bot:<16}{arrow} {s.score:+.2f}  {s.note}")
    print(f" ---- MASTER BOT: score {total:+.2f}  ->  {side}  (confidence {confidence}%) ----")

    msg = f"{name} {side} | score {total:+.2f} | conf {confidence}% | {state}"

    sn = sniper_bot(name, df, side)
    print(" ---- SNIPER BOT (time + entry) ----")
    print(f"  Time now     : {sn['session']}")
    if sn["to_best"]:
        print(f"  Best window  : in about {sn['to_best']:.0f} hour(s) (UTC)")
    if sn["news"]:
        print("  !! US jobs report (NFP) window - huge spikes, AVOID trading !!")
    n, wr, avg_r = sn["bt"]
    if n:
        print(f"  Past check   : this setup on last data = {n} trades, {wr:.0f}% hit TP1 before SL, avg {avg_r:+.2f}R"
              + ("  (few trades, low trust)" if n < 20 else "") + "  [break-even = 40%]")
    if side != "WAIT":
        for w in sn["why"]: print(f"   - {w}")
        print(f"  SNIPER SAYS  : {sn['status']}")
        if sn["trigger"] is not None and sn["status"].startswith(("ALMOST", "WAIT")):
            print(f"  Trigger level: {sn['trigger']:,.2f}")
    msg += f"\nSniper: {sn['status']} | {sn['session']}"

    if side == "WAIT":
        print(" No clean setup. Best action: WAIT, do not force a trade.")
        print(f" Watch: support {sup:,.2f}  |  resistance {res:,.2f}")
    else:
        entry = sn["entry"] if sn["entry"] is not None else price
        r = risk_bot(name, side, entry, a, balance, risk_pct, sup, res)
        verdict = "TRADE NOW" if sn["status"] == "ENTER NOW" else "NOT YET - follow the Sniper instruction"
        print(f" ==== FINAL VERDICT: {verdict} ====")
        print(f"  Entry        : {r['entry']:,.2f}")
        print(f"  Stop Loss    : {r['sl']:,.2f}")
        print(f"  Take Profit 1: {r['tp1']:,.2f}   (close half, move SL to entry)")
        print(f"  Take Profit 2: {r['tp2']:,.2f}")
        print(f"  Position size: {r['size']:.4f} {r['unit']}  (risking ${r['risk_usd']:.2f})")
        msg += f"\nEntry {r['entry']:,.2f} SL {r['sl']:,.2f} TP1 {r['tp1']:,.2f} TP2 {r['tp2']:,.2f}"
    return msg, f"{side}|{sn['status']}", side


def telegram(text):
    tok, chat = os.getenv("TG_TOKEN"), os.getenv("TG_CHAT")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text}, timeout=10)
        except Exception as e:
            print("Telegram error:", e)


STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_state.json")


def load_state():
    try:
        with open(STATE_FILE) as f: return json.load(f)
    except Exception:
        return {}


def save_state(st):
    try:
        with open(STATE_FILE, "w") as f: json.dump(st, f)
    except Exception:
        pass


def run(balance, risk):
    state = load_state()
    for name, fn in (("BTCUSD", get_btc), ("XAUUSD", get_gold)):
        try:
            msg, key, side = analyze(name, fn(), balance, risk)
            last = state.get(name)
            # Telegram only when the signal CHANGED (first plain WAIT is not sent)
            if key != last and not (last is None and side == "WAIT"):
                telegram(msg)
                print(f"[{name}] Telegram sent (signal changed)")
            else:
                print(f"[{name}] No change, no Telegram message")
            state[name] = key
        except Exception as e:
            print(f"\n[{name}] error: {e}")
    save_state(state)
    print("\nNote: analysis only, not financial advice. Always use the Stop Loss.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="repeat every N seconds")
    ap.add_argument("--balance", type=float, default=1000)
    ap.add_argument("--risk", type=float, default=1.0, help="risk percent per trade")
    a = ap.parse_args()
    while True:
        run(a.balance, a.risk)
        if not a.watch: break
        time.sleep(a.watch)
