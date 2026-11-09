# -*- coding: utf-8 -*-
"""
fake_market -- a made-up NSE session so the AI layer runs without the trading app.

The private app fills these in-memory caches from Kite / yfinance and its own scanner.
Here every number is generated from a seeded random walk, so runs are repeatable.
NOTHING here is real market data and NOTHING here is the app's strategy: the conviction
score, tiers, PACE bands and scanner states are stand-ins with made-up rules.

    m = Market(workdir)        # builds history DBs + today's session up to 11:30
    m.ctx                      # ai_tools.Ctx the tools read from
    m.step()                   # one more minute: prices move, sampler runs
"""
import csv
import math
import os
import random
import sqlite3
import threading
import types
from collections import deque
from datetime import datetime, timedelta

import pandas as pd

import ai_tools as T

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY = "2026-11-06"                  # a Friday; fixed so every run gives the same answers
START = "11:30"                       # the session is pre-simulated up to this time

# symbol -> (sector index, previous close). Real F&O names, made-up prices.
UNIVERSE = {
    "HDFCBANK": ("NIFTY BANK", 1712.0), "ICICIBANK": ("NIFTY BANK", 1288.0),
    "AXISBANK": ("NIFTY BANK", 1176.0), "KOTAKBANK": ("NIFTY BANK", 1840.0),
    "SBIN": ("NIFTY PSU BANK", 812.0), "BANKBARODA": ("NIFTY PSU BANK", 246.0),
    "PNB": ("NIFTY PSU BANK", 104.0),
    "INFY": ("NIFTY IT", 1495.0), "TCS": ("NIFTY IT", 3120.0), "WIPRO": ("NIFTY IT", 248.0),
    "HCLTECH": ("NIFTY IT", 1530.0),
    "MARUTI": ("NIFTY AUTO", 12650.0), "M&M": ("NIFTY AUTO", 3240.0), "TATAMOTORS": ("NIFTY AUTO", 690.0),
    "HINDUNILVR": ("NIFTY FMCG", 2420.0), "ITC": ("NIFTY FMCG", 412.0), "NESTLEIND": ("NIFTY FMCG", 2260.0),
    "TATASTEEL": ("NIFTY METAL", 162.0), "HINDALCO": ("NIFTY METAL", 702.0),
    "NYKAA": ("NIFTY CONSUMER", 352.0), "DMART": ("NIFTY CONSUMER", 4010.0),
    "RELIANCE": ("NIFTY ENERGY", 2985.0), "ONGC": ("NIFTY ENERGY", 248.0), "NTPC": ("NIFTY ENERGY", 362.0),
}
INDEX_BASE = {"NIFTY 50": 25120.0, "NIFTY BANK": 56300.0, "NIFTY PSU BANK": 7150.0, "NIFTY IT": 36900.0,
              "NIFTY AUTO": 26800.0, "NIFTY FMCG": 55900.0, "NIFTY METAL": 9850.0,
              "NIFTY CONSUMER": 11900.0, "NIFTY ENERGY": 35600.0, "INDIA VIX": 12.4}
FNO_SYMBOLS = sorted(UNIVERSE)
SECTOR_MAP = {s: v[0] for s, v in UNIVERSE.items()}

# Day "stories" so the demo has something to find (drift in % per minute, from 09:15).
DRIFT = {"NYKAA": -0.017, "TATASTEEL": 0.012, "SBIN": 0.008, "INFY": -0.006, "BANKBARODA": 0.010,
         "ITC": 0.002, "MARUTI": -0.003, "HINDALCO": 0.007}
# After START: NYKAA slides ~0.1% a minute, so "alert me when nykaa goes below 348" fires in the demo.
DRIFT_LIVE = {"NYKAA": -0.10}
OPEN_IS_HIGH = {"WIPRO"}              # opens at the day high and slides: an "open = high" stock
OPEN_IS_LOW = {"PNB"}                 # opens at the day low and climbs: an "open = low" stock

_LOCKS = ("SESSION_RESULTS_LOCK", "SESSION_RESULTS_SHORT_LOCK", "OI_CACHE_LOCK", "SESSION_HILO_LOCK",
          "IMPLIED_BUDGET_LOCK", "_INTRADAY_CACHE_LOCK", "MARKET_REGIME_LOCK", "SCAN_LOCK", "KANBAN_LOCK",
          "WATCH_LOCK", "ALERT_LOCK", "AD_BREADTH_LOCK", "SECTOR_INTEL_LOCK", "LIVE_TICK_LOCK")


def _weekdays_before(day, n):
    d, out = datetime.strptime(day, "%Y-%m-%d"), []
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
    return sorted(out)


# ── stand-ins for the private engine functions (made-up rules) ────────────────
def fake_band(pace):
    """Stand-in PACE band names. The real app's bands are private."""
    if pace is None:
        return None
    return "LOW" if pace < 0.4 else "MID" if pace < 0.8 else "HIGH"


def _fake_conviction(state):
    def conv(sym, rvol, pct, sector):
        with state.SCAN_LOCK:                       # the real one takes SCAN_LOCK too
            h = sum(map(ord, sym)) % 7
            score = 30 + h * 9
            return score, ("SUPER" if score > 75 else "READY" if score > 50 else "LOADING"), h % 4, h % 3
    return conv


class Market:
    def __init__(self, workdir, seed=7, start=START):
        self.rng = random.Random(seed)
        # one generator per stock: adding a stock never changes another stock's day
        self._rs = {sym: random.Random("%d-%s" % (seed, sym)) for sym in UNIVERSE}
        self.dir = workdir
        os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
        self.past = _weekdays_before(TODAY, 25)
        self.saga_db = os.path.join(workdir, "data", "saga_obs.db")
        self.playbook_db = os.path.join(workdir, "data", "playbook.db")
        self._history()
        h, m = map(int, start.split(":"))
        self.now = datetime.strptime(TODAY, "%Y-%m-%d").replace(hour=h, minute=m)
        self._start = self.now
        self.state = self._state()
        self.ctx = T.Ctx(self.state, saga_db=self.saga_db, playbook_db=self.playbook_db,
                         conviction=_fake_conviction(self.state), short_conviction=_fake_conviction(self.state),
                         breadth=self._breadth, today=TODAY, now=self.now)
        self.ctx._pband = fake_band
        self._bars = {s: [] for s in UNIVERSE}       # (time, o, h, l, c, v)
        self._fv = {s: 0.0 for s in UNIVERSE}          # futures cumulative volume
        t = self.now.replace(hour=9, minute=15)
        self._open_px = {}
        while t <= self.now:
            self._minute(t)
            T.sample_once(self.ctx, t.timestamp())
            t += timedelta(minutes=1)

    # ── history: 25 past sessions in CSV + SQLite ────────────────────────────
    def _history(self):
        r = self.rng
        # Walk BACKWARDS from yesterday's close, so the last session ends exactly at the
        # universe's previous close and there is no jump into today.
        series = {}
        for s, (_sec, pdc) in UNIVERSE.items():
            cl = [pdc]
            for _ in self.past[1:]:
                cl.append(cl[-1] / (1 + r.gauss(0.0005, 0.013)))
            series[s] = cl[::-1]
        rows = []
        for i, d in enumerate(self.past):
            for s in UNIVERSE:
                c = series[s][i]
                c0 = series[s][i - 1] if i else c / (1 + r.gauss(0, 0.01))
                o = c0 * (1 + r.gauss(0, 0.004))
                hi, lo = max(o, c) * (1 + abs(r.gauss(0, 0.006))), min(o, c) * (1 - abs(r.gauss(0, 0.006)))
                rows.append((d, s, o, hi, lo, c, r.uniform(2e5, 6e6), (c / c0 - 1) * 100))
        with open(os.path.join(self.dir, "data", "daily_ohlc.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
            for d, s, o, h, l, c, v, _p in rows:
                w.writerow([d, s, round(o, 2), round(h, 2), round(l, 2), round(c, 2), int(v)])
        for p in (self.saga_db, self.playbook_db):
            if os.path.exists(p):
                os.remove(p)
        c = sqlite3.connect(self.saga_db)
        with open(os.path.join(HERE, "schema.sql"), encoding="utf-8") as f:
            c.executescript(f.read())
        for d, s, o, h, l, cl, v, p in rows:
            c.execute("INSERT INTO symbol_day(day,sym,open,high,low,close,volume,day_pct,oi_sess,max_pace,rvol_max)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (d, s, o, h, l, cl, v, p, r.gauss(0, 4), r.uniform(0.2, 1.4), r.uniform(0.5, 4.0)))
        stages = ["STAGE_A", "STAGE_B"]                 # stand-in names
        for d in self.past[-10:]:
            for s in r.sample(FNO_SYMBOLS, 3):
                st, side, t = r.choice(stages), r.choice(["LONG", "SHORT"]), "%02d:%02d:00" % (r.randint(9, 14), r.randint(0, 59))
                m60 = r.gauss(0.1, 0.8)
                c.execute("INSERT INTO signals(day,t,sym,stage,side) VALUES(?,?,?,?,?)", (d, t, s, st, side))
                c.execute("INSERT INTO outcomes(day,sym,stage,t,side,m15,m30,m60,res_10,src,bars_n)"
                          " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (d, s, st, t, side, m60 / 3, m60 / 2, m60, "WIN" if m60 > 0.5 else "LOSS" if m60 < -0.5
                           else "FLAT", "minute", 60))
        c.commit()
        c.close()
        c = sqlite3.connect(self.playbook_db)
        c.execute("CREATE TABLE oi_eod(date TEXT, symbol TEXT, oi_abs REAL, sess_pct REAL, tag TEXT)")
        c.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, date TEXT, symbol TEXT, tier TEXT,"
                  " pct_15m REAL, pct_30m REAL, pct_60m REAL)")
        oi = {s: r.uniform(1e6, 5e7) for s in UNIVERSE}
        for d in self.past:
            for s in UNIVERSE:
                oi[s] *= 1 + r.gauss(0, 0.03)
                c.execute("INSERT INTO oi_eod VALUES(?,?,?,?,?)", (d, s, oi[s], 0, ""))
        c.commit()
        c.close()

    # ── today's live caches (the shapes the private app keeps in state.py) ───
    def _state(self):
        s = types.SimpleNamespace()
        s.FNO_SYMBOLS, s.SECTOR_MAP = list(FNO_SYMBOLS), dict(SECTOR_MAP)
        s.BASE_DIR = s.DATA_DIR = os.path.join(self.dir, "data")
        s.DAILY_OHLC_CSV = os.path.join(self.dir, "data", "daily_ohlc.csv")
        for name in _LOCKS:
            setattr(s, name, threading.Lock())
        s.OI_HISTORY_LOCK, s.IV_HISTORY_LOCK = threading.RLock(), threading.RLock()
        for name in ("SESSION_RESULTS", "SESSION_RESULTS_SHORT", "OI_CACHE", "SESSION_HILO", "IMPLIED_BUDGET",
                     "_INTRADAY_CACHE", "MARKET_REGIME_INPUTS", "OI_HISTORY", "LIVE_VOL_HISTORY", "IV_HISTORY"):
            setattr(s, name, {})
        s.SCAN_STATE = {"finished": "", "results": []}
        s.SECTOR_INTEL = {"data": {}}
        s.AD_BREADTH = deque(maxlen=400)
        s.MARKET_REGIME = "MIXED"
        s.KANBAN = [{"symbol": "SBIN", "stage": "setup", "direction": "LONG"},
                    {"symbol": "INFY", "stage": "watch", "direction": "SHORT"}]
        s.WATCHLIST = {"SBIN", "INFY", "NYKAA", "TATASTEEL"}
        s.ALERT_QUEUE = deque(maxlen=50)
        for sym, (_sec, pdc) in UNIVERSE.items():
            s.IMPLIED_BUDGET[sym] = pdc * 0.02          # stand-in "expected day range"
        return s

    def _breadth(self):
        res = self.state.SESSION_RESULTS
        up = sum(1 for r in res.values() if (r.get("intra_pct") or 0) > 0)
        return {"ok": True, "now": {"hi": up // 4, "lo": (len(res) - up) // 4, "st": 0},
                "w15": {"hi": up // 2, "lo": (len(res) - up) // 2}, "w60": {"hi": up, "lo": len(res) - up},
                "day": {"hi": up, "lo": len(res) - up}, "bias": "UP" if up * 2 > len(res) else "DOWN"}

    def _minute(self, t):
        """Close the bar for minute t and refresh every cache like the app's scan would."""
        r, st = self.rng, self.state
        k = (t.hour * 60 + t.minute) - (9 * 60 + 15)
        ts = t.timestamp()
        for sym, (sec, pdc) in UNIVERSE.items():
            bars, rs = self._bars[sym], self._rs[sym]
            if not bars:
                o = pdc * (1 + rs.gauss(0, 0.005))
                self._open_px[sym] = o
                prev = o
            else:
                prev = bars[-1][4]
            drift = DRIFT.get(sym, 0.0) / 100
            if t > self._start and sym in DRIFT_LIVE:
                drift = DRIFT_LIVE[sym] / 100
            if sym in OPEN_IS_HIGH:
                drift = -0.0002
            if sym in OPEN_IS_LOW:
                drift = 0.0002
            c = prev * (1 + drift + rs.gauss(0, 0.0007))
            o = prev
            hi = max(o, c) * (1 + abs(rs.gauss(0, 0.0003)))
            lo = min(o, c) * (1 - abs(rs.gauss(0, 0.0003)))
            if sym in OPEN_IS_HIGH:
                hi = min(hi, self._open_px[sym])
            if sym in OPEN_IS_LOW:
                lo = max(lo, self._open_px[sym])
            v = max(1.0, rs.gauss(3000, 900) * (3 if k < 15 else 1))
            bars.append((t, o, hi, lo, c, v))
            self._fv[sym] += v * r.uniform(0.3, 0.6)
        with st._INTRADAY_CACHE_LOCK:
            for sym, bars in self._bars.items():
                idx = pd.DatetimeIndex([b[0] for b in bars]).tz_localize("Asia/Kolkata")
                st._INTRADAY_CACHE[sym] = pd.DataFrame(
                    {"Open": [b[1] for b in bars], "High": [b[2] for b in bars], "Low": [b[3] for b in bars],
                     "Close": [b[4] for b in bars], "Volume": [b[5] for b in bars]}, index=idx)
        long_, short_ = {}, {}
        for sym, (sec, pdc) in UNIVERSE.items():
            bars = self._bars[sym]
            px = bars[-1][4]
            hi, lo = max(b[2] for b in bars), min(b[3] for b in bars)
            pv = sum((b[2] + b[3] + b[4]) / 3 * b[5] for b in bars)
            vol = sum(b[5] for b in bars)
            pct = (px / pdc - 1) * 100
            rvol = round(vol / (3000.0 * len(bars)) * (1.6 if sym in DRIFT else 1.0), 2)
            state = "BREAKING" if pct > 1.2 else "PRIMED" if pct > 0.6 else ""
            long_[sym] = {"price": round(px, 2), "intra_pct": round(pct, 2), "rvol": rvol, "vwap": round(pv / vol, 2),
                          "state": state, "sector": sec, "score": int(8 + pct * 3) % 19, "max_score": 18,
                          "proximity": int(abs(pct) * 30) % 100, "prox_name": "PDH" if pct > 0 else "PDL",
                          "day_open": round(self._open_px[sym], 2), "pdc": pdc}
            short_[sym] = {"state": "BREAKING SHORT" if pct < -1.0 else "", "score": int(8 - pct * 3) % 19,
                           "max_score": 18}
            with st.SESSION_HILO_LOCK:
                st.SESSION_HILO[sym] = {"hi": hi * 1.0005, "lo": lo * 0.9995}
            oi_open = round(pct * 1.5 + r.gauss(0, 0.3), 2)
            st.OI_CACHE[sym] = {"oi_from_open": oi_open, "oi_session": oi_open, "oi_15m": round(r.gauss(0, 0.6), 2),
                                "streak": abs(int(pct * 2)) % 6, "vol_quad": "LBUILD" if pct > 0 else "SBUILD",
                                "iv_dir": "rising" if pct < 0 else "flat"}
            st.OI_HISTORY.setdefault(sym, deque(maxlen=200)).append({"ts": ts, "raw": 1e6 * (1 + oi_open / 100)})
            st.LIVE_VOL_HISTORY.setdefault(sym, deque(maxlen=400)).append((ts, self._fv[sym]))
            st.IV_HISTORY.setdefault(sym, deque(maxlen=200)).append({"ts": ts, "iv": 18 + abs(pct) * 2 + r.gauss(0, 0.2)})
        with st.SESSION_RESULTS_LOCK:
            st.SESSION_RESULTS.clear()
            st.SESSION_RESULTS.update(long_)
        with st.SESSION_RESULTS_SHORT_LOCK:
            st.SESSION_RESULTS_SHORT.clear()
            st.SESSION_RESULTS_SHORT.update(short_)
        # indices: base level moved by the members' average % (VIX moves the other way)
        data = {}
        for name, base in INDEX_BASE.items():
            mem = [long_[s]["intra_pct"] for s, (sec, _p) in UNIVERSE.items() if sec == name or name == "NIFTY 50"]
            pct = (sum(mem) / len(mem)) if mem else 0.0
            if name == "INDIA VIX":
                pct = -sum(x["intra_pct"] for x in long_.values()) / len(long_) * 4
            px = base * (1 + pct / 100)
            old = st.SECTOR_INTEL["data"].get(name) or {"day_high": px, "day_low": px, "pv": 0.0, "n": 0}
            n, pv = old["n"] + 1, old["pv"] + px
            data[name] = {"price": round(px, 2), "pct": round(pct, 2), "day_high": max(old["day_high"], px),
                          "day_low": min(old["day_low"], px), "vwap_v": round(pv / n, 2),
                          "above_vwap": px >= pv / n, "pv": pv, "n": n}
        with st.SECTOR_INTEL_LOCK:
            st.SECTOR_INTEL = {"data": data}
        adv = sum(1 for x in long_.values() if x["intra_pct"] > 0)
        st.AD_BREADTH.append({"ts": ts, "time": t.strftime("%H:%M"), "advance": adv,
                              "decline": len(long_) - adv, "ad_ratio": adv / max(1, len(long_) - adv)})
        with st.MARKET_REGIME_LOCK:
            n50 = data["NIFTY 50"]
            st.MARKET_REGIME = "BULL" if n50["pct"] > 0.3 else "BEAR" if n50["pct"] < -0.3 else "MIXED"
            st.MARKET_REGIME_INPUTS.update({"nifty_pct": n50["pct"], "vix_pct": data["INDIA VIX"]["pct"],
                                            "nifty_above": n50["above_vwap"]})
        for sym, x in long_.items():
            if x["state"] == "BREAKING" and not any(a["sym"] == sym for a in st.ALERT_QUEUE):
                st.ALERT_QUEUE.append({"sym": sym, "state": "BREAKING", "severity": "A", "reason": "stand-in",
                                       "time": t.strftime("%H:%M:%S"), "date": TODAY})
        with st.SCAN_LOCK:
            st.SCAN_STATE["finished"] = t.strftime("%H:%M:%S")

    def step(self, minutes=1):
        """Advance the fake clock: new bars, caches refreshed, sampler runs (like the live app)."""
        for _ in range(minutes):
            self.now += timedelta(minutes=1)
            self.ctx._now = self.now
            self._minute(self.now)
            T.sample_once(self.ctx, self.now.timestamp())
        return self.now
