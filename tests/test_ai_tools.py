# -*- coding: utf-8 -*-
"""
Tests for the Telegram AI v2 (ai_tools.py, ai_chat.py, ai_llm.FakeRouter).
No network, no Kite, no Gemini, no real database: a fake state module and
temporary SQLite files built from fake_market/schema.sql (the columns the tools read).

Run from the TradEdge folder:   python -m pytest tests -q
"""
import os
import re
import sqlite3
import sys
import threading
import types
from collections import deque
from datetime import datetime

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # TradEdge folder

import ai_tools as T                                         # noqa: E402
import ai_chat as C                                          # noqa: E402
from ai_llm import FakeRouter, Route, RouterError           # noqa: E402

TODAY = "2026-11-04"
NOW = datetime(2026, 11, 4, 9, 34, 30)
DAYS = ["2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"]
BANK, IT, PSU = "NIFTY BANK", "NIFTY IT", "NIFTY PSU BANK"


# ── fixtures ─────────────────────────────────────────────────────────────────
def _saga_db(path):
    c = sqlite3.connect(path)
    with open(os.path.join(os.path.dirname(HERE), "fake_market", "schema.sql"), encoding="utf-8") as f:
        c.executescript(f.read())
    # AAA rises 1% a day, BBB falls 2%, CCC flat; plus a half-written TODAY row
    for i, d in enumerate(DAYS + [TODAY]):
        for sym, close, vol, pace, rv in (
                ("AAA", 100 * 1.01 ** i, 1e6 * (i + 1), 1.0 + i / 10, 2.0),
                ("BBB", 100 * 0.98 ** i, 5e5, 0.5, 1.0),
                ("CCC", 100.0, 9e6 if d == TODAY else 2e6, 0.8, 4.0)):
            c.execute("INSERT INTO symbol_day(day,sym,close,high,low,volume,max_pace,rvol_max,"
                      "day_pct,oi_sess) VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (d, sym, close, close * 1.02, close * 0.98, vol, pace, rv, 1.0, 2.0))
    n = 0
    for d in DAYS[1:]:                       # SAGA: 25 signals on 5 days, each logged twice
        for k in range(5):
            sym = "S%02d" % n
            n += 1
            for tt in ("10:%02d:00" % k, "11:00:00"):
                c.execute("INSERT INTO signals(day,t,sym,stage,side) VALUES(?,?,?,?,?)",
                          (d, tt, sym, "SAGA", "LONG"))
                c.execute("INSERT INTO outcomes(day,sym,stage,t,side,m15,m30,m60,res_10,src,bars_n)"
                          " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                          (d, sym, "SAGA", tt, "LONG", 0.1, 0.2, 0.3 if k % 2 else -0.1,
                           "WIN" if k % 2 else "LOSS", "minute", 60))
    for i, d in enumerate(DAYS[:3]):         # SHARK: only 3 -> too few
        c.execute("INSERT INTO signals(day,t,sym,stage,side) VALUES(?,?,?,?,?)",
                  (d, "10:00:00", "SH%d" % i, "SHARK", "SHORT"))
        c.execute("INSERT INTO outcomes(day,sym,stage,t,side,m60,bars_n) VALUES(?,?,?,?,?,?,?)",
                  (d, "SH%d" % i, "SHARK", "10:00:00", "SHORT", 0.5, 60))
    c.commit()
    c.close()


def _ohlc_csv(path):
    """daily_ohlc.csv with the same prices as symbol_day (plus today's row, which tools must ignore)."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
        for i, d in enumerate(DAYS + [TODAY]):
            for sym, close, vol in (("AAA", 100 * 1.01 ** i, 1e6 * (i + 1)), ("BBB", 100 * 0.98 ** i, 5e5),
                                    ("CCC", 100.0, 9e6 if d == TODAY else 2e6)):
                w.writerow([d, sym, close, close * 1.02, close * 0.98, close, vol])


def _playbook_db(path):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE oi_eod(date TEXT, symbol TEXT, oi_abs REAL, sess_pct REAL, tag TEXT)")
    c.execute("CREATE TABLE trades(id INTEGER PRIMARY KEY, date TEXT, symbol TEXT, tier TEXT,"
              " pct_15m REAL, pct_30m REAL, pct_60m REAL)")
    for i, d in enumerate(DAYS):
        c.execute("INSERT INTO oi_eod VALUES(?,?,?,?,?)", (d, "AAA", 1000 * (1 + 0.1 * i), 0, ""))
        c.execute("INSERT INTO oi_eod VALUES(?,?,?,?,?)", (d, "BBB", 1000.0, 0, ""))
        for k in range(4):
            c.execute("INSERT INTO trades(date,symbol,tier,pct_15m,pct_30m,pct_60m) VALUES(?,?,?,?,?,?)",
                      (d, "X%d" % k, "READY", 0.1, 0.2, 0.4))
    c.commit()
    c.close()


def _bars(closes, vols, end="09:34"):
    idx = pd.date_range(end=TODAY + " " + end, periods=len(closes), freq="min", tz="Asia/Kolkata")
    return pd.DataFrame({"Close": closes, "Volume": vols}, index=idx)


def _state():
    s = types.SimpleNamespace()
    s.FNO_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "M&M"]
    s.SECTOR_MAP = {"AAA": BANK, "BBB": IT, "CCC": PSU, "DDD": IT, "M&M": "NIFTY AUTO"}
    for name in ("SESSION_RESULTS", "SESSION_RESULTS_SHORT", "OI_CACHE", "SESSION_HILO",
                 "IMPLIED_BUDGET", "_INTRADAY_CACHE", "MARKET_REGIME_INPUTS"):
        setattr(s, name, {})
    # plain (non-reentrant) locks: a tool that held one while calling conviction
    # (which takes SCAN_LOCK) would deadlock -> test_current_candidates catches it
    for name in ("SESSION_RESULTS_LOCK", "SESSION_RESULTS_SHORT_LOCK", "OI_CACHE_LOCK",
                 "SESSION_HILO_LOCK", "IMPLIED_BUDGET_LOCK", "_INTRADAY_CACHE_LOCK",
                 "MARKET_REGIME_LOCK", "SCAN_LOCK", "KANBAN_LOCK", "WATCH_LOCK", "ALERT_LOCK",
                 "AD_BREADTH_LOCK", "SECTOR_INTEL_LOCK"):
        setattr(s, name, threading.Lock())
    s.OI_HISTORY_LOCK = threading.RLock()
    s.SCAN_STATE = {"finished": "09:34:00", "results": []}
    s.SESSION_RESULTS.update({
        "AAA": {"price": 105.0, "intra_pct": 2.0, "rvol": 3.0, "vwap": 103.0, "state": "BREAKING",
                "sector": BANK, "score": 14, "max_score": 18, "proximity": 85, "prox_name": "R1"},
        "BBB": {"price": 85.0, "intra_pct": -1.5, "rvol": 1.2, "vwap": 88.0, "state": "—", "sector": IT,
                "score": 6, "max_score": 18, "proximity": 20, "prox_name": "S1"},
        "CCC": {"price": 100.5, "intra_pct": 0.3, "rvol": 5.0, "vwap": 100.0, "state": "PRIMED",
                "sector": PSU, "score": 11, "max_score": 18, "proximity": 60, "prox_name": "PDH"},
        "DDD": {"price": 50.0, "intra_pct": 0.0, "rvol": None, "vwap": None, "state": "", "sector": IT},
        "M&M": {"price": 3000.0, "intra_pct": 1.0, "rvol": 2.0, "vwap": 2990.0, "state": "",
                "sector": "NIFTY AUTO"},
    })
    s.SESSION_RESULTS_SHORT.update({"BBB": {"state": "BREAKING SHORT"}})
    s.OI_CACHE.update({"AAA": {"oi_from_open": 0.0, "oi_session": 9.9, "streak": 3, "vol_quad": "LBUILD",
                               "iv_dir": "rising"},
                       "BBB": {"oi_from_open": 4.0, "oi_15m": 1.0, "vol_quad": "SBUILD", "iv_dir": "flat"},
                       "CCC": {"oi_session": 2.0, "vol_quad": "LBUILD"}, "DDD": "not a dict"})
    s.SESSION_HILO.update({"AAA": {"hi": 106, "lo": 100}, "CCC": {"hi": 101, "lo": 100}})
    s.IMPLIED_BUDGET.update({"AAA": 3.0, "CCC": 2.0})
    n = 20
    yday = pd.date_range("2026-11-03 15:28", periods=2, freq="min", tz="Asia/Kolkata")
    aaa = _bars([100 + i for i in range(n)], [100] * n)                 # +1 per minute
    s._INTRADAY_CACHE.update({
        "AAA": pd.concat([pd.DataFrame({"Close": [90, 91], "Volume": [9999, 9999]}, index=yday), aaa]),
        "BBB": _bars([100 - i * 0.5 for i in range(n)], [10] * n),       # falling
        "CCC": _bars([100.0] * n, [10] * 15 + [100] * 5),               # flat, volume spike last 5
        "DDD": _bars([50.0] * 5, [1] * 5, end="09:20"),                 # stale: last bar 14 min old
    })
    s.MARKET_REGIME = "BULL"
    s.MARKET_REGIME_INPUTS.update({"nifty_pct": 0.42, "vix_pct": -3.1, "nifty_above": True})
    s.SECTOR_INTEL = {"data": {"NIFTY 50": {"pct": 0.42, "above_vwap": True},
                               "NIFTY BANK": {"pct": 0.61, "above_vwap": True},
                               "INDIA VIX": {"pct": -3.1}}}
    base = NOW.timestamp()
    s.AD_BREADTH = deque([{"ts": base - 60 * m, "time": "%02d:%02d" % divmod(9 * 60 + 34 - m, 60),
                           "advance": 100 + (40 - m), "decline": 100 - (40 - m),
                           "ad_ratio": (100 + 40 - m) / (100 - 40 + m)} for m in range(40, -1, -1)])
    s.OI_HISTORY = {"AAA": deque([{"ts": base - 60 * m, "raw": 1000 + (30 - m) * 10} for m in (30, 27, 15, 3, 0)]),
                    "BBB": deque([{"ts": base - 60 * m, "raw": 1000.0} for m in (30, 0)])}
    s.KANBAN = [{"symbol": "AAA", "stage": "setup", "direction": "LONG"},
                {"symbol": "BBB", "stage": "watch", "direction": "SHORT"}]
    s.WATCHLIST = {"AAA", "BBB", "CCC"}
    s.ALERT_QUEUE = deque([
        {"sym": "AAA", "state": "BREAKING", "severity": "A", "reason": "R1 break", "time": "09:30:01", "date": TODAY},
        {"sym": "BBB", "state": "SHORT", "severity": "B", "reason": "x", "time": "09:31:00", "date": TODAY},
        {"sym": "AAA", "state": "OLD", "severity": "A", "reason": "y", "time": "15:00:00", "date": "2026-11-03"}])
    return s


def _conv_factory(state):
    def conv(sym, rvol, pct, sector):
        with state.SCAN_LOCK:                                # real code takes SCAN_LOCK too
            return ({"AAA": 80, "CCC": 60}.get(sym, 40), "SUPER" if sym == "AAA" else "READY", 2, 1)
    return conv


BREADTH = {"ok": True, "w15": {"hi": 34, "lo": 9}, "w60": {"hi": 50, "lo": 20},
           "day": {"hi": 80, "lo": 30}, "bias": "UP", "now": {"hi": 3, "lo": 1, "st": 0}}


@pytest.fixture
def ctx(tmp_path):
    sdb, pdb = str(tmp_path / "saga_obs.db"), str(tmp_path / "playbook.db")
    _saga_db(sdb)
    _playbook_db(pdb)
    st = _state()
    st.DAILY_OHLC_CSV = str(tmp_path / "data" / "daily_ohlc.csv")
    _ohlc_csv(st.DAILY_OHLC_CSV)
    for sym, r in st.SESSION_RESULTS.items():
        r["day_open"] = {"AAA": 101.0, "BBB": 86.0, "CCC": 100.5, "DDD": 50.0, "M&M": 3010.0}[sym]
        r["pdc"] = {"AAA": 102.9, "BBB": 86.3, "CCC": 100.2, "DDD": 50.0, "M&M": 2970.0}[sym]
    return T.Ctx(st, saga_db=sdb, playbook_db=pdb, conviction=_conv_factory(st),
                 short_conviction=_conv_factory(st), breadth=lambda: BREADTH, today=TODAY, now=NOW)


def ok(ctx, name, **args):
    r, err = T.run(ctx, name, args)
    assert err is None, err
    text = T.format_reply(r)
    assert isinstance(text, str) and text
    return r, text


def _all_locks_free(st):
    for n in dir(st):
        if n.endswith("_LOCK"):
            lk = getattr(st, n)
            assert lk.acquire(blocking=False), "%s left held" % n
            lk.release()


# ── validation ───────────────────────────────────────────────────────────────
def test_every_tool_has_an_implementation():
    assert set(T._IMPL) == T.TOOL_NAMES and len(T.TOOLS) == 19


def test_validate_enums_bounds_and_defaults(ctx):
    a = T.validate(ctx, "top_symbols", {"metric": "VOLUME", "days": 99, "limit": "3"})
    assert a == {"metric": "volume", "days": 20, "end_day": None, "order": "top", "limit": 3, "sector": None,
                 "show": []}
    assert T.validate(ctx, "live_top", {"metric": "pct", "limit": 0})["limit"] == 1
    assert T.validate(ctx, "live_top", {"metric": "pct", "window_min": 2})["window_min"] == 5
    assert T.validate(ctx, "live_top", {"metric": "rvol", "window_min": 15})["window_min"] == 0


@pytest.mark.parametrize("name,args,msg", [
    ("top_symbols", {"metric": "profit", "days": 5}, "metric"),
    ("top_symbols", {"metric": "gain", "days": "abc"}, "number"),
    ("live_symbol", {"symbol": "NOTREAL"}, "not in the F&O"),
    ("live_symbol", {}, "Which stock"),
    ("level_breaks_history", {"kind": "prev_high", "day": "03-11-2026"}, "YYYY-MM-DD"),
    ("live_vwap", {"side": "above", "sector": "PHARMA"}, "Which sector"),
    ("signal_performance", {"by": "stage", "value": "MOON"}, "one of"),
    ("place_order", {}, "Unknown tool"),
])
def test_validate_rejects(ctx, name, args, msg):
    r, err = T.run(ctx, name, args)
    assert r is None and msg in err


def test_symbol_and_sector_normalised(ctx):
    assert T.validate(ctx, "live_symbol", {"symbol": " aaa.ns "})["symbol"] == "AAA"
    assert T.validate(ctx, "live_vwap", {"side": "Above", "sector": "BANK"})["sector"] == BANK   # BANK -> NIFTY BANK
    assert T.validate(ctx, "live_vwap", {"side": "above", "sector": "it"})["sector"] == IT
    assert T.validate(ctx, "live_vwap", {"side": "above", "sector": "psu"})["sector"] == PSU      # unique fuzzy
    assert T.validate(ctx, "live_vwap", {"side": "above", "sector": BANK})["sector"] == BANK


def test_show_columns_validated(ctx):
    a = T.validate(ctx, "live_top", {"metric": "pct", "show": ["price", "RVOL", "rvol", "pct", "junk",
                                                                "oi", "vwap", "pace"]})
    assert a["show"] == ["price", "rvol", "oi", "vwap"]          # deduped, metric dropped, max 4
    assert T.validate(ctx, "live_top", {"metric": "pct", "show": "price and rvol"})["show"] == ["price", "rvol"]


# ── history ──────────────────────────────────────────────────────────────────
def test_gain_is_compounded_and_excludes_today(ctx):
    r, text = ok(ctx, "top_symbols", metric="gain", days=5)
    assert (r["from"], r["to"], r["days"]) == ("2026-10-28", "2026-11-03", 5)
    top = dict(r["rows"])
    assert top["AAA"] == pytest.approx((1.01 ** 5 - 1) * 100, rel=1e-6)   # not 5 x day_pct
    assert r["rows"][0][0] == "AAA" and r["rows"][-1][0] == "BBB"
    assert "(10-28 → 11-03)" in text and "close 100.00 → 105.10" in text
    assert "📌 Notes" in text and "Market over the same 5 sessions" in text


def test_gain_window_shrinks_when_history_short(ctx):
    r, _ = ok(ctx, "top_symbols", metric="gain", days=20)
    assert r["days"] == 5


def test_history_show_columns(ctx):
    r, text = ok(ctx, "top_symbols", metric="gain", days=5, show=["price", "volume"])
    assert r["extra"]["AAA"]["price"] == pytest.approx(100 * 1.01 ** 5)
    assert "PRICE 105.10" in text and "VOL 60.0L" in text and "extra columns are values on 2026-11-03" in text


def test_volume_bottom_and_sector(ctx):
    r, _ = ok(ctx, "top_symbols", metric="volume", days=3, order="bottom")
    assert r["rows"][0][0] == "BBB"
    assert dict(r["rows"])["CCC"] == 2e6                     # today's 9e6 row ignored
    r, _ = ok(ctx, "top_symbols", metric="volume", days=3, sector="IT")
    assert [s for s, _ in r["rows"]] == ["BBB"]


def test_pace_rvol(ctx):
    r, _ = ok(ctx, "top_symbols", metric="pace", days=2)
    assert r["rows"][0] == ("AAA", pytest.approx(1.5))
    r, _ = ok(ctx, "top_symbols", metric="rvol", days=2)
    assert r["rows"][0][0] == "CCC"


def test_oi_change_from_playbook(ctx):
    r, _ = ok(ctx, "top_symbols", metric="oi", days=3, show=["price"])
    assert r["rows"][0][0] == "AAA"
    assert r["rows"][0][1] == pytest.approx((1.5 / 1.2 - 1) * 100)
    assert r["extra"]["AAA"]["price"] is not None


def test_level_breaks_history(ctx):
    r, _ = ok(ctx, "level_breaks_history", kind="prev_high")
    assert r["day"] == "2026-11-03" and r["prev_day"] == "2026-11-02"
    assert [x["sym"] for x in r["rows"]] == ["AAA"]
    assert r["rows"][0]["closed_beyond"] is False
    r, _ = ok(ctx, "level_breaks_history", kind="prev_low", day="2026-10-30")
    assert [x["sym"] for x in r["rows"]] == ["BBB"]


def test_symbol_history(ctx):
    r, text = ok(ctx, "symbol_history", symbol="AAA", days=3)
    assert [x["day"] for x in r["rows"]] == ["2026-11-03", "2026-11-02", "2026-10-30"]
    assert "AAA · last 3 sessions" in text


@pytest.fixture(autouse=True)
def _tiny_sectors(monkeypatch):
    """The fixture has 1-2 stocks per sector; real sectors have 5-30. Tests that check
    the small-sector rule set MIN_SECTOR_N back themselves."""
    monkeypatch.setattr(T, "MIN_SECTOR_N", 1)


def test_sector_history(ctx):
    r, text = ok(ctx, "sector_history", days=5)
    assert [x["sector"] for x in r["rows"]] == [BANK, PSU, IT]
    assert "Equal-weight" in text


def test_signal_performance_counts_one_per_stock_day(ctx):
    r, text = ok(ctx, "signal_performance", by="stage")
    g = {x["group"]: x for x in r["groups"]}
    assert g["SAGA"]["n"] == 25 and g["SAGA"]["days"] == 5
    assert g["SHARK"].get("too_few") is True
    assert g["SAGA"]["ci_lo"] <= g["SAGA"]["m60"] <= g["SAGA"]["ci_hi"]
    assert g["SAGA"]["win_share"] == pytest.approx(10 / 25 * 100)
    assert "too few to judge" in text and "not a prediction" in text


def test_signal_performance_side_and_tier(ctx):
    r, _ = ok(ctx, "signal_performance", by="stage", side="SHORT")
    assert [x["group"] for x in r["groups"]] == ["SHARK"]
    r, text = ok(ctx, "signal_performance", by="tier", value="ready")
    assert r["groups"][0]["n"] == 24 and r["groups"][0]["m60"] == pytest.approx(0.4)
    assert "no side column" in text


def test_missing_db_is_a_message_not_a_crash(ctx, tmp_path):
    ctx.saga_db = str(tmp_path / "nope.db")
    r, err = T.run(ctx, "symbol_history", {"symbol": "AAA"})
    assert r is None and "not found" in err


# ── live ─────────────────────────────────────────────────────────────────────
def test_live_top_day_metrics(ctx):
    r, text = ok(ctx, "live_top", metric="pct")
    assert r["rows"][0][0] == "AAA" and "as of scan 09:34:00" in text
    assert "1 AAA  +2.00%" in text                              # line 1: rank, symbol, main number
    assert "  → 105.00" in text or "→ 105.00" in text            # line 2: then -> now (price always shown)
    r, _ = ok(ctx, "live_top", metric="oi")
    assert dict(r["rows"])["AAA"] == 0.0                     # 0.0 kept, not replaced by 9.9
    assert "DDD" not in dict(r["rows"])
    r, _ = ok(ctx, "live_top", metric="pace")
    assert r["rows"] == [("AAA", 2.0), ("CCC", 0.5)]
    r, _ = ok(ctx, "live_top", metric="volume")
    assert dict(r["rows"])["AAA"] == 2000.0                  # yesterday's 9999 bars excluded
    assert "DDD" not in dict(r["rows"])                      # stale bars skipped
    _all_locks_free(ctx.state)


def test_live_top_with_price_and_rvol(ctx):
    r, text = ok(ctx, "live_top", metric="pct", show=["price", "rvol"])
    assert r["extra"]["AAA"] == {"rvol": 3.0}                    # price lives on line 2, not as a column
    assert "AAA  +2.00%  RVOL 3.00x" in text and "105.00" in text


def test_live_top_last_15_min(ctx):
    r, text = ok(ctx, "live_top", metric="pct", window_min=15)
    rows = dict(r["rows"])
    assert rows["AAA"] == pytest.approx((119 / 104 - 1) * 100)   # close now vs 15 bars ago
    assert r["rows"][-1][0] == "BBB" and "DDD" not in rows
    assert "last 15 min" in text and "1 skipped: no fresh bars" in text
    assert r["info"]["AAA"][:2] == (119.0, 104.0)               # NOW and THEN from the same bars
    assert "(09:20 → 09:35)" in text                            # one comparison time, in the title
    assert "104.00 → 119.00" in text                            # then -> now on line 2


def _restart_ctx(ctx, frames, now):
    ctx.state._INTRADAY_CACHE.clear()
    ctx.state._INTRADAY_CACHE.update(frames)
    ctx._now = now
    return ctx


def _frame(stamps, closes, vols=None):
    idx = pd.DatetimeIndex([TODAY + " " + t for t in stamps]).tz_localize("Asia/Kolkata")
    return pd.DataFrame({"Close": closes, "Volume": vols or [10] * len(closes)}, index=idx)


def _mins(a, b):
    out, (h, m) = [], map(int, a.split(":"))
    end = tuple(map(int, b.split(":")))
    while (h, m) <= end:
        out.append("%02d:%02d" % (h, m))
        m += 1
        if m == 60:
            h, m = h + 1, 0
    return out


def test_window_skips_restart_gap(ctx):
    # app ran 09:15-09:20, was stopped, restarted at 09:40 -> bars 09:40-09:50 only
    st = _mins("09:15", "09:20") + _mins("09:40", "09:50")
    gap = _frame(st, [100.0] * 6 + [110.0 + i for i in range(11)])
    ok_ = _frame(_mins("09:15", "09:50"), [100.0 + i * 0.1 for i in range(36)])
    _restart_ctx(ctx, {"AAA": gap, "CCC": ok_}, datetime(2026, 11, 4, 9, 50, 30))
    r, text = ok(ctx, "live_top", metric="pct", window_min=15)
    assert dict(r["rows"]).keys() == {"CCC"}                    # AAA would be +20% vs the pre-restart bar
    assert "1 skipped: data gap in the window (app restarted?)" in text
    r, _ = ok(ctx, "live_top", metric="pct", window_min=5)      # window fully after the restart: fine
    assert dict(r["rows"])["AAA"] == pytest.approx((120 / 115 - 1) * 100)


def test_window_handles_5min_warmup_bars(ctx):
    # startup warm-up gave 5-min bars 09:15..09:40, then the Kite poller 1-min bars 09:45..09:50
    st = ["09:15", "09:20", "09:25", "09:30", "09:35", "09:40"] + _mins("09:45", "09:50")
    closes = [100, 101, 102, 103, 104, 105] + [106, 107, 108, 109, 110, 111]
    _restart_ctx(ctx, {"AAA": _frame(st, [float(x) for x in closes])}, datetime(2026, 11, 4, 9, 50, 30))
    r, _ = ok(ctx, "live_top", metric="pct", window_min=15)
    # last bar closes 09:51 -> mark 09:36 -> base = the 09:30 5-min bar (closes 09:35), close 103
    # (counting 15 BARS back would have reached the 09:15 bar = 11% instead of 7.8%)
    assert dict(r["rows"])["AAA"] == pytest.approx((111 / 103 - 1) * 100)
    assert r["info"]["AAA"][2].strftime("%H:%M") == "09:35"


def test_window_app_started_late_is_skipped(ctx):
    _restart_ctx(ctx, {"AAA": _frame(_mins("10:00", "10:05"), [100.0] * 6)}, datetime(2026, 11, 4, 10, 5, 30))
    r, err = T.run(ctx, "live_top", {"metric": "pct", "window_min": 15})
    assert r is None and "data gap" in err


def test_window_volume_skips_gap(ctx):
    st = _mins("09:15", "09:30") + _mins("09:45", "09:50")
    _restart_ctx(ctx, {"AAA": _frame(st, [100.0] * len(st), [10] * len(st))},
                 datetime(2026, 11, 4, 9, 50, 30))
    r, err = T.run(ctx, "live_top", {"metric": "volume", "window_min": 10})
    assert r is None and "data gap" in err                      # the 10-min window spans the restart


def test_oi_window_ignores_reading_from_before_restart(ctx):
    base = NOW.timestamp()
    ctx.state.OI_HISTORY["AAA"] = deque([{"ts": base - 3600, "raw": 500.0},   # before the stop
                                         {"ts": base - 120, "raw": 1000.0}, {"ts": base, "raw": 1010.0}])
    ctx.state.OI_HISTORY["CCC"] = deque([{"ts": base - 60 * m, "raw": 1000.0 + m} for m in (18, 15, 12, 0)])
    r, _ = ok(ctx, "live_top", metric="oi", window_min=15)
    assert "AAA" not in dict(r["rows"]) and "CCC" in dict(r["rows"])   # AAA would have shown +102%


def test_live_top_window_early_morning(ctx):
    r, text = ok(ctx, "live_top", metric="pct", window_min=60)
    assert r["mins_used"] == 19 and "only 19 min of bars" in text


def test_live_top_volume_spike(ctx):
    r, text = ok(ctx, "live_top", metric="volume", window_min=5)
    assert r["rows"][0][0] == "CCC" and r["rows"][0][1] == pytest.approx(10.0)
    vn, vp, pn, pt, _t = r["info"]["CCC"]
    assert (vn, vp, pn, pt) == (500.0, 50.0, 100.0, 100.0)      # last 5 min, the 5 min before, prices
    assert "CCC  10.00x" in text and "100.00 → 100.00 (+0.00%)" in text   # spike + prices + move only
    assert "VOL" not in text.split("SPIKE =")[0].split("📌")[0]          # no raw volume columns
    assert "(09:30 → 09:35)" in text and "volume/min earlier today" in text


def test_move_window_shows_volume_and_oi_window_shows_before_after(ctx):
    r, text = ok(ctx, "live_top", metric="pct", window_min=15)
    assert r["info"]["AAA"][3] == 1500.0                        # kept in data, not shown
    ctx.state.OI_HISTORY["AAA"][-1]["ltp"] = 110.0
    ctx.state.OI_HISTORY["AAA"][2]["ltp"] = 100.0                 # reading 15 min ago
    r, text = ok(ctx, "live_top", metric="oi", window_min=15)
    on, ot, pn, pt, _t = r["info"]["AAA"]
    assert (on, ot) == (1300, 1150) and "fut 100.00 → 110.00 (+10.00%)" in text


# ── "from that list" + filters ───────────────────────────────────────────────
PREV = [{"symbols": ["AAA", "CCC", "M&M"], "time": "09:30", "q": "stocks above vwap now"},
        {"symbols": ["BBB"], "time": "09:25", "q": "below vwap"}]


def test_from_previous_limits_to_that_list(ctx):
    r, err = T.run(ctx, "live_top", {"metric": "rvol", "from_previous": 1}, prev=PREV)
    assert err is None and [s for s, _ in r["rows"]] == ["CCC", "AAA", "M&M"]
    text = T.format_reply(r)
    assert text.split("\n")[1] == "↳ from answer #1 (09:30, the 3 stocks shown)"
    r, _ = T.run(ctx, "live_top", {"metric": "pct", "from_previous": 2}, prev=PREV)
    assert [s for s, _ in r["rows"]] == ["BBB"]
    assert ctx.scope is None                                     # cleared after the call


def test_filters_apply_before_limit(ctx):
    args = {"side": "above", "filters": [{"col": "rvol", "op": ">", "value": 2.5}], "limit": 1}
    r, err = T.run(ctx, "live_vwap", args, prev=PREV)
    assert err is None and [x["sym"] for x in r["rows"]] == ["AAA"] and r["n"] == 2   # AAA, CCC pass
    assert "where rvol > 2.5" in T.format_reply(r)
    r, _ = T.run(ctx, "live_vwap", {"side": "above", "from_previous": 1,
                                    "filters": [{"col": "price", "op": "<", "value": 1000}]}, prev=PREV)
    assert [x["sym"] for x in r["rows"]] == ["AAA", "CCC"]       # M&M (3000) filtered out


def test_from_previous_with_sector_and_other_tools(ctx):
    r, _ = T.run(ctx, "current_candidates", {"from_previous": 1, "sector": "BANK"}, prev=PREV)
    assert [x["sym"] for x in r["rows"]] == ["AAA"]
    r, _ = T.run(ctx, "watchlist_status", {"from_previous": 1}, prev=PREV)
    assert {x["sym"] for x in r["rows"]} == {"AAA", "CCC"}
    r, err = T.run(ctx, "top_symbols", {"metric": "gain", "days": 5, "from_previous": 1,
                                        "filters": [{"col": "pct", "op": ">", "value": 0}]}, prev=PREV)
    assert err is None and [s for s, _ in r["rows"]] == ["AAA"]      # CCC's last day was flat (0%)


@pytest.mark.parametrize("args,msg", [
    ({"metric": "pct", "from_previous": 3}, "no earlier list #3"),
    ({"metric": "pct", "filters": [{"col": "mood", "op": ">", "value": 1}]}, "Can't filter on 'mood'"),
    ({"metric": "pct", "filters": [{"col": "rvol", "op": "==", "value": 1}]}, ">, >=, < or <="),
    ({"metric": "pct", "filters": [{"col": "rvol", "op": ">", "value": "lots"}]}, "needs a number"),
])
def test_scope_validation(ctx, args, msg):
    r, err = T.run(ctx, "live_top", args, prev=PREV)
    assert r is None and msg in err
    assert ctx.scope is None


def test_from_previous_without_memory(ctx):
    r, err = T.run(ctx, "live_vwap", {"side": "above", "from_previous": 1}, prev=[])
    assert "keep your last 5 answers" in err


def test_scope_ignored_on_tools_without_lists(ctx):
    r, err = T.run(ctx, "market_mood", {"from_previous": 1}, prev=PREV)
    assert err is None and "scope" not in r


def test_live_top_oi_window(ctx):
    r, _ = ok(ctx, "live_top", metric="oi", window_min=15)
    assert dict(r["rows"])["AAA"] == pytest.approx((1300 / 1150 - 1) * 100)


def test_live_level_breaks_values(ctx):
    st = ctx.state
    st.SESSION_RESULTS["CCC"]["price"] = 103.0
    st.SESSION_RESULTS["BBB"]["price"] = 80.0
    r, text = ok(ctx, "live_level_breaks", kind="prev_high", show=["rvol"])
    assert [x["sym"] for x in r["rows"]] == ["CCC"]
    assert r["rows"][0]["dist_pct"] == pytest.approx((103 / 102 - 1) * 100)
    assert "RVOL" in text
    r, _ = ok(ctx, "live_level_breaks", kind="prev_low")
    assert [x["sym"] for x in r["rows"]] == ["BBB"]
    assert r["missing_levels"] == 2                           # DDD and M&M have no history rows


def test_live_vwap(ctx):
    r, _ = ok(ctx, "live_vwap", side="above")
    assert [x["sym"] for x in r["rows"]] == ["AAA", "CCC", "M&M"]
    r, _ = ok(ctx, "live_vwap", side="below", sector="IT")
    assert [x["sym"] for x in r["rows"]] == ["BBB"]


def _bands(p):
    return "QUIET" if p < 0.5 else "NORMAL" if p < 1 else "ACTIVE" if p < 1.5 else "HOT"


def test_live_symbol_both_scores_and_pulse(ctx):
    ctx.state.SESSION_RESULTS["BBB"].update({"pct_score": 33, "oi_pulse": 12})
    ctx.state.SESSION_RESULTS_SHORT["BBB"].update({"pct_score": 38, "oi_pulse": 46, "score": 7, "max_score": 18})
    r, text = ok(ctx, "live_symbol", symbol="BBB")
    assert r["short_state"] == "BREAKING SHORT" and r["oi_open"] == 4.0
    assert "LONG  33%  (6/18)" in text and "SHORT 38%  (7/18)" in text and "BREAKING SHORT" in text
    assert "stronger side now: SHORT (scanner score)" in text
    assert "OI pulse  long 12 · short 46" in text
    assert "open 86.00 → now 85.00" in text


def test_live_symbol_pace_path(ctx):
    ctx._pband = _bands
    r, text = ok(ctx, "live_symbol", symbol="AAA")       # AAA bars rise 1/min, budget 3.0
    pp = r["pace_path"]
    assert pp["bands"] == [("09:17", "NORMAL"), ("09:18", "ACTIVE"), ("09:20", "HOT")]
    assert pp["last"] == ("09:34", "HIGH") and pp["first"] == "QUIET"
    assert "broken 3 times since open: 09:17 NORMAL · 09:18 ACTIVE · 09:20 HOT" in text
    assert "last change 09:34 (new day HIGH)" in text and "live PACE uses futures" in text


def test_live_symbol_without_budget_has_no_path(ctx):
    r, text = ok(ctx, "live_symbol", symbol="BBB")        # no IMPLIED_BUDGET for BBB
    assert r["pace_path"] is None and "broken" not in text


def test_current_candidates_long_short_no_deadlock(ctx):
    done = {}

    def go():
        done["long"] = T.run(ctx, "current_candidates", {"side": "LONG", "show": ["price"]})
        done["short"] = T.run(ctx, "current_candidates", {"side": "SHORT"})
    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(5)
    assert not th.is_alive(), "deadlock: a lock was held while calling conviction"
    r, err = done["long"]
    assert err is None and [x["sym"] for x in r["rows"]] == ["AAA", "CCC"]
    assert r["rows"][0]["tier"] == "SUPER" and r["pair"]["AAA"][1] == 105.0
    assert "AAA  SUPER 80  RVOL 3.00x" in T.format_reply(r)
    assert "not advice" in T.format_reply(r)
    r, err = done["short"]
    assert err is None and [x["sym"] for x in r["rows"]] == ["BBB"]
    assert r["rows"][0]["rvol"] == 1.2
    _all_locks_free(ctx.state)


def test_watchlist_status(ctx):
    r, text = ok(ctx, "watchlist_status")
    assert [x["sym"] for x in r["rows"]] == ["AAA", "CCC", "BBB"]       # by score
    bbb = [x for x in r["rows"] if x["sym"] == "BBB"][0]
    assert bbb["dir"] == "SHORT" and bbb["state"] == "BREAKING SHORT"   # short side state
    assert [x for x in r["rows"] if x["sym"] == "CCC"][0]["stage"] == "watch"   # name-only entry
    r, t2 = ok(ctx, "watchlist_status", sort_by="oi")
    assert r["rows"][0]["sym"] == "BBB" and "1 BBB  +4.0%" in t2      # sort column is the main number
    r, _ = ok(ctx, "watchlist_status", side="LONG", only_flagged=True)
    assert [x["sym"] for x in r["rows"]] == ["AAA", "CCC"]
    assert "R1" in text and "14/18" in text


def test_watchlist_empty(ctx):
    ctx.state.KANBAN, ctx.state.WATCHLIST = [], set()
    r, err = T.run(ctx, "watchlist_status", {})
    assert "empty" in err


def test_recent_alerts_today_only_and_deque(ctx):
    r, text = ok(ctx, "recent_alerts")
    assert [a["time"] for a in r["rows"]] == ["09:31:00", "09:30:01"]
    r, _ = ok(ctx, "recent_alerts", symbol="AAA")
    assert r["n"] == 1 and r["rows"][0]["reason"] == "R1 break"


def test_live_sector_rank(ctx):
    r, text = ok(ctx, "live_sector_rank", metric="price")
    assert r["rows"][0]["sector"] == BANK and r["rows"][0]["index_pct"] == 0.61
    assert "index +0.61%" in text and "Equal-weight" in text
    r, _ = ok(ctx, "live_sector_rank", metric="price", window_min=15)
    assert r["rows"][0]["sector"] == BANK and r["rows"][-1]["sector"] == IT
    r, _ = ok(ctx, "live_sector_rank", metric="breadth", order="bottom")
    assert r["rows"][0]["sector"] == IT


def test_market_mood(ctx):
    r, text = ok(ctx, "market_mood")
    assert r["regime"] == "BULL" and r["quads"] == {"LBUILD": 2, "SBUILD": 1}
    assert (r["flags_long"], r["flags_short"]) == (2, 1)
    assert r["ago"]["adv"] == 110                              # 30 min ago
    for s in ("BULL", "NIFTY +0.42% above VWAP", "BANKNIFTY +0.61%", "VIX -3.10%",
              "15 min 34 / 9", "LBUILD 2", "Leading BANK"):
        assert s in text, s


def test_market_mood_survives_missing_parts(ctx):
    ctx._breadth = lambda: 1 / 0
    del ctx.state.SECTOR_INTEL_LOCK
    r, text = ok(ctx, "market_mood")
    assert r["breadth"] is None and "Market mood" in text


def test_breadth_trend(ctx):
    r, text = ok(ctx, "breadth_trend", window_min=30)
    assert r["trend"] == "improving" and r["covered_min"] == 30
    assert "IMPROVING" in text and len(r["rows"]) >= 6


def test_market_closed_messages(ctx):
    ctx.state.SESSION_RESULTS.clear()
    for name, args in (("live_top", {"metric": "pct"}), ("market_mood", {}),
                       ("live_vwap", {"side": "above"})):
        r, err = T.run(ctx, name, args)
        assert r is None and "No live scan results" in err
    ctx._now = datetime(2026, 11, 4, 16, 0)                    # all bars stale after close
    r, err = T.run(ctx, "live_top", {"metric": "pct", "window_min": 15})
    assert "no fresh bars" in err


def test_tool_bug_is_contained(ctx, monkeypatch):
    monkeypatch.setitem(T._IMPL, "market_mood", lambda ctx: 1 / 0)
    r, err = T.run(ctx, "market_mood", {})
    assert r is None and "ZeroDivisionError" in err


def test_every_tool_formats(ctx):
    calls = [("top_symbols", {"metric": m, "days": 3, "show": ["price"]}) for m in ("gain", "volume", "oi", "pace", "rvol")]
    calls += [("live_top", {"metric": m, "show": ["price", "rvol", "oi", "state"]})
              for m in ("pct", "volume", "rvol", "oi", "pace")]
    calls += [("level_breaks_history", {"kind": "prev_low"}), ("symbol_history", {"symbol": "BBB"}),
              ("sector_history", {"days": 3}), ("signal_performance", {"by": "tier"}),
              ("live_level_breaks", {"kind": "prev_low"}), ("live_vwap", {"side": "below"}),
              ("live_symbol", {"symbol": "M&M"}), ("current_candidates", {"show": ["sector", "prox", "score"]}),
              ("watchlist_status", {"show": ["volume", "pace"]}), ("recent_alerts", {}),
              ("live_sector_rank", {"metric": "oi", "window_min": 15}), ("market_mood", {}),
              ("breadth_trend", {}), ("cannot_answer", {"reason": "x"})]
    for name, args in calls:
        ok(ctx, name, **args)


def test_table_alignment():
    out = T.table(["#", "SYMBOL", "DAY%"], [["1", "BEL", "+4.71%"], ["10", "MUTHOOTFIN", "+7.33%"]])
    lines = out.split("\n")
    assert len({len(l) for l in lines}) == 1                   # every line same width
    assert lines[1].endswith("+4.71%") and lines[2].endswith("+7.33%")


def test_today_follows_clock():
    c = T.Ctx(types.SimpleNamespace())
    assert len(c.today) == 10 and c.today[4] == "-"


# ── Telegram bot logic ───────────────────────────────────────────────────────
class FakeTG:
    def __init__(self):
        self.sent, self.buttons, self.callbacks, self.fail = [], [], [], False

    def send(self, chat, text, buttons=None):
        if self.fail:
            return False
        self.sent.append((chat, text))
        self.buttons.append(buttons)
        return True

    def answer_callback(self, cq_id, text=""):
        self.callbacks.append(cq_id)


class RecRouter(FakeRouter):
    def __init__(self, table):
        super().__init__(table)
        self.prompts = []

    def route(self, q):
        self.prompts.append(q)
        return super().route(q.split("\nNEW: ")[-1])


def _bot(ctx, tmp_path, table):
    return C.Bot(ctx, RecRouter(table), FakeTG(), ["111"], str(tmp_path / "logs"))


def _msg(text, chat=111):
    return {"update_id": 1, "message": {"chat": {"id": chat}, "text": text}}


def test_bot_routes_and_logs(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"top gainers 5 days": ("top_symbols", {"metric": "gain", "days": 5})})
    b.handle(_msg("top gainers 5 days"))
    assert b.tg.sent and "Top" in b.tg.sent[0][1]
    logs = os.listdir(str(tmp_path / "logs"))
    line = open(os.path.join(str(tmp_path / "logs"), logs[0]), encoding="utf-8").read()
    assert '"tool": "top_symbols"' in line and '"clean_args"' in line


def test_bot_followup_gets_previous_context(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"top gainers now": ("live_top", {"metric": "pct"}),
                             "and in IT?": ("live_top", {"metric": "pct", "sector": "NIFTY IT"})})
    b.handle(_msg("top gainers now"))
    b.handle(_msg("and in IT?"))
    assert "PREVIOUS ANSWERS (1 = most recent):" in b.router.prompts[1]
    assert "'top gainers now' -> live_top" in b.router.prompts[1] and '"metric": "pct"' in b.router.prompts[1]
    assert "· IT" in b.tg.sent[1][1]


def test_bot_chain_from_that_list(ctx, tmp_path):
    table = {"stocks above vwap now": ("live_vwap", {"side": "above"}),
             "from that, which have rvol above 2?": ("live_top", {"metric": "rvol", "from_previous": 1,
                                                               "filters": [{"col": "rvol", "op": ">", "value": 2}]}),
             "of these, only bank": ("live_top", {"metric": "rvol", "from_previous": 1, "sector": "BANK"})}
    b = _bot(ctx, tmp_path, table)
    for q in table:
        b.handle(_msg(q))
    a1, a2, a3 = [t for _, t in b.tg.sent]
    assert "3 stocks above VWAP" in a1
    assert "from answer #1" in a2 and "where rvol > 2" in a2 and "M&M" not in a2   # M&M rvol = 2, not > 2
    assert "AAA" in a3 and "CCC" not in a3                                          # CCC is PSU BANK
    assert "2. [" in b.router.prompts[2] and "(3 stocks)" in b.router.prompts[2]   # both earlier answers sent


def test_bot_memory_keeps_last_5(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"q%d" % i: ("live_top", {"metric": "pct", "limit": i + 1}) for i in range(6)})
    for i in range(6):                                       # 6 = the per-minute rate limit
        b.handle(_msg("q%d" % i))
    mem = b.recent("111")
    assert len(mem) == 5 and mem[0]["q"] == "q5" and mem[-1]["q"] == "q1"
    assert b.router.prompts[-1].count("\n") == 6                   # header + 5 answers + NEW


def test_bot_followup_expires(ctx, tmp_path, monkeypatch):
    b = _bot(ctx, tmp_path, {"top gainers now": ("live_top", {"metric": "pct"})})
    b.handle(_msg("top gainers now"))
    t0 = C.time.time()
    monkeypatch.setattr(C.time, "time", lambda: t0 + C.FOLLOWUP_S + 1)
    b.handle(_msg("top gainers now"))
    assert not b.router.prompts[1].startswith("PREVIOUS ANSWERS")


def test_bot_blocks_unknown_chat_silently(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    b.handle(_msg("hi", chat=999))
    assert b.tg.sent == []


def test_bot_help_skips_llm(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    b.handle(_msg("/help"))
    assert "I never give buy/sell advice" in b.tg.sent[0][1] and b.router.prompts == []


def test_bot_rate_limit(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    for _ in range(C.MAX_PER_MIN + 2):
        b.handle(_msg("anything"))
    assert sum("Slow down" in t for _, t in b.tg.sent) == 2


def test_bot_router_errors_are_friendly(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"q": RouterError("quota", "429"), "d": RouterError("quota_day", "429"),
                             "m": RouterError("model_missing", "404"), "k": RouterError("key", "403")})
    for q in ("q", "d", "m", "k"):
        b.handle(_msg(q))
    per_min, day, model, key = [t for _, t in b.tg.sent]
    assert "per-minute limit" in per_min and "resets about 13:30 IST" in day
    assert "model not found" in model and "rejected the API key" in key
    assert all("not configured" not in t for t in (per_min, day, model, key))


def test_bot_advice_goes_to_cannot_answer(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"should i buy AAA": ("cannot_answer", {"reason": "No buy/sell advice."})})
    b.handle(_msg("should i buy AAA"))
    assert "can't answer" in b.tg.sent[0][1] and "/help" in b.tg.sent[0][1]


def test_bot_bad_model_args_shown_not_crashed(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"x": ("top_symbols", {"metric": "money"})})
    b.handle(_msg("x"))
    assert "metric" in b.tg.sent[0][1]


def test_bot_ignores_non_text(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    b.handle({"update_id": 2, "message": {"chat": {"id": 111}, "photo": []}})
    assert b.tg.sent == []


# ── Telegram HTML sending ────────────────────────────────────────────────────
class FakeSession:
    def __init__(self, reject_html=False):
        self.posts, self.reject = [], reject_html

    def post(self, url, data, timeout):
        self.posts.append(data)
        ok = not (self.reject and data.get("parse_mode"))
        return type("R", (), {"json": lambda s: {"ok": ok}})()


def _tg(sess):
    t = C.Telegram.__new__(C.Telegram)
    t._s, t._base = sess, "x/"
    return t


def test_send_escapes_ampersand_in_pre_block():
    s = FakeSession()
    _tg(s).send("1", "1. M&M  +1.00%\n2. GVT&D <x>")
    d = s.posts[0]
    assert d["parse_mode"] == "HTML"
    assert d["text"] == "<pre>1. M&amp;M  +1.00%\n2. GVT&amp;D &lt;x&gt;</pre>"


def test_send_falls_back_to_plain_text():
    s = FakeSession(reject_html=True)
    _tg(s).send("1", "M&M")
    assert len(s.posts) == 2 and s.posts[1]["text"] == "M&M" and "parse_mode" not in s.posts[1]


def test_chunks_split_on_lines():
    text = "\n".join("line %04d %s" % (i, "x" * 50) for i in range(200))
    parts = C.chunks(text, 1000)
    assert all(len(p) <= 1000 for p in parts) and "\n".join(parts) == text


# ── digests ──────────────────────────────────────────────────────────────────
def test_parse_and_due_slots():
    slots = C.parse_slots("09:20, 11:00,bad,15:35")
    assert slots == ["09:20", "11:00", "15:35"]
    sent = set()
    assert C.due_slots(slots, datetime(2026, 11, 4, 11, 4), sent) == ["11:00"]   # late restart ok
    assert C.due_slots(slots, datetime(2026, 11, 4, 11, 30), sent) == []         # too late
    assert C.due_slots(slots, datetime(2026, 11, 4, 10, 59), sent) == []
    sent.add(("2026-11-04", "11:00"))
    assert C.due_slots(slots, datetime(2026, 11, 4, 11, 1), sent) == []          # once a day
    assert C.due_slots(slots, datetime(2026, 11, 7, 11, 0), set()) == []         # Saturday


def test_digest_sends_mood_and_eod_extras(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    assert b.digest("11:00") is True
    assert "11:00 digest" in b.tg.sent[0][1] and "Top 5" not in b.tg.sent[0][1]
    assert b.digest("15:35") is True
    eod = b.tg.sent[1][1]
    assert "Top 5 gainers" in eod and "Top 5 losers" in eod and "sectors by" in eod
    assert b.router.prompts == []                              # no LLM call


def test_digest_skipped_without_live_data(ctx, tmp_path):
    ctx.state.SESSION_RESULTS.clear()
    b = _bot(ctx, tmp_path, {})
    assert b.digest("09:20") is False and b.tg.sent == []


# ── thread loop + start ──────────────────────────────────────────────────────
def test_loop_skips_backlog_and_survives_errors(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(C.time, "sleep", lambda s: None)

    class TG(FakeTG):
        def __init__(self):
            super().__init__()
            self.plan = [[{"update_id": 5, "message": {"chat": {"id": 111}, "text": "/help"}}],
                         RuntimeError("network down"),
                         [{"update_id": 6, "message": {"chat": {"id": 111}, "text": "/help"}}],
                         KeyboardInterrupt()]
            self.offsets = []

        def get_updates(self, offset):
            self.offsets.append(offset)
            x = self.plan.pop(0)
            if isinstance(x, BaseException):
                raise x
            return x
    b = C.Bot(ctx, FakeRouter({}), TG(), ["111"], str(tmp_path / "logs"))
    with pytest.raises(KeyboardInterrupt):
        C._loop(b)
    assert b.tg.offsets == [-1, 6, 6, 7]
    assert len(b.tg.sent) == 1


def test_start_without_token_does_not_start(monkeypatch, capsys):
    fake = types.SimpleNamespace(_CFG_TG_TOKEN="", _CFG_TG_CHAT="", SECTOR_MAP={})
    monkeypatch.setitem(sys.modules, "state", fake)
    monkeypatch.setattr(C, "_started", False)
    monkeypatch.setattr(C, "_cfg", lambda n, d="": d)
    C.start()
    assert "not started" in capsys.readouterr().out


def test_route_object():
    r = Route("live_top", {"metric": "pct"}, "m", 12, 5)
    assert (r.name, r.args["metric"], r.tokens) == ("live_top", "pct", 12)


# ── notes (code-computed facts) ──────────────────────────────────────────────
def _market(ctx, rows):
    """rows: (sym, sector, pct, rvol, above_vwap)"""
    st = ctx.state
    st.SESSION_RESULTS.clear()
    st.SECTOR_MAP.clear()
    for sym, sec, pct, rv, above in rows:
        st.SECTOR_MAP[sym] = sec
        st.SESSION_RESULTS[sym] = {"price": 100 + pct, "pdc": 100.0, "intra_pct": pct, "rvol": rv,
                                   "vwap": (99.0 if above else 101.0) + pct, "sector": sec, "state": ""}
    st.FNO_SYMBOLS = [r[0] for r in rows]


def test_notes_cluster_against_market_and_rotation(ctx):
    rows = [("PH%d" % i, "NIFTY PHARMA", 3.0 - i * 0.2, 2.5, True) for i in range(4)]
    rows += [("BK%d" % i, "NIFTY BANK", -1.0, 1.0, False) for i in range(5)]
    rows += [("IT%d" % i, "NIFTY IT", -0.8, 1.0, False) for i in range(5)]
    rows += [("AU%d" % i, "NIFTY AUTO", -0.5, 1.0, False) for i in range(5)]
    _market(ctx, rows)
    r, text = ok(ctx, "live_top", metric="pct", limit=4)
    notes = r["notes"]
    assert notes[0] == "4 of 4 are PHARMA (PH0, PH1, PH2, PH3)"
    assert "against the market".upper() in notes[1].upper() and "relative strength" in notes[1]
    assert any("PHARMA +2.70% while 3 of 4 sectors are red → sector rotation" in n for n in notes)
    assert any("4 of 4 above VWAP · 4 with RVOL ≥ 2 → volume-backed" in n for n in notes)
    assert "📌 Notes" in text


def test_notes_market_wide_and_thin_volume(ctx):
    rows = [("S%02d" % i, "NIFTY IT" if i % 2 else "NIFTY BANK", 1.0 + i * 0.01, 1.0, True) for i in range(12)]
    _market(ctx, rows)
    r, _ = ok(ctx, "live_top", metric="pct", limit=5)
    assert any("market-wide rise" in n for n in r["notes"])
    assert any("thin volume" in n for n in r["notes"])
    _market(ctx, [("S%02d" % i, "NIFTY IT", -1.0 - i * 0.01, 1.0, False) for i in range(12)])
    r, _ = ok(ctx, "live_top", metric="pct", order="bottom", limit=5)
    assert any("5 of 5 below VWAP" in n for n in r["notes"])     # losers check the other VWAP side
    assert any("market-wide fall" in n for n in r["notes"])


def test_notes_never_advise_and_keep_skip_line(ctx):
    r, _ = ok(ctx, "live_top", metric="pct", window_min=15)
    joined = " ".join(r["notes"]).lower()
    for w in ("buy", "sell", "target", "should"):
        assert w not in joined
    assert r["notes"][-1].startswith("1 skipped")                 # data-quality line always last, never cut


def test_notes_total_shown(ctx):
    r, _ = ok(ctx, "live_vwap", side="above", limit=2)
    assert any("3 stocks in total above VWAP; top 2 shown" in n for n in r["notes"])



# ── real-data lessons (05-11-2026 export) ────────────────────────────────────
def test_since_open_uses_todays_open_not_yesterdays_close(ctx):
    r, text = ok(ctx, "live_top", metric="open", order="bottom")
    rows = dict(r["rows"])
    assert rows["M&M"] == pytest.approx((3000 / 3010 - 1) * 100)      # vs open 3010, not prev close 2970
    assert r["rows"][0][0] == "BBB"
    assert "losers since open" in text and "open 3010.0 → now 3000.0" in text


def test_prices_come_from_daily_ohlc_even_if_symbol_day_has_no_close(ctx):
    c = sqlite3.connect(ctx.saga_db)
    c.execute("UPDATE symbol_day SET close=NULL, high=NULL, low=NULL, volume=NULL")   # like the real DB
    c.commit()
    c.close()
    r, text = ok(ctx, "top_symbols", metric="gain", days=5)
    assert r["rows"][0][0] == "AAA" and "close 100.00 → 105.10" in text
    r, _ = ok(ctx, "level_breaks_history", kind="prev_high")
    assert [x["sym"] for x in r["rows"]] == ["AAA"]


def test_missing_latest_session_is_said_and_ram_daily_fills_it(ctx):
    import csv as _csv
    rows = [r for r in _csv.reader(open(ctx.state.DAILY_OHLC_CSV)) if not r[0].startswith("2026-11-03")]
    with open(ctx.state.DAILY_OHLC_CSV, "w", newline="") as f:
        _csv.writer(f).writerows(rows)                                  # 15:45 backfill did not run yet
    r, text = ok(ctx, "top_symbols", metric="gain", days=3)
    assert r["to"] == "2026-11-02" and "2026-11-03 not available yet" in text
    idx = pd.DatetimeIndex(["2026-11-02", "2026-11-03", TODAY]).tz_localize("Asia/Kolkata")
    ctx.state._D_SESSION_LOCK = threading.Lock()
    ctx.state._D_SESSION_CACHE = {s: pd.DataFrame({"Open": [1, 1, 1], "High": [h, h, h], "Low": [1, 1, 1],
                                                   "Close": [c0, c1, 999], "Volume": [1, 1, 1]}, index=idx)
                                  for s, c0, c1, h in (("AAA", 104.06, 120.0, 130), ("BBB", 90.4, 80.0, 95))}
    r, text = ok(ctx, "top_symbols", metric="gain", days=1)
    assert r["to"] == "2026-11-03" and dict(r["rows"])["AAA"] == pytest.approx((120 / 104.06 - 1) * 100, rel=1e-3)
    assert "11-03 from the app's daily cache" in text                   # today's partial bar (999) ignored


def test_oi_history_names_the_real_dates(ctx):
    r, text = ok(ctx, "top_symbols", metric="oi", days=3)
    assert "end-of-day OI" in text and r["from"] < r["to"]


def test_pace_says_which_sessions_are_missing(ctx):
    c = sqlite3.connect(ctx.saga_db)
    c.execute("DELETE FROM symbol_day WHERE day IN ('2026-10-30', '2026-11-02')")
    c.commit()
    c.close()
    r, text = ok(ctx, "top_symbols", metric="pace", days=4)
    assert r["days"] == 2 and "PACE data exists for 2 of the last 4 sessions (missing: 10-30, 11-02)" in text


def test_empty_result_explains_why(ctx):
    r, err = T.run(ctx, "top_symbols", {"metric": "gain", "days": 5, "sector": "NIFTY AUTO"})
    assert r is None and err.startswith("No stocks found") and "→" in err



# ── batch 05-11: more cases ──────────────────────────────────────────────────
def test_schema_is_small(ctx):
    """Every tool's text goes with every question: keep the declarations lean (tokens = cost + latency)."""
    import json as _j
    size = len(_j.dumps(T.TOOLS))
    assert size < 18500, size          # v16 live_screen (conditions) ≈ +450 tokens; trim before raising


def test_slash_commands_never_reach_the_llm(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {})
    for cmd in ("/help", "/check", "/check@TradEdgeBot", "/foo"):
        b.handle(_msg(cmd))
    assert b.router.prompts == []
    help_, check, check2, unknown = [t for _, t in b.tg.sent]
    assert "I never give buy/sell advice" in help_
    assert check.startswith("TradEdge data check") and "TOOLS (dry run)" in check and check2.startswith("TradEdge")
    assert "Unknown command /foo" in unknown


def test_check_report_lists_sources_and_tools(ctx):
    text = T.check_report(ctx)
    for part in ("Scan results", "OI cache", "1-min bars", "PACE (hi/lo)", "A/D history",
                 "daily_ohlc.csv", "symbol_day", "outcomes", "oi_eod", "ready"):
        assert part in text, part
    off = T.check_report(ctx, live=False)
    assert "Scan results" not in off and "daily_ohlc.csv" in off


def test_gap_metric(ctx):
    r, text = ok(ctx, "live_top", metric="gap")
    assert dict(r["rows"])["M&M"] == pytest.approx((3010 / 2970 - 1) * 100)
    assert "gap-up" in text and "prev close 2970.0 → open 3010.0" in text


def test_new_day_high_in_window(ctx):
    r, text = ok(ctx, "live_top", metric="new_high", window_min=15)
    assert [s for s, _ in r["rows"]] == ["AAA"]                  # only AAA keeps making highs
    assert "new day highs" in text and "new day high now" in text
    r, _ = ok(ctx, "live_top", metric="new_low", window_min=15)
    assert [s for s, _ in r["rows"]] == ["BBB"]


def test_compare_symbols(ctx):
    r, text = ok(ctx, "compare_symbols", symbols=["aaa", "BBB", "AAA"])
    assert [x["sym"] for x in r["rows"]] == ["AAA", "BBB"]
    assert text.startswith("AAA vs BBB") and "Last 15m" in text and "Since open" in text
    r, err = T.run(ctx, "compare_symbols", {"symbols": ["AAA"]})
    assert "2 or 3 stocks" in err


def test_end_day_history(ctx):
    r, text = ok(ctx, "top_symbols", metric="gain", days=2, end_day="2026-10-30")
    assert r["to"] == "2026-10-30" and r["from"] == "2026-10-29"


def test_symbol_suggestions(ctx):
    ctx.state.FNO_SYMBOLS += ["TATAMOTORS", "TATASTEEL", "TATAPOWER"]
    r, err = T.run(ctx, "live_symbol", {"symbol": "TATA"})
    assert "Did you mean: TATAMOTORS, TATAPOWER, TATASTEEL?" in err
    r, err = T.run(ctx, "live_symbol", {"symbol": "TATAMOTRS"})
    assert "TATAMOTORS" in err


def test_pre_market_and_weekend_messages(ctx):
    ctx.state.SESSION_RESULTS.clear()
    ctx._now = datetime(2026, 11, 4, 9, 5)
    assert "opens at 09:15" in T.run(ctx, "live_top", {"metric": "pct"})[1]
    ctx._now = datetime(2026, 11, 7, 11, 0)
    assert "weekend" in T.run(ctx, "live_top", {"metric": "pct"})[1]


def test_live_symbol_5_and_15_min(ctx):
    r, text = ok(ctx, "live_symbol", symbol="AAA")
    w5, w15 = r["recent"][5], r["recent"][15]
    # 'now' is the LIVE price (105), so the window line agrees with the card's Price line
    assert w5["pct"] == pytest.approx((105 / 114 - 1) * 100) and w15["then"] == 104.0 and w5["now"] == 105.0
    assert w5["ext"] == "new day HIGH"
    assert "Last  5 min" in text and "Last 15 min" in text and "104.00 → 105.00" in text


def test_watchlist_score_percent_and_short_side(ctx):
    ctx.state.SESSION_RESULTS["AAA"]["pct_score"] = 78
    ctx.state.SESSION_RESULTS["CCC"]["pct_score"] = 61
    ctx.state.SESSION_RESULTS_SHORT["BBB"]["pct_score"] = 90                # BBB is a SHORT watch
    r, text = ok(ctx, "watchlist_status")
    assert [x["sym"] for x in r["rows"]] == ["BBB", "AAA", "CCC"]
    assert "1 BBB  90%" in text and "2 AAA  78%" in text


class _FakeKite:
    def __init__(self, bars):
        self.bars, self.calls = bars, []

    def historical_data(self, tok, frm, to, interval):
        self.calls.append((tok, frm[:10], interval))
        return self.bars.get(tok, [])


def test_kite_fill_only_missing_stocks_once_a_day(ctx, monkeypatch):
    monkeypatch.setattr(T, "_KITE_DAILY", {})
    monkeypatch.setattr(T, "_KITE_STATE", {"day": None, "running": False, "fetched": 0, "failed": [], "missing": []})
    import csv as _csv
    rows = [r for r in _csv.reader(open(ctx.state.DAILY_OHLC_CSV)) if not (r[0] == "2026-11-03" and r[1] == "CCC")]
    with open(ctx.state.DAILY_OHLC_CSV, "w", newline="") as f:
        _csv.writer(f).writerows(rows)                                       # CCC has no 11-03 bar
    ctx.state.FNO_SYMBOLS = ["AAA", "BBB", "CCC", "NIFTY"]
    k = _FakeKite({3: [{"open": 100, "high": 103, "low": 99, "close": 102, "volume": 5}]})
    ctx.state.kite_mgr = types.SimpleNamespace(kite=k, _equity_tokens={"AAA": 1, "BBB": 2, "CCC": 3})
    st = T.kite_fill(ctx, sleep=lambda s: None)
    assert st["missing"] == ["CCC"] and st["fetched"] == 1 and k.calls == [(3, "2026-11-03", "day")]
    assert T.kite_fill(ctx, sleep=lambda s: None)["fetched"] == 1 and len(k.calls) == 1   # once a day
    ctx.state.SESSION_RESULTS["CCC"]["price"] = 103.5
    r, text = ok(ctx, "live_level_breaks", kind="prev_high")
    assert [x["sym"] for x in r["rows"]] == ["CCC"] and r["rows"][0]["level"] == 103   # Kite bar used


def test_level_breaks_names_missing_and_skips_indices(ctx, monkeypatch):
    monkeypatch.setattr(T, "_KITE_STATE", {"day": TODAY, "running": False, "fetched": 0, "failed": ["DDD"],
                                           "missing": ["DDD"]})
    ctx.state.SESSION_RESULTS["NIFTY"] = {"price": 24000.0, "intra_pct": 0.1}
    ctx.state.SECTOR_MAP["NIFTY"] = "INDEX"
    r, text = ok(ctx, "live_level_breaks", kind="prev_high")
    assert r["missing_syms"] == ["DDD", "M&M"]                                  # NIFTY not counted
    assert "No 11-03 bar yet for 2 stocks: DDD, M&M" in text


def test_bottom_title_and_sector_metric_default(ctx):
    r, text = ok(ctx, "top_symbols", metric="volume", days=2, order="bottom")
    assert text.startswith("Lowest ")
    assert T.validate(ctx, "live_sector_rank", {"order": "bottom"})["metric"] == "price"


# ── v9: every way of asking × every list tool (built from the SCHEMA, so a new tool or
#    option that the code ignores fails here, not in the user's Telegram) ─────────────────
LIST_CALLS = {                       # one base question per tool that returns a ranked list
    "top_symbols": {"metric": "gain", "days": 5}, "level_breaks_history": {"kind": "prev_high"},
    "sector_history": {"days": 5}, "live_top": {"metric": "pct"},
    "live_level_breaks": {"kind": "prev_high"}, "live_vwap": {"side": "above"},
    "current_candidates": {}, "watchlist_status": {}, "recent_alerts": {},
    "live_sector_rank": {"metric": "price"},
    "live_screen": {"conditions": [{"left": "price", "op": ">", "value": 0}]},
}
NO_ORDER_OK = {"recent_alerts": "newest first is the only useful order"}
SECTOR_ROWS = {"sector_history", "live_sector_rank"}        # rows ARE sectors: no sector filter


def _props(name):
    return next(t for t in T.TOOLS if t["name"] == name)["parameters"]["properties"]


def test_matrix_every_list_tool_has_the_basic_options():
    with_limit = {t["name"] for t in T.TOOLS if "limit" in t["parameters"]["properties"]}
    assert with_limit == set(LIST_CALLS), "add the new list tool to LIST_CALLS"
    for name in LIST_CALLS:
        p = _props(name)
        assert "limit" in p, name
        assert "order" in p or name in NO_ORDER_OK, "%s: 'lowest / weakest' has nowhere to go" % name
    for t in T.TOOLS:
        assert t["name"] == "cannot_answer" or "unsupported" in t["parameters"]["properties"], t["name"]


def _sector_ctx(ctx):
    """Bigger sector groups so sector rows survive MIN_SECTOR_N and order checks have >1 row."""
    st = ctx.state
    for sym, sec, px, op, pct in (("E1", "NIFTY METAL", 98.0, 100.0, -1.0), ("E2", "NIFTY METAL", 97.0, 100.0, -2.0),
                                  ("E3", "NIFTY METAL", 99.0, 100.0, 1.0), ("F1", "NIFTY FMCG", 102.0, 100.0, -0.5),
                                  ("F2", "NIFTY FMCG", 103.0, 100.0, -0.2), ("F3", "NIFTY FMCG", 101.0, 100.0, 0.4),
                                  ("O1", "OTHER", 90.0, 100.0, -9.0), ("C1", "COMMODITIES", 80.0, 100.0, -5.0)):
        st.SESSION_RESULTS[sym] = {"price": px, "intra_pct": pct, "day_open": op, "rvol": 1.0, "vwap": px}
        st.SECTOR_MAP[sym] = sec
        st.FNO_SYMBOLS.append(sym)
    return ctx


def test_matrix_options_change_the_answer(ctx):
    _sector_ctx(ctx)
    checked = {"limit": 0, "order": 0, "sector": 0}
    for name, base in LIST_CALLS.items():
        p = _props(name)
        r0, err = T.run(ctx, name, dict(base, limit=20))
        assert err is None, (name, err)
        T.format_reply(r0)
        rows = T.result_symbols(r0) or [x.get("sector") for x in r0["rows"]]
        if len(rows) >= 2:                                   # limit really limits
            r1, _ = T.run(ctx, name, dict(base, limit=1))
            assert len(r1["rows"]) == 1, name
            checked["limit"] += 1
        if "order" in p and len(rows) >= 2:                  # bottom really reverses
            rb, _ = T.run(ctx, name, dict(base, order="bottom", limit=20))
            rows_b = T.result_symbols(rb) or [x.get("sector") for x in rb["rows"]]
            assert rows_b != rows, "%s ignores order=bottom" % name
            checked["order"] += 1
        if "sector" in p and name not in SECTOR_ROWS and T.result_symbols(r0):
            sec = ctx.state.SECTOR_MAP[T.result_symbols(r0)[0]]
            rs, err = T.run(ctx, name, dict(base, sector=sec, limit=20))
            assert err is None, (name, err)
            assert all(ctx.state.SECTOR_MAP.get(s) == sec for s in T.result_symbols(rs)), name
            checked["sector"] += 1
    assert checked["limit"] >= 7 and checked["order"] >= 6 and checked["sector"] >= 5, checked


def test_screenshot_top_5_sector_fallen_from_open(ctx, monkeypatch):
    """05-11 10:40: 'top 5 sector fallen from open' showed ALL sectors vs YESTERDAY."""
    monkeypatch.setattr(T, "MIN_SECTOR_N", 3)
    _sector_ctx(ctx)
    for s in ("AAA", "BBB", "CCC", "DDD", "M&M"):
        ctx.state.SESSION_RESULTS[s]["day_open"] = ctx.state.SESSION_RESULTS[s]["price"]
    r, text = ok(ctx, "live_sector_rank", metric="open", order="bottom", limit=5)
    assert [x["sector"] for x in r["rows"]] == ["NIFTY METAL", "NIFTY FMCG"]   # only real sectors
    assert r["rows"][0]["avg"] == pytest.approx(-2.0)                          # since OPEN, not vs yday (-0.67)
    assert text.startswith("Weakest 2 sectors by avg % since today's open")
    assert "left out (< 3 stocks or catch-all)" in text and "COMMODITIES (1)" in text and "OTHER (1)" in text
    r, _ = ok(ctx, "live_sector_rank", metric="open", window_min=15)
    assert r["window_min"] == 0


def test_sector_limit_says_how_many_hidden(ctx):
    _sector_ctx(ctx)
    r, text = ok(ctx, "live_sector_rank", metric="price", limit=2)
    assert len(r["rows"]) == 2 and ("2 of %d sectors shown" % r["n"]) in text


def test_honesty_guard_model_note_is_shown(ctx):
    r, text = ok(ctx, "live_top", metric="pct", unsupported="FII data")
    assert "⚠️ Can't do yet: FII data — this is the closest answer." in text


def test_honesty_guard_code_notes(ctx):
    r, text = ok(ctx, "live_top", metric="rvol", window_min=15, limit=50, show=["rvol", "delivery"])
    u = r["unsupported"]
    assert "last 15 min for rvol (whole day shown)" in u and "max 20 rows" in u and "column delivery here" in u
    assert "⚠️ Can't do yet" in text
    r, text = ok(ctx, "live_top", metric="pct")
    assert r["unsupported"] == [] and "⚠️" not in text


def test_watchlist_limit_sector_order(ctx):
    r, _ = ok(ctx, "watchlist_status", limit=1)
    assert len(r["rows"]) == 1 and r["n"] >= 2
    r_top, _ = ok(ctx, "watchlist_status", sort_by="pct")
    r_bot, _ = ok(ctx, "watchlist_status", sort_by="pct", order="bottom")
    assert T.result_symbols(r_top)[0] == T.result_symbols(r_bot)[-1]
    r, err = T.run(ctx, "watchlist_status", {"sector": "NIFTY AUTO"})
    assert r is None or all(ctx.state.SECTOR_MAP[s] == "NIFTY AUTO" for s in T.result_symbols(r))


def test_closest_first_title(ctx):
    _r, text = ok(ctx, "live_vwap", side="above", order="bottom")
    assert text.split("\n")[0].endswith("closest first")


def test_question_log_has_reply_title_and_notes(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"q": ("live_top", {"metric": "pct", "unsupported": "news"})})
    reply, rec = b.answer("q", "1")
    assert rec["reply_head"] == reply.split("\n")[0] and rec["unsupported"] == ["news"] and rec["rows"] >= 1


# ── v10: metric engine (what × how) + fix_route guards ───────────────────────────────
def _engine_data(ctx):
    """Futures ticks, IV readings and a PACE budget that crosses a band in the last 5 min."""
    st = ctx.state
    now = NOW.timestamp()
    st.LIVE_TICK_LOCK = threading.Lock()
    st.LIVE_VOL_HISTORY = {
        "AAA": deque([(now - 60 * m, v) for m, v in ((40, 500), (30, 1000), (15, 2000), (1, 5000))]),  # 3000 vs 1000
        "BBB": deque([(now - 60 * m, v) for m, v in ((31, 1000), (16, 3000), (0, 4000))]),             # 1000 vs 2000
        "CCC": deque([(now - 60 * m, v) for m, v in ((10, 100), (0, 900))]),       # only 10 min of ticks
        "DDD": deque([(now - 600, 7000)])}                                         # stale: no tick for 10 min
    st.IV_HISTORY_LOCK = threading.RLock()
    st.IV_HISTORY = {"AAA": deque([{"ts": now - 60 * m, "iv": 20 + (30 - m) / 6.0} for m in range(30, -1, -3)]),
                     "BBB": deque([{"ts": now - 60 * m, "iv": 30 - (30 - m) / 10.0} for m in range(30, -1, -3)])}
    st.IMPLIED_BUDGET["AAA"] = 30.0          # AAA PACE = k/30: QUIET until 09:30, NORMAL after
    ctx._pband = _bands
    return ctx


def test_fut_volume_window_and_day(ctx):
    _engine_data(ctx)
    r, text = ok(ctx, "live_top", metric="fut_volume", window_min=15)
    assert r["rows"] == [("AAA", pytest.approx(3.0)), ("BBB", pytest.approx(0.5))]
    assert "1 stocks skipped" in r["note"]                          # CCC: not enough ticks; DDD stale
    assert "futures volume rise" in text and "fut vol prev 15 min 1000 → last 15 min 3000" in text
    r, _ = ok(ctx, "live_top", metric="fut_volume")
    assert [s for s, _ in r["rows"]] == ["AAA", "BBB", "CCC"]
    raw = {"metric": "fut_volume", "window_min": 60}
    clean = T.validate(ctx, "live_top", raw)
    assert clean["window_min"] == 40 and "max 40 min window for fut_volume (40 min shown)" in T._unsupported(
        "live_top", raw, clean)


@pytest.fixture(autouse=True)
def _fresh_samples(monkeypatch):
    """The sampler store is module-level (one per app): give every test an empty one."""
    monkeypatch.setattr(T, "_SAMPLES", {})
    monkeypatch.setattr(T, "_ISAMPLES", {})
    monkeypatch.setattr(T, "_PACE", {"day": None, "last": {}, "events": {}, "runs": [], "loaded": False, "hb": 0})
    monkeypatch.setattr(T, "_SAMPLE_META", {"since": None, "last": None, "n": 0})


def _feed(ctx, minutes, price_of, hilo_of=None):
    """Run the real sampler once a minute for `minutes` minutes ending at NOW.
    price_of(sym, k) / hilo_of(sym, k): values k minutes before NOW (None = leave as is)."""
    st, now = ctx.state, NOW.timestamp()
    keep = {s: dict(r) for s, r in st.SESSION_RESULTS.items()}
    keep_hilo = {s: dict(v) for s, v in st.SESSION_HILO.items()}
    for k in range(minutes, -1, -1):
        for sym, r in keep.items():
            v = price_of(sym, k)
            st.SESSION_RESULTS[sym] = dict(r, price=v if v is not None else r["price"])
        if hilo_of:
            for sym in keep_hilo:
                h = hilo_of(sym, k)
                if h:
                    st.SESSION_HILO[sym] = {"hi": h[0], "lo": h[1]}
        T.sample_once(ctx, now - 60 * k)
    for sym, r in keep.items():                   # leave NOW's values in place
        st.SESSION_RESULTS[sym] = dict(r, price=price_of(sym, 0) or r["price"])


def test_pace_cross_and_pace_window(ctx):
    """PACE from samples of the dashboard's own PACE (futures hi/lo ÷ budget = 3)."""
    _engine_data(ctx)
    ctx.state.IMPLIED_BUDGET["AAA"] = 3.0
    rng = lambda sym, k: (100 + (2.7 + 0.1 * (20 - k)) if sym == "AAA" else 101, 100)  # noqa: E731
    _feed(ctx, 20, lambda s, k: None, rng)          # AAA PACE 0.9 -> 1.57: NORMAL -> HOT
    r, text = ok(ctx, "live_top", metric="pace_cross", window_min=15)
    assert [s for s, _ in r["rows"]] == ["AAA"]                       # CCC flat: no band change
    assert r["rows"][0][1] == pytest.approx(1.5 / 3)
    assert "ACTIVE → HOT · crossed 09:32" in text and "PACE 1.07 → 1.57" in text
    assert text.startswith("1 stock PACE band crossed · last 15 min")
    assert "price 105.00" in text and "day +2.00%" in text
    r, _ = ok(ctx, "live_top", metric="pace", window_min=5)
    assert dict(r["rows"])["CCC"] == 0 and r["rows"][0][0] == "AAA"
    r, _ = ok(ctx, "live_top", metric="pace_cross")
    assert r["window_min"] == 15                                       # needs a window: default 15


def test_vwap_cross_newest_first(ctx):
    # AAA: below VWAP 103 until 6 min ago, above since; M&M: above until 4 min ago? no - always above
    _feed(ctx, 20, lambda s, k: {"AAA": 102.0 if k > 6 else 105.0, "BBB": 85.0,
                                 "CCC": 99.5 if k > 3 else 100.5}.get(s))
    r, text = ok(ctx, "live_top", metric="vwap_up", window_min=15)
    assert [s for s, _ in r["rows"]] == ["CCC", "AAA"]                 # CCC crossed 09:31, AAA 09:28
    assert "2 stocks crossed ABOVE VWAP · newest first" in text.split("\n")[0]
    assert "crossed 09:31" in text and "price 99.50 → 100.50" in text and "day +0.30%" in text
    r, _ = ok(ctx, "live_top", metric="vwap_up", window_min=15, order="bottom")
    assert [s for s, _ in r["rows"]] == ["AAA", "CCC"]
    r, err = T.run(ctx, "live_top", {"metric": "vwap_down", "window_min": 15})
    assert err == "No stock crossed BELOW VWAP in the last 15 min."


def test_yesterday_high_and_open_cross(ctx):
    pdh = T._pdh(ctx)["AAA"]
    _feed(ctx, 20, lambda s, k: (pdh - 1 if k > 4 else pdh + 1) if s == "AAA" else None)
    r, text = ok(ctx, "live_top", metric="pdh_up", window_min=10)
    assert [s for s, _ in r["rows"]] == ["AAA"] and "crossed ABOVE yesterday's high" in text
    r, err = T.run(ctx, "live_top", {"metric": "open_down", "window_min": 10})
    assert err is None or err.startswith("No ")


def test_samples_after_restart_say_when_ready(ctx):
    _feed(ctx, 5, lambda s, k: None)
    r, err = T.run(ctx, "live_top", {"metric": "vwap_up", "window_min": 15})
    assert "Collecting live samples since 09:29 (app start) — a 15-min window is ready at 09:45" in err
    r, err = T.run(ctx, "live_top", {"metric": "pace", "window_min": 5})
    assert err is None or err.startswith("No ")


def test_every_stock_row_has_price_and_day_pct(ctx):
    """user 05-11: 'any question, if a stock is there, price and % must be there'."""
    _engine_data(ctx)
    _feed(ctx, 20, lambda s, k: None, lambda s, k: (100 + 0.1 * (20 - k), 100))
    res = ctx.state.SESSION_RESULTS
    calls = [("live_top", {"metric": m, "window_min": w}) for m in T.LIVE_METRICS for w in (0, 15)]
    calls += [(n, a) for n, a in LIST_CALLS.items() if n.startswith(("live_", "current", "watch", "recent"))
              and n != "live_sector_rank"]
    checked = 0
    for name, args in calls:
        r, err = T.run(ctx, name, args)
        if err:
            continue
        text = T.format_reply(r)
        lines = text.split("\n")
        for sym in T.result_symbols(r) or [a.get("sym") for a in r.get("rows", []) if isinstance(a, dict)]:
            i = next(j for j, l in enumerate(lines) if re.match(r"^\s*\d+ %s\b" % re.escape(sym), l))
            card = lines[i] + " | " + lines[i + 1]
            day_is_main = name == "live_top" and args.get("metric") == "pct" and not args.get("window_min")
            assert day_is_main or "day " in lines[i] or "day " in lines[i + 1], (name, args, card)
            assert re.search(r"\d+\.\d{1,2}\b(?!%)", lines[i + 1]), (name, args, card)   # a price
            checked += 1
    assert checked >= 40, checked


def _f(v):
    return None if v is None else float(v)


def test_iv_vwap_score_metrics(ctx):
    _engine_data(ctx)
    r, text = ok(ctx, "live_top", metric="iv", window_min=15)
    assert dict(r["rows"])["AAA"] == pytest.approx(2.5) and dict(r["rows"])["BBB"] == pytest.approx(-1.5)
    assert "+2.5 pts" in text and "IV 22.5 → 25.0" in text
    r, _ = ok(ctx, "live_top", metric="iv")
    assert r["rows"][0][0] == "BBB"                                    # highest IV level now
    r, text = ok(ctx, "live_top", metric="vwap")
    assert r["rows"][0][0] == "AAA" and "VWAP 103.00 → now 105.00" in text
    ctx.state.SESSION_RESULTS["CCC"]["pct_score"] = 90
    r, text = ok(ctx, "live_top", metric="score", limit=2)
    assert r["rows"][0][0] == "CCC" and "90%" in text and "PRIMED" in text


def test_matrix_every_metric_every_how(ctx):
    """Each metric × (no window, window 15): what the capability table promises must happen,
    and anything it cannot do must be SAID in the reply."""
    _engine_data(ctx)
    _feed(ctx, 20, lambda s, k: None, lambda s, k: (100 + 0.1 * (20 - k), 100))
    ran = 0
    for m, cap in T.LIVE_METRICS.items():
        for w in (0, 15):
            raw = {"metric": m, "window_min": w}
            clean = T.validate(ctx, "live_top", raw)
            notes = T._unsupported("live_top", raw, clean)
            if w and cap["window"]:
                assert clean["window_min"] == 15 and notes == [], (m, notes)
            elif w:
                assert clean["window_min"] == 0 and any("whole day shown" in u for u in notes), m
            elif cap.get("needs"):
                assert clean["window_min"] == 15, m
            r, err = T.run(ctx, "live_top", raw)
            assert err is None or err.startswith("No "), (m, w, err)     # fixture lacks a few windows
            if r is None:
                continue
            ran += 1
            text = T.format_reply(r)
            assert ("last 15 min" in text.split("\n")[0]) == bool(clean["window_min"]), (m, w, text[:60])
            assert ("⚠️ Can't do yet" in text) == bool(notes), m
            for order in ("top", "bottom"):
                rr, err = T.run(ctx, "live_top", dict(raw, order=order, limit=1, sector="NIFTY BANK"))
                assert err is None or err.startswith("No "), (m, w, order, err)
                if rr:
                    assert len(rr["rows"]) <= 1 and all(ctx.state.SECTOR_MAP[s] == "NIFTY BANK" for s, _ in rr["rows"])
    assert ran >= 26, ran


def test_cols_in_text():
    assert T._cols_in_text("stocks above vwap now with volume and rvol") == ["volume", "rvol"]
    assert T._cols_in_text("highest volume today saathe oi pan batavo") == ["oi"]
    assert T._cols_in_text("top gainers rvol pan batavo") == ["rvol"]
    assert T._cols_in_text("top gainers with fut vol") == []
    assert T._cols_in_text("top gainers last 15 min") == []


def _prev(tool, args, q="x"):
    return [{"tool": tool, "args": args, "symbols": ["AAA", "BBB"], "time": "09:30", "q": q}]


def test_fix_route_the_9_eval_misses(ctx):
    f = T.fix_route
    # columns named in the question, dropped by the model
    n, a, fx = f(ctx, "stocks above vwap now with volume and rvol", "live_vwap", {"side": "above"})
    assert a["show"] == ["volume", "rvol"] and fx
    # 'with volume' follow-up that came back as from_previous on the same question
    pv = _prev("live_vwap", T.validate(ctx, "live_vwap", {"side": "above"}))
    n, a, fx = f(ctx, "with volume", "live_vwap", {"side": "above", "from_previous": 1}, pv)
    assert "from_previous" not in a and a["show"] == ["volume"]
    # 'top 3 only' = refresh of the same list
    pv = _prev("live_top", T.validate(ctx, "live_top", {"metric": "pct"}))
    n, a, _ = f(ctx, "top 3 only", "live_top", {"metric": "pct", "limit": 3, "from_previous": 1}, pv)
    assert "from_previous" not in a and a["limit"] == 3
    # explicit 'from that' + a condition = the earlier list, keep it
    n, a, _ = f(ctx, "from that, which have rvol above 2", "live_top",
                {"metric": "pct", "from_previous": 1, "filters": [{"col": "rvol", "op": ">", "value": 2}]}, pv)
    assert a["from_previous"] == 1
    n, a, _ = f(ctx, "aama thi top 3", "live_top", {"metric": "pct", "limit": 3, "from_previous": 1}, pv)
    assert a["from_previous"] == 1
    # a DIFFERENT question on that list keeps from_previous
    n, a, _ = f(ctx, "which one has highest volume from that", "live_top",
                {"metric": "volume", "from_previous": 1}, pv)
    assert a["from_previous"] == 1
    # past year for a day not yet reached this year
    n, a, fx = f(ctx, "top gainers on 30th oct", "top_symbols", {"metric": "gain", "days": 1, "end_day": "2025-10-30"})
    assert a["end_day"] == "2026-10-30"
    n, a, _ = f(ctx, "top gainers on 30th dec", "top_symbols", {"metric": "gain", "end_day": "2025-12-30"})
    assert a["end_day"] == "2025-12-30"                              # Dec 2026 has not happened: keep
    # today -> live
    n, a, fx = f(ctx, "top gainers today by delivery percentage", "top_symbols",
                 {"metric": "gain", "days": 1, "unsupported": "delivery percentage"})
    assert n == "live_top" and a == {"metric": "pct", "unsupported": "delivery percentage"}
    n, a, _ = f(ctx, "aaje kon yesterday high todyu", "level_breaks_history", {"kind": "prev_high"})
    assert n == "level_breaks_history"                                # 'yesterday' = a past word: untouched
    n, a, _ = f(ctx, "which broke previous high today", "level_breaks_history", {"kind": "prev_high"})
    assert n == "live_level_breaks"
    n, a, _ = f(ctx, "top gainers last 5 days", "top_symbols", {"metric": "gain", "days": 5})
    assert n == "top_symbols"


def test_sector_typo_from_the_model(ctx):
    assert T.validate(ctx, "watchlist_status", {"sector": "NIFTS BANK"})["sector"] == "NIFTY BANK"
    with pytest.raises(T.ToolError):
        T.validate(ctx, "watchlist_status", {"sector": "CHEMICALS"})


def test_bot_applies_fix_route_and_logs_it(ctx, tmp_path):
    b = _bot(ctx, tmp_path, {"top gainers today": ("top_symbols", {"metric": "gain", "days": 1})})
    reply, rec = b.answer("top gainers today", "1")
    assert rec["tool"] == "live_top" and rec["fixes"] == ["today -> live tool"]
    assert reply.startswith("Top ")


def test_check_shows_samples_and_gemini(ctx, tmp_path):
    _feed(ctx, 3, lambda s, k: None)
    b = _bot(ctx, tmp_path, {})
    b.router.last_error = ("12:14", "gemini-3.5-flash-lite", "quota_day", "429 ...")
    b.router.dead = {"gemini-2.5-flash-lite"}
    reply, _ = b.answer("/check", "1")
    assert "Live samples   4 taken · since 09:31" in reply
    assert "last error 12:14 · gemini-3.5-flash-lite · quota_day" in reply
    assert "not found (skipped): gemini-2.5-flash-lite" in reply


def test_sampler_never_raises_and_reports(ctx, monkeypatch):
    monkeypatch.setattr(T, "_live_pace", lambda c: 1 / 0)
    assert T.sample_once(ctx) == 0 and "ZeroDivisionError" in T._SAMPLE_META["error"]
