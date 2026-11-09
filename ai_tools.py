# -*- coding: utf-8 -*-
"""
ai_tools.py -- the ONLY things the Telegram AI can do. READ-ONLY.  v2, 04-11-2026.

WHY THIS FILE EXISTS
    The LLM never touches data. It picks one tool name + arguments (ai_llm.py),
    this file validates those arguments, runs a fixed query, and builds the reply
    text itself. So the model can never write SQL, never invent a number, and
    never place an order -- there is no tool for that.

TWO DATA SOURCES
    history  -> SQLite, opened mode=ro (saga_obs.db, playbook.db)
    live     -> in-memory caches in state.py, COPIED under their lock, lock
                released, then filtered / sorted. Never two locks at once.

NOT ADVICE
    current_candidates() lists what YOUR rules engine already flagged, ranked by
    YOUR conviction score. market_mood() shows YOUR regime label and the numbers
    behind it. Nothing here says buy or sell.

v2 (04-11-2026): `show` extra columns on every list, `window_min` (last N
minutes from 1-min bars / OI history), watchlist, alerts, sectors, market mood,
breadth trend, monospace tables (Telegram <pre>, see ai_chat.py).
"""
import json
import os
import random
from collections import deque
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta

# ── limits (validation, not strategy) ─────────────────────────────────────────
MAX_LIMIT = 20
MAX_DAYS = 20
MAX_SHOW = 4
MIN_PERF_SIGNALS = 20      # signal_performance refuses below this many signals
MIN_PERF_DAYS = 5          # ... or below this many distinct sessions
BOOT_N = 1000              # day-block bootstrap resamples
BAR_STALE_S = 180          # a symbol whose last 1-min bar is older than this is skipped
_FLAG_STATES = ("SHAKTIMAN", "BREAKING", "PRIMED", "ARMING")


# ═════════════════════════════════════════════════════════════════════════════
# Context: where the data lives. Tests pass a fake state + temp DB paths.
# ═════════════════════════════════════════════════════════════════════════════
class Ctx:
    def __init__(self, state, saga_db=None, playbook_db=None, conviction=None,
                 short_conviction=None, breadth=None, today=None, now=None):
        self.state = state
        base = getattr(state, "BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
        self.saga_db = saga_db or os.path.join(base, "data", "saga_obs.db")
        self.playbook_db = playbook_db or getattr(state, "PLAYBOOK_DB",
                                                  os.path.join(base, "playbook.db"))
        self._conv, self._sconv, self._breadth = conviction, short_conviction, breadth
        self._today, self._now = today, now      # tests pin these; live code follows the clock
        # per-call scope ("from that list" + filters). Thread-local: the digest thread
        # and the chat thread share one Ctx and may run tools at the same time.
        self._tl = threading.local()

    @property
    def scope(self):
        return getattr(self._tl, "scope", None)

    @property
    def today(self):
        """Recomputed on every call: the thread runs for days."""
        return self._today or datetime.now().strftime("%Y-%m-%d")

    def now(self):
        return self._now or datetime.now()

    def conviction(self, side):
        """(sym, rvol, intra_pct, sector) -> (score, tier, streak, peers).
        Lazy import: oi_engine is heavy and tests do not need it."""
        if side == "SHORT":
            if self._sconv is None:
                import oi_engine
                self._sconv = oi_engine._calc_short_conviction
            return self._sconv
        if self._conv is None:
            import oi_engine
            self._conv = oi_engine._calc_conviction
        return self._conv

    def pband(self, pace):
        """saga_engine.pband_of: the dashboard's own PACE band names. Lazy import."""
        if getattr(self, "_pband", None) is None:
            try:
                import saga_engine
                self._pband = saga_engine.pband_of
            except Exception:
                self._pband = lambda p: None
        return self._pband(pace)

    def breadth(self):
        """oi_engine.breadth(): new day highs / lows now, 15 min, 60 min."""
        if self._breadth is None:
            import oi_engine
            self._breadth = oi_engine.breadth
        return self._breadth()


class ToolError(Exception):
    """Bad arguments or missing data. The message is shown to the user as is."""


def _ro(path):
    if not os.path.isfile(path):
        raise ToolError("Database not found: %s" % os.path.basename(path))
    uri = "file:%s?mode=ro" % path.replace("\\", "/").replace("?", "%3f").replace("#", "%23")
    c = sqlite3.connect(uri, uri=True, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def _f(v):
    try:
        if v is None or v == "":
            return None
        x = float(v)
        return None if x != x else x        # NaN -> None
    except (TypeError, ValueError):
        return None


def _snap(lock, obj):
    """Copy a shared container under its lock and release it at once.
    Shallow copy; callers only READ the inner objects."""
    with lock:
        return dict(obj) if isinstance(obj, dict) else list(obj)


def _secmap(ctx):
    return getattr(ctx.state, "SECTOR_MAP", {}) or {}


# ═════════════════════════════════════════════════════════════════════════════
# Tool declarations (sent to Gemini). Descriptions are the router's only guide.
# ═════════════════════════════════════════════════════════════════════════════
def _obj(props, required=()):
    return {"type": "OBJECT", "properties": props, "required": list(required)}


_LIMIT = {"type": "INTEGER", "description": "Rows 1-20, default 10."}
_SECTOR = {"type": "STRING", "description": "Exact name from the sector list. Empty = all."}
_SIDE = {"type": "STRING", "enum": ["LONG", "SHORT"], "description": "Default LONG."}
_ORDER = {"type": "STRING", "enum": ["top", "bottom"], "description": "Default top."}
# fut_vol5 = futures volume last 5 min ÷ the 5 min before; vol5 = stock (cash) volume spike last 5 min
LIVE_COLS = ["price", "pct", "rvol", "volume", "fut_vol5", "vol5", "oi", "vwap", "pace", "state", "sector",
             "streak", "score", "prox"]
HIST_COLS = ["price", "pct", "volume", "rvol", "pace", "oi", "sector"]


def _show(cols):
    return {"type": "ARRAY", "items": {"type": "STRING", "enum": cols},
            "description": "Rule 5. Max 4."}


# What each live_top metric can do. ONE table drives validation, the honesty notes and
# the matrix test: a metric that claims "window" must really change with window_min.
#   window : 'last N min' works (value = change inside the window)
#   needs  : the metric only makes sense with a window (default 15)
#   max_w  : longest window the data behind it allows
LIVE_METRICS = {
    "price":      {"window": False, "what": "price level (₹)"},
    "pct":        {"window": True,  "max_w": 120, "what": "% vs yesterday's close (window: % move)"},
    "open":       {"window": False, "what": "% since today's open"},
    "gap":        {"window": False, "what": "opening gap vs yesterday's close"},
    "volume":     {"window": True,  "max_w": 120, "what": "equity volume (window: spike x)"},
    # ticks are pruned at 90 min, and a window compares N min with the N before -> 2N < 90
    "fut_volume": {"window": True,  "max_w": 40, "what": "FUTURES volume (window: last N min vs the N before)"},
    "rvol":       {"window": False, "what": "relative volume"},
    "oi":         {"window": True,  "max_w": 120, "what": "% OI change (window: change in N min)"},
    "pace":       {"window": True,  "max_w": 120, "what": "PACE (window: PACE gained in N min)"},
    "pace_cross": {"window": True,  "needs": True, "max_w": 120, "what": "PACE moved into a new band in N min"},
    "pace_change": {"window": True, "needs": True, "max_w": 375, "what": "PACE VALUE changed (2 decimals) in N min"},
    "pace_low":   {"window": True, "needs": True, "max_w": 375, "what": "PACE changed by a new day LOW in N min"},
    "pace_high":  {"window": True, "needs": True, "max_w": 375, "what": "PACE changed by a new day HIGH in N min"},
    "vwap_up":    {"window": True,  "needs": True, "max_w": 120, "what": "crossed ABOVE VWAP in N min"},
    "vwap_down":  {"window": True,  "needs": True, "max_w": 120, "what": "crossed BELOW VWAP in N min"},
    "pdh_up":     {"window": True,  "needs": True, "max_w": 120, "what": "crossed ABOVE yesterday's high in N min"},
    "pdl_down":   {"window": True,  "needs": True, "max_w": 120, "what": "crossed BELOW yesterday's low in N min"},
    "open_up":    {"window": True,  "needs": True, "max_w": 120, "what": "crossed ABOVE today's open in N min"},
    "open_down":  {"window": True,  "needs": True, "max_w": 120, "what": "crossed BELOW today's open in N min"},
    "new_high":   {"window": True,  "needs": True, "max_w": 120, "what": "made a new day high in N min"},
    "new_low":    {"window": True,  "needs": True, "max_w": 120, "what": "made a new day low in N min"},
    "vwap":       {"window": False, "what": "% distance from VWAP"},
    "score":      {"window": False, "what": "long scanner score %"},
    "iv":         {"window": True,  "max_w": 50, "what": "ATM IV (window: change in IV points)"},
    "near_low":   {"window": False, "near": True, "what": "% above TODAY's low (0 = at / breaking it)"},
    "near_high":  {"window": False, "near": True, "what": "% below TODAY's high (0 = at / breaking it)"},
    "near_vwap":  {"window": False, "near": True, "what": "% away from VWAP (either side)"},
    "near_pdh":   {"window": False, "near": True, "what": "% away from YESTERDAY's high"},
    "near_pdl":   {"window": False, "near": True, "what": "% away from YESTERDAY's low"},
}

# FIELDS a condition may use (each stock, live). The model may only combine these; code computes
# every value. This is the answer to "any price condition": one small language, not one tool each.
FIELDS = {
    "price": "last price", "open": "today's open", "high": "today's high", "low": "today's low",
    "vwap": "VWAP", "pdc": "previous close", "pdh": "previous day high", "pdl": "previous day low",
    "pct": "day % vs previous close", "open_pct": "% since open", "gap": "opening gap %",
    "range_pct": "today's high-low range %", "pace": "PACE", "rvol": "RVOL",
    "fut_vol5": "futures vol 5 min ÷ 5 before", "vol5": "stock vol 5-min spike", "oi": "OI change %",
    "iv": "ATM IV", "score": "long score %"}
FIELD_LABEL = {"price": "price", "open": "open", "high": "high", "low": "low", "vwap": "VWAP", "pdc": "prev close",
               "pdh": "prev high", "pdl": "prev low", "pct": "day", "open_pct": "since open", "gap": "gap",
               "range_pct": "range", "pace": "PACE", "rvol": "RVOL", "fut_vol5": "fut vol 5m", "vol5": "stock vol 5m",
               "oi": "OI", "iv": "IV", "score": "score"}
PRICE_FIELDS = {"price", "open", "high", "low", "vwap", "pdc", "pdh", "pdl"}
EQ_TOL_PCT = 0.05            # "open = high" = within 0.05% (user 06-11): real prices rarely match to the paisa
_UNIVERSE = {"type": "STRING", "enum": ["all", "watchlist", "setups"],
             "description": "Default all."}
CONDITIONS = {"type": "ARRAY", "description": "Rule 14. All must hold (AND).", "items": {"type": "OBJECT", "properties": {
    "left": {"type": "STRING", "enum": list(FIELDS)},
    "op": {"type": "STRING", "enum": ["=", ">", "<", ">=", "<="]},
    "right": {"type": "STRING", "description": "Another field name (as left), or empty to use value."},
    "value": {"type": "NUMBER"},
    "pct": {"type": "NUMBER", "description": "Offset % on right (0.5% below open -> -0.5)."}},
    "required": ["left", "op"]}}

_ORDER_DIST = {"type": "STRING", "enum": ["top", "bottom"],
               "description": "bottom = closest first."}
_UNSUPPORTED = {"type": "STRING",
                "description": "Rule 13."}
_WINDOW = {"type": "INTEGER",
           "description": "'last N min': 5-120. Empty = whole day."}

TOOLS = [
    {"name": "top_symbols",
     "description": ("PAST completed sessions (database). Rank stocks over the last N sessions by: gain "
                     "(% price change over the period), volume (average daily volume), oi (% change in "
                     "futures OI), pace (highest PACE), rvol (highest relative volume)."),
     "parameters": _obj({
         "metric": {"type": "STRING", "enum": ["gain", "volume", "oi", "pace", "rvol"]},
         "days": {"type": "INTEGER", "description": "Past sessions, 1-20. 'this week' = 5, 'yesterday' = 1."},
         "end_day": {"type": "STRING", "description": "YYYY-MM-DD only when the user names a date ('on 3rd Nov'). "
                                                      "Empty = up to the latest session."},
         "order": _ORDER, "limit": _LIMIT, "sector": _SECTOR, "show": _show(HIST_COLS)},
         ["metric", "days"])},
    {"name": "level_breaks_history",
     "description": ("PAST completed session: stocks whose high went above the previous session's high "
                     "(prev_high) or low below the previous session's low (prev_low) ON THAT PAST DAY. "
                     "Not for today."),
     "parameters": _obj({
         "kind": {"type": "STRING", "enum": ["prev_high", "prev_low"]},
         "day": {"type": "STRING", "description": "YYYY-MM-DD, or empty = latest completed session."},
         "order": _ORDER_DIST, "limit": _LIMIT, "sector": _SECTOR, "show": _show(HIST_COLS)}, ["kind"])},
    {"name": "symbol_history",
     "description": "PAST sessions for ONE stock: close, % change, volume, OI, PACE, RVOL per day.",
     "parameters": _obj({
         "symbol": {"type": "STRING", "description": "NSE symbol, e.g. RELIANCE."},
         "days": {"type": "INTEGER", "description": "1-20. Default 5."}}, ["symbol"])},
    {"name": "sector_history",
     "description": "PAST sessions: rank SECTORS by average % gain of their F&O stocks over the last N sessions.",
     "parameters": _obj({
         "days": {"type": "INTEGER", "description": "1-20. 'this week' = 5."},
         "order": _ORDER, "limit": _LIMIT}, ["days"])},
    {"name": "signal_performance",
     "description": ("HOW A SIGNAL TYPE PERFORMED in the past. by=stage (SAGA ladder stages: AAHAT, "
                     "SANKET, SAGA, SHARK, RAAKH) or by=tier (conviction tiers LOADING, READY, SUPER). "
                     "Returns sample size, average forward move and a confidence range."),
     "parameters": _obj({
         "by": {"type": "STRING", "enum": ["stage", "tier"]},
         "value": {"type": "STRING", "description": "Stage or tier name. Empty = all of them."},
         "sessions": {"type": "INTEGER", "description": "Past sessions, 5-120. Default 30."},
         "side": {"type": "STRING", "enum": ["LONG", "SHORT", "BOTH"], "description": "Default BOTH."}},
         ["by"])},
    {"name": "live_top",
     "description": ("TODAY / RIGHT NOW. Rank stocks by a metric (rule 6): pct, open, gap, volume (equity), "
                     "fut_volume (futures), rvol, oi, pace, pace_cross (PACE entered a new band), new_high, "
                     "new_low, vwap (distance), score, iv, pace_change (PACE value moved, e.g. 1.25 -> 1.27; "
                     "pace_low / pace_high = by a new day low / high); "
                     "JUST CROSSED in N min: vwap_up, vwap_down, pdh_up "
                     "(yesterday high), pdl_down (yesterday low), open_up, open_down; NEAR a level (closest "
                     "first): near_low / near_high (today), near_vwap, near_pdh / near_pdl (yesterday). "
                     "window_min = last N minutes."),
     "parameters": _obj({
         "metric": {"type": "STRING", "enum": list(LIVE_METRICS)}, "universe": _UNIVERSE,
         "window_min": _WINDOW, "order": _ORDER, "limit": _LIMIT, "sector": _SECTOR,
         "show": _show(LIVE_COLS)}, ["metric"])},
    {"name": "live_level_breaks",
     "description": ("TODAY / RIGHT NOW: stocks trading above YESTERDAY's high (prev_high) or below "
                     "YESTERDAY's low (prev_low). 'Yesterday's high/low' is the LEVEL; the question is about "
                     "today."),
     "parameters": _obj({
         "kind": {"type": "STRING", "enum": ["prev_high", "prev_low"]},
         "order": _ORDER_DIST, "limit": _LIMIT, "sector": _SECTOR, "show": _show(LIVE_COLS)}, ["kind"])},
    {"name": "live_vwap",
     "description": "TODAY / RIGHT NOW: stocks trading above or below VWAP, furthest first.",
     "parameters": _obj({
         "side": {"type": "STRING", "enum": ["above", "below"]},
         "order": _ORDER_DIST, "limit": _LIMIT, "sector": _SECTOR, "show": _show(LIVE_COLS)}, ["side"])},
    {"name": "live_symbol",
     "description": ("TODAY / RIGHT NOW for ONE stock ('NYKAA study', 'RELIANCE now', 'TCS details'): "
                     "price, open, VWAP, RVOL, long AND short scanner score and state, OI pulse, OI change, "
                     "proximity, PACE with its band changes since open and the last new day high/low."),
     "parameters": _obj({"symbol": {"type": "STRING"}}, ["symbol"])},
    {"name": "current_candidates",
     "description": ("TODAY / RIGHT NOW: stocks the scanner has flagged (BREAKING, SHAKTIMAN, PRIMED, "
                     "ARMING) on the long or short side, ranked by conviction score. For 'any setups', "
                     "'signals now', 'best stocks right now'."),
     "parameters": _obj({"side": _SIDE, "order": _ORDER, "limit": _LIMIT, "sector": _SECTOR,
                         "show": _show(LIVE_COLS)})},
    {"name": "watchlist_status",
     "description": ("The trader's WATCHLIST (kanban) right now: stage, direction, scanner state, score, "
                     "proximity, % move, OI from open. sort_by picks the order; only_flagged keeps only "
                     "BREAKING / PRIMED / ARMING / SHAKTIMAN."),
     "parameters": _obj({
         "sort_by": {"type": "STRING", "enum": ["score", "prox", "oi", "pct", "rvol"],
                     "description": "Default score."},
         "side": {"type": "STRING", "enum": ["LONG", "SHORT", "BOTH"], "description": "Default BOTH."},
         "only_flagged": {"type": "BOOLEAN"}, "order": _ORDER, "limit": _LIMIT, "sector": _SECTOR,
         "show": _show(LIVE_COLS)})},
    {"name": "recent_alerts",
     "description": "TODAY's alerts already fired (bell / Telegram), newest first, optionally for ONE stock.",
     "parameters": _obj({"symbol": {"type": "STRING", "description": "Empty = all stocks."},
                         "limit": _LIMIT})},
    {"name": "live_sector_rank",
     "description": ("TODAY / RIGHT NOW: rank SECTORS. metric price (average % vs YESTERDAY's close), "
                     "open (average % since TODAY's OPEN: 'from open', 'opening thi'), breadth (% of its "
                     "stocks up), oi (average OI change). window_min for 'last 15 min'. 'top 5 sectors' -> limit."),
     "parameters": _obj({
         "metric": {"type": "STRING", "enum": ["price", "open", "breadth", "oi"]},
         "window_min": _WINDOW, "order": _ORDER, "limit": _LIMIT}, ["metric"])},
    {"name": "live_value",
     "description": ("TODAY / NOW the VALUE of fields for 1-5 stocks or indices: 'what is low of nykaa', 'nykaa "
                     "high and vwap', 'pdh of tcs', 'nifty high', 'vix now'. Fields as in rule 14."),
     "parameters": _obj({"symbols": {"type": "ARRAY", "items": {"type": "STRING"},
                                     "description": "NSE symbols or index names (NIFTY, BANKNIFTY, VIX)."},
                         "fields": {"type": "ARRAY", "items": {"type": "STRING", "enum": list(FIELDS)},
                                    "description": "Empty = price, high, low."}}, ["symbols"])},
    {"name": "live_screen",
     "description": "TODAY / NOW: stocks matching price CONDITIONS (rule 14): open = high, open = low, above pdh...",
     "parameters": _obj({"conditions": CONDITIONS, "universe": _UNIVERSE,
                         "order_by": {"type": "STRING", "description": "A field name. Default pct."},
                         "order": _ORDER, "limit": _LIMIT, "sector": _SECTOR, "show": _show(LIVE_COLS)},
                        ["conditions"])},
    {"name": "compare_symbols",
     "description": ("TODAY / RIGHT NOW side by side for 2 or 3 stocks: 'compare TCS and INFY', 'RELIANCE vs "
                     "HDFCBANK'. Day %, since-open %, last 15 min, VWAP, RVOL, OI, long/short score."),
     "parameters": _obj({"symbols": {"type": "ARRAY", "items": {"type": "STRING"},
                                     "description": "2-3 NSE symbols."}}, ["symbols"])},
    {"name": "market_mood",
     "description": ("TODAY / RIGHT NOW whole-market summary: regime label, NIFTY / BANKNIFTY vs VWAP, VIX, "
                     "advance/decline and its trend, new highs vs lows, OI build-up counts, long vs short "
                     "flags, IV direction, leading and lagging sector. For 'market kevu che', 'market mood', "
                     "'breadth', 'overall market'."),
     "parameters": _obj({})},
    {"name": "breadth_trend",
     "description": ("TODAY: how advance/decline changed over the last N minutes (improving or "
                     "weakening), sampled every few minutes."),
     "parameters": _obj({"window_min": {"type": "INTEGER", "description": "10-120. Default 30."}})},
    {"name": "cannot_answer",
     "description": ("Use when no other tool fits: chit-chat, buy/sell advice, targets, predictions, news, "
                     "FII/DII, PCR, anything not in the data. Give a short reason."),
     "parameters": _obj({"reason": {"type": "STRING"}}, ["reason"])},
]
LIVE_FILTER_COLS = ["price", "pct", "open", "gap", "rvol", "volume", "fut_vol5", "vol5", "oi", "vwap", "pace",
                    "streak"]
HIST_FILTER_COLS = ["price", "pct", "volume", "rvol", "pace", "oi"]
HIST_TOOLS = {"top_symbols", "level_breaks_history"}
MAX_FILTERS = 3


def _filters_schema(cols):
    return {"type": "ARRAY", "description": "Numeric conditions (rule 11). Max 3.",
        "items": {"type": "OBJECT", "properties": {
            "col": {"type": "STRING", "enum": cols},
            "op": {"type": "STRING", "enum": [">", ">=", "<", "<="]},
            "value": {"type": "NUMBER"}}, "required": ["col", "op", "value"]}}


_FROM_PREV = {"type": "INTEGER", "description": "Rule 11."}

for _t in TOOLS:          # the model reads tool descriptions more closely than general rules
    if "show" in _t["parameters"]["properties"]:
        _t["parameters"]["properties"]["from_previous"] = _FROM_PREV
        _t["parameters"]["properties"]["filters"] = _filters_schema(
            HIST_FILTER_COLS if _t["name"] in HIST_TOOLS else LIVE_FILTER_COLS)
for _t in TOOLS:          # honesty guard: the model reports what it could not map (shown in the reply)
    if _t["name"] != "cannot_answer":
        _t["parameters"]["properties"]["unsupported"] = _UNSUPPORTED
SCOPED_TOOLS = {t["name"] for t in TOOLS if "from_previous" in t["parameters"]["properties"]}
TOOL_NAMES = {t["name"] for t in TOOLS}


# ═════════════════════════════════════════════════════════════════════════════
# Validation: every argument is checked in code. The model's output is a guess.
# ═════════════════════════════════════════════════════════════════════════════
def _enum(args, key, allowed, default=None):
    v = args.get(key, default)
    if v is None or v == "":
        v = default
    if isinstance(v, str):
        for a in allowed:
            if v.strip().lower() == str(a).lower():
                return a
    if v in allowed:
        return v
    raise ToolError("'%s' must be one of: %s" % (key, ", ".join(map(str, allowed))))


def _int(args, key, lo, hi, default):
    v = args.get(key, default)
    if v is None or v == "":
        v = default
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        raise ToolError("'%s' must be a number" % key)
    return max(lo, min(hi, v))


def _window(args):
    w = _int(args, "window_min", 0, 120, 0)
    return 0 if w == 0 else max(5, w)


def _bool(args, key):
    v = args.get(key)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def _symbol(ctx, args, required=True):
    s = str(args.get("symbol") or "").upper().strip().replace(".NS", "")
    if not s:
        if required:
            raise ToolError("Which stock? Send the NSE symbol, e.g. RELIANCE.")
        return None
    universe = set(getattr(ctx.state, "FNO_SYMBOLS", []) or [])
    if universe and s not in universe:
        import difflib
        near = [u for u in sorted(universe) if u.startswith(s)][:5] or \
            difflib.get_close_matches(s, sorted(universe), n=4, cutoff=0.6)
        raise ToolError("%s is not in the F&O list.%s" % (
            s, (" Did you mean: " + ", ".join(near) + "?") if near else ""))
    return s


def _sector(ctx, args):
    s = str(args.get("sector") or "").strip().upper()
    if not s:
        return None
    sectors = {str(v).upper() for v in _secmap(ctx).values() if v}
    if not sectors or s in sectors:
        return s
    if ("NIFTY " + s) in sectors:              # user / model said BANK, map is NIFTY BANK
        return "NIFTY " + s
    near = sorted(x for x in sectors if s in x or x in s)
    if len(near) == 1:
        return near[0]
    import difflib
    close = difflib.get_close_matches(s, sorted(sectors), n=2, cutoff=0.8)
    if len(close) == 1 or (len(close) == 2 and difflib.SequenceMatcher(None, s, close[0]).ratio() >
                           difflib.SequenceMatcher(None, s, close[1]).ratio() + 0.05):
        return close[0]                          # a typo, not a different sector
    raise ToolError("Which sector? %s" % (", ".join(near or sorted(sectors))))


def _in_sector(ctx, sym, sector):
    return not sector or str(_secmap(ctx).get(sym, "")).upper() == sector


def _keep(ctx, sym, sector):
    """Sector filter + 'from that list' restriction."""
    sc = ctx.scope
    return _in_sector(ctx, sym, sector) and (not sc or sc.get("only") is None or sym in sc["only"])


_OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}


def _scope(ctx, items, sym_of, hist_day=None):
    """Apply the user's numeric filters ('rvol above 2', 'price below 500') to
    candidate rows BEFORE ranking and the limit, so 'top 10 of the filtered' is right.
    Live tools read current values; history tools read the window's last session."""
    sc = ctx.scope
    if not sc or not sc.get("filters") or not items:
        return items
    syms = [sym_of(x) for x in items]
    cols = sorted({f["col"] for f in sc["filters"]})
    if hist_day:
        c = _ro(ctx.saga_db)
        try:
            vals = _hist_cols(c, hist_day, syms, cols, ctx)
        finally:
            c.close()
    else:
        vals = _live_cols(ctx, syms, cols)
    keep = []
    for x, sym in zip(items, syms):
        v = vals.get(sym) or {}
        if all(_f(v.get(f["col"])) is not None and _OPS[f["op"]](_f(v.get(f["col"])), f["value"])
               for f in sc["filters"]):
            keep.append(x)
    return keep


def _scope_hint(ctx):
    sc = ctx.scope
    if not sc:
        return ""
    bits = []
    if sc.get("only") is not None:
        bits.append("within the %d stocks of answer #%d" % (len(sc["only"]), sc["from"]))
    if sc.get("filters"):
        bits.append("with " + ", ".join("%s %s %g" % (f["col"], f["op"], f["value"]) for f in sc["filters"]))
    return (" (" + "; ".join(bits) + ")") if bits else ""


def _cols(args, allowed, skip=()):
    v = args.get("show") or []
    if isinstance(v, str):
        v = [x for x in v.replace(" and ", ",").split(",")]
    out = []
    for x in v:
        x = str(x).strip().lower()
        if x in allowed and x not in out and x not in skip:
            out.append(x)
    return out[:MAX_SHOW]


def _with_price(cols):
    """Live ranked lists always show the price first (what the trader looks at next)."""
    return (["price"] + [c for c in cols if c != "price"])[:MAX_SHOW]


def validate(ctx, name, args):
    """Returns clean args or raises ToolError. Unknown keys are dropped."""
    if name not in TOOL_NAMES:
        raise ToolError("Unknown tool '%s'" % name)
    a = dict(args or {})
    lim = lambda: _int(a, "limit", 1, MAX_LIMIT, 10)          # noqa: E731
    if name == "top_symbols":
        m = _enum(a, "metric", ["gain", "volume", "oi", "pace", "rvol"])
        end = str(a.get("end_day") or "").strip()
        if end:
            try:
                datetime.strptime(end, "%Y-%m-%d")
            except ValueError:
                raise ToolError("end_day must be YYYY-MM-DD")
        return {"metric": m, "days": _int(a, "days", 1, MAX_DAYS, 5), "end_day": end or None,
                "order": _enum(a, "order", ["top", "bottom"], "top"), "limit": lim(),
                "sector": _sector(ctx, a), "show": _cols(a, HIST_COLS, {m})}
    if name == "level_breaks_history":
        day = str(a.get("day") or "").strip()
        if day:
            try:
                datetime.strptime(day, "%Y-%m-%d")
            except ValueError:
                raise ToolError("day must be YYYY-MM-DD")
        return {"kind": _enum(a, "kind", ["prev_high", "prev_low"]), "day": day or None,
                "order": _enum(a, "order", ["top", "bottom"], "top"),
                "limit": lim(), "sector": _sector(ctx, a), "show": _cols(a, HIST_COLS, {"pct"})}
    if name == "symbol_history":
        return {"symbol": _symbol(ctx, a), "days": _int(a, "days", 1, MAX_DAYS, 5)}
    if name == "sector_history":
        return {"days": _int(a, "days", 1, MAX_DAYS, 5),
                "order": _enum(a, "order", ["top", "bottom"], "top"),
                "limit": _int(a, "limit", 1, MAX_LIMIT, MAX_LIMIT)}
    if name == "signal_performance":
        by = _enum(a, "by", ["stage", "tier"])
        allowed = (["AAHAT", "SANKET", "SAGA", "SHARK", "RAAKH"] if by == "stage"
                   else ["LOADING", "READY", "SUPER"])
        v = str(a.get("value") or "").strip()
        return {"by": by, "value": _enum({"v": v}, "v", allowed) if v else None,
                "sessions": _int(a, "sessions", 5, 120, 30),
                "side": _enum(a, "side", ["LONG", "SHORT", "BOTH"], "BOTH")}
    if name == "live_top":
        m = _enum(a, "metric", list(LIVE_METRICS))
        cap = LIVE_METRICS[m]
        w = min(_window(a), cap["max_w"]) if cap["window"] else 0
        if cap.get("needs"):
            w = w or 15
        return {"metric": m, "window_min": w, "order": _enum(a, "order", ["top", "bottom"], "top"),
                "limit": lim(), "sector": _sector(ctx, a), "universe": _universe(a),
                "show": _with_price(_cols(a, LIVE_COLS, set() if w else {m}))}
    if name == "live_level_breaks":
        return {"kind": _enum(a, "kind", ["prev_high", "prev_low"]),
                "order": _enum(a, "order", ["top", "bottom"], "top"), "limit": lim(),
                "sector": _sector(ctx, a), "show": _cols(a, LIVE_COLS, {"price", "pct"})}
    if name == "live_vwap":
        return {"side": _enum(a, "side", ["above", "below"]),
                "order": _enum(a, "order", ["top", "bottom"], "top"), "limit": lim(),
                "sector": _sector(ctx, a), "show": _cols(a, LIVE_COLS, {"vwap"})}
    if name == "live_symbol":
        return {"symbol": _symbol(ctx, a)}
    if name == "current_candidates":
        return {"side": _enum(a, "side", ["LONG", "SHORT"], "LONG"),
                "order": _enum(a, "order", ["top", "bottom"], "top"), "limit": lim(),
                "sector": _sector(ctx, a), "show": _cols(a, LIVE_COLS, {"state", "rvol", "pct"})}
    if name == "watchlist_status":
        return {"sort_by": _enum(a, "sort_by", ["score", "prox", "oi", "pct", "rvol"], "score"),
                "side": _enum(a, "side", ["LONG", "SHORT", "BOTH"], "BOTH"),
                "only_flagged": _bool(a, "only_flagged"),
                "order": _enum(a, "order", ["top", "bottom"], "top"),
                "limit": _int(a, "limit", 1, MAX_LIMIT, MAX_LIMIT), "sector": _sector(ctx, a),
                "show": _cols(a, LIVE_COLS, {"state", "score", "prox", "pct"})}
    if name == "recent_alerts":
        return {"symbol": _symbol(ctx, a, required=False), "limit": lim()}
    if name == "live_sector_rank":
        m = _enum(a, "metric", ["price", "open", "breadth", "oi"], "price")
        return {"metric": m, "window_min": 0 if m == "open" else _window(a),
                "order": _enum(a, "order", ["top", "bottom"], "top"),
                "limit": _int(a, "limit", 1, MAX_LIMIT, MAX_LIMIT)}
    if name == "live_value":
        raw = a.get("symbols") or []
        if isinstance(raw, str):
            raw = re.split(r",| and | & ", raw)
        out, seen = [], set()
        for x in list(raw)[:5]:
            x = str(x).strip()
            if not x:
                continue
            key = x.upper().replace("-", " ")
            if key in INDEX_ALIASES or key in MARKET_INDICES or key.startswith("NIFTY") or "VIX" in key:
                name_ = "IDX:" + resolve_index(ctx, x)
            else:
                name_ = _symbol(ctx, {"symbol": x})
            if name_ not in seen:
                seen.add(name_)
                out.append(name_)
        if not out:
            raise ToolError("Which stock? e.g. 'what is low of NYKAA'.")
        fl = a.get("fields") or []
        if isinstance(fl, str):
            fl = re.split(r",| and ", fl)
        fields = []
        for f_ in fl:
            f_ = {"close": "price", "ltp": "price", "prev_close": "pdc", "yesterday_close": "pdc",
                  "day_high": "high", "day_low": "low"}.get(str(f_).strip().lower(), str(f_).strip().lower())
            if f_ not in FIELDS:
                raise ToolError("I don't have '%s'. I can give: %s" % (f_, ", ".join(FIELDS)))
            if f_ not in fields:
                fields.append(f_)
        return {"symbols": out, "fields": fields or ["price", "high", "low"]}
    if name == "live_screen":
        return {"conditions": clean_conditions(a.get("conditions")), "universe": _universe(a),
                "order_by": _enum(a, "order_by", list(FIELDS), "pct"),
                "order": _enum(a, "order", ["top", "bottom"], "top"), "limit": lim(),
                "sector": _sector(ctx, a), "show": _cols(a, LIVE_COLS, {"price", "pct"})}
    if name == "compare_symbols":
        raw = a.get("symbols") or []
        if isinstance(raw, str):
            raw = raw.replace(" vs ", ",").replace(" and ", ",").split(",")
        syms = []
        for x in raw:
            sym = _symbol(ctx, {"symbol": x})
            if sym not in syms:
                syms.append(sym)
        if len(syms) < 2:
            raise ToolError("Send 2 or 3 stocks to compare, e.g. 'compare TCS and INFY'.")
        return {"symbols": syms[:3]}
    if name == "market_mood":
        return {}
    if name == "breadth_trend":
        return {"window_min": _int(a, "window_min", 10, 120, 30)}
    if name == "cannot_answer":
        return {"reason": str(a.get("reason") or "")[:200]}
    raise ToolError("Unknown tool '%s'" % name)


# ═════════════════════════════════════════════════════════════════════════════
# Shared live readers (each copies under ONE lock, then releases)
# ═════════════════════════════════════════════════════════════════════════════
def _live_results(ctx, side="LONG", required=True):
    st = ctx.state
    if side == "SHORT":
        res = _snap(st.SESSION_RESULTS_SHORT_LOCK, st.SESSION_RESULTS_SHORT)
    else:
        res = _snap(st.SESSION_RESULTS_LOCK, st.SESSION_RESULTS)
    if not res and required:
        now = ctx.now()
        if now.weekday() >= 5:
            raise ToolError("Market is closed today (weekend) — ask a history question, e.g. 'top gainers last 5 days'.")
        if now.hour * 60 + now.minute < 555:
            raise ToolError("Market opens at 09:15 — the first scan lands a minute or two after.")
        raise ToolError("No live scan results yet (market closed or app just started).")
    return res


def _as_of(ctx):
    try:
        with ctx.state.SCAN_LOCK:
            return str(ctx.state.SCAN_STATE.get("finished") or "")
    except Exception:
        return ""


def _live_pace(ctx):
    st = ctx.state
    hilo = _snap(st.SESSION_HILO_LOCK, st.SESSION_HILO)
    bud = _snap(st.IMPLIED_BUDGET_LOCK, st.IMPLIED_BUDGET)
    out = {}
    for s, d in hilo.items():
        b = _f(bud.get(s))
        hi, lo = _f((d or {}).get("hi")), _f((d or {}).get("lo"))
        if b and b > 0 and hi is not None and lo is not None:
            out[s] = (hi - lo) / b
    return out


MAX_BAR_MIN = 5            # yfinance warm-up bars are 5-min; Kite poller bars are 1-min
GAP_TOL_MIN = 3            # base bar may end this much before the target time, no more
OI_GAP_S = 6 * 60          # OI readings ~every 3 min; a base older than this = restart gap
OPEN_STAMP = (9, 16)       # a first bar stamped by 09:16 is the session open


def _bars_today(ctx):
    """{sym: {"c": closes, "v": volumes, "t": bar start times, "end": bar close times}}
    for TODAY's bars, fresh symbols only. Returns (bars, stale_count).

    The cache can mix 5-min yfinance bars (startup warm-up) with 1-min Kite bars,
    and has holes when the app was stopped. So every window below is measured by
    TIME, never by counting bars. A bar's close time = its start + its own length;
    the length is the gap to the next bar if that is <= 5 min, else 1 min (a gap
    after it means the app was down, not that the bar was long).
    Copies the dict of DataFrame references under the lock; reads outside it."""
    memo = getattr(ctx._tl, "bars", None)
    if memo is not None:
        return memo
    st = ctx.state
    cache = _snap(st._INTRADAY_CACHE_LOCK, st._INTRADAY_CACHE)
    now = ctx.now()
    out, stale = {}, 0
    for s, df in cache.items():
        try:
            keep = [str(t)[:10] == ctx.today for t in df.index]
            d = df[keep]
            if not len(d):
                continue
            ts = [(x.tz_localize(None) if getattr(x, "tzinfo", None) else x).to_pydatetime() for x in d.index]
            if (now - ts[-1]).total_seconds() > BAR_STALE_S:
                stale += 1
                continue
            ends = []
            for i, t in enumerate(ts):
                gap = (ts[i + 1] - t) if i + 1 < len(ts) else timedelta(minutes=1)
                ends.append(t + (gap if gap <= timedelta(minutes=MAX_BAR_MIN) else timedelta(minutes=1)))
            out[s] = {"c": [float(x) for x in d["Close"]], "v": [float(x) for x in d["Volume"]],
                      "t": ts, "end": ends}
        except Exception:
            continue
    if hasattr(ctx._tl, "bars"):
        ctx._tl.bars = (out, stale)
    return out, stale


def _live_volume(ctx):
    bars, _ = _bars_today(ctx)
    return {s: sum(b["v"]) for s, b in bars.items() if sum(b["v"]) > 0}


def _window_moves(ctx, minutes):
    """% change over the last `minutes` of CLOCK time, per symbol.

    base = the latest bar that closed at or before (last close - minutes).
    * base ends more than GAP_TOL_MIN before that mark -> data gap (restart): SKIP.
    * no bar that old and the first bar is the session open -> measure from the open
      (early morning) and report the minutes actually covered.
    Returns (pct {sym: v}, info {sym: (now_px, base_px, base_end_time, volume_since_base)},
             min minutes covered, stale_count, gap_count)."""
    bars, stale = _bars_today(ctx)
    out, info, used, gaps = {}, {}, [], 0
    for s, b in bars.items():
        c, ends, ts = b["c"], b["end"], b["t"]
        if len(c) < 2:
            continue
        mark = ends[-1] - timedelta(minutes=minutes)
        idx = [i for i, e in enumerate(ends[:-1]) if e <= mark]
        if idx:
            i = idx[-1]
            if (mark - ends[i]).total_seconds() > GAP_TOL_MIN * 60:
                gaps += 1
                continue
        elif (ts[0].hour, ts[0].minute) <= OPEN_STAMP:
            i = 0                                  # early morning: from the first bar of the day
        else:
            gaps += 1                              # app started late: nothing that old today
            continue
        if c[i] <= 0:
            continue
        # a hole INSIDE the window also makes the move unreliable
        if any((ts[j + 1] - ts[j]).total_seconds() > (MAX_BAR_MIN + GAP_TOL_MIN) * 60
               for j in range(i, len(ts) - 1)):
            gaps += 1
            continue
        out[s] = (c[-1] / c[i] - 1) * 100.0
        info[s] = (c[-1], c[i], ends[i], sum(b["v"][i + 1:]))
        used.append(int((ends[-1] - ends[i]).total_seconds() // 60))
    return out, info, (min(used) if used else 0), stale, gaps


def _window_volume(ctx, minutes):
    """Volume per minute in the last N clock minutes, as a multiple of the stock's own
    volume per minute earlier today ('spike' x > 1 = more active than its day so far).
    Also returns, per stock, the raw volume of this window and of the N minutes before
    it, and price now / price at the window start -- the before-and-after the trader
    needs without asking. Skips a stock if the window has a data gap or less than N
    minutes came before it.
    Returns (spike {sym: x}, info {sym: (vol_now, vol_prev, px_now, px_then, then_end)},
             stale_count, gap_count)."""
    bars, stale = _bars_today(ctx)
    out, info, gaps = {}, {}, 0
    for s, b in bars.items():
        v, ts, ends, c = b["v"], b["t"], b["end"], b["c"]
        mark = ends[-1] - timedelta(minutes=minutes)
        recent = [j for j in range(len(v)) if ends[j] > mark]
        before = [j for j in range(len(v)) if ends[j] <= mark]
        if not recent or not before:
            continue
        if (mark - ends[before[-1]]).total_seconds() > GAP_TOL_MIN * 60 or any(
                (ts[j + 1] - ts[j]).total_seconds() > (MAX_BAR_MIN + GAP_TOL_MIN) * 60
                for j in range(before[-1], len(ts) - 1)):
            gaps += 1
            continue
        mins_before = sum((ends[j] - ts[j]).total_seconds() / 60 for j in before)
        if mins_before < minutes:
            continue
        rate_before = sum(v[j] for j in before) / mins_before
        vol_now = sum(v[j] for j in recent)
        prev_mark = mark - timedelta(minutes=minutes)
        prev = [j for j in before if ends[j] > prev_mark]
        if rate_before > 0:
            out[s] = (vol_now / minutes) / rate_before
            info[s] = (vol_now, sum(v[j] for j in prev) if prev else None, c[-1], c[before[-1]],
                       ends[before[-1]])
    return out, info, stale, gaps


# Live samples: every 60 s the bot copies each stock's price, VWAP and PACE (the SAME
# numbers the dashboard shows: futures day high/low ÷ budget). The app keeps no history
# of them, and rebuilding from bars broke on restarts (Yahoo 5-min warm-up + Kite 1-min).
# RAM only, ~2 h kept; after a restart the window is honestly "not ready yet".
SAMPLE_KEEP = 130                    # samples per stock ≈ minutes
SAMPLE_GAP_S = 150                   # a hole longer than this = missed samples (restart / stall)
_SAMPLES = {}                        # sym -> deque[(epoch, price, vwap, pace)]
_SAMPLE_META = {"since": None, "last": None, "n": 0}
_SAMPLE_LOCK = threading.Lock()


# PACE changes: every time the dashboard PACE (2 decimals, as shown) moves, one event is kept for
# the WHOLE day and appended to data/ai_pace_YYYY-MM-DD.jsonl, so a restart keeps the history.
# Minutes when the app was off are recorded as gaps (a change then cannot be seen).
_PACE = {"day": None, "last": {}, "events": {}, "runs": [], "loaded": False, "hb": 0, "hl": {}}
PACE_HB_S = 300


def _fut_hilo(ctx):
    st = ctx.state
    out = {}
    for s, d in _snap(st.SESSION_HILO_LOCK, st.SESSION_HILO).items():
        hi, lo = _f((d or {}).get("hi")), _f((d or {}).get("lo"))
        if hi is not None and lo is not None:
            out[s] = (hi, lo)
    return out


def _pace_file(ctx):
    st = ctx.state
    base = getattr(st, "DATA_DIR", None) or (os.path.dirname(getattr(st, "DAILY_OHLC_CSV", "") or "") or None)
    return os.path.join(base, "ai_pace_%s.jsonl" % ctx.today) if base else None


def _pace_write(ctx, rec):
    path = _pace_file(ctx)
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass                                   # history is a convenience: never stop sampling for it


def _pace_load(ctx):
    """Today's events from the file (once per process / day). Called under _SAMPLE_LOCK."""
    P = _PACE
    if P["day"] != ctx.today:
        P.update(day=ctx.today, last={}, events={}, runs=[], loaded=False, hb=0, hl={})
    if P["loaded"]:
        return
    P["loaded"] = True
    path = _pace_file(ctx)
    if not path or not os.path.exists(path):
        return
    try:
        for line in open(path, encoding="utf-8"):
            x = json.loads(line)
            if "run" in x:
                P["runs"].append([x["run"], x["run"]])
            elif "hb" in x and P["runs"]:
                P["runs"][-1][1] = max(P["runs"][-1][1], x["hb"])
            elif "s" in x:
                P["events"].setdefault(x["s"], []).append((x["t"], x["f"], x["v"], x.get("side"), x.get("lvl")))
                P["last"][x["s"]] = x["v"]
                if x.get("hl"):
                    P["hl"][x["s"]] = tuple(x["hl"])
                if P["runs"]:
                    P["runs"][-1][1] = max(P["runs"][-1][1], x["t"])
            elif "base" in x:
                P["last"].update(x["base"])
                P["hl"].update({k: tuple(v) for k, v in (x.get("hl") or {}).items()})
    except Exception:
        pass


def _pace_track(ctx, pace, now, new_run, hilo=None):
    """Called by sample_once under _SAMPLE_LOCK with {sym: live PACE} and the futures day
    {sym: (high, low)} it came from. A change is tagged LOW (the low extended), HIGH, or BOTH."""
    hilo = hilo or {}
    _pace_load(ctx)
    P = _PACE
    if new_run or not P["runs"]:
        P["runs"].append([now, now])
        _pace_write(ctx, {"run": now})
    P["runs"][-1][1] = now
    first = {}
    for sym, v in pace.items():
        if v is None:
            continue
        r = round(float(v), 2)
        last = P["last"].get(sym)
        hl = hilo.get(sym)
        if last is None:
            P["last"][sym] = first[sym] = r
            if hl:
                P["hl"][sym] = hl
            continue
        if r != last:
            side, lvl = None, None
            old = P["hl"].get(sym)
            if hl and old:
                lo_brk, hi_brk = hl[1] < old[1], hl[0] > old[0]
                side = "BOTH" if lo_brk and hi_brk else "LOW" if lo_brk else "HIGH" if hi_brk else None
                lvl = hl[1] if side == "LOW" else hl[0] if side == "HIGH" else None
            P["events"].setdefault(sym, []).append((now, last, r, side, lvl))
            P["last"][sym] = r
            if hl:
                P["hl"][sym] = hl
            _pace_write(ctx, {"s": sym, "t": now, "f": last, "v": r, "side": side, "lvl": lvl,
                              "hl": list(hl) if hl else None})
    if first:
        _pace_write(ctx, {"base": first, "hl": {k: list(P["hl"][k]) for k in first if k in P["hl"]}})
    if now - P["hb"] >= PACE_HB_S:
        P["hb"] = now
        _pace_write(ctx, {"hb": now})


def pace_history(ctx, sym):
    """{"n", "events": [(HH:MM, from, to)], "last", "flat_min" (before the last change),
    "since" (HH:MM tracking began), "gaps": [(HH:MM, HH:MM)]} or None."""
    with _SAMPLE_LOCK:
        if _PACE["day"] != ctx.today:
            return None
        ev = list(_PACE["events"].get(sym) or ())
        runs = [list(x) for x in _PACE["runs"]]
        cur = _PACE["last"].get(sym)
    if not runs:
        return None
    hm = lambda t: datetime.fromtimestamp(t).strftime("%H:%M")      # noqa: E731
    gaps = [(hm(a[1]), hm(b[0])) for a, b in zip(runs, runs[1:]) if b[0] - a[1] > SAMPLE_GAP_S]
    flat, seen = None, None
    if ev:
        flat, seen = _flat_before(ev, len(ev) - 1, runs)
    return {"n": len(ev), "events": [(hm(e[0]), e[1], e[2]) for e in ev], "last": hm(ev[-1][0]) if ev else None,
            "last_epoch": ev[-1][0] if ev else None, "now_epoch": runs[-1][1],
            "flat_min": flat, "seen_since": seen, "since": hm(runs[0][0]), "gaps": gaps, "now": cur,
            "sides": [e[3] if len(e) > 3 else None for e in ev]}


OPEN_HM = (9, 16)


def _flat_before(ev, i, runs):
    """(minutes PACE was flat before change i, None) when the bot watched the whole stretch
    without a gap; else (None, 'HH:MM') = the bot only watched since then (restart / late start).
    A watch that began by 09:16 covers the whole day, so 'flat since open' is real."""
    t = ev[i][0]
    run = next((r for r in runs if r[0] <= t <= r[1] + SAMPLE_GAP_S), None)
    start = run[0] if run else t
    if i > 0 and ev[i - 1][0] >= start:
        return int((t - ev[i - 1][0]) // 60), None
    first_run = runs[0][0] if runs else start
    opened = datetime.fromtimestamp(first_run)
    if start == first_run and (opened.hour, opened.minute) <= OPEN_HM:
        return int((t - first_run) // 60), None
    return None, datetime.fromtimestamp(start).strftime("%H:%M")


def _pace_meta(ctx):
    with _SAMPLE_LOCK:
        runs = [list(x) for x in _PACE["runs"]] if _PACE["day"] == ctx.today else []
    hm = lambda t: datetime.fromtimestamp(t).strftime("%H:%M")      # noqa: E731
    return {"gaps": [(hm(a[1]), hm(b[0])) for a, b in zip(runs, runs[1:]) if b[0] - a[1] > SAMPLE_GAP_S]}


def _hm_dur(m):
    return ("%dh %02dm" % divmod(m, 60)) if m >= 60 else "%d min" % m


def pace_line(ctx, sym):
    """One line for PACE replies: 'PACE changed 6× · last 14:37 (flat 4h 25m before)'."""
    h = pace_history(ctx, sym)
    if not h:
        return ""
    if not h["n"]:
        return "PACE unchanged since %s" % h["since"]
    out = "PACE changed %d× · last %s" % (h["n"], h["last"])
    if h["flat_min"] is not None:
        out += " (flat %s before)" % _hm_dur(h["flat_min"])
    elif h.get("seen_since"):
        out += " (no change seen since %s, bot started then)" % h["seen_since"]
    return out


def _pace_changes(ctx, minutes, side=None):
    """{sym: epoch of the latest change in the window}, info {sym: (from, to, HH:MM, flat_min or None,
    n today, seen_since or None, side, level)}. side 'LOW' / 'HIGH' keeps only changes made by a
    new day low / high (BOTH counts for either)."""
    now = _epoch(ctx)
    mark = now - minutes * 60
    with _SAMPLE_LOCK:
        last = _SAMPLE_META["last"]
        runs = [list(x) for x in _PACE["runs"]] if _PACE["day"] == ctx.today else []
        evs = {k: list(v) for k, v in _PACE["events"].items()} if _PACE["day"] == ctx.today else {}
    if not runs or last is None or now - last > SAMPLE_GAP_S:
        raise ToolError("No live samples yet — the bot samples PACE every minute while the market is open.")
    if runs[-1][0] > mark + SAMPLE_GAP_S:
        ready = datetime.fromtimestamp(runs[-1][0] + minutes * 60 + 60).strftime("%H:%M")
        raise ToolError("Collecting PACE since %s (app start) — a %d-min window is ready at %s."
                        % (datetime.fromtimestamp(runs[-1][0]).strftime("%H:%M"), minutes, ready))
    vals, info = {}, {}
    for sym, ev in evs.items():
        idx = [i for i, e in enumerate(ev) if e[0] > mark and
               (side is None or (len(e) > 3 and e[3] in (side, "BOTH")))]
        if not idx:
            continue
        i = idx[-1]
        e = ev[i]
        first_in = ev[idx[0]]
        flat, seen = _flat_before(ev, idx[0], runs)
        vals[sym] = e[0]
        info[sym] = (first_in[1], e[2], datetime.fromtimestamp(e[0]).strftime("%H:%M"), flat, len(ev), seen,
                     e[3] if len(e) > 3 else None, e[4] if len(e) > 4 else None)
    return vals, info


def sample_once(ctx, now_epoch=None):
    """One sample of every stock. Called by ai_chat's sampler thread. Never raises."""
    try:
        res = _live_results(ctx, required=False)
        if not res:
            return 0
        pace = _live_pace(ctx)
        now = now_epoch if now_epoch is not None else _epoch(ctx)
        rows = {}
        for sym, r in res.items():
            px = _f(r.get("price"))
            if px:
                rows[sym] = (now, px, _f(r.get("vwap")), pace.get(sym))
        with _SAMPLE_LOCK:
            last = _SAMPLE_META["last"]
            new_run = last is None or now - last > SAMPLE_GAP_S
            if new_run:
                _SAMPLE_META["since"] = now          # first sample, or after a stall: a new run
            _pace_track(ctx, pace, now, new_run, _fut_hilo(ctx))
            for sym, x in rows.items():
                dq = _SAMPLES.get(sym)
                if dq is None:
                    dq = _SAMPLES[sym] = deque(maxlen=SAMPLE_KEEP)
                dq.append(x)
            _SAMPLE_META["last"] = now
            _SAMPLE_META["n"] += 1
        _sample_indices(ctx, now)
        return len(rows)
    except Exception as e:                   # never kill the sampler thread, but never hide it either
        with _SAMPLE_LOCK:
            _SAMPLE_META["error"] = "%s: %s" % (type(e).__name__, e)
        return 0


_ISAMPLES = {}                       # index name -> deque[(epoch, price, day_high, day_low, vwap, pct)]
INDEX_ALIASES = {"VIX": "INDIA VIX", "INDIAVIX": "INDIA VIX", "NIFTY": "NIFTY 50", "NIFTY50": "NIFTY 50",
                 "BANKNIFTY": "NIFTY BANK", "BANK NIFTY": "NIFTY BANK"}
MARKET_INDICES = {"NIFTY 50", "INDIA VIX"}           # not sectors: 'any sector' skips them


def _sample_indices(ctx, now):
    """Indices + VIX from the scanner's SECTOR_INTEL (price, day high/low, VWAP, day %)."""
    st = ctx.state
    lock = getattr(st, "SECTOR_INTEL_LOCK", None)
    if lock is None:
        return 0
    with lock:
        data = {k: dict(v) for k, v in ((st.SECTOR_INTEL or {}).get("data") or {}).items() if isinstance(v, dict)}
    n = 0
    with _SAMPLE_LOCK:
        for name, v in data.items():
            px = _f(v.get("price"))
            if not px:
                continue
            dq = _ISAMPLES.get(name)
            if dq is None:
                dq = _ISAMPLES[name] = deque(maxlen=SAMPLE_KEEP)
            if dq and dq[-1][1:] == (px, _f(v.get("day_high")), _f(v.get("day_low")), _f(v.get("vwap_v")),
                                     _f(v.get("pct"))):
                dq.append((now,) + dq[-1][1:])          # same reading: SECTOR_INTEL not refreshed yet
            else:
                dq.append((now, px, _f(v.get("day_high")), _f(v.get("day_low")), _f(v.get("vwap_v")),
                           _f(v.get("pct"))))
            n += 1
    return n


def index_refresh_s():
    """Median seconds between CHANGED NIFTY 50 readings (how often SECTOR_INTEL refreshes), or None."""
    with _SAMPLE_LOCK:
        xs = list(_ISAMPLES.get("NIFTY 50") or ())
    ch = [xs[i][0] for i in range(1, len(xs)) if xs[i][1:] != xs[i - 1][1:]]
    gaps = sorted(b - a for a, b in zip(ch, ch[1:]))
    return gaps[len(gaps) // 2] if gaps else None


def index_names(ctx):
    with _SAMPLE_LOCK:
        have = sorted(k for k in _ISAMPLES if not k.startswith("_"))
    if have:
        return have
    st = ctx.state
    try:
        with st.SECTOR_INTEL_LOCK:
            return sorted(((st.SECTOR_INTEL or {}).get("data") or {}).keys())
    except Exception:
        return []


def resolve_index(ctx, name):
    """'vix' -> 'INDIA VIX', 'it' -> 'NIFTY IT', 'any' -> 'ANY' (every sector). Raises ToolError."""
    s = str(name or "").strip().upper()
    if s in ("ANY", "ANY SECTOR", "ALL", "SECTOR", "SECTORS", ""):
        return "ANY"
    s = INDEX_ALIASES.get(s.replace("-", " "), s)
    names = index_names(ctx)
    if not names or s in names:
        return s
    if "NIFTY " + s in names:
        return "NIFTY " + s
    import difflib
    close = difflib.get_close_matches(s, names, n=1, cutoff=0.75)
    if close:
        return close[0]
    raise ToolError("Unknown index %s. Known: %s" % (s, ", ".join(names)))


INDEX_METRICS = {
    "price":        {"window": False, "what": "index level"},
    "pct":          {"window": False, "what": "day % change"},
    "move":         {"window": True, "needs": True, "what": "% move in the last N min"},
    "day_high_up":  {"window": True, "needs": True, "event": True, "what": "made a new DAY HIGH in N min"},
    "day_low_down": {"window": True, "needs": True, "event": True, "what": "made a new DAY LOW in N min"},
    "vwap_up":      {"window": True, "needs": True, "event": True, "what": "crossed ABOVE VWAP in N min"},
    "vwap_down":    {"window": True, "needs": True, "event": True, "what": "crossed BELOW VWAP in N min"},
    "near_high":    {"window": False, "near": True, "what": "% below today's high"},
    "near_low":     {"window": False, "near": True, "what": "% above today's low"},
}


def index_values(ctx, metric, window_min, names):
    """{index: (value, price now, day %)} for indices in `names` ('ANY' = every sector index).
    Event metrics: only indices where the event happened (value = price now)."""
    with _SAMPLE_LOCK:
        snap = {k: list(v) for k, v in _ISAMPLES.items() if not k.startswith("_")}
    want = [k for k in snap if k not in MARKET_INDICES] if "ANY" in names else [n for n in names if n in snap]
    if not snap:
        raise ToolError("No index samples yet — they are taken every minute in market hours.")
    now = _epoch(ctx)
    out = {}
    for name in want:
        xs = snap[name]
        if not xs or now - xs[-1][0] > SAMPLE_GAP_S:
            continue
        last = xs[-1]
        px, hi, lo, vw, pct = last[1:]
        if not INDEX_METRICS[metric]["window"]:
            if metric == "pct":
                out[name] = (pct, px, pct)
            elif metric == "price":
                out[name] = (px, px, pct)
            elif metric == "near_high" and hi:
                out[name] = (max(0.0, (hi - px) / hi * 100.0), px, pct)
            elif metric == "near_low" and lo:
                out[name] = (max(0.0, (px - lo) / lo * 100.0), px, pct)
            continue
        mark = now - window_min * 60
        base_i = max((i for i, x in enumerate(xs) if x[0] <= mark), default=None)
        if base_i is None or mark - xs[base_i][0] > SAMPLE_GAP_S:
            continue
        b = xs[base_i]
        if metric == "move":
            out[name] = ((px / b[1] - 1) * 100.0, px, pct)
        elif metric == "day_high_up" and hi and b[2] and hi > b[2]:
            out[name] = (px, px, pct)
        elif metric == "day_low_down" and lo and b[3] and lo < b[3]:
            out[name] = (px, px, pct)
        elif metric == "vwap_up" and vw and b[4] and b[1] <= b[4] and px > vw:
            out[name] = (px, px, pct)
        elif metric == "vwap_down" and vw and b[4] and b[1] >= b[4] and px < vw:
            out[name] = (px, px, pct)
    return out


def _sample_windows(ctx, minutes):
    """{sym: (base, [samples after base], last)} for stocks with an unbroken run of samples
    covering the window. Raises ToolError (with the time it will be ready) when none do."""
    now = _epoch(ctx)
    mark = now - minutes * 60
    with _SAMPLE_LOCK:
        snap = {s: list(dq) for s, dq in _SAMPLES.items()}
        since, last = _SAMPLE_META["since"], _SAMPLE_META["last"]
    out = {}
    for sym, xs in snap.items():
        if not xs or now - xs[-1][0] > SAMPLE_GAP_S:
            continue
        base_i = max((i for i, x in enumerate(xs) if x[0] <= mark), default=None)
        if base_i is None or mark - xs[base_i][0] > SAMPLE_GAP_S:
            continue
        run = xs[base_i:]
        if any(b[0] - a[0] > SAMPLE_GAP_S for a, b in zip(run, run[1:])):
            continue
        out[sym] = (run[0], run[1:], run[-1])
    if not out:
        if since is None or last is None or now - last > SAMPLE_GAP_S:
            raise ToolError("No live samples yet — the bot samples price / VWAP / PACE every minute "
                            "while the market is open.")
        ready = datetime.fromtimestamp(since + minutes * 60 + 60).strftime("%H:%M")
        raise ToolError("Collecting live samples since %s (app start) — a %d-min window is ready at %s."
                        % (datetime.fromtimestamp(since).strftime("%H:%M"), minutes, ready))
    return out


def _px_then(ctx, syms, minutes):
    """{sym: price at the window start} from the samples (empty if none)."""
    if not minutes:
        return {}
    try:
        w = _sample_windows(ctx, minutes)
    except ToolError:
        return {}
    return {s: w[s][0][1] for s in syms if s in w}


def day_hilo_times(ctx, sym):
    """(high, 'HH:MM', low, 'HH:MM') from today's 1-min bars + live price + open, or None."""
    st = ctx.state
    df = _snap(st._INTRADAY_CACHE_LOCK, st._INTRADAY_CACHE).get(sym)
    if df is None:
        return None
    try:
        d = df[[str(t)[:10] == ctx.today for t in df.index]]
        if not len(d):
            return None
        hs = d["High"] if "High" in d else d["Close"]
        ls = d["Low"] if "Low" in d else d["Close"]
        hi, lo = float(hs.max()), float(ls.min())
        ht, lt = str(hs.idxmax())[11:16], str(ls.idxmin())[11:16]
    except Exception:
        return None
    r = _live_results(ctx, required=False).get(sym) or {}
    px, op = _f(r.get("price")), _f(r.get("day_open"))
    if px is not None and px > hi:
        hi, ht = px, "now"
    if px is not None and px < lo:
        lo, lt = px, "now"
    if op is not None and op > hi:                     # opened at the high, bars began a minute later
        hi, ht = op, "open"
    if op is not None and op < lo:
        lo, lt = op, "open"
    return hi, ht, lo, lt


def index_fields(ctx, name):
    """{field: value} for an index from the scanner's SECTOR_INTEL (price, high, low, vwap, pct)."""
    with _SAMPLE_LOCK:
        xs = list(_ISAMPLES.get(name) or ())
    if xs:
        _t, px, hi, lo, vw, pct = xs[-1]
    else:
        try:
            with ctx.state.SECTOR_INTEL_LOCK:
                v = dict(((ctx.state.SECTOR_INTEL or {}).get("data") or {}).get(name) or {})
        except Exception:
            v = {}
        px, hi, lo, vw, pct = (_f(v.get(k)) for k in ("price", "day_high", "day_low", "vwap_v", "pct"))
    if px is None:
        return None
    return {"price": px, "high": hi, "low": lo, "vwap": vw, "pct": pct,
            "range_pct": (hi / lo - 1) * 100.0 if hi and lo else None}


def _day_hilo(ctx):
    """{sym: (today's high, today's low)} from today's 1-min bars (High/Low when present)."""
    st = ctx.state
    cache = _snap(st._INTRADAY_CACHE_LOCK, st._INTRADAY_CACHE)
    out = {}
    for s, df in cache.items():
        try:
            d = df[[str(t)[:10] == ctx.today for t in df.index]]
            if not len(d):
                continue
            hi = float((d["High"] if "High" in d else d["Close"]).max())
            lo = float((d["Low"] if "Low" in d else d["Close"]).min())
            out[s] = (hi, lo)
        except Exception:
            continue
    return out


def _near(ctx, metric):
    """{sym: distance %}, info {sym: (level, price)}. Price beyond the level counts as 0
    for today's high/low (that is a break, the strongest 'near')."""
    res = _live_results(ctx)
    vals, info = {}, {}
    if metric in ("near_low", "near_high"):
        hl = _day_hilo(ctx)
        for s, r in res.items():
            px = _f(r.get("price"))
            if not px or s not in hl:
                continue
            hi, lo = max(hl[s][0], px), min(hl[s][1], px)        # live price may be past the last bar
            lv = lo if metric == "near_low" else hi
            vals[s] = abs(px / lv - 1) * 100.0
            info[s] = (lv, px)
        return vals, info
    if metric == "near_vwap":
        lv_of = {s: _f(r.get("vwap")) for s, r in res.items()}
    else:
        lv_of = (_pdh if metric == "near_pdh" else _pdl)(ctx)
    for s, r in res.items():
        px, lv = _f(r.get("price")), lv_of.get(s)
        if px and lv:
            vals[s] = abs(px / lv - 1) * 100.0
            info[s] = (lv, px)
    return vals, info


def _crosses(ctx, minutes, level_of, up):
    """Stocks whose price crossed a level inside the window and is still beyond it now.
    level_of(sym, sample) -> level at that sample (VWAP moves; yesterday high / open do not).
    Returns (vals {sym: cross epoch}, info {sym: (beyond %, level now, price then, price now, HH:MM)})."""
    vals, info = {}, {}
    for sym, (base, inside, last) in _sample_windows(ctx, minutes).items():
        lb, ln = level_of(sym, base), level_of(sym, last)
        if not lb or not ln:
            continue
        was = base[1] <= lb if up else base[1] >= lb
        now_ok = last[1] > ln if up else last[1] < ln
        if not (was and now_ok):
            continue
        t = next((x for x in inside if level_of(sym, x) and
                  (x[1] > level_of(sym, x) if up else x[1] < level_of(sym, x))), last)
        vals[sym] = t[0]
        info[sym] = ((last[1] / ln - 1) * 100.0, ln, base[1], last[1],
                     datetime.fromtimestamp(t[0]).strftime("%H:%M"))
    return vals, info


def _epoch(ctx):
    return time.mktime(ctx.now().timetuple())


FUT_TICK_STALE_S = 5 * 60


def _fut_volume(ctx, minutes):
    """FUTURES volume from live KiteTicker ticks (state.LIVE_VOL_HISTORY: (epoch, day
    cumulative volume), pruned to 90 min). No window -> today's futures volume so far.
    Window -> volume of the last N min ÷ volume of the N min before (x > 1 = rising).
    Locks one symbol at a time: the tick thread writes these deques many times a second.
    Returns (vals, info {sym: (vol_now_window, vol_prev_window)}, skipped)."""
    st = ctx.state
    hist = getattr(st, "LIVE_VOL_HISTORY", None)
    lock = getattr(st, "LIVE_TICK_LOCK", None)
    if hist is None or lock is None:
        return {}, {}, 0
    with lock:
        syms = list(hist.keys())
    now = _epoch(ctx)
    vals, info, skipped = {}, {}, 0
    for sym in syms:
        with lock:
            ticks = list(hist.get(sym) or ())
        ticks = [(float(t), float(v)) for t, v in ticks if v]
        if not ticks or now - ticks[-1][0] > FUT_TICK_STALE_S:
            continue
        cur = ticks[-1][1]
        if not minutes:
            vals[sym] = cur
            continue

        def at(mark):
            v = None
            for t, cv in ticks:
                if t <= mark:
                    v = cv
                else:
                    break
            return v
        a, b = at(now - minutes * 60), at(now - 2 * minutes * 60)
        if a is None or b is None:                 # not enough tick history (restart / 90-min cap)
            skipped += 1
            continue
        v_now, v_prev = max(cur - a, 0.0), max(a - b, 0.0)
        if v_prev > 0:
            vals[sym] = v_now / v_prev
            info[sym] = (v_now, v_prev)
    return vals, info, skipped


def _pace_window(ctx, minutes, cross=False):
    """PACE over the last N min from the live samples (dashboard PACE: futures high/low ÷
    budget). Value = PACE gained. cross=True keeps only stocks whose band changed.
    PACE only goes UP during a day. Returns (vals, info {sym: (then, now, band_from,
    band_to, cross HH:MM, price then, price now)}, gaps)."""
    vals, info = {}, {}
    for sym, (base, inside, last) in _sample_windows(ctx, minutes).items():
        if base[3] is None or last[3] is None:
            continue
        b0, b1 = ctx.pband(base[3]), ctx.pband(last[3])
        if cross and b0 == b1:
            continue
        t = next((x for x in inside if x[3] is not None and ctx.pband(x[3]) != b0), None)
        vals[sym] = last[3] - base[3]
        info[sym] = (base[3], last[3], b0, b1, datetime.fromtimestamp(t[0]).strftime("%H:%M") if t else None,
                     base[1], last[1])
    return vals, info, 0


def _iv_values(ctx, minutes):
    """ATM IV per stock (state.IV_HISTORY: {"ts", "iv"} per OI pass, ~20 kept ≈ 55 min).
    No window -> latest IV. Window -> IV now minus IV at the window start (points).
    Returns (vals, info {sym: (iv_then, iv_now)})."""
    st = ctx.state
    lock = getattr(st, "IV_HISTORY_LOCK", None)
    if lock is None:
        return {}, {}
    with lock:
        hist = {s: list(dq) for s, dq in (st.IV_HISTORY or {}).items()}
    now = _epoch(ctx)
    vals, info = {}, {}
    for sym, xs in hist.items():
        xs = [x for x in xs if isinstance(x, dict) and _f(x.get("iv")) and _f(x.get("ts"))]
        if not xs or now - float(xs[-1]["ts"]) > OI_GAP_S:
            continue
        cur = _f(xs[-1]["iv"])
        if not minutes:
            vals[sym], info[sym] = cur, (None, cur)
            continue
        mark = now - minutes * 60
        base = [x for x in xs if float(x["ts"]) <= mark]
        if not base or mark - float(base[-1]["ts"]) > OI_GAP_S:
            continue
        b = _f(base[-1]["iv"])
        vals[sym], info[sym] = cur - b, (b, cur)
    return vals, info


def _new_extremes(ctx, minutes, high=True):
    """Stocks whose DAY high (or low) was set inside the last `minutes` of clock time.
    Value = % move over the same window (so the strongest push ranks first).
    Returns (vals, info {sym: (now, then, then_end, mins_since_extreme)}, used, stale, gaps)."""
    moves, minfo, used, stale, gaps = _window_moves(ctx, minutes)
    bars, _st = _bars_today(ctx)
    vals, info = {}, {}
    for s, mv in moves.items():
        b = bars.get(s)
        if not b:
            continue
        hs = b["c"]                                   # closes are enough for "new extreme" timing
        ends = b["end"]
        mark = ends[-1] - timedelta(minutes=minutes)
        before = [x for x, e in zip(hs, ends) if e <= mark]
        inside = [(x, e) for x, e in zip(hs, ends) if e > mark]
        if not before or not inside:
            continue
        ref = max(before) if high else min(before)
        ext = max(inside, key=lambda t: t[0]) if high else min(inside, key=lambda t: t[0])
        if (ext[0] > ref) if high else (ext[0] < ref):
            vals[s] = mv
            info[s] = (minfo[s][0], minfo[s][1], minfo[s][2], int((ends[-1] - ext[1]).total_seconds() // 60))
    return vals, info, used, stale, gaps


def _oi_pct(d):
    v = d.get("oi_from_open")              # 0.0 is a real value -- no `or`
    if v is None:
        v = d.get("oi_session")
    return _f(v)


def _oi_window(ctx, minutes):
    """% change of futures OI over the last N minutes from OI_HISTORY (~3-min readings).
    Returns (pct {sym: v}, info {sym: (oi_now, oi_then, fut_px_now, fut_px_then, then_ts)})."""
    st = ctx.state
    with st.OI_HISTORY_LOCK:
        hist = {s: list(dq) for s, dq in st.OI_HISTORY.items()}
    out, info = {}, {}
    for s, h in hist.items():
        h = [x for x in h if isinstance(x, dict) and _f(x.get("raw"))]
        if len(h) < 2:
            continue
        cut = h[-1]["ts"] - minutes * 60
        base = [x for x in h if x["ts"] <= cut]
        # OI_HISTORY survives restarts (oi_engine._load_oi_history): a reading from
        # before a stop must not be used as "15 min ago". Readings come every ~3 min.
        if base and base[-1]["raw"] > 0 and cut - base[-1]["ts"] <= OI_GAP_S:
            b0, b1 = base[-1], h[-1]
            out[s] = (b1["raw"] / b0["raw"] - 1) * 100.0
            info[s] = (b1["raw"], b0["raw"], _f(b1.get("ltp")), _f(b0.get("ltp")),
                       datetime.fromtimestamp(b0["ts"]))
    return out, info


def _score_num(r):
    """Sort key for the scanner score: pct_score, else raw points."""
    v = _f(r.get("pct_score"))
    return v if v is not None else _f(r.get("score"))


def _pct_txt(r):
    """Scanner score as the dashboard shows it (pct_score %), raw points only as a fallback."""
    ps = _f(r.get("pct_score"))
    if ps is not None:
        return "%.0f%%" % ps
    sc, mx = r.get("score"), r.get("max_score")
    return ("%s/%s" % (sc, mx)) if sc is not None and mx else None


def _live_cols(ctx, syms, cols, res=None):
    """{sym: {col: value}} for the requested extra columns."""
    if not cols or not syms:
        return {}
    res = res if res is not None else _live_results(ctx, required=False)
    oi = _snap(ctx.state.OI_CACHE_LOCK, ctx.state.OI_CACHE) if {"oi", "streak"} & set(cols) else {}
    pace = _live_pace(ctx) if "pace" in cols else {}
    vol = _live_volume(ctx) if "volume" in cols else {}
    fv5 = _fut_volume(ctx, 5)[0] if "fut_vol5" in cols else {}
    v5 = _window_volume(ctx, 5)[0] if "vol5" in cols else {}
    out = {}
    for s in syms:
        r, o = res.get(s) or {}, oi.get(s) if isinstance(oi.get(s), dict) else {}
        px, vw = _f(r.get("price")), _f(r.get("vwap"))
        row = {}
        for c in cols:
            if c == "price":
                row[c] = px
            elif c == "pct":
                row[c] = _f(r.get("intra_pct"))
            elif c == "rvol":
                row[c] = _f(r.get("rvol"))
            elif c == "volume":
                row[c] = vol.get(s)
            elif c == "fut_vol5":
                row[c] = fv5.get(s)
            elif c == "vol5":
                row[c] = v5.get(s)
            elif c == "oi":
                row[c] = _oi_pct(o)
            elif c == "vwap":
                row[c] = (px / vw - 1) * 100.0 if px and vw else None
            elif c == "pace":
                row[c] = pace.get(s)
            elif c == "state":
                row[c] = r.get("state") or ""
            elif c == "sector":
                row[c] = r.get("sector") or _secmap(ctx).get(s, "")
            elif c == "streak":
                row[c] = o.get("streak")
            elif c == "score":
                row[c] = _pct_txt(r)
            elif c == "open":
                o = _f(r.get("day_open"))
                row[c] = (px / o - 1) * 100.0 if px and o else None
            elif c == "gap":
                o, pc = _f(r.get("day_open")), _f(r.get("pdc"))
                row[c] = (o / pc - 1) * 100.0 if o and pc else None
            elif c == "prox":
                p = r.get("proximity")
                row[c] = ("%s %s" % (p, r.get("prox_name") or "")).strip() if p is not None else None
        out[s] = row
    return out


def _hist_cols(c, day, syms, cols, ctx):
    """Extra columns for a past session: price / pct / volume from daily_ohlc.csv (the raw
    daily bars), rvol / pace / oi from symbol_day (only those are reliable there)."""
    if not cols or not syms or not day:
        return {}
    days, data = _ohlc(ctx)
    bar = data.get(day, {})
    prev = _prev_of(days, day)
    pbar = data.get(prev, {}) if prev else {}
    rows = {}
    if {"rvol", "pace", "oi"} & set(cols):
        rows = {r["sym"]: r for r in c.execute("SELECT sym, rvol_max, max_pace, oi_sess FROM symbol_day "
                                               "WHERE day=?", (day,))}
    out = {}
    for s in syms:
        b, pb, r = bar.get(s), pbar.get(s), rows.get(s)
        row = {}
        for k in cols:
            if k == "sector":
                row[k] = _secmap(ctx).get(s, "")
            elif k == "price":
                row[k] = b["c"] if b else None
            elif k == "volume":
                row[k] = b["v"] if b else None
            elif k == "pct":
                row[k] = (b["c"] / pb["c"] - 1) * 100.0 if b and pb and pb["c"] else None
            else:
                row[k] = _f(r[{"rvol": "rvol_max", "pace": "max_pace", "oi": "oi_sess"}[k]]) if r else None
        out[s] = row
    return out


def _day_pair(ctx, syms):
    """{sym: (previous close, price now)} -- the 'then -> now' line for today's lists."""
    res = _live_results(ctx, required=False)
    out = {}
    for s in syms:
        r = res.get(s) or {}
        out[s] = (_f(r.get("pdc")), _f(r.get("price")))
    return out


def _day_universe(res):
    return {s: _f(r.get("intra_pct")) for s, r in res.items() if _f(r.get("intra_pct")) is not None}


# ═════════════════════════════════════════════════════════════════════════════
# Notes: short facts about the list, computed by code (never by the LLM), so they
# are always true for the numbers shown. Descriptive only -- never buy/sell.
# ═════════════════════════════════════════════════════════════════════════════
MAX_NOTES = 4
CLUSTER_MIN = 3            # this many of the list from one sector = a cluster
WIDE_SHARE = 0.6           # this share of the market moving the same way = market-wide


def _short(sec):
    return str(sec or "").replace("NIFTY ", "")


def _notes(ctx, rows, universe, when, live=True, signed=True, skipped="", total=None, total_label="",
           confirm="vwap+rvol"):
    """rows: [(sym, value)] in display order. universe: {sym: value} for the WHOLE market
    measured the same way (same window), or None when a market comparison makes no sense."""
    notes, late = [], []                 # late = outlier notes, shown after the confirmation line
    rows = [(s, v) for s, v in rows if s]
    if not rows:
        return [skipped] if skipped else []
    sm = _secmap(ctx)
    # 1. sector cluster
    by = {}
    for s, _ in rows:
        if sm.get(s):
            by.setdefault(sm[s], []).append(s)
    cl = max(by.items(), key=lambda x: len(x[1])) if by else None
    if cl and len(cl[1]) >= CLUSTER_MIN and len(rows) >= 4:
        names = ", ".join(cl[1][:4]) + (", …" if len(cl[1]) > 4 else "")
        notes.append("%d of %d are %s (%s)" % (len(cl[1]), len(rows), _short(cl[0]), names))
    vals = [v for _, v in rows if v is not None]
    if signed and universe and vals:
        d = 1 if sum(vals) / len(vals) > 0 else -1
        u = [v for v in universe.values() if v is not None]
        up, dn = sum(1 for v in u if v > 0), sum(1 for v in u if v < 0)
        avg = sum(u) / len(u) if u else 0.0
        same = (up if d > 0 else dn) / max(len(u), 1)
        against = (dn if d > 0 else up) / max(len(u), 1)
        head = "Market %s: %d up / %d down, avg %s" % (when, up, dn, _p(avg))
        if against >= WIDE_SHARE:
            notes.append(head + " → these move AGAINST the market (relative %s)" %
                         ("strength" if d > 0 else "weakness"))
        elif same >= WIDE_SHARE:
            notes.append(head + " → market-wide %s" % ("rise" if d > 0 else "fall"))
        else:
            notes.append(head + " → mixed market, moves look stock-specific")
        # 2. sector going the other way from most sectors
        secs = {}
        for s, v in universe.items():
            if sm.get(s) and v is not None:
                secs.setdefault(sm[s], []).append(v)
        savg = {k: sum(x) / len(x) for k, x in secs.items() if x}
        if cl and len(cl[1]) >= CLUSTER_MIN and cl[0] in savg and len(savg) >= 3:
            a = savg[cl[0]]
            other = [k for k, x in savg.items() if (x < 0 if a > 0 else x > 0)]
            if len(other) >= 2 * len(savg) / 3:
                notes.append("%s %s while %d of %d sectors are %s → sector rotation" % (
                    _short(cl[0]), _p(a), len(other), len(savg), "red" if a > 0 else "green"))
        # 3. outlier vs the market (same direction as the market = outperforming it)
        top_s, top_v = rows[0]
        if top_v and avg and abs(top_v) >= 3 * abs(avg):
            late.append("%s's %s is %.0fx the market average" % (top_s, _p(top_v), abs(top_v / avg)))
    # 4. outlier vs #2 (only if not already said vs the market)
    if not late and len(vals) >= 2 and vals[1] and abs(vals[0]) >= 2 * abs(vals[1]):
        late.append("%s is far ahead: %.1fx the #2" % (rows[0][0], abs(vals[0] / vals[1])))
    # 5. confirmation (live lists): VWAP side + RVOL
    if live and confirm == "rvol":
        ex = _live_cols(ctx, [s for s, _ in rows], ["rvol"])
        rv = sum(1 for s, _ in rows if (_f((ex.get(s) or {}).get("rvol")) or 0) >= 2)
        notes.append("%d of %d with day RVOL ≥ 2" % (rv, len(rows)))
    elif live and confirm:
        ex = _live_cols(ctx, [s for s, _ in rows], ["vwap", "rvol"])
        d = 1 if (sum(vals) / len(vals) if vals else 0) >= 0 else -1
        side = sum(1 for s, _ in rows if (_f((ex.get(s) or {}).get("vwap")) or 0) * d > 0)
        rv = sum(1 for s, _ in rows if (_f((ex.get(s) or {}).get("rvol")) or 0) >= 2)
        notes.append("%d of %d %s VWAP · %d with RVOL ≥ 2 → %s" % (
            side, len(rows), "above" if d > 0 else "below", rv,
            "volume-backed" if rv * 2 >= len(rows) else "thin volume"))
    notes += late
    if total and total > len(rows):
        notes.append("%d stocks in total %s; top %d shown" % (total, total_label, len(rows)))
    notes = notes[:MAX_NOTES]
    if skipped:                              # data-quality line is never dropped
        notes.append(skipped)
    return notes


# ═════════════════════════════════════════════════════════════════════════════
# HISTORY tools (read-only SQLite)
# ═════════════════════════════════════════════════════════════════════════════
def _last_days(c, n, table="symbol_day", col="day", before=None):
    q = "SELECT DISTINCT %s d FROM %s" % (col, table)
    p = []
    if before:
        q += " WHERE %s < ?" % col
        p.append(before)
    q += " ORDER BY d DESC LIMIT ?"
    p.append(n)
    return [r["d"] for r in c.execute(q, p)]


# ── daily_ohlc.csv: the raw daily bars (EQUITY), written at 15:45 by backfill_ohlc.py.
# Checked on real data (05-11-2026): every trading day present, ~211 stocks. symbol_day
# has close/volume on only some days and misses whole sessions, so prices and volume
# history come from here. Parsed once and re-read only when the file changes.
_OHLC_CACHE = {"key": None, "days": [], "data": {}}
_OHLC_LOCK = threading.Lock()


def _ohlc_path(ctx):
    st = ctx.state
    return getattr(st, "DAILY_OHLC_CSV", None) or os.path.join(
        getattr(st, "BASE_DIR", os.path.dirname(os.path.abspath(__file__))), "data", "daily_ohlc.csv")


def _ohlc(ctx):
    """(sorted days, {day: {sym: {o,h,l,c,v}}}). Raises ToolError if the file is missing."""
    import csv
    path = _ohlc_path(ctx)
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        raise ToolError("daily_ohlc.csv not found (%s)" % path)
    with _OHLC_LOCK:
        if _OHLC_CACHE["key"] == key:
            return _OHLC_CACHE["days"], _OHLC_CACHE["data"]
    data = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d, sym = (r.get("date") or "")[:10], (r.get("symbol") or "").strip()
            c = _f(r.get("close"))
            if not d or not sym or not c or c <= 0:
                continue
            data.setdefault(d, {})[sym] = {"o": _f(r.get("open")), "h": _f(r.get("high")),
                                           "l": _f(r.get("low")), "c": c, "v": _f(r.get("volume"))}
    days = sorted(data)
    with _OHLC_LOCK:
        _OHLC_CACHE.update(key=key, days=days, data=data)
    return days, data


def _prev_of(days, day):
    i = [d for d in days if d < day]
    return i[-1] if i else None


def _prev_weekday(today):
    d = datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _ram_daily(ctx, after):
    """{day: {sym: bar}} for completed sessions AFTER `after`, from the app's in-memory
    daily bars (scan_engine.fetch_daily: 60-day yfinance, auto_adjust=False, cached per
    session in state._D_SESSION_CACHE). daily_ohlc.csv is only written at 15:45, so in the
    morning yesterday is missing there -- but the running app already has it."""
    st = ctx.state
    if not hasattr(st, "_D_SESSION_LOCK"):
        return {}
    with st._D_SESSION_LOCK:
        cache = dict(getattr(st, "_D_SESSION_CACHE", {}) or {})
    out = {}
    for sym, df in cache.items():
        try:
            for t, row in df.iterrows():
                d = str(t)[:10]
                if after < d < ctx.today and _f(row.get("Close")):
                    out.setdefault(d, {})[sym] = {"o": _f(row.get("Open")), "h": _f(row.get("High")),
                                                  "l": _f(row.get("Low")), "c": _f(row.get("Close")),
                                                  "v": _f(row.get("Volume"))}
        except Exception:
            continue
    return out


def _ohlc_days(ctx, use_kite=True):
    """Completed sessions (daily_ohlc.csv + the app's RAM daily bars + the once-a-day Kite
    fill) and a note when the latest expected session is still missing.
    Merge is PER STOCK: a day the CSV has for 170 stocks still gets the other 40 from
    RAM/Kite. CSV wins, then RAM (Yahoo), then Kite -- never overwrite a bar we have."""
    days, data = _ohlc(ctx)
    past = [d for d in days if d < ctx.today]
    recent = past[-1] if past else ""
    extra = _ram_daily(ctx, past[-2] if len(past) > 1 else "")   # last CSV day may be partial
    if use_kite:
        with _KITE_LOCK:
            kd = {d: dict(v) for d, v in _KITE_DAILY.items() if d < ctx.today and d >= recent}
        for d, bars in kd.items():
            day = extra.setdefault(d, {})
            for sym, b in bars.items():
                day.setdefault(sym, b)
    note, added = "", []
    if extra:
        data = dict(data)
        for d, bars in extra.items():
            have = data.get(d, {})
            new_syms = {s: b for s, b in bars.items() if s not in have}
            if new_syms:
                merged = dict(have)
                merged.update(new_syms)
                data[d] = merged
                added.append(d)
        past = sorted(set(past) | set(added))
        if added:
            note = "%s from the app's daily cache (daily_ohlc.csv updates at 15:45)" % ", ".join(
                d[5:] + (" (some stocks)" if d in days else "") for d in sorted(added))
    if not past:
        raise ToolError("daily_ohlc.csv has no completed sessions yet.")
    exp = _prev_weekday(ctx.today)
    if past[-1] < exp:
        note = ("prices end %s — %s not available yet (daily_ohlc.csv updates at 15:45; "
                "NSE holiday?)" % (past[-1], exp))
    return past, data, note


def _gains(ctx, days):
    """{sym: compounded % change over the last `days` completed sessions}, window days,
    {sym: (base close, last close)}, note. Compounded: last close / close BEFORE the window - 1."""
    past, data, note = _ohlc_days(ctx)
    if len(past) < 2:
        raise ToolError("Need at least 2 sessions in daily_ohlc.csv.")
    win = past[-days:] if len(past) > days else past[1:]
    base = past[past.index(win[0]) - 1]
    b0, b1 = data[base], data[win[-1]]
    closes = {s: (b0[s]["c"], b1[s]["c"]) for s in b1 if s in b0}
    vals = {s: (c1 / c0 - 1) * 100.0 for s, (c0, c1) in closes.items()}
    return vals, list(reversed(win)), closes, note


def _empty(what, why):
    raise ToolError("No %s found — %s." % (what, why))


def _ending(ctx, end_day):
    """A Ctx whose history ends at end_day: every history reader excludes 'today and later',
    so 'today' becomes the day after end_day. Same state and scope (thread-local) object."""
    import copy
    c = copy.copy(ctx)
    c._today = (datetime.strptime(end_day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return c


def top_symbols(ctx, metric, days, end_day, order, limit, sector, show):
    if end_day:
        ctx = _ending(ctx, end_day)
    desc = order == "top"
    note, pair = "", {}
    if metric == "oi":
        out, d0, d1, note = _top_oi(ctx, days)
        nd = None
    elif metric in ("gain", "volume"):
        if metric == "gain":
            vals, win, closes, note = _gains(ctx, days)
            pair = closes
        else:
            past, data, note = _ohlc_days(ctx)
            win = list(reversed(past[-days:]))
            acc = {}
            for d in win:
                for s, b in data[d].items():
                    if b["v"]:
                        acc.setdefault(s, []).append(b["v"])
            vals = {s: sum(v) / len(v) for s, v in acc.items()}
        out = list(vals.items())
        d0, d1, nd = win[-1], win[0], len(win)
    else:
        # PACE / RVOL exist only in symbol_day, which skips some sessions: say how many it has.
        past, _data, _n = _ohlc_days(ctx)
        c = _ro(ctx.saga_db)
        try:
            sdd = _last_days(c, 60, before=ctx.today)
            span = sorted(set(past) | set(sdd))[-days:]          # trading calendar from both sources
            have = [d for d in sdd if d in span]
            if not have:
                _empty("%s data" % metric.upper(), "symbol_day has no session since %s" % span[0])
            col = {"pace": "max_pace", "rvol": "rvol_max"}[metric]
            qm = ",".join("?" * len(have))
            out = [(r["sym"], r["v"]) for r in c.execute(
                "SELECT sym, MAX(%s) v FROM symbol_day WHERE day IN (%s) GROUP BY sym" % (col, qm), have)
                if r["v"] is not None]
        finally:
            c.close()
        d0, d1, nd = min(have), max(have), len(have)
        miss = [d for d in span if d not in have]
        if miss:
            note = "%s data exists for %d of the last %d sessions (missing: %s)" % (
                metric.upper(), len(have), len(span), ", ".join(m[5:] for m in miss))
        vals = dict(out)
    universe = dict(out)
    out = [x for x in out if _keep(ctx, x[0], sector)]
    out = _scope(ctx, out, lambda x: x[0], hist_day=d1)
    if not out:
        _empty("stocks", "no rows matched for %s → %s%s" % (d0, d1, _scope_hint(ctx)))
    out.sort(key=lambda x: x[1], reverse=desc)
    n = len(out)
    out = out[:limit]
    if metric != "gain":                    # every stock row shows price + %: close then -> now over the window
        try:
            _v, _w, closes, _nt = _gains(ctx, nd or days)
            pair = closes
        except ToolError:
            pair = {}
    c = _ro(ctx.saga_db)
    try:
        extra = _hist_cols(c, d1, [s for s, _ in out], show, ctx)
    finally:
        c.close()
    notes = _notes(ctx, out, universe if metric == "gain" else None,
                   "over the same %d session%s" % (nd or days, "" if (nd or days) == 1 else "s"), live=False,
                   signed=metric == "gain", skipped=note)
    return {"tool": "top_symbols", "metric": metric, "order": order, "days": nd, "from": d0,
            "to": d1, "sector": sector, "rows": out, "n": n, "show": show, "extra": extra,
            "pair": {s: pair[s] for s, _ in out if s in pair}, "notes": notes,
            "secs": {x: _secmap(ctx).get(x, "") for x, _ in out}}


def _top_oi(ctx, days):
    """End-of-day OI change. oi_eod is saved only on SOME days (real data: 6 of 35), so the
    reply always names the two dates actually compared."""
    c = _ro(ctx.playbook_db)
    try:
        od = sorted(r["d"] for r in c.execute("SELECT DISTINCT date d FROM oi_eod WHERE date < ?",
                                              (ctx.today,)))
        if len(od) < 2:
            _empty("OI history", "playbook.db oi_eod has fewer than 2 saved days")
        d1 = od[-1]
        try:
            past, _d, _n = _ohlc_days(ctx)
            cal = [d for d in past if d <= d1]
            target = cal[-1 - days] if len(cal) > days else cal[0]
        except ToolError:
            target = od[max(0, len(od) - 1 - days)]
        older = [d for d in od if d <= target]
        d0 = older[-1] if older else od[0]
        if d0 == d1:
            d0 = od[-2]
        rows = c.execute(
            "SELECT a.symbol sym, a.oi_abs o1, b.oi_abs o0 FROM oi_eod a "
            "JOIN oi_eod b ON b.symbol=a.symbol AND b.date=? "
            "WHERE a.date=? AND a.oi_abs>0 AND b.oi_abs>0", (d0, d1)).fetchall()
    finally:
        c.close()
    note = ""
    if d0 != target:
        note = ("End-of-day OI is saved only on some days (%d in playbook.db): compared %s → %s, "
                "not exactly %d sessions" % (len(od), d0, d1, days))
    return [(r["sym"], (r["o1"] / r["o0"] - 1) * 100.0) for r in rows], d0, d1, note


def level_breaks_history(ctx, kind, day, order, limit, sector, show):
    past, data, note = _ohlc_days(ctx)
    if not day:
        day = past[-1]
    if day not in data:
        _empty("data for %s" % day, "not a session in daily_ohlc.csv (last is %s)" % past[-1])
    p = _prev_of(past, day)
    if not p:
        _empty("previous session", "nothing before %s in daily_ohlc.csv" % day)
    bar, pb = data[day], data[p]
    out = []
    for s, b in bar.items():
        q = pb.get(s)
        if not q or not _keep(ctx, s, sector) or not b["h"] or not q["h"]:
            continue
        hi = kind == "prev_high"
        lvl, x = (q["h"], b["h"]) if hi else (q["l"], b["l"])
        if not lvl or not x or (x <= lvl if hi else x >= lvl):
            continue
        out.append({"sym": s, "ext_pct": abs(x / lvl - 1) * 100.0,
                    "closed_beyond": bool(b["c"] > lvl if hi else b["c"] < lvl),
                    "day_pct": (b["c"] / q["c"] - 1) * 100.0, "lvl": lvl, "x": x, "close": b["c"]})
    out.sort(key=lambda x: x["ext_pct"], reverse=(order == "top"))
    out = _scope(ctx, out, lambda x: x["sym"], hist_day=day)
    n = len(out)
    out = out[:limit]
    c = _ro(ctx.saga_db)
    try:
        extra = _hist_cols(c, day, [x["sym"] for x in out], show, ctx)
    finally:
        c.close()
    return {"tool": "level_breaks_history", "kind": kind, "day": day, "prev_day": p, "sector": sector,
            "order": order,
            "rows": out, "n": n, "show": show, "extra": extra,
            "notes": _notes(ctx, [(x["sym"], x["day_pct"]) for x in out], None, "", live=False,
                            skipped=note if day == past[-1] else "")}


def symbol_history(ctx, symbol, days):
    """One stock, day by day: price / volume from daily_ohlc.csv, OI / PACE / RVOL from
    symbol_day. A day present in only one source still shows, with '—' for the rest."""
    past, data, note = _ohlc_days(ctx)
    c = _ro(ctx.saga_db)
    try:
        sd = {r["day"]: dict(r) for r in c.execute(
            "SELECT day, day_pct, oi_sess, max_pace, rvol_max FROM symbol_day WHERE sym=? AND day < ? "
            "ORDER BY day DESC LIMIT ?", (symbol, ctx.today, days + 10))}
    finally:
        c.close()
    alld = sorted({d for d in past if symbol in data[d]} | set(sd), reverse=True)[:days]
    if not alld:
        _empty("history for %s" % symbol, "not in daily_ohlc.csv or symbol_day")
    rows = []
    for d in alld:
        b = data.get(d, {}).get(symbol)
        p = _prev_of(past, d)
        pb = data.get(p, {}).get(symbol) if p else None
        x = sd.get(d, {})
        pct = (b["c"] / pb["c"] - 1) * 100.0 if b and pb else _f(x.get("day_pct"))
        rows.append({"day": d, "close": b["c"] if b else None, "day_pct": pct,
                     "volume": b["v"] if b else None, "oi_sess": x.get("oi_sess"),
                     "max_pace": x.get("max_pace"), "rvol_max": x.get("rvol_max")})
    return {"tool": "symbol_history", "symbol": symbol, "rows": rows, "note": note}


def sector_history(ctx, days, order, limit):
    vals, win, _cl, note = _gains(ctx, days)
    if not _secmap(ctx):
        _empty("sector data", "SECTOR_MAP is empty")
    out, left = _rank_sectors(ctx, vals)
    if not out:
        _empty("sector data", "no sector has %d+ stocks with prices" % MIN_SECTOR_N)
    for x in out:
        x["best_pct"] = x["best_v"]
    out.sort(key=lambda x: x["avg"], reverse=(order == "top"))
    return {"tool": "sector_history", "days": len(win), "from": win[-1], "to": win[0], "order": order,
            "rows": out[:limit], "n": len(out), "left_out": left, "note": note}


def _bootstrap_ci(by_day, seed=7):
    """Day-block bootstrap of the mean: resample whole SESSIONS, not signals,
    because signals on the same day move together. Returns (lo, hi) 90%."""
    days = list(by_day)
    if len(days) < 2:
        return None, None
    rnd = random.Random(seed)
    means = []
    for _ in range(BOOT_N):
        vals = []
        for _ in days:
            vals.extend(by_day[rnd.choice(days)])
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    return means[int(0.05 * len(means))], means[int(0.95 * len(means)) - 1]


def signal_performance(ctx, by, value, sessions, side):
    if by == "stage":
        c = _ro(ctx.saga_db)
        try:
            ds = _last_days(c, sessions, table="outcomes", before=ctx.today)
            if not ds:
                raise ToolError("No scored outcomes in saga_obs.db yet.")
            qm = ",".join("?" * len(ds))
            # v_signals_1pd = one row per symbol per day (highest stage, earliest
            # time). Counting raw `signals` would count one move dozens of times.
            q = ("SELECT o.day, o.stage grp, o.side, o.m15, o.m30, o.m60, o.res_10 res, o.src "
                 "FROM outcomes o JOIN v_signals_1pd s ON s.day=o.day AND s.sym=o.sym "
                 "AND s.stage=o.stage AND s.t=o.t "
                 "WHERE o.day IN (%s) AND o.bars_n > 0 AND o.side IN ('LONG','SHORT')" % qm)
            p = list(ds)
            if value:
                q += " AND o.stage=?"
                p.append(value)
            if side != "BOTH":
                q += " AND o.side=?"
                p.append(side)
            rows = [dict(r) for r in c.execute(q, p)]
        finally:
            c.close()
        note = "one signal per stock per day; moves signed in trade direction"
    else:
        c = _ro(ctx.playbook_db)
        try:
            ds = _last_days(c, sessions, table="trades", col="date", before=ctx.today)
            if not ds:
                raise ToolError("No logged trades in playbook.db yet.")
            qm = ",".join("?" * len(ds))
            q = ("SELECT date day, tier grp, pct_15m m15, pct_30m m30, pct_60m m60, NULL res, "
                 "NULL src FROM trades WHERE date IN (%s) AND tier IS NOT NULL" % qm)
            p = list(ds)
            if value:
                q += " AND tier=?"
                p.append(value)
            rows = [dict(r) for r in c.execute(q, p)]
        finally:
            c.close()
        side = "as logged"
        note = "playbook.db trades has no side column; % moves are as stored"

    groups = {}
    for r in rows:
        groups.setdefault(r["grp"], []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        m60 = [r for r in rs if _f(r["m60"]) is not None]
        n, nd = len(m60), len({r["day"] for r in m60})
        item = {"group": g, "n": n, "days": nd}
        if n < MIN_PERF_SIGNALS or nd < MIN_PERF_DAYS:
            item["too_few"] = True
            out.append(item)
            continue
        by_day = {}
        for r in m60:
            by_day.setdefault(r["day"], []).append(_f(r["m60"]))
        for k in ("m15", "m30", "m60"):
            vs = [_f(r[k]) for r in rs if _f(r[k]) is not None]
            item[k] = sum(vs) / len(vs) if vs else None
        item["pos_share"] = sum(1 for r in m60 if _f(r["m60"]) > 0) / n * 100.0
        item["ci_lo"], item["ci_hi"] = _bootstrap_ci(by_day)
        dec = [r["res"] for r in rs if r.get("res") in ("WIN", "LOSS") and r.get("src") == "minute"]
        if len(dec) >= MIN_PERF_SIGNALS:
            item["win_share"] = dec.count("WIN") / len(dec) * 100.0
            item["n_decided"] = len(dec)
        out.append(item)
    if not out:
        raise ToolError("No matching signals in the last %d sessions." % len(ds))
    return {"tool": "signal_performance", "by": by, "value": value, "side": side,
            "sessions": len(ds), "from": ds[-1], "to": ds[0], "groups": out, "note": note}


# ═════════════════════════════════════════════════════════════════════════════
# LIVE tools (in-memory caches)
# ═════════════════════════════════════════════════════════════════════════════
def _skip_note(stale, gaps):
    parts = []
    if gaps:
        parts.append("%d skipped: data gap in the window (app restarted?)" % gaps)
    if stale:
        parts.append("%d skipped: no fresh bars" % stale)
    return " · ".join(parts)


def live_top(ctx, metric, window_min, order, limit, sector, show):
    desc = order == "top"
    note, used = "", 0
    info, kind = {}, ""
    if window_min and metric == "pct":
        vals, info, used, stale, gaps = _window_moves(ctx, window_min)
        note, kind = _skip_note(stale, gaps), "move"
    elif window_min and metric == "volume":
        vals, info, stale, gaps = _window_volume(ctx, window_min)
        note, kind = _skip_note(stale, gaps), "spike"
    elif window_min and metric == "oi":
        vals, info = _oi_window(ctx, window_min)
        kind = "oi"
    elif metric == "fut_volume":
        vals, info, skipped = _fut_volume(ctx, window_min)
        kind = "futspike" if window_min else "futvol"
        if skipped:
            note = "%d stocks skipped: not enough futures ticks for %d+%d min" % (skipped, window_min, window_min)
    elif metric in ("pace_change", "pace_low", "pace_high"):
        vals, info = _pace_changes(ctx, window_min, {"pace_low": "LOW", "pace_high": "HIGH"}.get(metric))
        kind = "pacechg"
    elif metric in ("pace_cross",) or (metric == "pace" and window_min):
        vals, info, gaps = _pace_window(ctx, window_min, cross=metric == "pace_cross")
        note, kind = _skip_note(0, gaps), "pacecross" if metric == "pace_cross" else "pacewin"
    elif LIVE_METRICS.get(metric, {}).get("near"):
        vals, info = _near(ctx, metric)
        kind = "near"
    elif metric in _CROSS_LEVELS:
        lv_fn, up = _CROSS_LEVELS[metric](ctx)
        vals, info = _crosses(ctx, window_min, lv_fn, up)
        kind = "cross"
    elif metric == "vwap":
        vals, info = {}, {}
        for s, r in _live_results(ctx).items():
            px, vw = _f(r.get("price")), _f(r.get("vwap"))
            if px and vw:
                vals[s], info[s] = (px / vw - 1) * 100.0, (vw, px)
        kind = "vwap"
    elif metric == "score":
        vals, info = {}, {}
        for s, r in _live_results(ctx).items():
            v = _score_num(r)
            if v is not None:
                vals[s], info[s] = v, (_pct_txt(r), str(r.get("state") or ""))
        kind = "score"
    elif metric == "iv":
        vals, info = _iv_values(ctx, window_min)
        kind = "ivwin" if window_min else "iv"
    elif metric in ("new_high", "new_low"):
        vals, info, used, stale, gaps = _new_extremes(ctx, window_min, metric == "new_high")
        note, kind = _skip_note(stale, gaps), "extreme"
    elif metric == "gap":
        res = _live_results(ctx)
        vals, info = {}, {}
        for s, r in res.items():
            o, pc = _f(r.get("day_open")), _f(r.get("pdc"))
            if o and pc:
                vals[s] = (o / pc - 1) * 100.0
                info[s] = (pc, o)
        kind = "gap"
    elif metric == "open":
        res = _live_results(ctx)
        vals, info = {}, {}
        for s, r in res.items():
            o, px = _f(r.get("day_open")), _f(r.get("price"))
            if o and o > 0 and px:
                vals[s] = (px / o - 1) * 100.0
                info[s] = (o, px)
        kind = "open"
    elif metric == "price":
        vals = {s: _f(r.get("price")) for s, r in _live_results(ctx).items()}
    elif metric in ("pct", "rvol"):
        key = "intra_pct" if metric == "pct" else "rvol"
        vals = {s: _f(r.get(key)) for s, r in _live_results(ctx).items()}
    elif metric == "oi":
        oi = _snap(ctx.state.OI_CACHE_LOCK, ctx.state.OI_CACHE)
        vals = {s: _oi_pct(d) for s, d in oi.items() if isinstance(d, dict)}
    elif metric == "pace":
        vals = _live_pace(ctx)
    else:
        vals = _live_volume(ctx)
    universe = {s: v for s, v in vals.items() if v is not None}
    rows = [(s, v) for s, v in universe.items() if _keep(ctx, s, sector)]
    rows = _scope(ctx, rows, lambda x: x[0])
    if not rows and kind == "pacechg":
        raise ToolError("No stock changed PACE%s in the last %d min%s%s." % (
            {"pace_low": " by a new day low", "pace_high": " by a new day high"}.get(metric, ""), window_min, (" in " + _short(sector)) if sector else "", _scope_hint(ctx)))
    if not rows and kind in ("cross", "pacecross"):
        raise ToolError("No stock %s in the last %d min%s%s." % (
            _CROSS_NAME.get(metric, "moved into a new PACE band"), window_min,
            (" in " + _short(sector)) if sector else "", _scope_hint(ctx)))
    if not rows:
        if note:
            raise ToolError("No usable %s data for the last %d min — %s." % (metric, window_min, note))
        raise ToolError("No live %s data%s." % (
            metric, " for the last %d min (market closed?)" % window_min if window_min else "") +
            _scope_hint(ctx))
    if metric == "new_low" or kind == "near":
        desc = not desc                          # deepest fall / closest first
    rows.sort(key=lambda x: x[1], reverse=desc)
    n = len(rows)
    rows = rows[:limit]
    show = [c for c in show if c != "price"]   # price is always on line 2 (then -> now)
    syms = [s for s, _ in rows]
    if kind == "move":
        pair = {s: (info[s][1], info[s][0]) for s in syms}
    elif kind == "spike":
        pair = {s: (info[s][3], info[s][2]) for s in syms}
    elif kind == "oi":
        pair = {s: (info[s][3], info[s][2]) for s in syms}
    elif kind in ("open", "gap"):
        pair = {s: info[s] for s in syms}
    elif kind == "extreme":
        pair = {s: (info[s][1], info[s][0]) for s in syms}
    elif kind in ("vwap", "iv", "ivwin", "near"):
        pair = {s: info[s] for s in syms}
    elif kind in ("pacewin", "pacecross"):
        pair = {s: (info[s][5], info[s][6]) for s in syms}
    elif kind == "cross":
        pair = {s: (info[s][2], info[s][3]) for s in syms}
    elif kind in ("futspike", "score", "pacechg"):
        pair = {}
    else:
        pair = _day_pair(ctx, syms)
    signed = metric in ("pct", "open", "gap") or kind in ("oi", "extreme")
    when = ("in the same %d min" % window_min) if window_min else "today"
    if metric == "pct" and not window_min:
        when = "today"
    uni = universe if metric in ("pct", "open", "gap") else None
    if metric == "open":
        when = "since open"
    if metric == "gap":
        when = "at the open"
    if kind == "extreme":
        uni = None
    if kind in ("futspike", "futvol", "pacewin", "pacecross", "score", "iv", "ivwin", "vwap", "cross", "near",
                "pacechg"):
        signed = kind in ("ivwin", "vwap", "cross")
        uni = None
    res_now = _live_results(ctx, required=False)
    px = {s: (_f((res_now.get(s) or {}).get("price")), _f((res_now.get(s) or {}).get("intra_pct"))) for s in syms}
    if window_min and kind not in ("move", "spike", "oi", "extreme", "pacewin", "pacecross", "cross"):
        then = _px_then(ctx, syms, window_min)
        pair = {s: (then.get(s), px[s][0]) for s in syms} if then else pair
    note_rows = [(s, info[s][0]) for s in syms] if kind == "cross" else \
        [(s, info[s][1]) for s in syms] if kind == "pacechg" else rows
    notes = _notes(ctx, note_rows, uni, when, live=True, signed=signed, skipped=note,
                   confirm="rvol" if kind in ("spike", "futspike") or metric in ("volume", "rvol") else "vwap+rvol")
    pace_hist = {s: pace_line(ctx, s) for s in syms} if metric in ("pace", "pace_cross", "pace_change", "pace_low",
                                                                   "pace_high") else {}
    return {"tool": "live_top", "metric": metric, "window_min": window_min, "kind": kind, "px": px,
            "pace_hist": pace_hist, "pace_meta": _pace_meta(ctx) if pace_hist else None,
            "info": {s: info[s] for s in syms if s in info}, "n": n, "pair": pair, "notes": notes,
            "mins_used": used, "order": order, "sector": sector, "rows": rows, "show": show,
            "extra": _live_cols(ctx, syms, show), "note": note, "as_of": _as_of(ctx)}


def _cross_vwap(up):
    return lambda ctx: ((lambda sym, x: x[2]), up)


def _cross_fixed(get, up):
    """A level that does not move during the day (yesterday's high/low, today's open)."""
    def make(ctx):
        lv = get(ctx)
        return (lambda sym, x: lv.get(sym)), up
    return make


def _pdh(ctx):
    return {s: v[0] for s, v in _prev_levels(ctx)[1].items()}


def _pdl(ctx):
    return {s: v[1] for s, v in _prev_levels(ctx)[1].items()}


def _opens(ctx):
    return {s: _f(r.get("day_open")) for s, r in _live_results(ctx).items()}


_CROSS_LEVELS = {"vwap_up": _cross_vwap(True), "vwap_down": _cross_vwap(False),
                 "pdh_up": _cross_fixed(_pdh, True), "pdl_down": _cross_fixed(_pdl, False),
                 "open_up": _cross_fixed(_opens, True), "open_down": _cross_fixed(_opens, False)}
_CROSS_NAME = {"vwap_up": "crossed ABOVE VWAP", "vwap_down": "crossed BELOW VWAP",
               "pdh_up": "crossed ABOVE yesterday's high", "pdl_down": "crossed BELOW yesterday's low",
               "open_up": "crossed ABOVE today's open", "open_down": "crossed BELOW today's open"}


def _kick_kite_fill(ctx):
    """Start the once-a-day Kite fill in the background (never blocks a reply)."""
    with _KITE_LOCK:
        if _KITE_STATE["running"] or _KITE_STATE["day"] == ctx.today:
            return
    threading.Thread(target=kite_fill, args=(ctx,), name="ai_kite_fill", daemon=True).start()


def _is_index(ctx, sym):
    return str(_secmap(ctx).get(sym, "")).upper() == "INDEX" or sym in (
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")


def _prev_levels(ctx):
    """Previous completed session's high / low per symbol. Same source order as the
    history tools: Kite fill + RAM daily bars (current) -> daily_ohlc.csv -> symbol_day."""
    try:
        past, data, _note = _ohlc_days(ctx)
        d = past[-1]
        lv = {s: (b["h"], b["l"]) for s, b in data[d].items() if b["h"] and b["l"]}
        if lv:
            return d, lv
    except ToolError:
        pass
    c = _ro(ctx.saga_db)
    try:
        ds = _last_days(c, 1, before=ctx.today)
        if not ds:
            raise ToolError("No previous session found for yesterday's levels.")
        rows = c.execute("SELECT sym, high, low FROM symbol_day WHERE day=?", (ds[0],)).fetchall()
    finally:
        c.close()
    return ds[0], {r["sym"]: (_f(r["high"]), _f(r["low"])) for r in rows}


# ── Kite daily fill ──────────────────────────────────────────────────────────
# The morning gap: daily_ohlc.csv is written at 15:45, and the Yahoo daily cache in RAM
# sometimes lacks yesterday for some stocks. For exactly those stocks, fetch yesterday's
# daily bar from Kite historical_data ONCE per day (exact NSE bar), cache it in RAM.
# Read-only API use; ~3 requests/sec to respect Kite's historical-data limit.
KITE_SLEEP_S = 0.35
_KITE_DAILY = {}                    # day -> {sym: bar}
_KITE_STATE = {"day": None, "running": False, "fetched": 0, "failed": [], "missing": []}
_KITE_LOCK = threading.Lock()


def _missing_prev(ctx):
    """(expected previous session, [stocks with no bar for it]) — indices excluded."""
    try:
        past, data, _n = _ohlc_days(ctx, use_kite=True)
    except ToolError:
        return None, []
    exp = _prev_weekday(ctx.today)
    day = past[-1] if past[-1] >= exp else exp
    have = data.get(day, {})
    uni = [u for u in (getattr(ctx.state, "FNO_SYMBOLS", []) or []) if not _is_index(ctx, u)]
    return day, sorted(u for u in uni if u not in have)


def kite_fill(ctx, sleep=time.sleep):
    """Fetch the previous session's daily bar from Kite for stocks that have none.
    Runs at most once per day (call it from a background thread). Never raises."""
    with _KITE_LOCK:
        if _KITE_STATE["running"] or _KITE_STATE["day"] == ctx.today:
            return dict(_KITE_STATE)
        _KITE_STATE.update(running=True, day=ctx.today, fetched=0, failed=[], missing=[])
    try:
        km = getattr(ctx.state, "kite_mgr", None)
        toks = getattr(km, "_equity_tokens", None) or {}
        day, missing = _missing_prev(ctx)
        with _KITE_LOCK:
            _KITE_STATE["missing"] = list(missing)
        if not day or not missing or km is None or not getattr(km, "kite", None):
            return dict(_KITE_STATE)
        got, failed = {}, []
        for sym in missing:
            tok = toks.get(sym)
            if not tok:
                failed.append(sym)
                continue
            try:
                rows = km.kite.historical_data(int(tok), "%s 00:00:00" % day, "%s 23:59:59" % day, "day")
                r = (rows or [None])[-1]
                if r and _f(r.get("close")):
                    got[sym] = {"o": _f(r.get("open")), "h": _f(r.get("high")), "l": _f(r.get("low")),
                                "c": _f(r.get("close")), "v": _f(r.get("volume"))}
                else:
                    failed.append(sym)
            except Exception:
                failed.append(sym)
            sleep(KITE_SLEEP_S)
        with _KITE_LOCK:
            _KITE_DAILY.setdefault(day, {}).update(got)
            _KITE_STATE.update(fetched=len(got), failed=failed)
        print("[AI chat] Kite daily fill %s: %d fetched, %d failed" % (day, len(got), len(failed)))
        return dict(_KITE_STATE)
    finally:
        with _KITE_LOCK:
            _KITE_STATE["running"] = False


def live_level_breaks(ctx, kind, order, limit, sector, show):
    pday, lv = _prev_levels(ctx)
    res = _live_results(ctx)
    out = []
    for s, r in res.items():
        px = _f(r.get("price"))
        hi_lo = lv.get(s)
        if not px or not hi_lo or not _keep(ctx, s, sector):
            continue
        ph, pl = hi_lo
        if kind == "prev_high" and ph and px > ph:
            out.append({"sym": s, "dist_pct": (px / ph - 1) * 100.0, "price": px, "level": ph,
                        "pct": _f(r.get("intra_pct"))})
        elif kind == "prev_low" and pl and px < pl:
            out.append({"sym": s, "dist_pct": (1 - px / pl) * 100.0, "price": px, "level": pl,
                        "pct": _f(r.get("intra_pct"))})
    out.sort(key=lambda x: x["dist_pct"], reverse=(order == "top"))
    out = _scope(ctx, out, lambda x: x["sym"])
    n = len(out)
    out = out[:limit]
    miss = sorted(s for s in res if s not in lv and not _is_index(ctx, s))
    if miss:
        _kick_kite_fill(ctx)
    return {"tool": "live_level_breaks", "kind": kind, "prev_day": pday, "sector": sector, "order": order,
            "rows": out, "n": n, "missing_levels": len(miss), "missing_syms": miss,
            "kite": dict(_KITE_STATE),
            "show": show, "extra": _live_cols(ctx, [x["sym"] for x in out], show, res),
            "notes": _notes(ctx, [(x["sym"], x["pct"]) for x in out], _day_universe(res), "today",
                            live=True, total=n, total_label="beyond yesterday's level"),
            "as_of": _as_of(ctx)}


def live_vwap(ctx, side, order, limit, sector, show):
    res = _live_results(ctx)
    out = []
    for s, r in res.items():
        px, vw = _f(r.get("price")), _f(r.get("vwap"))
        if not px or not vw or not _keep(ctx, s, sector):
            continue
        d = (px / vw - 1) * 100.0
        if (side == "above" and d > 0) or (side == "below" and d < 0):
            out.append({"sym": s, "dist_pct": d, "price": px, "vwap": vw, "pct": _f(r.get("intra_pct"))})
    out.sort(key=lambda x: abs(x["dist_pct"]), reverse=(order == "top"))
    out = _scope(ctx, out, lambda x: x["sym"])
    n = len(out)
    out = out[:limit]
    return {"tool": "live_vwap", "side": side, "sector": sector, "order": order, "rows": out, "n": n,
            "show": show, "extra": _live_cols(ctx, [x["sym"] for x in out], show, res),
            "notes": _notes(ctx, [(x["sym"], x["pct"]) for x in out], _day_universe(res), "today",
                            live=True, total=n, total_label="%s VWAP" % side),
            "as_of": _as_of(ctx)}


def _pace_path(ctx, symbol):
    """Today's PACE path rebuilt from the 1-min bars: running (high - low) / locked budget,
    the same odometer as thin_backtest.simulate_pace. Returns
    {"bands": [(HH:MM, band)], "first": band, "last": (HH:MM, "HIGH"|"LOW"), "budget", "gap"}
    or None. Live PACE uses FUTURES high/low; these bars are EQUITY, so times can differ a
    little from the dashboard -- the reply says so."""
    st = ctx.state
    with st.IMPLIED_BUDGET_LOCK:
        budget = _f(st.IMPLIED_BUDGET.get(symbol))
    with st._INTRADAY_CACHE_LOCK:
        df = st._INTRADAY_CACHE.get(symbol)
    if not budget or budget <= 0 or df is None:
        return None
    try:
        d = df[[str(t)[:10] == ctx.today for t in df.index]]
        if not len(d):
            return None
        hi_col = d["High"] if "High" in d else d["Close"]
        lo_col = d["Low"] if "Low" in d else d["Close"]
        times = [str(t)[11:16] for t in d.index]
        stamps = [(x.tz_localize(None) if getattr(x, "tzinfo", None) else x).to_pydatetime() for x in d.index]
        hs, ls = [float(x) for x in hi_col], [float(x) for x in lo_col]
    except Exception:
        return None
    hi = lo = None
    bands, last, prev_band, n_new, series = [], None, None, 0, []
    for k, (t, h, l) in enumerate(zip(times, hs, ls)):
        new_hi, new_lo = hi is None or h > hi, lo is None or l < lo
        if hi is not None and (new_hi or new_lo):
            last = (t, "HIGH" if new_hi else "LOW")
            n_new += 1
        hi, lo = h if hi is None else max(hi, h), l if lo is None else min(lo, l)
        b = ctx.pband((hi - lo) / budget)
        series.append((stamps[k] + timedelta(minutes=1), (hi - lo) / budget, b))
        if b and b != prev_band:
            if prev_band is not None:
                bands.append((t, b))
            prev_band = b
    first = ctx.pband((hs[0] - ls[0]) / budget) if hs else None
    return {"bands": bands, "first": first, "now_band": prev_band, "last": last, "n_new": n_new,
            "budget": budget, "start": times[0], "gap": times[0] > "09:16", "series": series}


def _symbol_window(ctx, symbol, minutes):
    """Last N clock minutes for one stock: move %, price then -> now, new day high/low inside
    the window, volume spike x. Same time-based lookup and gap checks as the list tools."""
    moves, info, used, stale, gaps = _window_moves(ctx, minutes)
    if symbol not in moves:
        return {"gap": bool(gaps), "stale": bool(stale)} if (gaps or stale) else None
    now, then, t0, _vol = info[symbol]
    live = _f((_live_results(ctx, required=False).get(symbol) or {}).get("price"))
    if live and then:
        now = live
        moves = dict(moves)
        moves[symbol] = (live / then - 1) * 100.0
    spike, sinfo, _s1, _g1 = _window_volume(ctx, minutes)
    ext = None
    for hi in (True, False):
        v, _i, _u, _st, _g = _new_extremes(ctx, minutes, hi)
        if symbol in v:
            ext = "new day HIGH" if hi else "new day LOW"
    return {"pct": moves[symbol], "now": now, "then": then, "at": t0.strftime("%H:%M"),
            "spike": spike.get(symbol), "ext": ext, "used": used}


def live_symbol(ctx, symbol):
    st = ctx.state
    with st.SESSION_RESULTS_LOCK:
        r = dict(st.SESSION_RESULTS.get(symbol) or {})
    with st.SESSION_RESULTS_SHORT_LOCK:
        sh = dict(st.SESSION_RESULTS_SHORT.get(symbol) or {})
    with st.OI_CACHE_LOCK:
        oi = st.OI_CACHE.get(symbol)
        oi = dict(oi) if isinstance(oi, dict) else {}
    if not r and not oi:
        if _is_index(ctx, symbol):
            raise ToolError("%s is an index — ask 'market mood' for NIFTY / BANKNIFTY." % symbol)
        raise ToolError("No live data for %s yet." % symbol)
    px, vw = _f(r.get("price")), _f(r.get("vwap"))
    sc, mx = r.get("score"), r.get("max_score")
    ls, ss = _f(r.get("pct_score")), _f(sh.get("pct_score"))
    recent = {w: _symbol_window(ctx, symbol, w) for w in (5, 15)}
    stronger = None
    if ls is not None and ss is not None and ls != ss:
        stronger = "LONG" if ls > ss else "SHORT"
    return {"tool": "live_symbol", "symbol": symbol, "price": px, "pct": _f(r.get("intra_pct")),
            "open": _f(r.get("day_open")), "vwap": vw, "vwap_dist": (px / vw - 1) * 100.0 if px and vw else None,
            "rvol": _f(r.get("rvol")), "state": r.get("state") or "", "short_state": sh.get("state") or "",
            "sector": r.get("sector") or _secmap(ctx).get(symbol, ""),
            "long_pct": ls, "short_pct": ss, "stronger": stronger,
            "score": ("%s/%s" % (sc, mx)) if sc is not None and mx else sc,
            "short_score": ("%s/%s" % (sh.get("score"), sh.get("max_score")))
                           if sh.get("score") is not None and sh.get("max_score") else None,
            "pulse_long": _f(r.get("oi_pulse")), "pulse_short": _f(sh.get("oi_pulse")),
            "prox": r.get("proximity"), "prox_name": r.get("prox_name") or "",
            "oi_open": _oi_pct(oi), "oi_15m": _f(oi.get("oi_15m")), "oi_30m": _f(oi.get("oi_30m")),
            "streak": oi.get("streak"), "pace": _live_pace(ctx).get(symbol),
            "pace_path": _pace_path(ctx, symbol), "pace_hist": pace_history(ctx, symbol),
            "hilo": day_hilo_times(ctx, symbol), "prev": _prev_levels(ctx)[1].get(symbol), "pdc": _f(r.get("pdc")),
            "fields": (stock_fields(ctx, set(FIELDS), only={symbol}) or {}).get(symbol) or {},
            "recent": recent, "as_of": _as_of(ctx)}


def current_candidates(ctx, side, order, limit, sector, show):
    res = _live_results(ctx, side)
    longs = res if side == "LONG" else _live_results(ctx, "LONG")
    picks = []
    for s, r in res.items():
        stt = str(r.get("state") or "").upper()
        if any(f in stt for f in _FLAG_STATES) and _keep(ctx, s, sector):
            picks.append((s, stt, longs.get(s) or r))
    picks = _scope(ctx, picks, lambda x: x[0])
    conv = ctx.conviction(side)
    out = []
    for s, stt, base in picks:                   # no lock held here: conv() takes its own
        rv, ip = _f(base.get("rvol")) or 1.0, _f(base.get("intra_pct")) or 0.0
        sec = base.get("sector") or _secmap(ctx).get(s, "")
        try:
            score, tier, streak, peers = conv(s, rv, ip, sec)
        except Exception:
            score, tier, streak, peers = 0, "LOADING", 0, 0
        out.append({"sym": s, "state": stt, "score": score, "tier": tier, "streak": streak,
                    "peers": peers, "rvol": rv, "pct": ip})
    out.sort(key=lambda x: x["score"], reverse=(order == "top"))
    n = len(out)
    out = out[:limit]
    show = [c for c in show if c != "price"]
    return {"tool": "current_candidates", "side": side, "sector": sector, "order": order, "rows": out, "n": n,
            "show": show, "extra": _live_cols(ctx, [x["sym"] for x in out], show, longs),
            "pair": _day_pair(ctx, [x["sym"] for x in out]),
            "notes": _notes(ctx, [(x["sym"], x["pct"]) for x in out], _day_universe(longs), "today",
                            live=True),
            "as_of": _as_of(ctx)}


def watchlist_status(ctx, sort_by, side, only_flagged, order, limit, sector, show):
    st = ctx.state
    with st.KANBAN_LOCK:
        kb = [dict(k) for k in st.KANBAN if isinstance(k, dict)]
    with st.WATCH_LOCK:
        names = set(st.WATCHLIST)
    items = {k["symbol"]: k for k in kb if k.get("symbol")}
    for n in names - set(items):
        items[n] = {"symbol": n, "stage": "watch", "direction": "LONG"}
    if not items:
        raise ToolError("Your watchlist is empty.")
    longs = _live_results(ctx, required=False)
    shorts = _live_results(ctx, "SHORT", required=False)
    oi = _snap(st.OI_CACHE_LOCK, st.OI_CACHE)
    out = []
    for s, k in items.items():
        d = str(k.get("direction") or "LONG").upper()
        if (side != "BOTH" and d != side) or not _keep(ctx, s, sector):
            continue
        r = longs.get(s) or {}
        stt = ((shorts.get(s) or {}).get("state") if d == "SHORT" else r.get("state")) or ""
        if only_flagged and not any(f in stt.upper() for f in _FLAG_STATES):
            continue
        o = oi.get(s) if isinstance(oi.get(s), dict) else {}
        sc, mx = r.get("score"), r.get("max_score")
        out.append({"sym": s, "stage": k.get("stage") or "watch", "dir": d, "state": stt,
                    "score": _score_num(shorts.get(s) or {} if d == "SHORT" else r),
                    "score_txt": _pct_txt(shorts.get(s) or {} if d == "SHORT" else r),
                    "prox": r.get("proximity"), "prox_name": r.get("prox_name") or "",
                    "pct": _f(r.get("intra_pct")), "rvol": _f(r.get("rvol")), "oi": _oi_pct(o),
                    "price": _f(r.get("price"))})
    out = _scope(ctx, out, lambda x: x["sym"])
    key = {"score": "score", "prox": "prox", "oi": "oi", "pct": "pct", "rvol": "rvol"}[sort_by]
    sgn = -1 if order == "top" else 1                    # missing values always last
    out.sort(key=lambda x: (_f(x[key]) is None, sgn * (_f(x[key]) or 0)))
    n = len(out)
    out = out[:limit]
    if not out:
        raise ToolError("No watchlist stocks match%s." % ((" in " + _short(sector)) if sector else ""))
    return {"tool": "watchlist_status", "sort_by": sort_by, "side": side, "only_flagged": only_flagged,
            "order": order, "sector": sector, "rows": out, "n": n, "show": show,
            "extra": _live_cols(ctx, [x["sym"] for x in out], show, longs), "as_of": _as_of(ctx)}


def recent_alerts(ctx, symbol, limit):
    st = ctx.state
    q = _snap(st.ALERT_LOCK, st.ALERT_QUEUE)       # works for list or deque
    out = [a for a in q if isinstance(a, dict) and str(a.get("date", "")) == ctx.today
           and (not symbol or str(a.get("sym", "")).upper() == symbol)]
    out.sort(key=lambda a: str(a.get("time", "")), reverse=True)
    res = _live_results(ctx, required=False)
    px = {str(a.get("sym", "")): (_f((res.get(str(a.get("sym", ""))) or {}).get("price")),
                                   _f((res.get(str(a.get("sym", ""))) or {}).get("intra_pct"))) for a in out[:limit]}
    return {"tool": "recent_alerts", "symbol": symbol, "rows": out[:limit], "n": len(out), "px": px}


def _sector_values(ctx, metric, window_min):
    """{sym: value} feeding the sector average, + window note."""
    note = ""
    if metric == "oi":
        if window_min:
            return _oi_window(ctx, window_min)[0], note
        oi = _snap(ctx.state.OI_CACHE_LOCK, ctx.state.OI_CACHE)
        return {s: _oi_pct(d) for s, d in oi.items() if isinstance(d, dict)}, note
    if metric == "open":
        out = {}
        for s, r in _live_results(ctx).items():
            px, op = _f(r.get("price")), _f(r.get("day_open"))
            if px and op:
                out[s] = (px / op - 1) * 100.0
        return out, note
    if window_min:
        vals, info, used, stale, gaps = _window_moves(ctx, window_min)
        if used and used < window_min:
            note = "only %d min of bars so far" % used
        sk = _skip_note(stale, gaps)
        if sk:
            note = (note + " · " if note else "") + sk
        return vals, note
    return {s: _f(r.get("intra_pct")) for s, r in _live_results(ctx).items()}, note


MIN_SECTOR_N = 3                      # a 1-stock "sector" is one stock, not a sector move
JUNK_SECTORS = {"OTHER", "OTHERS", "MISC", "UNKNOWN", "INDEX"}


def _rank_sectors(ctx, vals):
    """([sector rows], ["NAME (n)" left out]). Sectors with < MIN_SECTOR_N stocks and
    catch-all buckets are left out of the ranking and named in a footnote instead."""
    groups = {}
    for s, v in vals.items():
        sec = _secmap(ctx).get(s)
        if sec and v is not None:
            groups.setdefault(sec, []).append((s, v))
    out, left = [], []
    for sec, xs in sorted(groups.items()):
        if len(xs) < MIN_SECTOR_N or _short(sec).upper() in JUNK_SECTORS:
            left.append("%s (%d)" % (_short(sec), len(xs)))
            continue
        best = max(xs, key=lambda x: x[1])
        out.append({"sector": sec, "avg": sum(v for _, v in xs) / len(xs), "n": len(xs),
                    "up": sum(1 for _, v in xs if v > 0), "best": best[0], "best_v": best[1]})
    return out, left


def live_sector_rank(ctx, metric, window_min, order, limit):
    vals, note = _sector_values(ctx, metric if metric in ("oi", "open") else "price", window_min)
    out, left = _rank_sectors(ctx, vals)
    if not out:
        raise ToolError("No live sector data yet%s." % (" for the last %d min" % window_min if window_min else ""))
    keyf = (lambda x: x["up"] / x["n"]) if metric == "breadth" else (lambda x: x["avg"])
    out.sort(key=keyf, reverse=(order == "top"))
    idx = {}
    if not window_min and metric == "price" and hasattr(ctx.state, "SECTOR_INTEL_LOCK"):
        with ctx.state.SECTOR_INTEL_LOCK:
            idx = {k: _f((v or {}).get("pct")) for k, v in
                   ((ctx.state.SECTOR_INTEL or {}).get("data") or {}).items()}
    for x in out:
        x["index_pct"] = idx.get(x["sector"])
    return {"tool": "live_sector_rank", "metric": metric, "window_min": window_min, "order": order,
            "rows": out[:limit], "n": len(out), "left_out": left, "note": note, "as_of": _as_of(ctx)}


def _ad_points(ctx):
    st = ctx.state
    return [p for p in _snap(st.AD_BREADTH_LOCK, st.AD_BREADTH) if isinstance(p, dict) and p.get("ts")]


def _ad_ago(points, minutes, now_ts):
    cut = now_ts - minutes * 60
    older = [p for p in points if p["ts"] <= cut]
    return older[-1] if older else None


def compare_symbols(ctx, symbols):
    res = _live_results(ctx)
    shorts = _live_results(ctx, "SHORT", required=False)
    oi = _snap(ctx.state.OI_CACHE_LOCK, ctx.state.OI_CACHE)
    moves, _i, _u, _st, _g = _window_moves(ctx, 15)
    rows = []
    for sym in symbols:
        r = res.get(sym) or {}
        o = oi.get(sym) if isinstance(oi.get(sym), dict) else {}
        px, vw, op = _f(r.get("price")), _f(r.get("vwap")), _f(r.get("day_open"))
        rows.append({"sym": sym, "price": px, "pct": _f(r.get("intra_pct")),
                     "open": (px / op - 1) * 100.0 if px and op else None, "m15": moves.get(sym),
                     "vwap": (px / vw - 1) * 100.0 if px and vw else None, "rvol": _f(r.get("rvol")),
                     "oi": _oi_pct(o), "long": _pct_txt(r), "short": _pct_txt(shorts.get(sym) or {}),
                     "state": r.get("state") or "", "sstate": (shorts.get(sym) or {}).get("state") or ""})
    return {"tool": "compare_symbols", "rows": rows, "as_of": _as_of(ctx)}


def market_mood(ctx):
    st = ctx.state
    res = _live_results(ctx)
    shorts = _live_results(ctx, "SHORT", required=False)
    regime, rin = "", {}
    if hasattr(st, "MARKET_REGIME_LOCK"):
        with st.MARKET_REGIME_LOCK:
            regime, rin = getattr(st, "MARKET_REGIME", ""), dict(getattr(st, "MARKET_REGIME_INPUTS", {}) or {})
    idx = {}
    if hasattr(st, "SECTOR_INTEL_LOCK"):
        with st.SECTOR_INTEL_LOCK:
            idx = {k: dict(v or {}) for k, v in ((st.SECTOR_INTEL or {}).get("data") or {}).items()}
    adv = sum(1 for r in res.values() if (_f(r.get("intra_pct")) or 0) > 0)
    dec = sum(1 for r in res.values() if (_f(r.get("intra_pct")) or 0) < 0)
    pts = _ad_points(ctx) if hasattr(st, "AD_BREADTH_LOCK") else []
    ago = _ad_ago(pts, 30, pts[-1]["ts"]) if pts else None
    try:
        br = ctx.breadth() or {}
    except Exception:
        br = {}
    oi = _snap(st.OI_CACHE_LOCK, st.OI_CACHE)
    quads, ivd = {}, {}
    for d in oi.values():
        if not isinstance(d, dict):
            continue
        q = d.get("vol_quad") or d.get("tag")
        if q:
            quads[str(q)] = quads.get(str(q), 0) + 1
        v = d.get("iv_dir")
        if v:
            ivd[str(v)] = ivd.get(str(v), 0) + 1
    fl = sum(1 for r in res.values() if any(f in str(r.get("state") or "").upper() for f in _FLAG_STATES))
    fs = sum(1 for r in shorts.values() if any(f in str(r.get("state") or "").upper() for f in _FLAG_STATES))
    secs, _left = _rank_sectors(ctx, {s: _f(r.get("intra_pct")) for s, r in res.items()})
    secs.sort(key=lambda x: x["avg"], reverse=True)
    return {"tool": "market_mood", "regime": regime, "regime_inputs": rin,
            "nifty": idx.get("NIFTY 50"), "bank": idx.get("NIFTY BANK"), "vix": idx.get("INDIA VIX"),
            "adv": adv, "dec": dec, "total": len(res),
            "ago": ({"adv": ago.get("advance"), "dec": ago.get("decline"), "time": ago.get("time")}
                    if ago else None),
            "breadth": br if br.get("ok") else None, "quads": quads, "iv_dir": ivd,
            "flags_long": fl, "flags_short": fs,
            "lead": secs[0] if secs else None, "lag": secs[-1] if len(secs) > 1 else None,
            "as_of": _as_of(ctx)}


def breadth_trend(ctx, window_min):
    pts = _ad_points(ctx)
    if not pts:
        raise ToolError("No advance/decline history yet (market closed or app just started).")
    now_ts = pts[-1]["ts"]
    win = [p for p in pts if p["ts"] >= now_ts - window_min * 60]
    step = max(1, len(win) // 6)
    sample = win[::step]
    if sample[-1] is not win[-1]:
        sample.append(win[-1])
    first, last = win[0], win[-1]
    r0, r1 = _f(first.get("ad_ratio")), _f(last.get("ad_ratio"))
    trend = "flat"
    if r0 and r1:
        trend = "improving" if r1 > r0 * 1.1 else ("weakening" if r1 < r0 / 1.1 else "flat")
    return {"tool": "breadth_trend", "window_min": window_min,
            "covered_min": int((now_ts - first["ts"]) / 60), "trend": trend,
            "rows": [{"time": p.get("time"), "adv": p.get("advance"), "dec": p.get("decline"),
                      "ratio": _f(p.get("ad_ratio"))} for p in sample]}


def cannot_answer(ctx, reason):
    return {"tool": "cannot_answer", "reason": reason}


def _universe(a):
    u = str(a.get("universe") or "all").lower()
    return u if u in ("all", "watchlist", "setups") else "all"


def clean_conditions(raw):
    """Model output -> [{left, op, right|None, value|None, pct}] or ToolError. Never trusted:
    unknown fields, a missing right side, or more than 6 conditions are refused."""
    if isinstance(raw, dict):
        raw = [raw]
    if not raw:
        raise ToolError("No condition given.")
    out = []
    for c in list(raw)[:7]:
        c = dict(c or {})
        left, op = str(c.get("left") or "").lower(), str(c.get("op") or "").strip()
        right = str(c.get("right") or "").lower() or None
        if op == "==":
            op = "="
        if left not in FIELDS or (right and right not in FIELDS):
            raise ToolError("Can't watch '%s' — I know: %s" % (right if left in FIELDS else left, ", ".join(FIELDS)))
        if op not in ("=", ">", "<", ">=", "<="):
            raise ToolError("Unknown comparison '%s'" % op)
        val = _f(c.get("value"))
        if right is None and val is None:
            raise ToolError("'%s %s ?' — compare with a field or a number." % (left, op))
        out.append({"left": left, "op": op, "right": right, "value": None if right else val,
                    "pct": _f(c.get("pct")) or 0.0})
    if len(out) > 6:
        raise ToolError("Max 6 conditions in one question / alert.")
    return out


def stock_fields(ctx, need, only=None):
    """{sym: {field: value}} for the fields in `need` (computed lazily: futures ticks, bars and
    OI are read only when a condition uses them)."""
    res = _live_results(ctx)
    syms = [s for s in res if (only is None or s in only) and not _is_index(ctx, s)]
    hl = _day_hilo(ctx) if need & {"high", "low", "range_pct"} else {}
    pv = _prev_levels(ctx)[1] if need & {"pdh", "pdl"} else {}
    pace = _live_pace(ctx) if "pace" in need else {}
    fv5 = _fut_volume(ctx, 5)[0] if "fut_vol5" in need else {}
    v5 = _window_volume(ctx, 5)[0] if "vol5" in need else {}
    oi = _snap(ctx.state.OI_CACHE_LOCK, ctx.state.OI_CACHE) if "oi" in need else {}
    iv = _iv_values(ctx, 0)[0] if "iv" in need else {}
    out = {}
    for s in syms:
        r = res[s]
        px, op, pc = _f(r.get("price")), _f(r.get("day_open")), _f(r.get("pdc"))
        f = {"price": px, "open": op, "vwap": _f(r.get("vwap")), "pdc": pc, "pct": _f(r.get("intra_pct")),
             "rvol": _f(r.get("rvol")), "score": _score_num(r)}
        f["open_pct"] = (px / op - 1) * 100.0 if px and op else None
        f["gap"] = (op / pc - 1) * 100.0 if op and pc else None
        if s in hl and px:
            hi, lo = max(hl[s][0], px), min(hl[s][1], px)
            if op:                                   # the open is part of the day's range by definition
                hi, lo = max(hi, op), min(lo, op)
            f["high"], f["low"] = hi, lo
            f["range_pct"] = (hi / lo - 1) * 100.0 if lo else None
        if s in pv:
            f["pdh"], f["pdl"] = pv[s][0], pv[s][1]
        f["pace"], f["fut_vol5"], f["vol5"], f["iv"] = pace.get(s), fv5.get(s), v5.get(s), iv.get(s)
        if "oi" in need:
            o = oi.get(s)
            f["oi"] = _oi_pct(o) if isinstance(o, dict) else None
        out[s] = f
    return out


def cond_holds(f, c):
    """True / False, or None when a needed value is missing (the stock is skipped, not failed)."""
    L = f.get(c["left"])
    R = f.get(c["right"]) if c["right"] else c["value"]
    if L is None or R is None:
        return None
    if c["right"] and c.get("pct"):
        R = R * (1 + c["pct"] / 100.0)
    if c["op"] == "=":
        if c["right"] in PRICE_FIELDS or c["left"] in PRICE_FIELDS:
            return R != 0 and abs(L / R - 1) * 100.0 <= EQ_TOL_PCT
        return abs(L - R) <= max(abs(R) * EQ_TOL_PCT / 100.0, 1e-9)
    return {">": L > R, "<": L < R, ">=": L >= R, "<=": L <= R}[c["op"]]


def _cv(field, v):
    if v is None:
        return "—"
    if field in PRICE_FIELDS:
        return _px2(v)
    if field in ("pct", "open_pct", "gap", "range_pct", "oi"):
        return _p(v)
    if field in ("pace", "rvol", "fut_vol5", "vol5"):
        return _x(v)
    return "%.1f" % v


def cond_words(c):
    r = c["right"] or ("%g" % c["value"])
    if c["right"] and c.get("pct"):
        r = "%s %s%g%%" % (c["right"], "+" if c["pct"] > 0 else "−", abs(c["pct"]))
    op = {"=": "=", ">": ">", "<": "<", ">=": "≥", "<=": "≤"}[c["op"]]
    return "%s %s %s%s" % (c["left"], op, r, " (±%g%%)" % EQ_TOL_PCT if c["op"] == "=" else "")


def cond_values(f, c):
    """'open 152.40 = high 152.40' with the live numbers, for the reply."""
    op = {"=": "=", ">": ">", "<": "<", ">=": "≥", "<=": "≤"}[c["op"]]
    R = f.get(c["right"]) if c["right"] else c["value"]
    rr = c["right"] or ""
    if c["right"] and c.get("pct") and R is not None:
        R = R * (1 + c["pct"] / 100.0)
        rr = "%s%+g%%" % (c["right"], c["pct"])
    return ("%s %s %s %s %s" % (c["left"], _cv(c["left"], f.get(c["left"])), op, rr,
                                _cv(c["right"] or c["left"], R))).replace("  ", " ")


def screen(ctx, conditions, only=None, sector=None):
    """[(sym, fields)] matching ALL conditions, plus how many stocks lacked data."""
    need = {c["left"] for c in conditions} | {c["right"] for c in conditions if c["right"]} | {"price", "pct"}
    rows, missing = [], 0
    for s, f in stock_fields(ctx, need, only).items():
        if not _keep(ctx, s, sector):
            continue
        ok = [cond_holds(f, c) for c in conditions]
        if any(x is None for x in ok):
            missing += 1
            continue
        if all(ok):
            rows.append((s, f))
    return rows, missing


def live_value(ctx, symbols, fields):
    """The exact values asked for, one line per stock / index — the answer first, nothing to dig for."""
    need = set(fields) | {"price", "pct"}
    stocks = [s for s in symbols if not s.startswith("IDX:")]
    f_all = stock_fields(ctx, need, only=set(stocks)) if stocks else {}
    rows = []
    for s in symbols:
        if s.startswith("IDX:"):
            name = s[4:]
            f = index_fields(ctx, name)
            if f is None:
                rows.append({"sym": name, "missing": "no live index data yet"})
                continue
            bad = [x for x in fields if x not in f]
            rows.append({"sym": name, "f": f, "times": {}, "na": bad, "index": True})
            continue
        f = f_all.get(s)
        if f is None:
            rows.append({"sym": s, "missing": "no live data yet"})
            continue
        t = day_hilo_times(ctx, s) if {"high", "low"} & set(fields) else None
        times = {"high": t[1], "low": t[3]} if t else {}
        rows.append({"sym": s, "f": f, "times": times, "na": []})
    if all("missing" in r for r in rows):
        raise ToolError("No live data for %s yet." % ", ".join(r["sym"] for r in rows))
    return {"tool": "live_value", "rows": rows, "fields": fields, "as_of": _as_of(ctx)}


def live_screen(ctx, conditions, order_by, order, limit, sector, show):
    hit, missing = screen(ctx, conditions, sector=sector)
    hit = _scope(ctx, hit, lambda x: x[0])
    key = lambda x: (x[1].get(order_by) is None, -(x[1].get(order_by) or 0) if order == "top"   # noqa: E731
                     else (x[1].get(order_by) or 0))
    hit.sort(key=key)
    n = len(hit)
    if not hit:
        raise ToolError("No stock matches %s right now%s%s." % (" and ".join(cond_words(c) for c in conditions),
                                                               (" in " + _short(sector)) if sector else "",
                                                               _scope_hint(ctx)))
    hit = hit[:limit]
    show = [c for c in show if c not in ("price", "pct")]
    syms = [s for s, _ in hit]
    hm = ctx.now().hour * 60 + ctx.now().minute
    early = hm < 9 * 60 + 30 and any(c["left"] in ("high", "low", "open") or c["right"] in ("high", "low", "open")
                                     for c in conditions)
    return {"tool": "live_screen", "conditions": conditions, "order_by": order_by, "order": order, "sector": sector,
            "rows": [(s, f.get(order_by)) for s, f in hit], "fields": {s: f for s, f in hit}, "n": n,
            "missing": missing, "early": early, "show": show, "extra": _live_cols(ctx, syms, show),
            "as_of": _as_of(ctx)}


_IMPL = {"top_symbols": top_symbols, "level_breaks_history": level_breaks_history,
         "symbol_history": symbol_history, "sector_history": sector_history,
         "signal_performance": signal_performance, "live_top": live_top,
         "live_level_breaks": live_level_breaks, "live_vwap": live_vwap, "live_symbol": live_symbol,
         "current_candidates": current_candidates, "watchlist_status": watchlist_status,
         "recent_alerts": recent_alerts, "live_sector_rank": live_sector_rank,
         "market_mood": market_mood, "breadth_trend": breadth_trend, "cannot_answer": cannot_answer,
         "compare_symbols": compare_symbols, "live_screen": live_screen, "live_value": live_value}
assert set(_IMPL) == TOOL_NAMES


def scope_args(ctx, name, args, prev=None):
    """'From that list' + numeric filters -> scope dict, or None.
    prev = the chat's recent answers, most recent first: [{"symbols", "time", "q"}, ...]."""
    if name not in SCOPED_TOOLS:
        return None
    a = dict(args or {})
    n = _int(a, "from_previous", 0, 5, 0)
    only = None
    if n:
        if not prev or n > len(prev) or not prev[n - 1].get("symbols"):
            raise ToolError("There is no earlier list #%d to work from — I keep your last 5 answers "
                            "for 15 minutes. Ask the full question again." % n)
        only = set(prev[n - 1]["symbols"])
    allowed = HIST_FILTER_COLS if name in HIST_TOOLS else LIVE_FILTER_COLS
    fl = []
    for f in (a.get("filters") or [])[:MAX_FILTERS]:
        if not isinstance(f, dict):
            raise ToolError("Could not read the condition %r" % (f,))
        col = str(f.get("col") or "").lower()
        op = str(f.get("op") or "").strip()
        if col not in allowed:
            raise ToolError("Can't filter on '%s'. Use: %s" % (col, ", ".join(allowed)))
        if op not in _OPS:
            raise ToolError("Condition must use >, >=, < or <=")
        v = _f(f.get("value"))
        if v is None:
            raise ToolError("Condition on %s needs a number" % col)
        fl.append({"col": col, "op": op, "value": v})
    if not n and not fl:
        return None
    return {"from": n, "only": only, "filters": fl,
            "base_time": prev[n - 1].get("time") if n else None,
            "base_q": prev[n - 1].get("q") if n else None}


# ═════════════════════════════════════════════════════════════════════════════
# fix_route: code corrects the parts of the model's pick that a plain rule always gets
# right (dates, columns named in the question, today vs history, redundant 'from that').
# The model judges INTENT; code owns exact details. Each fix is logged, never hidden.
# ═════════════════════════════════════════════════════════════════════════════
_COL_WORDS = {"price": "price", "rvol": "rvol", "volume": "volume", "vol": "volume", "oi": "oi",
              "vwap": "vwap", "pace": "pace", "score": "score", "prox": "prox", "proximity": "prox",
              "streak": "streak", "sector": "sector", "state": "state"}
_COL_TRIGGER = re.compile(r"\b(with|and|saathe|sathe|saath|pan|bhi|show|plus|also|ane)\b", re.I)
_TODAY = re.compile(r"\b(today|todays|aaje|aaj|abhi|hmna|hamna|now|right now|live|intraday)\b", re.I)
_PAST = re.compile(r"\b(yesterday|yday|kal|gaye|days?|week|weeks|month|session|sessions|divas|"
                   r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{1,2}(st|nd|rd|th))\b", re.I)
_WATCH_W = re.compile(r"\b(watch\s*list|watchlist|kanban|my\s+list|mari\s+list|meri\s+list|watch\s+stocks?)\b", re.I)
_SETUP_W = re.compile(r"\b(setups?|flagged|candidates|scanner\s+stocks?)\b", re.I)
_FROM_WORDS = re.compile(r"\b(from (that|this|these|those|the|it)|of (these|those|them)|among|aama|ama|emathi|"
                         r"ema|inme|isme|usme|in that list|out of)\b", re.I)
_CORE = ("metric", "kind", "side", "window_min", "days", "end_day", "day", "sort_by", "only_flagged",
         "sector", "by", "symbol")


_POSTPOS = {"pan", "bhi", "saathe", "sathe", "saath"}     # Gujarati/Hindi: 'oi pan batavo' = 'also show oi'


def _cols_in_text(text):
    """Column words the user asked to SEE: after 'with / and / show', or just before a
    Gujarati/Hindi postposition ('rvol pan batavo', 'oi saathe'). In question order."""
    words = re.findall(r"[a-z&]+", (text or "").lower())
    out, on = [], False
    for i, w in enumerate(words):
        if _COL_TRIGGER.fullmatch(w):
            on = True
            if w in _POSTPOS:                       # look back over the column words just before it
                j = i - 1
                while j >= 0 and (words[j] in _COL_WORDS or words[j] == "and"):
                    c = _COL_WORDS.get(words[j])
                    if c and c not in out and not (j and words[j - 1] in ("fut", "futures")):
                        out.append(c)
                    j -= 1
            continue
        c = _COL_WORDS.get(w)
        if on and c and c not in out and not (i and words[i - 1] in ("fut", "futures")):
            out.append(c)
    return out


def fix_route(ctx, text, name, args, prev=None):
    """(name, args, fixes). Never raises: a fix that cannot apply is skipped."""
    args, fixes = dict(args or {}), []
    try:
        # 1. today never means history (model sent 'gainers today by ...' to days=1)
        if name in ("top_symbols", "level_breaks_history") and _TODAY.search(text or "") \
                and not _PAST.search(text or "") and not args.get("end_day") and not args.get("day") \
                and _int(args, "days", 1, MAX_DAYS, 1) == 1:
            keep = {k: args[k] for k in ("order", "limit", "sector", "unsupported", "from_previous", "filters")
                    if k in args}
            if name == "top_symbols":
                m = {"gain": "pct"}.get(str(args.get("metric") or "gain").lower(), str(args.get("metric") or "pct"))
                name, args = "live_top", dict(keep, metric=m)
            else:
                name, args = "live_level_breaks", dict(keep, kind=args.get("kind") or "prev_high")
            fixes.append("today -> live tool")
        # 2. a past year for a day that has not come yet this year -> this year
        for k in ("end_day", "day"):
            v = str(args.get(k) or "")
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v) and v[:4] < ctx.today[:4]:
                cand = ctx.today[:4] + v[4:]
                if cand < ctx.today:
                    args[k] = cand
                    fixes.append("%s year %s -> %s" % (k, v[:4], cand[:4]))
        # 2b. high <-> low / up <-> down flipped by the model
        if name == "live_top" and args.get("metric"):
            m2, note = direction_fix(text, str(args["metric"]))
            if note:
                args["metric"] = m2
                fixes.append(note)
        # 2c. 'watchlist / setups' said but not passed
        if name in ("live_top", "live_screen") and not args.get("universe"):
            if _WATCH_W.search(text or ""):
                args["universe"] = "watchlist"
                fixes.append("universe watchlist (your words)")
            elif _SETUP_W.search(text or ""):
                args["universe"] = "setups"
                fixes.append("universe setups (your words)")
        # 3. columns the user named but the model dropped
        allowed = {x.lower() for x in _schema_enum(name, "show")}
        if allowed:
            have = [str(x).lower() for x in (args.get("show") or [])] if not isinstance(args.get("show"), str) \
                else [x.strip().lower() for x in args["show"].split(",")]
            add = [c for c in _cols_in_text(text) if c in allowed and c not in have]
            if add:
                args["show"] = have + add
                fixes.append("show +" + ",".join(add))
        # 4. 'from_previous' on the SAME question as that answer = a refresh, not a subset
        n = _int(args, "from_previous", 0, 5, 0)
        if n and prev and n <= len(prev) and prev[n - 1].get("tool") == name \
                and not args.get("filters") and not _FROM_WORDS.search(text or ""):
            clean = validate(ctx, name, args)
            pa = prev[n - 1].get("args") or {}
            if all(clean.get(k) == pa.get(k) for k in _CORE if k in clean or k in pa):
                args.pop("from_previous", None)
                fixes.append("from_previous dropped (same question, fresh data)")
    except ToolError:
        pass
    except Exception:
        pass
    return name, args, fixes


def result_symbols(r):
    """Symbols of a list answer, in display order (stored for 'from that' follow-ups)."""
    out = []
    for x in (r or {}).get("rows") or []:
        if isinstance(x, (tuple, list)) and x:
            out.append(x[0])
        elif isinstance(x, dict) and x.get("sym"):
            out.append(x["sym"])
    return out


def _schema_enum(name, key):
    for t in TOOLS:
        if t["name"] == name:
            p = t["parameters"]["properties"].get(key) or {}
            return (p.get("items") or {}).get("enum") or p.get("enum") or []
    return []


def _unsupported(name, raw, clean):
    """Parts of the question this answer does NOT cover -- from the model (rule 13) AND
    from code: whatever validation had to drop or clamp. Shown in the reply, so the bot
    never quietly answers a different question."""
    raw = dict(raw or {})
    out = []
    u = str(raw.get("unsupported") or "").strip()
    if u and name != "cannot_answer":
        out.append(u[:120])
    lim = _f(raw.get("limit"))
    if lim and lim > MAX_LIMIT and "limit" in clean:
        out.append("max %d rows" % MAX_LIMIT)
    w = _f(raw.get("window_min"))
    if w and w > 0 and "window_min" in clean and not clean["window_min"]:
        out.append("last %d min for %s (whole day shown)" % (w, clean.get("metric", name)))
    if w and clean.get("window_min") and w > clean["window_min"]:
        out.append("max %d min window for %s (%d min shown)" % (clean["window_min"], clean.get("metric", name),
                                                                clean["window_min"]))
    if "show" in clean:
        allowed = {x.lower() for x in _schema_enum(name, "show")}
        want = raw.get("show") or []
        if isinstance(want, str):
            want = want.replace(" and ", ",").split(",")
        bad = [str(x).strip() for x in want if str(x).strip() and str(x).strip().lower() not in allowed]
        if bad:
            out.append("column %s here" % ", ".join(bad))
        if len([x for x in want if str(x).strip()]) > MAX_SHOW:
            out.append("max %d extra columns" % MAX_SHOW)
    return out


def universe_symbols(ctx, universe):
    """Live members of 'watchlist' (Kanban + watch list, read NOW, so dashboard edits count at
    once) or 'setups' (scanner-flagged, long or short). None = all stocks."""
    st = ctx.state
    if universe == "watchlist":
        out = set()
        try:
            with st.KANBAN_LOCK:
                out |= {k.get("symbol") for k in st.KANBAN if isinstance(k, dict) and k.get("symbol")}
        except Exception:
            pass
        try:
            with st.WATCH_LOCK:
                out |= set(st.WATCHLIST)
        except Exception:
            pass
        return out
    if universe == "setups":
        out = set()
        for side in ("LONG", "SHORT"):
            for sym, r in _live_results(ctx, side, required=False).items():
                if any(f in str(r.get("state") or "").upper() for f in _FLAG_STATES):
                    out.add(sym)
        return out
    return None


# Direction pairs: the model sometimes picks the mirror ('new day low' -> new_high, 06-11 09:47).
# Code compares the pick with the user's own words and flips it back. Both kinds of words = leave.
MIRROR = {"new_high": "new_low", "pdh_up": "pdl_down", "near_high": "near_low", "near_pdh": "near_pdl",
          "pace_high": "pace_low", "vwap_up": "vwap_down", "open_up": "open_down",
          "day_high_up": "day_low_down"}
MIRROR.update({v: k for k, v in list(MIRROR.items())})
_LEVEL_HIGH = re.compile(r"\b(high|highs|top|uchu|upar\s*nu)\b", re.I)
_LEVEL_LOW = re.compile(r"\b(low|lows|bottom|nichu)\b", re.I)
_MOVE_UP = re.compile(r"\b(above|over|up|upar|upper|rises?|rising|climbs?|jumps?|crosses\s+above|goes\s+above|"
                      r"bullish)\b", re.I)
_MOVE_DOWN = re.compile(r"\b(below|under|beneath|down|niche|neeche|nicha|falls?|falling|drops?|slips?|sinks?|"
                        r"crosses\s+below|"
                        r"goes\s+below|bearish|tute|tut[iy]\w*)\b", re.I)
_LEVEL_METRICS = {"new_high", "new_low", "pdh_up", "pdl_down", "near_high", "near_low", "near_pdh", "near_pdl",
                  "pace_high", "pace_low", "day_high_up", "day_low_down"}
_HIGH_SIDE = {"new_high", "pdh_up", "near_high", "near_pdh", "pace_high", "vwap_up", "open_up", "day_high_up"}


def direction_fix(text, metric):
    """(metric, note) — the mirror metric when the user's words clearly say the other side."""
    if metric not in MIRROR:
        return metric, None
    if metric in _LEVEL_METRICS:
        hi, lo = bool(_LEVEL_HIGH.search(text or "")), bool(_LEVEL_LOW.search(text or ""))
    else:
        hi, lo = bool(_MOVE_UP.search(text or "")), bool(_MOVE_DOWN.search(text or ""))
    want_high = hi and not lo
    want_low = lo and not hi
    if (want_low and metric in _HIGH_SIDE) or (want_high and metric not in _HIGH_SIDE):
        return MIRROR[metric], "direction %s -> %s (your words)" % (metric, MIRROR[metric])
    return metric, None


def run(ctx, name, args, prev=None, only=None, all_rows=False):
    """Validate + execute. Returns (result_dict, error_text). Never raises:
    one bad question must never kill the listener thread.
    only = restrict to these symbols; all_rows = no row limit (the alert checker needs every match)."""
    t0 = time.time()
    try:
        clean = validate(ctx, name, args)
        if all_rows and "limit" in clean:
            clean["limit"] = 10 ** 6
        sc = scope_args(ctx, name, args, prev)
        uni = universe_symbols(ctx, str((args or {}).get("universe") or "all").lower()) \
            if name in ("live_top", "live_screen") else None
        if uni is not None:
            if not uni:
                raise ToolError("Your %s is empty right now." % ("watchlist" if args.get("universe") == "watchlist"
                                                               else "scanner setups list"))
            only = set(uni) if only is None else set(only) & set(uni)
        if only is not None:
            sc = dict(sc or {"from": 0, "filters": [], "base_time": None, "base_q": None})
            sc["only"] = set(only) if sc.get("only") is None else (set(sc["only"]) & set(only))
        ctx._tl.scope = sc
        ctx._tl.bars = None                      # memo slot: bars are read once per call
        try:
            out = _IMPL[name](ctx, **{k: v for k, v in clean.items() if k != "universe"})
        finally:
            ctx._tl.scope = None
            del ctx._tl.bars
        if sc:
            out["scope"] = {k: v for k, v in sc.items() if k != "only"}
            out["scope"]["base_n"] = len(sc["only"]) if sc["only"] is not None else None
        out["unsupported"] = _unsupported(name, args, clean)
        out["_ms"] = int((time.time() - t0) * 1000)
        out["_args"] = dict(clean, **({"from_previous": sc["from"], "filters": sc["filters"]} if sc else {}))
        return out, None
    except ToolError as e:
        return None, str(e)
    except sqlite3.Error as e:
        return None, "Database error: %s" % e
    except Exception as e:                       # bug in a tool: say so, keep running
        return None, "Internal error in %s: %s: %s" % (name, type(e).__name__, e)


# ═════════════════════════════════════════════════════════════════════════════
# Reply text. Built by code from the numbers -- the LLM never writes a number.
# Plain text; ai_chat.py sends it inside <pre> so the columns line up.
# ═════════════════════════════════════════════════════════════════════════════
def _p(v, d=2):
    return "—" if v is None else ("%+.*f%%" % (d, v))


def _n(v):
    if v is None:
        return "—"
    v = float(v)
    if v >= 1e7:
        return "%.2fCr" % (v / 1e7)
    if v >= 1e5:
        return "%.1fL" % (v / 1e5)
    return "%.0f" % v


def _x(v):
    return "—" if v is None else "%.2fx" % v


def _px(v):
    return "—" if v is None else "%.2f" % v


def _s(v):
    return "—" if v in (None, "") else str(v)


COL_LABEL = {"price": "PRICE", "pct": "DAY%", "open": "OPEN%", "gap": "GAP%", "rvol": "RVOL", "volume": "VOL", "oi": "OI%",
             "vwap": "VWAP%", "pace": "PACE", "state": "STATE", "sector": "SECTOR",
             "streak": "STRK", "score": "SCORE", "prox": "PROX", "fut_vol5": "FUTVOL5m", "vol5": "VOL5m"}
COL_FMT = {"price": _px, "pct": _p, "open": _p, "gap": _p, "rvol": _x, "volume": _n, "oi": lambda v: _p(v, 1), "vwap": _p,
           "pace": _x, "state": lambda v: _s(v)[:14], "sector": lambda v: _s(v).replace("NIFTY ", "")[:10],
           "streak": _s, "score": _s, "prox": lambda v: _s(v)[:12], "fut_vol5": _x, "vol5": _x}


def table(headers, rows):
    """Left-align text, right-align numbers; widths from the data."""
    rows = [[_s(c) if not isinstance(c, str) else c for c in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]
    num = [i > 0 and all((r[i][:1] in "+-0123456789." or r[i] == "—") for r in rows) for i in range(len(headers))]

    def line(cells):
        return "  ".join((c.rjust(w[i]) if num[i] else c.ljust(w[i])) for i, c in enumerate(cells)).rstrip()
    return "\n".join([line(headers)] + [line(r) for r in rows])


def _with_extra(r, headers, rows, syms):
    show, extra = r.get("show") or [], r.get("extra") or {}
    if not show:
        return headers, rows
    headers = headers + [COL_LABEL[c] for c in show]
    rows = [row + [COL_FMT[c]((extra.get(s) or {}).get(c)) for c in show] for row, s in zip(rows, syms)]
    return headers, rows


def _uni(r):
    u = (r.get("_args") or {}).get("universe")
    return {"watchlist": " · watchlist only", "setups": " · setups only"}.get(u, "")


def _sec(r):
    return ((" · " + r["sector"].replace("NIFTY ", "")) if r.get("sector") else "") + _uni(r)


def _asof(r):
    return ("\nas of scan %s" % r["as_of"]) if r.get("as_of") else ""


def _list(r, title, headers, rows, syms, empty="No matches."):
    if not rows:
        return title + "\n" + empty
    h, rr = _with_extra(r, headers, rows, syms)
    return title + "\n" + table(h, rr)


def _px2(v):
    if v is None:
        return "—"
    return "%.1f" % v if abs(v) >= 1000 else "%.2f" % v


def _arrow(then, now):
    return "%s → %s" % (_px2(then), _px2(now))


def _extras_line(r, sym):
    show, ex = r.get("show") or [], (r.get("extra") or {}).get(sym) or {}
    return "  ".join("%s %s" % (COL_LABEL[c], COL_FMT[c](ex.get(c))) for c in show)


def _cards(title, items, notes=None, empty="No matches.", tail=""):
    """Two lines per stock:  rank SYMBOL  MAIN  [extras]  /  then -> now (+ detail).
    Phone-friendly: the wide details sit on their own line."""
    if not items:
        return title + "\n" + empty + tail
    ws = max(len(i[1]) for i in items)
    wm = max(len(i[2]) for i in items)
    wr = len(str(len(items)))
    lines = [title, ""]
    for rank, sym, main, extra, line2 in items:
        l1 = "%s %s  %s" % (str(rank).rjust(wr), sym.ljust(ws), main.rjust(wm))
        lines.append((l1 + ("  " + extra if extra else "")).rstrip())
        for part in (line2 or "").split("\n"):
            if part:
                lines.append(" " * (wr + 1) + part)
    if notes:
        lines += ["", "📌 Notes"] + ["• " + n for n in notes]
    return "\n".join(lines) + tail


def _window_span(r):
    """'14:06 → 14:21' from the most common base time of the rows."""
    inf, kind, w = r.get("info") or {}, r.get("kind"), r.get("window_min") or 0
    idx = {"move": 2, "extreme": 2, "spike": 4, "oi": 4}.get(kind)
    if not inf or idx is None:
        return ""
    hh = [v[idx].strftime("%H:%M") for v in inf.values()]
    t0 = max(set(hh), key=hh.count)
    t1 = (datetime.strptime(t0, "%H:%M") + timedelta(minutes=w)).strftime("%H:%M")
    return " (%s → %s)" % (t0, t1)


def _format_list(r):
    t = r["tool"]
    notes = r.get("notes")
    if t == "live_top":
        w, kind, pair = r.get("window_min") or 0, r.get("kind") or "", r.get("pair") or {}
        m = r["metric"]
        lab = {"price": "by price", "pct": "gainers" if r["order"] == "top" else "losers",
               "open": ("gainers" if r["order"] == "top" else "losers") + " since open",
               "gap": "gap-up" if r["order"] == "top" else "gap-down",
               "new_high": "new day highs", "new_low": "new day lows",
               "volume": "volume", "rvol": "RVOL", "oi": "OI change", "pace": "PACE",
               "fut_volume": "futures volume", "pace_cross": "PACE band crossed", "vwap": "distance from VWAP",
               "score": "long score", "iv": "ATM IV", "near_low": "near today's low",
               "near_high": "near today's high", "near_vwap": "near VWAP", "near_pdh": "near yesterday's high",
               "near_pdl": "near yesterday's low", "pace_change": "PACE changed",
               "pace_low": "PACE changed by a new day LOW",
               "pace_high": "PACE changed by a new day HIGH"}.get(m) or _CROSS_NAME.get(m, m)
        lab = {"futspike": "futures volume rise", "pacewin": "PACE gained",
               "ivwin": "IV change"}.get(kind, lab)
        if kind == "move":
            lab = "jump" if r["order"] == "top" else "fall"
        elif kind == "spike":
            lab = "volume spike"
        head = "Top" if (r["order"] == "top" or m in ("pct", "open", "gap") or kind in ("move", "extreme")) \
            else "Bottom"
        if kind == "pacecross":
            head = "%d stocks" % r["n"] if r["order"] == "top" else "Smallest"
        if kind == "pacechg":
            head = "%d stocks" % r["n"]
            lab += " · newest first" if r["order"] == "top" else " · earliest first"
        if kind == "cross":
            head = "%d stocks" % r["n"]
            lab += " · newest first" if r["order"] == "top" else " · earliest first"
        when = ("last %d min%s" % (w, _window_span(r))) if w else ("today" if m != "open" else "today")
        if r.get("mins_used") and w and r["mins_used"] < w:
            when += ", only %d min of bars so far" % r["mins_used"]
        if kind in ("cross", "pacecross", "pacechg") and r["order"] == "top":
            title = "%d stock%s %s · %s%s" % (r["n"], "" if r["n"] == 1 else "s", lab, when, _sec(r))
        else:
            title = "%s %d %s · %s%s" % (head, len(r["rows"]), lab, when, _sec(r))
        fmt = {"price": _px2, "pct": _p, "open": _p, "gap": _p, "new_high": _p, "new_low": _p, "oi": _p, "volume": _n,
               "rvol": _x, "pace": _x, "fut_volume": _n, "pace_cross": _x, "vwap": _p,
               "score": lambda v: "%.0f%%" % v, "iv": lambda v: "%.1f" % v}.get(m, _p)
        items = []
        for i, (s, v) in enumerate(r["rows"]):
            then, now = pair.get(s, (None, None))
            main = _x(v) if kind in ("spike", "futspike") else fmt(v)
            inf = (r.get("info") or {}).get(s)
            if kind in ("pacewin", "pacecross"):
                main = "+%.2fx" % v
            if kind == "cross" and inf:
                main = _p(inf[0])
            if kind == "near":
                main = "%.2f%%" % v
            if kind == "pacechg" and inf:
                main = "%.2f → %.2f" % (inf[0], inf[1])
            if kind == "ivwin":
                main = "%+.1f pts" % v
            if kind == "futspike" and inf:
                line2 = "fut vol prev %d min %s → last %d min %s" % (w, _n(inf[1]), w, _n(inf[0]))
            elif kind in ("pacewin", "pacecross") and inf:
                line2 = "PACE %.2f → %.2f · %s → %s" % (inf[0], inf[1], inf[2] or "—", inf[3] or "—")
                if inf[4]:
                    line2 += " · crossed %s" % inf[4]
            elif kind == "cross" and inf:
                line2 = "level %s · crossed %s" % (_px2(inf[1]), inf[4])
            elif kind == "pacechg" and inf:
                line2 = "changed %s" % inf[2]
                if inf[6] in ("LOW", "HIGH", "BOTH"):
                    line2 += " · %s break%s" % (inf[6], (" " + _px2(inf[7])) if inf[7] else "")
                line2 += "\n" + ("flat %s before" % _hm_dur(inf[3]) if inf[3] is not None else
                                 "unchanged since %s (bot started then — earlier not seen)" % inf[5])
            elif kind == "near" and inf:
                lvn = {"near_low": "today's low", "near_high": "today's high", "near_vwap": "VWAP",
                       "near_pdh": "yday high", "near_pdl": "yday low"}[m]
                line2 = "%s %s · price %s%s" % (lvn, _px2(inf[0]), _px2(inf[1]),
                                               " · AT / BREAKING" if v < 0.005 else "")
            elif kind == "vwap":
                line2 = "VWAP %s → now %s" % (_px2(then), _px2(now))
            elif kind == "score" and inf:
                line2 = inf[1] or "no flag"
            elif kind in ("iv", "ivwin"):
                line2 = ("IV %.1f → %.1f" % (then, now)) if then is not None else ""
            elif kind == "futvol":
                line2 = "futures, today so far"
            elif kind == "open":
                line2 = "open %s → now %s" % (_px2(then), _px2(now))
            elif kind == "gap":
                line2 = "prev close %s → open %s" % (_px2(then), _px2(now))
            elif kind == "extreme":
                ago = (r.get("info") or {}).get(s, (0, 0, 0, None))[3]
                line2 = "%s · new day %s %s" % (_arrow(then, now), "high" if m == "new_high" else "low",
                                                "now" if not ago else "%d min ago" % ago)
            elif kind in ("spike", "oi") or (not w and m != "pct"):
                mv = _p((now / then - 1) * 100.0) if then and now else "—"
                line2 = "%s (%s)" % (_arrow(then, now), mv)
                if kind == "oi":
                    line2 = "fut " + line2
            else:
                line2 = _arrow(then, now)
            if then is None and now is None and kind not in ("futspike", "pacewin", "pacecross", "score",
                                                            "futvol", "iv"):
                line2 = ""
            # Every stock row: price on line 2 (then -> now when we have it), day % on line 1.
            pnow, dpct = (r.get("px") or {}).get(s, (None, None))
            has_px = (kind in ("move", "spike", "oi", "extreme", "open", "gap", "near") or (
                not w and m in ("price", "pct", "volume", "rvol", "pace", "oi"))) and any(pair.get(s, (None, None)))
            if not has_px:
                then_px = pair.get(s, (None, None))[0] if kind in ("pacewin", "pacecross", "cross", "pacechg",
                                                                    "futspike", "futvol", "score", "iv",
                                                                    "ivwin") else None
                if kind == "vwap":
                    then_px = None
                pl = ("price %s" % _arrow(then_px, pnow)) if then_px else ("price %s" % _px2(pnow))
                line2 = pl + (" · " + line2 if line2 else "")
            extra = _extras_line(r, s)
            if not (m == "pct" and not w):
                extra = ("day %s  " % _p(dpct) + extra).strip()
            ph = (r.get("pace_hist") or {}).get(s)
            if kind == "pacechg":                       # line 2 already says when / flat: only the count
                n_today = inf[4] if inf else 0
                ph = ("PACE changed %d× today" % n_today) if n_today > 1 else None
            if ph:
                line2 = (line2 + "\n" + ph) if line2 else ph
            items.append((i + 1, s, main, extra, line2))
        tail = ""
        if kind == "spike":
            tail = "\nSPIKE = volume/min now ÷ its volume/min earlier today"
        elif kind == "futspike":
            tail = "\nRISE = futures volume last %d min ÷ the %d min before (live futures ticks)" % (w, w)
        elif kind in ("pacewin", "pacecross", "pacechg"):
            tail = "\nPACE = the dashboard's live PACE, sampled every minute. PACE only rises during the day."
            gaps = (pace_history_any(r) or {}).get("gaps")
            if gaps:
                tail += "\nApp was off %s: PACE changes then are not seen." % ", ".join("%s–%s" % g for g in gaps)
        elif kind == "cross":
            tail = "\nFrom 1-min samples of price and %s; still beyond the level now." % (
                "VWAP" if m.startswith("vwap") else "the level")
        out = _cards(title, items, notes, tail=tail)
        return out + _asof(r)
    if t == "top_symbols":
        m, pair = r["metric"], r.get("pair") or {}
        lab = {"gain": "gainers" if r["order"] == "top" else "losers", "volume": "by avg volume",
               "oi": "by OI change", "pace": "by max PACE", "rvol": "by max RVOL"}[m]
        fmt = {"gain": _p, "oi": _p, "volume": _n, "pace": _x, "rvol": _x}[m]
        if r["days"]:
            span = "last %d session%s (%s → %s)" % (r["days"], "" if r["days"] == 1 else "s",
                                                    r["from"][5:], r["to"][5:])
        else:
            span = "end-of-day OI %s → %s" % (r["from"][5:], r["to"][5:])
        head = "Top" if r["order"] == "top" or m == "gain" else "Lowest"     # 'losers' already says it
        title = "%s %d %s · %s%s" % (head, len(r["rows"]), lab, span, _sec(r))
        items = []
        for i, (s, v) in enumerate(r["rows"]):
            if s in pair:
                c0, c1 = pair[s]
                line2 = "close %s" % _arrow(c0, c1)
                if m != "gain" and c0:
                    line2 += " (%s)" % _p((c1 / c0 - 1) * 100.0)
            else:
                line2 = _short(_secmap_static(r, s))
            items.append((i + 1, s, fmt(v), _extras_line(r, s), line2))
        tail = ("\n(extra columns are values on %s)" % r["to"]) if r.get("show") else ""
        return _cards(title, items, notes, tail=tail)
    if t == "level_breaks_history":
        hi = r["kind"] == "prev_high"
        title = "%d stocks %s previous day's %s on %s%s" % (
            r["n"], "above" if hi else "below", "HIGH" if hi else "LOW", r["day"][5:], _sec(r))
        items = [(i + 1, x["sym"], "%.2f%%" % x["ext_pct"], _extras_line(r, x["sym"]),
                  "%s %s → %s %s · close %s %s" % ("prev high" if hi else "prev low", _px2(x["lvl"]),
                                                  "high" if hi else "low", _px2(x["x"]), _px2(x["close"]),
                                                  "held ✓" if x["closed_beyond"] else "back inside"))
                 for i, x in enumerate(r["rows"])]
        return _cards(title, items, notes, "None.")
    if t == "live_level_breaks":
        hi = r["kind"] == "prev_high"
        title = "%d stocks %s yesterday's %s (%s)%s" % (
            r["n"], "above" if hi else "below", "HIGH" if hi else "LOW", r["prev_day"][5:], _sec(r))
        items = [(i + 1, x["sym"], "%.2f%%" % x["dist_pct"], _extras_line(r, x["sym"]),
                  "%s %s → now %s · day %s" % ("yday high" if hi else "yday low", _px2(x["level"]),
                                              _px2(x["price"]), _p(x["pct"])))
                 for i, x in enumerate(r["rows"])]
        tail = ""
        if r.get("missing_levels"):
            ms = r.get("missing_syms") or []
            k = r.get("kite") or {}
            tail = "\nNo %s bar yet for %d stocks: %s%s" % (
                r["prev_day"][5:], len(ms), ", ".join(ms[:6]), (" (+%d)" % (len(ms) - 6)) if len(ms) > 6 else "")
            tail += ("\n(fetching them from Kite — ask again in ~30 s)" if k.get("running") or k.get("day") is None
                     else "\n(Kite had no bar for %d of them)" % len(k.get("failed") or []) if k.get("failed") else "")
        return _cards(title, items, notes, "None.", tail) + _asof(r)
    if t == "live_vwap":
        title = "%d stocks %s VWAP%s · furthest first" % (r["n"], r["side"], _sec(r))
        items = [(i + 1, x["sym"], _p(x["dist_pct"]), _extras_line(r, x["sym"]),
                  "VWAP %s → now %s · day %s" % (_px2(x.get("vwap")), _px2(x["price"]), _p(x["pct"])))
                 for i, x in enumerate(r["rows"])]
        return _cards(title, items, notes, "None.") + _asof(r)
    if t == "current_candidates":
        pair = r.get("pair") or {}
        title = "%d %s stocks flagged by your scanner%s · by conviction" % (r["n"], r["side"], _sec(r))
        items = []
        for i, x in enumerate(r["rows"]):
            then, now = pair.get(x["sym"], (None, None))
            items.append((i + 1, x["sym"], "%s %s" % (x["tier"], x["score"]),
                          ("RVOL %s  " % _x(x["rvol"]) + _extras_line(r, x["sym"])).strip(),
                          "%s · %s · day %s" % (x["state"][:14], _arrow(then, now), _p(x["pct"]))))
        out = _cards(title, items, notes, "Nothing flagged right now.")
        return out + "\nYour rules' flags, not advice." + _asof(r)
    if t == "watchlist_status":
        title = "Watchlist · %d stocks · by %s%s%s" % (
            r["n"], r["sort_by"], "" if r["side"] == "BOTH" else " · " + r["side"],
            " · flagged only" if r["only_flagged"] else "")
        items = []
        for i, x in enumerate(r["rows"]):
            main = {"score": _s(x["score_txt"]), "prox": _s(x["prox"]), "oi": _p(x["oi"], 1),
                    "pct": _p(x["pct"]), "rvol": _x(x["rvol"])}[r["sort_by"]]
            prox = ("%s %s" % (x["prox"], x["prox_name"])).strip() if x["prox"] is not None else "—"
            l2 = "price %s · day %s · %s %s · %s · prox %s" % (
                _px2(x.get("price")), _p(x["pct"]), "S" if x["dir"] == "SHORT" else "L", x["stage"],
                _s(x["state"])[:14], prox)
            items.append((i + 1, x["sym"], main, _extras_line(r, x["sym"]), l2))
        return _cards(title, items, None, "Nothing matches.") + _asof(r)
    if t == "live_sector_rank":
        w = r["window_min"]
        lab = {"price": "avg % vs yesterday's close", "open": "avg % since today's open",
               "breadth": "% of stocks up", "oi": "avg OI change"}[r["metric"]]
        when = ("last %d min" % w) if w else ("" if r["metric"] == "open" else "today")
        title = "%s %d sectors by %s%s" % ("Top" if r["order"] == "top" else "Weakest", len(r["rows"]), lab,
                                          (" · " + when) if when else "")
        items = []
        for i, x in enumerate(r["rows"]):
            main = ("%.0f%% up" % (100.0 * x["up"] / x["n"])) if r["metric"] == "breadth" else _p(x["avg"])
            l2 = "%d/%d up · top %s %s" % (x["up"], x["n"], x["best"], _p(x["best_v"], 1))
            if x.get("index_pct") is not None:
                l2 += " · index %s" % _p(x["index_pct"])
            items.append((i + 1, _short(x["sector"])[:12], main, "", l2))
        tail = "\nEqual-weight average of F&O stocks." + _sector_tail(r) + ("\n" + r["note"] if r.get("note") else "")
        return _cards(title, items, None, tail=tail) + _asof(r)
    return None


def _sector_tail(r):
    bits = []
    if r.get("n") and r["n"] > len(r["rows"]):
        bits.append("%d of %d sectors shown" % (len(r["rows"]), r["n"]))
    if r.get("left_out"):
        bits.append("left out (< %d stocks or catch-all): %s" % (MIN_SECTOR_N, ", ".join(r["left_out"])))
    return "".join("\n" + b for b in bits)


def pace_history_any(r):
    return r.get("pace_meta")


def _secmap_static(r, s):
    return (r.get("secs") or {}).get(s, "")


def _format(r):
    t = r["tool"]
    x = _format_list(r)
    if x is not None:
        return x
    if t == "cannot_answer":
        return ("I can't answer that from the data. " + (r.get("reason") or "")).strip() + \
               "\nSend /help to see what I can answer."
    if t == "symbol_history":
        rows = [[x["day"][5:], _px(_f(x["close"])), _p(_f(x["day_pct"])), _n(_f(x["volume"])),
                 _p(_f(x["oi_sess"]), 1), _x(_f(x["max_pace"])), _x(_f(x["rvol_max"]))] for x in r["rows"]]
        return "%s · last %d sessions\n%s%s" % (r["symbol"], len(rows), table(
            ["DAY", "CLOSE", "DAY%", "VOL", "OI%", "PACE", "RVOL"], rows),
            ("\n" + r["note"]) if r.get("note") else "")
    if t == "sector_history":
        rows = [[str(i + 1), x["sector"].replace("NIFTY ", "")[:12], _p(x["avg"]), "%d/%d" % (x["up"], x["n"]),
                 "%s %s" % (x["best"], _p(x["best_pct"], 1))] for i, x in enumerate(r["rows"])]
        return "%s %d sectors by avg gain · last %d sessions (%s → %s)\n%s\nEqual-weight average of F&O stocks.%s%s" % (
            "Top" if r.get("order", "top") == "top" else "Weakest", len(r["rows"]),
            r["days"], r["from"], r["to"], table(["#", "SECTOR", "AVG", "UP", "BEST"], rows),
            _sector_tail(r), ("\n" + r["note"]) if r.get("note") else "")
    if t == "signal_performance":
        head = "Signal performance by %s%s · %d sessions (%s → %s) · side %s" % (
            r["by"], (" = " + r["value"]) if r["value"] else "", r["sessions"], r["from"], r["to"], r["side"])
        lines = []
        for g in r["groups"]:
            if g.get("too_few"):
                lines.append("%s: only %d signals on %d days — too few to judge." % (g["group"], g["n"], g["days"]))
                continue
            s = ("%s: n=%d on %d days\n  avg 15m %s · 30m %s · 60m %s · 60m>0: %.0f%%"
                 % (g["group"], g["n"], g["days"], _p(g["m15"]), _p(g["m30"]), _p(g["m60"]), g["pos_share"]))
            if g.get("ci_lo") is not None:
                s += "\n  60m 90%% range %s to %s" % (_p(g["ci_lo"]), _p(g["ci_hi"]))
            if g.get("win_share") is not None:
                s += " · target-first %.0f%% of %d" % (g["win_share"], g["n_decided"])
            lines.append(s)
        return head + "\n" + "\n".join(lines) + \
            "\nNote: %s. Past behaviour, not a prediction. No costs included." % r["note"]
    if t == "live_symbol":
        L = ["%s%s" % (r["symbol"], (" · " + r["sector"]) if r["sector"] else "")]
        line = "Price %s  day %s" % (_px2(r["price"]), _p(r["pct"]))
        if r.get("open"):
            line += "  open %s → now %s" % (_px2(r["open"]), _px2(r["price"]))
        L.append(line)
        hl = r.get("hilo")
        if hl:
            L.append("Day   high %s (%s) · low %s (%s) · range %s" % (
                _px2(hl[0]), hl[1], _px2(hl[2]), hl[3], _p((hl[0] / hl[2] - 1) * 100.0) if hl[2] else "—"))
        pv = r.get("prev")
        if pv or r.get("pdc"):
            L.append("Prev  high %s · low %s · close %s" % (_px2(pv[0]) if pv else "—", _px2(pv[1]) if pv else "—",
                                                         _px2(r.get("pdc"))))
        L.append("VWAP %s (%s)  RVOL %s" % (_px2(r["vwap"]), _p(r["vwap_dist"]), _x(r["rvol"])))
        fl = r.get("fields") or {}
        extra = [("gap", "gap"), ("open_pct", "since open"), ("fut_vol5", "fut vol 5m"), ("vol5", "stock vol 5m"),
                 ("iv", "IV")]
        bits = ["%s %s" % (lab, _cv(k, fl.get(k))) for k, lab in extra if fl.get(k) is not None]
        if bits:
            L.append(" · ".join(bits))
        L.append("")
        pc = lambda v: "—" if v is None else "%.0f%%" % v                  # noqa: E731
        L.append("Scores  LONG  %-4s %-8s %s" % (pc(r["long_pct"]), "(%s)" % _s(r["score"]), r["state"] or "—"))
        L.append("        SHORT %-4s %-8s %s" % (pc(r["short_pct"]),
                                                "(%s)" % r["short_score"] if r.get("short_score") else "",
                                                r["short_state"] or "—"))
        if r.get("stronger"):
            L.append("→ stronger side now: %s (scanner score)" % r["stronger"])
        L.append("OI pulse  long %s · short %s" % (_s(r["pulse_long"] and round(r["pulse_long"])),
                                                   _s(r["pulse_short"] and round(r["pulse_short"]))))
        L.append("OI  from open %s · 15m %s · 30m %s · streak %s" % (
            _p(r["oi_open"], 1), _p(r["oi_15m"], 1), _p(r["oi_30m"], 1), _s(r["streak"])))
        prox = ("%s %s" % (r["prox"], r["prox_name"])).strip() if r["prox"] is not None else "—"
        L.append("Prox %s" % prox)
        L.append("")
        for w in (5, 15):
            x = (r.get("recent") or {}).get(w)
            if not x:
                continue
            if "pct" not in x:
                L.append("Last %2d min  — (%s)" % (w, "data gap after restart" if x.get("gap") else "no fresh bars"))
                continue
            bits = [_p(x["pct"]), "%s → %s" % (_px2(x["then"]), _px2(x["now"]))]
            if x.get("ext"):
                bits.append(x["ext"])
            if x.get("spike") is not None:
                bits.append("vol %s" % _x(x["spike"]))
            L.append("Last %2d min  %s" % (w, " · ".join(bits)))
        L.append("")
        pp = r.get("pace_path")
        ph = r.get("pace_hist")
        flat_now = ""
        if ph and ph.get("n") and ph.get("last_epoch"):
            mins = int((ph["now_epoch"] - ph["last_epoch"]) // 60)
            flat_now = " · flat since %s (%s)" % (ph["last"], _hm_dur(mins))
        elif ph and not ph.get("n"):
            flat_now = " · unchanged since %s" % ph["since"]
        L.append("PACE %s%s%s" % (_x(r["pace"]), ("  (now %s)" % pp["now_band"]) if pp and pp.get("now_band") else "",
                                  flat_now))
        if ph:
            if ph["n"]:
                evs = ph["events"]
                L.append("  changed %d× today (live, since %s): %s%s" % (
                    ph["n"], ph["since"], " · ".join("%s %.2f" % (t, v) for t, _f0, v in evs[-12:]),
                    (" (+%d earlier)" % (len(evs) - 12)) if len(evs) > 12 else ""))
                if ph["flat_min"] is not None:
                    L.append("  last change %s, flat %s before it" % (ph["last"], _hm_dur(ph["flat_min"])))
            else:
                L.append("  unchanged since %s (live samples)" % ph["since"])
            if ph["gaps"]:
                L.append("  app was off %s: changes then not seen" % ", ".join("%s–%s" % g for g in ph["gaps"]))
            pp = None                                   # live history replaces the bar rebuild
        if pp:
            if pp["bands"]:
                L.append("  broken %d time%s since open: %s" % (
                    len(pp["bands"]), "" if len(pp["bands"]) == 1 else "s",
                    " · ".join("%s %s" % (t, b) for t, b in pp["bands"])))
            else:
                L.append("  no band change since open (started %s)" % (pp["first"] or "—"))
            if pp["last"]:
                L.append("  last change %s (new day %s)" % pp["last"])
            L.append("  (rebuilt from equity 1-min bars%s; live PACE uses futures)" % (
                ", bars start %s" % pp["start"] if pp["gap"] else ""))
        return "\n".join(L) + _asof(r)
    if t == "live_value":
        out = []
        for x in r["rows"]:
            if "missing" in x:
                out.append("%s — %s" % (x["sym"], x["missing"]))
                continue
            f, bits = x["f"], []
            for fld in r["fields"]:
                if fld in x["na"]:
                    bits.append("%s — (not for an index)" % fld)
                    continue
                v = f.get(fld)
                tm = x["times"].get(fld)
                bits.append("%s %s%s" % (FIELD_LABEL.get(fld, fld), _cv(fld, v) if v is not None else "— (not available)",
                                        (" (%s)" % tm) if tm and v is not None else ""))
            tail = []
            if "price" not in r["fields"]:
                tail.append("price %s" % _px2(f.get("price")))
            if "pct" not in r["fields"]:
                tail.append("day %s" % _p(f.get("pct")))
            out.append("%s  %s" % (x["sym"], " · ".join(bits + tail)))
        return "\n".join(out) + _asof(r)
    if t == "live_screen":
        conds = r["conditions"]
        title = "%d stock%s · %s%s" % (r["n"], "" if r["n"] == 1 else "s",
                                       " and ".join(cond_words(c) for c in conds), _sec(r))
        items = []
        for i, (s, v) in enumerate(r["rows"]):
            f = r["fields"][s]
            main = _cv(r["order_by"], v) + ("" if r["order_by"] == "pct" else " " + r["order_by"])
            extra = ("day %s  " % _p(f.get("pct")) + _extras_line(r, s)).strip()
            vals = [cond_values(f, c) for c in conds]
            if not any(c["left"] == "price" or c["right"] == "price" for c in conds):
                vals.append("price %s" % _px2(f.get("price")))
            items.append((i + 1, s, main, extra, " · ".join(vals)))
        tail = ""
        if r.get("early"):
            tail += "\nFew minutes into the day: open, high and low are close for most stocks."
        if r.get("missing"):
            tail += "\n%d stocks skipped: a needed value is missing (no bars / levels yet)." % r["missing"]
        return _cards(title, items, None, tail=tail) + _asof(r)
    if t == "compare_symbols":
        rs = r["rows"]
        lines = [" vs ".join(x["sym"] for x in rs)]
        def row(label, key, fmt):
            return [label] + [fmt(x[key]) for x in rs]
        tab = [row("Price", "price", _px2), row("Day %", "pct", _p), row("Since open", "open", _p),
               row("Last 15m", "m15", _p), row("vs VWAP", "vwap", _p), row("RVOL", "rvol", _x),
               row("OI open", "oi", lambda v: _p(v, 1)), row("Long", "long", _s), row("Short", "short", _s),
               row("State L", "state", lambda v: _s(v)[:12]), row("State S", "sstate", lambda v: _s(v)[:12])]
        lines.append(table([""] + [x["sym"][:10] for x in rs], tab))
        return "\n".join(lines) + _asof(r)
    if t == "recent_alerts":
        title = "%d alerts today%s · newest first" % (r["n"], (" for " + r["symbol"]) if r["symbol"] else "")
        px = r.get("px") or {}
        items = []
        for i, a in enumerate(r["rows"]):
            sym = str(a.get("sym", ""))
            pnow, dpct = px.get(sym, (None, None))
            items.append((i + 1, sym, str(a.get("time", ""))[:5], "day %s" % _p(dpct),
                          "price %s · %s %s · %s" % (_px2(pnow), str(a.get("state", ""))[:16],
                                                     _s(a.get("severity")), str(a.get("reason", ""))[:40])))
        return _cards(title, items, None, "None yet.")
    if t == "market_mood":
        L = ["Market mood · " + (r["regime"] or "—") + " (your regime label)"]
        idx = []
        for nm, d in (("NIFTY", r["nifty"]), ("BANKNIFTY", r["bank"])):
            if d:
                av = d.get("above_vwap")
                idx.append("%s %s %s" % (nm, _p(_f(d.get("pct"))),
                                         "" if av is None else ("above VWAP" if av else "below VWAP")))
        if r["vix"]:
            idx.append("VIX %s" % _p(_f(r["vix"].get("pct"))))
        if idx:
            L.append(" · ".join(x.strip() for x in idx))
        ad = "A/D %d / %d of %d" % (r["adv"], r["dec"], r["total"])
        if r["ago"] and r["ago"]["adv"] is not None:
            ad += " (30 min ago %s / %s)" % (r["ago"]["adv"], r["ago"]["dec"])
        L.append(ad)
        b = r["breadth"]
        if b:
            L.append("New highs / lows · 15 min %s / %s · day %s / %s · bias %s" % (
                b["w15"]["hi"], b["w15"]["lo"], b["day"]["hi"], b["day"]["lo"], b.get("bias", "—")))
        if r["quads"]:
            L.append("OI: " + " · ".join("%s %d" % (k, v) for k, v in sorted(r["quads"].items(), key=lambda x: -x[1])[:5]))
        if r["iv_dir"]:
            L.append("IV: " + " · ".join("%s %d" % (k, v) for k, v in sorted(r["iv_dir"].items(), key=lambda x: -x[1])))
        L.append("Scanner flags · long %d · short %d" % (r["flags_long"], r["flags_short"]))
        if r["lead"]:
            s = "Leading %s %s" % (r["lead"]["sector"].replace("NIFTY ", ""), _p(r["lead"]["avg"]))
            if r["lag"]:
                s += " · Lagging %s %s" % (r["lag"]["sector"].replace("NIFTY ", ""), _p(r["lag"]["avg"]))
            L.append(s)
        return "\n".join(L) + _asof(r)
    if t == "breadth_trend":
        rows = [[_s(x["time"])[:5], _s(x["adv"]), _s(x["dec"]), "—" if x["ratio"] is None else "%.2f" % x["ratio"]]
                for x in r["rows"]]
        return "Breadth · last %d min (data covers %d) · %s\n%s" % (
            r["window_min"], r["covered_min"], r["trend"].upper(), table(["TIME", "ADV", "DEC", "A/D"], rows))
    return str(r)


def format_reply(r):
    """Reply text; answers built from an earlier list or with conditions say so on line 2."""
    text = _format(r)
    if r.get("order") == "bottom" and r.get("tool") in ("live_level_breaks", "level_breaks_history", "live_vwap"):
        head, _, rest = text.partition("\n")
        text = head + " · closest first" + ("\n" + rest if rest else "")
    if r.get("unsupported"):
        text += "\n\n⚠️ Can't do yet: %s — this is the closest answer." % "; ".join(r["unsupported"])
    sc = r.get("scope")
    if not sc:
        return text
    bits = []
    if sc.get("from"):
        bits.append("from answer #%d (%s, the %d stocks shown)" % (sc["from"], sc.get("base_time") or "?",
                                                                 sc.get("base_n") or 0))
    if sc.get("filters"):
        bits.append("where " + " and ".join("%s %s %g" % (f["col"], f["op"], f["value"]) for f in sc["filters"]))
    if not bits:                       # e.g. an alert's symbol scope: nothing to explain, no empty "↳" line
        return text
    head, _, rest = text.partition("\n")
    return head + "\n↳ " + " · ".join(bits) + ("\n" + rest if rest else "")


def _age(ts):
    if not ts:
        return "never"
    m = int((time.time() - float(ts)) // 60)
    return "%d min ago" % m if m < 120 else "%.1f h ago" % (m / 60.0)


CHECK_CALLS = [
    ("live_top", {"metric": "pct"}), ("live_top", {"metric": "open"}),
    ("live_top", {"metric": "pct", "window_min": 15}), ("live_top", {"metric": "volume", "window_min": 10}),
    ("live_top", {"metric": "oi", "window_min": 15}), ("live_top", {"metric": "new_high"}),
    ("live_vwap", {"side": "above"}), ("live_level_breaks", {"kind": "prev_high"}),
    ("current_candidates", {}), ("watchlist_status", {}), ("recent_alerts", {}),
    ("live_screen", {"conditions": [{"left": "open", "op": "=", "right": "high"}]}),
    ("live_value", {"symbols": ["NIFTY"], "fields": ["price", "high", "low"]}),
    ("live_top", {"metric": "fut_volume", "window_min": 15}), ("live_top", {"metric": "pace_cross", "window_min": 15}),
    ("live_top", {"metric": "iv", "window_min": 30}), ("live_top", {"metric": "vwap_up", "window_min": 15}),
    ("live_sector_rank", {"metric": "price", "window_min": 15}), ("live_sector_rank", {"metric": "open"}),
    ("market_mood", {}),
    ("breadth_trend", {}),
    ("top_symbols", {"metric": "gain", "days": 5}), ("top_symbols", {"metric": "oi", "days": 5}),
    ("top_symbols", {"metric": "pace", "days": 5}), ("level_breaks_history", {"kind": "prev_high"}),
    ("sector_history", {"days": 5}), ("signal_performance", {"by": "stage"}),
]


def check_report(ctx, live=True):
    """Data health per source + a real dry run of every tool. Code only, no LLM."""
    st = ctx.state
    L = ["TradEdge data check · %s" % ctx.now().strftime("%d-%m-%Y %H:%M")]
    if live:
        L += ["", "LIVE (in the running app)"]
        try:
            res = _live_results(ctx, required=False)
            L.append("%s Scan results   %d stocks · last scan %s" % ("✅" if res else "❌", len(res), _as_of(ctx) or "—"))
            with _SAMPLE_LOCK:
                meta = dict(_SAMPLE_META)
            if meta.get("last"):
                age = int(_epoch(ctx) - meta["last"])
                L.append("%s Live samples   %d taken · since %s · last %d s ago%s" % (
                    "✅" if age <= SAMPLE_GAP_S else "❌", meta["n"],
                    datetime.fromtimestamp(meta["since"]).strftime("%H:%M"), age,
                    (" · error " + meta["error"]) if meta.get("error") else ""))
            else:
                L.append("❌ Live samples   none yet (taken every minute 09:15-15:40)")
            rf = index_refresh_s()
            with _SAMPLE_LOCK:
                n_idx = len(_ISAMPLES)
            if n_idx:
                L.append("%s Index samples  %d indices · refresh ~%s" % (
                    "✅" if rf is None or rf <= 180 else "⚠️", n_idx,
                    "%d s" % rf if rf else "measuring"))
            oi = _snap(st.OI_CACHE_LOCK, st.OI_CACHE)
            L.append("%s OI cache       %d stocks · fetched %s" % (
                "✅" if oi else "❌", len(oi), _age(getattr(st, "_OI_LAST_FETCH", None))))
            ctx._tl.bars = None
            bars, stale = _bars_today(ctx)
            del ctx._tl.bars
            L.append("%s 1-min bars     %d fresh · %d stale (>3 min)" % (
                "✅" if bars and stale < len(bars) / 4 else "⚠️" if bars else "❌", len(bars), stale))
            hilo = _snap(st.SESSION_HILO_LOCK, st.SESSION_HILO)
            L.append("%s PACE (hi/lo)   %d stocks%s" % ("✅" if hilo else "❌", len(hilo),
                                                     "" if hilo else " · ticker in LTP mode?"))
            ad = _ad_points(ctx) if hasattr(st, "AD_BREADTH_LOCK") else []
            L.append("%s A/D history    %d points" % ("✅" if ad else "❌", len(ad)))
        except Exception as e:
            L.append("❌ live check failed: %s" % e)
    L += ["", "HISTORY"]
    try:
        days, data = _ohlc(ctx)
        past = [d for d in days if d < ctx.today]
        exp = _prev_weekday(ctx.today)
        ok = past and past[-1] >= exp
        L.append("%s daily_ohlc.csv last %s · %d stocks%s" % (
            "✅" if ok else "⚠️", past[-1] if past else "—", len(data.get(past[-1], {})) if past else 0,
            "" if ok else " · %s not written yet (15:45)" % exp))
    except ToolError as e:
        L.append("❌ %s" % e)
    try:
        day, miss = _missing_prev(ctx)
        k = dict(_KITE_STATE)
        L.append("%s prev session %s: %d stocks without a bar%s" % (
            "✅" if not miss else "⚠️", day or "—", len(miss),
            (" (%s)" % ", ".join(miss[:5]) + (" …" if len(miss) > 5 else "")) if miss else ""))
        if k.get("day"):
            L.append("   Kite fill today: %d fetched, %d failed%s" % (
                k.get("fetched", 0), len(k.get("failed") or []), " (running)" if k.get("running") else ""))
    except Exception as e:
        L.append("❌ prev-session check: %s" % e)
    try:
        c = _ro(ctx.saga_db)
        sd = _last_days(c, 10, before=ctx.today)
        pace = c.execute("SELECT COUNT(DISTINCT day) FROM symbol_day WHERE max_pace IS NOT NULL AND day IN (%s)"
                         % ",".join("?" * len(sd)), sd).fetchone()[0] if sd else 0
        outc = c.execute("SELECT COUNT(*), MAX(day) FROM outcomes").fetchone()
        c.close()
        L.append("%s symbol_day     last %s · PACE on %d of last 10 sessions" % (
            "✅" if sd else "❌", sd[0] if sd else "—", pace))
        L.append("%s outcomes       %d rows · last %s" % ("✅" if outc[0] else "❌", outc[0], outc[1] or "—"))
    except Exception as e:
        L.append("❌ saga_obs.db: %s" % e)
    try:
        c = _ro(ctx.playbook_db)
        od = c.execute("SELECT COUNT(DISTINCT date), MAX(date) FROM oi_eod").fetchone()
        c.close()
        L.append("%s oi_eod         %d saved days · last %s" % ("✅" if od[0] >= 2 else "❌", od[0], od[1] or "—"))
    except Exception as e:
        L.append("❌ playbook.db: %s" % e)
    L += ["", "TOOLS (dry run)"]
    good, bad = [], []
    for name, args in CHECK_CALLS:
        if not live and name in ("live_top", "live_vwap", "live_level_breaks", "current_candidates",
                                 "watchlist_status", "recent_alerts", "live_sector_rank", "live_screen", "live_value",
                                 "market_mood", "breadth_trend"):
            continue
        label = name + ("(%s)" % ",".join("%s=%s" % kv for kv in args.items()) if args else "")
        r, err = run(ctx, name, args)
        (good if r else bad).append((label, err))
    L.append("✅ %d ready" % len(good))
    for label, err in bad:
        L.append("❌ %s → %s" % (label, (err or "")[:90]))
    return "\n".join(L)


HELP_TEXT = """I answer from your TradEdge data. Examples:
• top gainers today with price and rvol
• which stock jumped most in last 15 min
• volume spike last 10 min · OI build-up last 30 min
• stocks above VWAP now · broke yesterday's high today
• market mood · breadth last 30 min
• most advancing sector last 15 min · best sector this week
• watchlist by score · watchlist OI since open
• any long signals now? · short setups in bank
• today's alerts for TCS
• RELIANCE now · TCS last 10 days
• top gainers last 5 days · highest OI build-up this week
• how did SAGA stage perform last 30 sessions?
Follow-ups (last 5 answers, 15 min): "and in IT?", "with volume",
"from that, rvol above 2" · "of these, only bank".
• compare TCS and INFY · gap-ups today · new day highs last 15 min
• stocks where open = high · open = low · above previous high and above vwap · range above 2%
• what is low of nykaa · nykaa high and vwap · pdh of tcs · nifty high · vix now
• fut vol rise last 15 min in FMCG · PACE crossed band last 5 min
• IV rising last 30 min · furthest from VWAP · top 5 sectors from open
ALERTS: start with "alert me" or "alert:" (default: today only, once)
• alert me when nykaa goes below 348 · alert me nifty below 24000
• alert me when HDFCBANK crosses above VWAP · alert me when stock open=high and down 0.5% from open
• alert: any FMCG stock fut vol 3x in 15 min always   (always / forever = every day)
• alert me when VIX moves 1% in 5 min · any sector makes a new day high
• alert me when any stock change pace · ... and pace > 1.4
• alert me when pace changes on low break with volume support   (volume support = fut vol 5 min ≥ 1.5x;
  also: with rvol (≥ 2) · with stock volume (5-min spike ≥ 2x) · or say your number: fut vol 3x)
• after any answer: "alert above result" (new names entering it) · "alert these when pace changes"
• alert 3 make it 1.5% · alert 3 delete · /alerts = list, change, pause, delete
/check = data health · /help = this list
I never give buy/sell advice."""
