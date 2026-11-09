# -*- coding: utf-8 -*-
"""
ai_alerts.py -- standing alerts written in plain words.  v1, 05-11-2026.

    "alert me when any FMCG stock's fut vol rises 3x in 15 min everyday"
      -> Gemini maps the sentence ONCE to a rule (create_alert tool, same metrics as the
         question tools) -> code validates it and asks with buttons for anything missing
         (how much / how long / how often) -> you confirm -> saved to data/ai_alerts.json
      -> every minute the sampler thread runs every active rule IN CODE (no LLM call, no
         quota) and pings once per stock per day (or again after 30 min, if you chose that).

One responsibility per class:
    RuleSpec        what each metric needs (condition? window?) and how to say it in words
    validate_rule   model output -> clean rule + list of missing pieces (never trusted)
    AlertStore      the JSON file: permanent IDs, atomic writes, survives restarts
    check_alerts    the minute check: run each rule, dedupe, build the ping text
"""
import json
import os
import re
import threading
import time
from datetime import datetime

import ai_tools as T

MAX_ACTIVE = 10
MAX_PING_ROWS = 10              # one ping lists at most 10 names (+N more)
NEAR_FROM = 9 * 60 + 30         # 'near today's low/high': in the first 15 min every stock is near both
GAP_REPEAT_S = 30 * 60          # "every time" = again after 30 min at the earliest
EVENT_WINDOW = 5                # crossings / new highs are looked for in the last 5 min
DRAFT_TTL_S = 10 * 60

# ── what each metric needs ───────────────────────────────────────────────────
# cond: a number is required ("how much?"), choices = the buttons offered
# event: the metric is a happening (crossed, new high) -> no number needed
STOCK_SPEC = {
    "price":      {"cond": True, "unit": "", "level": True},
    "pct":        {"cond": True, "unit": "%", "choices": [1, 2, 3], "signed": True},
    "open":       {"cond": True, "unit": "%", "choices": [1, 2, 3], "signed": True},
    "gap":        {"cond": True, "unit": "%", "choices": [1, 2, 3], "signed": True},
    "volume":     {"cond": True, "unit": "x", "choices": [2, 3, 5], "needs_window": True},
    "fut_volume": {"cond": True, "unit": "x", "choices": [2, 3, 5], "needs_window": True},
    "rvol":       {"cond": True, "unit": "x", "choices": [2, 3, 5]},
    "oi":         {"cond": True, "unit": "%", "choices": [3, 5, 10], "signed": True},
    "pace":       {"cond": True, "unit": "", "choices": [1.5, 2, 2.5]},
    "iv":         {"cond": True, "unit": " pts", "choices": [1, 2, 3], "signed": True},
    "vwap":       {"cond": True, "unit": "%", "choices": [0.5, 1, 2], "signed": True},
    "score":      {"cond": True, "unit": "%", "choices": [70, 80, 90]},
    "near_low":   {"cond": True, "unit": "%", "choices": [0.2, 0.3, 0.5], "near": True},
    "near_high":  {"cond": True, "unit": "%", "choices": [0.2, 0.3, 0.5], "near": True},
    "near_vwap":  {"cond": True, "unit": "%", "choices": [0.1, 0.2, 0.3], "near": True},
    "near_pdh":   {"cond": True, "unit": "%", "choices": [0.2, 0.3, 0.5], "near": True},
    "near_pdl":   {"cond": True, "unit": "%", "choices": [0.2, 0.3, 0.5], "near": True},
    "pace_cross": {"event": True}, "pace_change": {"event": True}, "pace_low": {"event": True},
    "pace_high": {"event": True},
    "new_high": {"event": True}, "new_low": {"event": True},
    "vwap_up": {"event": True}, "vwap_down": {"event": True}, "pdh_up": {"event": True},
    "pdl_down": {"event": True}, "open_up": {"event": True}, "open_down": {"event": True},
}
INDEX_SPEC = {
    "price":        {"cond": True, "unit": "", "level": True},
    "pct":          {"cond": True, "unit": "%", "choices": [0.5, 1, 2], "signed": True},
    "move":         {"cond": True, "unit": "%", "choices": [0.3, 0.5, 1], "signed": True, "needs_window": True},
    "near_high":    {"cond": True, "unit": "%", "choices": [0.1, 0.2, 0.3], "near": True},
    "near_low":     {"cond": True, "unit": "%", "choices": [0.1, 0.2, 0.3], "near": True},
    "day_high_up": {"event": True}, "day_low_down": {"event": True},
    "vwap_up": {"event": True}, "vwap_down": {"event": True},
}
STOCK_SPEC["screen"] = {"event": True}       # only CONDITIONS (open = high ...): fires while they all hold
assert set(STOCK_SPEC) - {"screen"} == set(T.LIVE_METRICS) and set(INDEX_SPEC) == set(T.INDEX_METRICS)

STOCK_WORDS = {
    "screen": "where", "price": "price", "pct": "day %", "open": "% since open", "gap": "opening gap", "volume": "volume spike",
    "fut_volume": "futures volume rise", "rvol": "RVOL", "oi": "OI change", "pace": "PACE",
    "iv": "ATM IV", "vwap": "distance from VWAP", "score": "long score",
    "near_low": "trades within {v} of today's low (or breaks it)",
    "near_high": "trades within {v} of today's high (or breaks it)",
    "near_vwap": "trades within {v} of VWAP", "near_pdh": "trades within {v} of yesterday's high",
    "near_pdl": "trades within {v} of yesterday's low", "pace_cross": "PACE moves into a new band",
    "pace_change": "PACE value changes (e.g. 1.25 → 1.27)",
    "pace_low": "PACE changes by a new day LOW (low break)",
    "pace_high": "PACE changes by a new day HIGH (high break)",
    "new_high": "makes a new day high", "new_low": "makes a new day low",
    "vwap_up": "crosses ABOVE VWAP", "vwap_down": "crosses BELOW VWAP",
    "pdh_up": "crosses ABOVE yesterday's high", "pdl_down": "crosses BELOW yesterday's low",
    "open_up": "crosses ABOVE today's open", "open_down": "crosses BELOW today's open",
}
INDEX_WORDS = {
    "price": "level", "pct": "day %", "move": "moves", "near_high": "trades within {v} of today's high",
    "near_low": "trades within {v} of today's low", "day_high_up": "makes a new day high",
    "day_low_down": "makes a new day low", "vwap_up": "crosses ABOVE VWAP", "vwap_down": "crosses BELOW VWAP",
}
FILTER_WORDS = {"fut_vol5": "fut vol (5 min vs 5 before)", "vol5": "stock vol (5 min spike)", "rvol": "rvol"}
OPS = {">=": "≥", "<=": "≤", "abs>=": "±", "between": "between", "cross": "crosses"}

# ── words decided by CODE, not the model ("at the end I write today or everyday") ──
# Defaults (user 05-11): TODAY only and ONCE. Only words that change them are read:
# 'always / forever / everyday / roj' -> every day; 'every time / har vakhte' -> repeat (30-min gap).
_TODAY_W = re.compile(r"\b(today only|only today|aaje j)\b", re.I)
_EVERY_W = re.compile(r"\b(always|forever|for ever|every\s*day|everyday|daily|roj|har\s*roj|rojnu|hamesha|"
                      r"kayam)\b", re.I)
_ONCE_W = re.compile(r"\b(once|ek\s*var|ekvar|one time|only once)\b", re.I)
_REPEAT_W = re.compile(r"\b(every\s*time|each\s*time|har\s*vakhte|whenever|jyare\s*pan|again)\b", re.I)
_BREAK_W = re.compile(r"\b(break\w*|broke|tut\w*|tod\w*|new\s+(day\s+)?(low|high))\b", re.I)
_NEAR_W = re.compile(r"\b(near|around|hanging|within|close\s+to|aaspaas|paase)\b", re.I)
_LOW_BRK_W = re.compile(r"\b(low\s*break\w*|breaks?\s+(the\s+|today'?s?\s+|day\s+)?low|new\s+(day\s+)?low|"
                        r"low\s*(tut\w*|tod\w*))\b", re.I)
_HIGH_BRK_W = re.compile(r"\b(high\s*break\w*|breaks?\s+(the\s+|today'?s?\s+|day\s+)?high|new\s+(day\s+)?high|"
                         r"high\s*(tut\w*|tod\w*))\b", re.I)
VOLUME_COLS = {"fut_vol5", "vol5", "rvol", "volume"}
VOLUME_DEFAULTS = [   # (filter column, words, default number) — order matters: specific words first
    ("vol5", re.compile(r"\b(stock|cash|equity|share)\s*vol(ume)?\b", re.I), 2.0),
    ("rvol", re.compile(r"\brvol\b", re.I), 2.0),
    ("fut_vol5", re.compile(r"\bfut(ure)?s?\s*vol(ume)?\b", re.I), 1.5),
    # generic words: futures volume, but only when no other volume kind was named
    ("fut_vol5?", re.compile(r"\b(volume\s*(support|backed|confirm\w*|saathe|sathe)|vol\s*support|"
                             r"with\s+volume)\b", re.I), 1.5),
]
_ORDER_W = re.compile(r"\b(order|orders|buy|sell|trade|trading|place)\b", re.I)
_PACE_CHANGE_W = re.compile(r"\b(chang\w*|badl\w*|moves?|moving|increas\w*|vadh\w*)\b", re.I)
QUERY_PREFIX = re.compile(r"^\s*alert\s+(?:me\s+)?(?:on\s+)?(?:above|this|that|it)\b(?:\s+(?:result|list|answer|one))?"
                          r"\s*(.*)$", re.I | re.S)
THESE_PREFIX = re.compile(r"^\s*alert\s+(?:me\s+)?(?:these|them|those|aa|ama|aama|emne)\b\s*(.*)$", re.I | re.S)
LIST_TOOLS = {"live_top", "live_level_breaks", "live_vwap", "current_candidates", "watchlist_status",
              "live_sector_rank"}
CREATE_PREFIX = re.compile(r"^\s*alert\s*(:|me\b|-)\s*", re.I)
EDIT_PREFIX = re.compile(r"^\s*alert\s*#?\s*(\d+)\b\s*(.*)$", re.I | re.S)
_CMD_WORDS = {"delete": "delete", "remove": "delete", "off": "delete", "stop": "pause", "pause": "pause",
              "resume": "resume", "start": "resume", "on": "resume", "kadhi": "delete", "bandh": "pause"}


# ═════════════════════════════════════════════════════════════════════════════
# Gemini tools (alert mode only: question mode never sees these)
# ═════════════════════════════════════════════════════════════════════════════
_RULE_PROPS = {
    "target": {"type": "STRING", "enum": ["stocks", "index"],
               "description": "stocks = F&O stocks; index = NIFTY 50, sector indices, INDIA VIX."},
    "metric": {"type": "STRING", "enum": sorted(set(STOCK_SPEC) | set(INDEX_SPEC)),
               "description": ("stocks: any live_top metric (pct, open, gap, volume, fut_volume, rvol, oi, pace, "
                               "pace_cross, new_high, new_low, vwap, score, iv, vwap_up, vwap_down, pdh_up, "
                               "pdl_down, open_up, open_down, near_low, near_high, near_vwap, near_pdh, "
                               "near_pdl). index: pct, move (% in N min), day_high_up, day_low_down, vwap_up, "
                               "vwap_down, near_high, near_low.")},
    "window_min": {"type": "INTEGER", "description": "'in N min' -> N. Empty if not said."},
    "symbols": {"type": "ARRAY", "items": {"type": "STRING"},
                "description": "stocks: NSE symbols named ('HDFCBANK'). index: index names ('INDIA VIX', "
                               "'NIFTY IT') or ['ANY'] for 'any sector'. Empty = any stock."},
    "sector": {"type": "STRING", "description": "stocks only: 'any FMCG stock' -> NIFTY FMCG."},
    "universe": {"type": "STRING", "enum": ["all", "watchlist", "setups"],
                 "description": "'watchlist stock' -> watchlist; 'flagged / setups' -> setups. Default all."},
    "op": {"type": "STRING", "enum": [">=", "<=", "abs>=", "cross", "between"],
           "description": "up / above / more than -> >=; down / below / less -> <=; 'moves' either way -> abs>=; "
                          "'crosses 4200' (no direction) -> cross; 'between A and B' -> between. near_* -> <=."},
    "value": {"type": "NUMBER", "description": "The number the user said (1% -> 1, 3x -> 3, ₹348 -> 348). Empty if not said."},
    "value2": {"type": "NUMBER", "description": "Only for between: the second number."},
    "filters": {"type": "ARRAY", "description": "Extra numeric conditions, e.g. rvol > 2.",
                "items": {"type": "OBJECT", "properties": {
                    "col": {"type": "STRING", "enum": T.LIVE_FILTER_COLS},
                    "op": {"type": "STRING", "enum": [">", ">=", "<", "<="]},
                    "value": {"type": "NUMBER"}}, "required": ["col", "op", "value"]}},
    "conditions": T.CONDITIONS,
    "unsupported": {"type": "STRING", "description": "Part of the request no metric can watch (PCR, news...)."},
}
ALERT_TOOLS = [
    {"name": "create_alert", "description": "Turn the user's alert sentence into ONE rule.",
     "parameters": {"type": "OBJECT", "properties": _RULE_PROPS, "required": ["target", "metric"]}},
    {"name": "update_alert",
     "description": "Change an existing alert #id. Send ONLY the fields the user changes.",
     "parameters": {"type": "OBJECT", "properties": dict(_RULE_PROPS, id={"type": "INTEGER"},
                                                       add_symbols={"type": "ARRAY", "items": {"type": "STRING"}},
                                                       remove_symbols={"type": "ARRAY", "items": {"type": "STRING"}}),
                    "required": ["id"]}},
    {"name": "cannot_alert", "description": "Nothing in the sentence can be watched from the live data.",
     "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
]


def system_prompt(sectors, today=None):
    today = today or datetime.now().strftime("%Y-%m-%d (%A) %H:%M IST")
    return (
        "You turn ONE alert sentence from an Indian NSE F&O trader into ONE rule. Now is %s. "
        "English, Gujarati or Hinglish.\n"
        "1. 'alert #N ...' -> update_alert(id=N) with only the changed fields ('make it 1.5%%' -> value=1.5; "
        "'add ICICIBANK' -> add_symbols).\n"
        "2. Stocks vs index: VIX, NIFTY, BANKNIFTY, 'sector' / 'any sector' -> target=index. Anything about "
        "stocks / a stock / 'any FMCG stock' -> target=stocks (sector=...).\n"
        "3a. PACE: 'pace changes / change pace / pace badlay / pace moves / pace increases' (the VALUE moving, "
        "1.25 -> 1.27) -> pace_change, no number. 'pace changes and pace > 1.4' -> pace_change + filters "
        "[{col: pace, op: '>', value: 1.4}]. 'pace changes when low breaks' -> pace_low ('high breaks' -> "
        "pace_high). Only 'pace band' -> pace_cross. 'pace above 2' -> pace (a level).\n"
        "3c. PRICE RELATIONS -> metric=screen + conditions (fields price, open, high, low, vwap, pdc, pdh, pdl, "
        "pct, open_pct, gap, range_pct, pace, rvol, fut_vol5, vol5, oi, iv, score): 'open = high' -> {left: open, "
        "op: '=', right: high}; 'decrease 0.5%% after open' -> {left: price, op: '<=', right: open, pct: -0.5}; "
        "'above previous high' -> {left: price, op: '>', right: pdh}. A metric (vwap_up, pace_change...) can "
        "also carry conditions. Never drop a condition: if one cannot be written, use cannot_alert.\n"
        "3b. VOLUME SUPPORT is a filter: 'with volume support' -> {col: fut_vol5, op: '>=', value: 1.5} "
        "(futures volume last 5 min ÷ the 5 before); 'with rvol' -> {col: rvol, op: '>=', value: 2}; 'with "
        "stock / cash volume' -> {col: vol5, op: '>=', value: 2}. A number the user says replaces the default.\n"
        "3. Crossings ('crosses / breaks / goes above VWAP', 'breaks yesterday high') -> vwap_up, vwap_down, "
        "pdh_up, pdl_down, open_up, open_down; 'new day high / breaks today's high' -> new_high (index: "
        "day_high_up); 'breaks today's low' / 'new day low' / 'low tute' -> new_low (index: day_low_down), NO "
        "number; only 'near / around / hanging / within X%% of today's low' -> near_low; 'PACE "
        "band' -> pace_cross.\n"
        "4a. A PRICE / LEVEL ('nykaa goes below 348', 'reliance above 3000', 'nifty below 24000', 'vix above "
        "16', 'TCS crosses 4200') -> metric=price, op <= / >= / cross, value=the number. NOT pct, NOT filters.\n"
        "4b. 'buy / sell X when ...': I only alert, never trade -> create the alert and put 'placing orders' in "
        "`unsupported`.\n"
        "4. 'fut vol rises 3x in 15 min' -> fut_volume, window_min=15, op >=, value 3. 'VIX up or down 1%% in "
        "5 min' -> target=index, symbols=['INDIA VIX'], metric=move, window_min=5, op abs>=, value 1.\n"
        "5. Leave value / window_min / op EMPTY when the user did not say them: the app asks with buttons. "
        "Never invent a number.\n"
        "6. Ignore 'today' / 'always' / 'everyday' / 'once' / 'every time': code reads those words itself.\n"
        "7. Known sectors (exact strings): %s. 'Any sector' -> symbols=['ANY'].\n"
        "8. If part cannot be watched (PCR, news, option chain), still map the rest and put that part in "
        "`unsupported`; if nothing can be watched -> cannot_alert."
    ) % (today, ", ".join(sorted(sectors)) or "unknown")


# ═════════════════════════════════════════════════════════════════════════════
# Validation: the model's rule is a guess; code checks every field
# ═════════════════════════════════════════════════════════════════════════════
def words_from_text(text):
    """expiry / repeat the user SAID (None = not said: create uses today / once, change keeps the old)."""
    t = text or ""
    expiry = "everyday" if _EVERY_W.search(t) else "today" if _TODAY_W.search(t) else None
    repeat = "every" if _REPEAT_W.search(t) else "once" if _ONCE_W.search(t) else None
    return expiry, repeat


def spec_of(rule):
    if rule.get("target") == "query":
        return {"event": True}
    return (INDEX_SPEC if rule.get("target") == "index" else STOCK_SPEC).get(rule.get("metric"))


def validate_rule(ctx, raw, text=""):
    """(rule, missing, notes). Raises T.ToolError on a field that is plainly wrong.
    missing: ordered slots to ask: 'value', 'window_min', 'repeat', 'expiry'."""
    a = dict(raw or {})
    target = str(a.get("target") or "stocks").lower()
    if target not in ("stocks", "index"):
        raise T.ToolError("target must be stocks or index")
    m = str(a.get("metric") or "").lower()
    if a.get("conditions") and target == "stocks" and not m:
        m = "screen"
    spec = (INDEX_SPEC if target == "index" else STOCK_SPEC).get(m)
    if not spec:
        hint = ", ".join(sorted(INDEX_SPEC if target == "index" else STOCK_SPEC))
        raise T.ToolError("Can't watch '%s' for %s. I can watch: %s" % (m or "?", target, hint))
    rule = {"target": target, "metric": m, "filters": [], "unsupported": str(a.get("unsupported") or "").strip()}
    if a.get("conditions") or m == "screen":
        if target != "stocks":
            raise T.ToolError("Price conditions work for stocks; for an index use its own alert words.")
        rule["conditions"] = T.clean_conditions(a.get("conditions"))
    if target == "index":
        names = [T.resolve_index(ctx, x) for x in (a.get("symbols") or [])] or ["ANY"]
        rule["symbols"] = sorted(set(names))
        cap = T.INDEX_METRICS[m]
    else:
        rule["symbols"] = sorted({T._symbol(ctx, {"symbol": x}) for x in (a.get("symbols") or []) if str(x).strip()})
        rule["sector"] = T._sector(ctx, a)
        rule["universe"] = T._universe(a)
        if rule["universe"] == "all" and T._WATCH_W.search(text or ""):
            rule["universe"] = "watchlist"                     # the user said it: code keeps it
        elif rule["universe"] == "all" and T._SETUP_W.search(text or ""):
            rule["universe"] = "setups"
        cap = T.LIVE_METRICS.get(m, {"window": False})          # 'screen' = conditions only
        if a.get("filters"):
            sc = T.scope_args(ctx, "live_top", {"filters": a["filters"]}) or {}
            rule["filters"] = sc.get("filters", [])
    # Guard (05-11 'alert me when any stock change pace' came back as metric=pace, asking 'How much?'):
    # PACE + change words = the VALUE moving. A number given with it = 'and PACE above that'.
    if m == "pace" and target == "stocks" and _PACE_CHANGE_W.search(text or "") and not a.get("conditions"):
        v = T._f(a.get("value"))
        if v is not None:
            rule["filters"].append({"col": "pace", "op": ">" if str(a.get("op") or ">=") != "<=" else "<",
                                    "value": v})
        m = rule["metric"] = "pace_change"
        spec, cap = STOCK_SPEC[m], T.LIVE_METRICS[m]
        a = dict(a, value=None, op=None)
    # Guard (06-11 09:45): 'nykaa break todays low' came back as near_low with no number -> a BREAK.
    if m in ("near_low", "near_high") and T._f(a.get("value")) is None and _BREAK_W.search(text or "") \
            and not _NEAR_W.search(text or ""):
        m = rule["metric"] = "new_low" if m == "near_low" else "new_high"
        spec, cap = (INDEX_SPEC if target == "index" else STOCK_SPEC)[m], T.LIVE_METRICS[m]
    if target == "index" and m in ("near_low", "near_high") and T._f(a.get("value")) is None \
            and _BREAK_W.search(text or "") and not _NEAR_W.search(text or ""):
        m = rule["metric"] = "day_low_down" if m == "near_low" else "day_high_up"
        spec, cap = INDEX_SPEC[m], T.INDEX_METRICS[m]
    # Guard (06-11 09:47): 'new day low' came back as new_high. The user's words decide the side.
    m2, _note = T.direction_fix(text, m)
    if m2 != m:
        m = rule["metric"] = m2
        spec = (INDEX_SPEC if target == "index" else STOCK_SPEC)[m]
        cap = (T.INDEX_METRICS if target == "index" else T.LIVE_METRICS).get(m, cap)
    # Guard: 'pace changes ... when low breaks' = which END of the range moved
    if m == "pace_change" and target == "stocks":
        if _LOW_BRK_W.search(text or ""):
            m = rule["metric"] = "pace_low"
        elif _HIGH_BRK_W.search(text or ""):
            m = rule["metric"] = "pace_high"
        spec, cap = STOCK_SPEC[m], T.LIVE_METRICS[m]
    # Guard: volume-support words the model left out -> the user's defaults (05-11):
    # 'volume support' = futures volume 5 min ≥ 1.5x; 'rvol' ≥ 2; 'stock volume' 5 min ≥ 2x
    if target == "stocks" and m not in ("volume", "fut_volume", "rvol") and m in T.LIVE_METRICS:
        have = {f["col"] for f in rule["filters"]}
        for col, rx, val in VOLUME_DEFAULTS:
            generic = col.endswith("?")
            col = col.rstrip("?")
            if rx.search(text or "") and col not in have and not (generic and have & VOLUME_COLS):
                rule["filters"].append({"col": col, "op": ">=", "value": val})
                have.add(col)
    # Guard (05-11 'nykaa goes below 348' came back as pct + filter price < 348): a metric with no
    # number plus ONE numeric condition = the user meant that condition. Promote it.
    if spec.get("cond") and T._f(a.get("value")) is None and len(rule["filters"]) == 1 \
            and rule["filters"][0]["col"] in STOCK_SPEC and target == "stocks":
        f = rule["filters"].pop()
        m = rule["metric"] = f["col"]
        spec, cap = STOCK_SPEC[m], T.LIVE_METRICS[m]
        a = dict(a, op=">=" if f["op"] in (">", ">=") else "<=", value=f["value"], window_min=0)
    if m == "screen":
        cap = {"window": False}
    w = T._int(a, "window_min", 0, cap.get("max_w", 120), 0) if cap.get("window") else 0
    if w and w < 5:
        w = 5
    if not w and cap.get("needs") and (spec or {}).get("event"):
        w = EVENT_WINDOW                       # crossings: look at the last 5 min, ping once
    rule["window_min"] = w
    missing = []
    if spec.get("cond"):
        op = str(a.get("op") or "").strip()
        if op not in OPS:
            op = "abs>=" if spec.get("signed") else ">="
        rule["op"], rule["value"] = op, T._f(a.get("value"))
        if op == "between":
            v1, v2 = rule["value"], T._f(a.get("value2"))
            if v1 is None or v2 is None:
                rule["op"] = ">="                          # one number only: treat as a plain level
            else:
                rule["value"], rule["value2"] = min(v1, v2), max(v1, v2)
        if op == "cross" and rule["value"] is not None:
            cur = current_value(ctx, rule)
            rule["op"] = "<=" if cur is not None and cur > rule["value"] else ">="
            if cur is None:
                rule["unsupported"] = (rule["unsupported"] + "; " if rule["unsupported"] else "") + \
                    "no live price now, so I assumed 'goes above'"
        rule = norm_cond(rule)
        if rule["value"] is None:
            missing.append("value")
    if spec.get("needs_window") and not w:
        missing.append("window_min")
    expiry, repeat = words_from_text(text)
    rule["expiry"], rule["repeat"] = expiry or "today", repeat or "once"      # never asked
    return rule, missing, ([rule["unsupported"]] if rule["unsupported"] else [])


def scope_problems(ctx, rule, text):
    """What the user NAMED but the rule does not cover (06-11 09:42: 'watchlist' dropped -> all stocks).
    Any problem = nothing is saved."""
    out = []
    if rule.get("target") != "stocks":
        return out
    t = text or ""
    if T._WATCH_W.search(t) and rule.get("universe") != "watchlist":
        out.append("you said 'watchlist' but the rule covers %s" % (rule.get("universe") or "all stocks"))
    if T._SETUP_W.search(t) and rule.get("universe") != "setups":
        out.append("you said 'setups / flagged' but the rule covers %s" % (rule.get("universe") or "all stocks"))
    syms = set(rule.get("symbols") or [])
    words = re.findall(r"[a-z0-9&-]+", t.lower())
    joined = {"".join(words[i:i + 2]) for i in range(len(words) - 1)} | set(words)
    fno = {str(x).upper() for x in (getattr(ctx.state, "FNO_SYMBOLS", []) or [])}
    named = {w.upper() for w in joined if len(w) >= 3 and w.upper() in fno and w.upper() not in _NOT_NAMES}
    missing = sorted(n for n in named if n not in syms and not T._is_index(ctx, n))
    if missing and (syms or not rule.get("sector")):
        out.append("you named %s but the rule watches %s" % (", ".join(missing), ", ".join(sorted(syms)) or "all stocks"))
    return out


_NOT_NAMES = {"ALL", "ANY", "HIGH", "LOW", "OPEN", "GAP", "NEW", "DAY", "BANK", "AUTO", "METAL", "IDEA", "PAGE",
              "VWAP", "PACE", "RVOL", "VOL", "FUT", "STOCK", "STOCKS", "ALERT", "WHEN", "BREAK", "TODAY", "NOW"}


def current_value(ctx, rule):
    """Live value of the rule's metric for its one stock / index (None if not exactly one or unknown)."""
    syms = rule.get("symbols") or []
    if len(syms) != 1 or syms[0] == "ANY":
        return None
    try:
        if rule["target"] == "index":
            v = T.index_values(ctx, "price", 0, syms).get(syms[0])
            if v:
                return v[0]
            with ctx.state.SECTOR_INTEL_LOCK:
                return T._f((((ctx.state.SECTOR_INTEL or {}).get("data") or {}).get(syms[0]) or {}).get("price"))
        r = T._live_results(ctx, required=False).get(syms[0]) or {}
        return T._f(r.get("price"))
    except Exception:
        return None


def norm_cond(rule):
    """One meaning per number: near_* = distance (<=, positive); 'down 2%' = (<=, -2);
    'moves 1%' = (abs>=, 1). Used after the model AND after a typed / tapped answer."""
    r = dict(rule)
    spec = spec_of(r) or {}
    v = r.get("value")
    if spec.get("near"):
        r["op"] = "<="
        r["value"] = None if v is None else abs(v)
        return r
    if v is None or spec.get("level") or r.get("op") == "between":
        return r                                      # a price is a price: no sign games
    if r.get("op") == "abs>=":
        r["value"] = abs(v)
    elif v < 0:
        r["op"] = "<="
    elif r.get("op") == "<=" and spec.get("signed"):
        r["value"] = -v                              # 'falls 2%' -> day % <= -2
    return r


# ── plain words, built by code (what you confirm is exactly what is checked) ──
def _num(v, unit):
    s = ("%g" % v) if v is not None else "?"
    return s + unit


def describe(rule):
    if rule.get("target") == "query":
        return 'NEW names entering: "%s"' % str(rule.get("q") or "")[:80]
    spec = spec_of(rule) or {}
    unit = spec.get("unit", "")
    if rule["target"] == "index":
        names = rule.get("symbols") or ["ANY"]
        who = "any sector index" if "ANY" in names else ", ".join(names)
        words = INDEX_WORDS[rule["metric"]]
    else:
        syms = rule.get("symbols") or []
        uni = {"watchlist": "watchlist", "setups": "flagged (setup)"}.get(rule.get("universe"))
        who = ", ".join(syms) if syms else ("any %s%s stock" % ((uni + " ") if uni else "", T._short(rule["sector"]))
                                            if rule.get("sector") else ("any %s stock" % uni if uni
                                                                         else "any F&O stock"))
        words = STOCK_WORDS.get(rule["metric"], "")
    w = rule.get("window_min") or 0
    if spec.get("near"):
        pred = words.format(v=_num(rule.get("value"), unit))
    elif rule.get("op") == "between":
        pred = "%s between %s and %s" % (words, _num(rule.get("value"), unit), _num(rule.get("value2"), unit))
    elif spec.get("cond"):
        op = rule.get("op", ">=")
        v = rule.get("value")
        num = ("±" + _num(v, unit)) if op == "abs>=" else ("%s %s" % (OPS[op], _num(v, unit)))
        pred = "%s %s" % (words, num) + (" in %d min" % w if w else "")
        if rule["metric"] in ("volume", "fut_volume") and w:
            pred = "%s %s (last %d min vs the %d before)" % (words, num, w, w) if rule["metric"] == "fut_volume" \
                else "%s %s in %d min" % (words, num, w)
    else:
        pred = words
    if rule.get("metric") == "screen":
        pred = "where"
    cw = " and ".join(T.cond_words(c) for c in rule.get("conditions") or [])
    if cw:
        pred = (pred + " " + cw) if pred == "where" else (pred + " and " + cw)
    fl = "".join(" and %s %s %g%s" % (FILTER_WORDS.get(f["col"], f["col"]), f["op"], f["value"],
                                       "x" if f["col"] in ("fut_vol5", "vol5", "rvol") else "")
                 for f in rule.get("filters") or [])
    return "%s %s%s" % (who, pred, fl)


def when_text(rule):
    exp = {"today": "today only", "everyday": "every trading day"}.get(rule.get("expiry"), "?")
    rep = {"once": "once per %s per day" % ("index" if rule["target"] == "index" else "stock"),
           "every": "every time (30-min gap)"}.get(rule.get("repeat"), "?")
    return "%s · %s" % (exp, rep)


# ── the questions, with buttons ──
def question(rule, slot):
    """(text, buttons) for one missing slot. buttons = [[(label, callback_data)]]."""
    spec = spec_of(rule) or {}
    cancel = [("❌ Cancel", "a:no")]
    if slot == "value" and spec.get("level"):
        return "Which price? Type the number (e.g. 348).", [cancel]
    if slot == "value":
        unit = spec.get("unit", "")
        ch = spec.get("choices") or [1, 2, 3]
        if rule.get("op") == "abs>=":
            btn = [("±" + _num(c, unit), "a:val:%g" % c) for c in ch]
        elif rule.get("op") == "<=" and not spec.get("near"):
            btn = [("-" + _num(c, unit), "a:val:%g" % -c) for c in ch]
        else:
            btn = [(_num(c, unit), "a:val:%g" % c) for c in ch]
        return ("How much? (%s)\nTap one or type a number." % describe(dict(rule, value=None)), [btn, cancel])
    if slot == "window_min":
        return "In how many minutes?\nTap one or type a number.", [[("5 min", "a:win:5"), ("15 min", "a:win:15"),
                                                                     ("30 min", "a:win:30")], cancel]
    if slot == "repeat":
        return "How often?", [[("Once per %s per day" % ("index" if rule["target"] == "index" else "stock"),
                                "a:rep:once"), ("Every time (30-min gap)", "a:rep:every")]]
    if slot == "expiry":
        return "How long?", [[("Today", "a:exp:today"), ("Everyday", "a:exp:everyday")]]
    raise ValueError(slot)


def confirm_text(rule, notes=(), old=None, aid=None):
    head = "🔔 Change alert #%s — please confirm" % aid if old else "🔔 New alert — please confirm"
    lines = [head]
    if old:
        lines += ["Was:  " + describe(old), "      " + when_text(old)]
        lines += ["Now:  " + describe(rule), "      " + when_text(rule)]
    else:
        early = any(c["left"] in ("open", "high", "low") or c.get("right") in ("open", "high", "low")
                    for c in rule.get("conditions") or [])
        start = "09:30" if rule.get("metric") in ("near_low", "near_high") or early else "09:15"
        lines += ["Watch: " + describe(rule), "Check: every minute, %s–15:30 · %s" % (start, when_text(rule))]
    for n in notes:
        lines.append("⚠️ Can't watch yet: %s — the rest is saved." % n)
    return "\n".join(lines), [[("✅ Save", "a:ok"), ("❌ Cancel", "a:no")]]


def fill(rule, slot, raw):
    """Apply one answer (button data or typed text). Returns the new rule or raises ValueError."""
    r = dict(rule)
    if slot == "value":
        r["value"] = float(str(raw).replace("%", "").replace("x", "").replace("pts", "").strip())
        r = norm_cond(r)
    elif slot == "window_min":
        w = int(float(str(raw).lower().replace("min", "").strip()))
        cap = (T.INDEX_METRICS if r["target"] == "index" else T.LIVE_METRICS)[r["metric"]]
        r["window_min"] = max(5, min(w, cap.get("max_w", 120)))
    elif slot == "repeat":
        if raw not in ("once", "every"):
            raise ValueError(raw)
        r["repeat"] = raw
    elif slot == "expiry":
        if raw not in ("today", "everyday"):
            raise ValueError(raw)
        r["expiry"] = raw
    return r


# ═════════════════════════════════════════════════════════════════════════════
# Storage: one JSON file; IDs permanent and never reused; atomic writes
# ═════════════════════════════════════════════════════════════════════════════
class AlertStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._d = {"next_id": 1, "alerts": []}
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict) and isinstance(d.get("alerts"), list):
                self._d = d
        except FileNotFoundError:
            pass
        except Exception as e:                       # a broken file must not lose new alerts: keep a copy
            print("[AI alerts] could not read %s (%s) — starting empty, old file kept as .bad" % (path, e))
            try:
                os.replace(path, path + ".bad")
            except Exception:
                pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)                 # a crash mid-write never leaves half a file

    def active(self, chat=None, day=None):
        with self._lock:
            out = [dict(a) for a in self._d["alerts"] if a.get("status") in ("on", "paused")
                   and (chat is None or a["chat"] == chat)]
        return [a for a in out if not _expired(a, day)]

    def get(self, aid):
        with self._lock:
            for a in self._d["alerts"]:
                if a["id"] == aid and a.get("status") != "deleted":
                    return dict(a)
        return None

    def add(self, chat, rule, text, day):
        with self._lock:
            n_on = sum(1 for a in self._d["alerts"] if a["chat"] == chat and a.get("status") in ("on", "paused")
                       and not _expired(a, day))
            if n_on >= MAX_ACTIVE:
                raise T.ToolError("You already have %d active alerts (max %d). Delete one with /alerts."
                                  % (n_on, MAX_ACTIVE))
            aid = self._d["next_id"]
            self._d["next_id"] = aid + 1
            self._d["alerts"].append(dict(rule, id=aid, chat=chat, text=text[:300], day=day, status="on",
                                          created=datetime.now().strftime("%Y-%m-%d %H:%M"), fired={},
                                          history=[]))
            self._save()
            return aid

    def update(self, aid, **fields):
        with self._lock:
            for a in self._d["alerts"]:
                if a["id"] == aid:
                    a.setdefault("history", []).append({"at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                        "changed": sorted(fields)})
                    a.update(fields)
                    self._save()
                    return dict(a)
        return None

    def mark_fired(self, aid, day, syms, at):
        with self._lock:
            for a in self._d["alerts"]:
                if a["id"] == aid:
                    f = a.setdefault("fired", {})
                    if f.get("_day") != day:          # new day: everyday alerts start fresh
                        f.clear()
                        f["_day"] = day
                    for s in syms:
                        f[s] = at
                    a["n_fired"] = a.get("n_fired", 0) + 1
                    a["last_fired"] = datetime.fromtimestamp(at).strftime("%H:%M")
                    self._save()
                    return


def _expired(a, day):
    return a.get("expiry") == "today" and day is not None and a.get("day") != day


# ═════════════════════════════════════════════════════════════════════════════
# The minute check: code only
# ═════════════════════════════════════════════════════════════════════════════
def _passes(rule, v):
    if not (spec_of(rule) or {}).get("cond"):
        return True
    op, x = rule.get("op"), rule.get("value")
    if v is None or x is None:
        return False
    if op == "abs>=":
        return abs(v) >= x
    if op == "between":
        return x <= v <= (rule.get("value2") if rule.get("value2") is not None else x)
    return v >= x if op == ">=" else v <= x


def hits_for(ctx, rule):
    """[(name, value)] matching the rule right now, plus the tool result (stocks) or None.
    Raises T.ToolError when the data is not there (not ready yet / no samples)."""
    if rule["target"] == "index":
        vals = T.index_values(ctx, rule["metric"], rule.get("window_min") or 0, rule.get("symbols") or ["ANY"])
        return [(k, v) for k, v in sorted(vals.items()) if _passes(rule, v[0])], vals
    if rule["target"] == "query":
        r, err = run_query(ctx, rule)
        if err:
            if err.startswith("No "):              # an empty list right now
                return [], None
            raise T.ToolError(err)
        return [(n, None) for n in names_of(r)], r
    if rule["metric"] == "screen":
        r, err = T.run(ctx, "live_screen", {"conditions": rule["conditions"], "sector": rule.get("sector"),
                                            "universe": rule.get("universe") or "all"},
                       only=rule.get("symbols") or None, all_rows=True)
        if err:
            if err.startswith("No stock "):
                return [], None
            raise T.ToolError(err)
        return [(s_, v) for s_, v in r["rows"]], r
    args = {"metric": rule["metric"], "window_min": rule.get("window_min") or 0, "sector": rule.get("sector")}
    if rule.get("filters"):
        args["filters"] = rule["filters"]
        cols = [f["col"] for f in rule["filters"] if f["col"] in VOLUME_COLS]
        if cols:
            args["show"] = sorted(set(cols), key=cols.index)[:4]     # the ping shows what was checked
    if rule.get("universe") in ("watchlist", "setups"):
        args["universe"] = rule["universe"]
    r, err = T.run(ctx, "live_top", args, only=rule.get("symbols") or None, all_rows=True)
    if err:
        if err.startswith("No stock "):            # a crossing that did not happen: nothing to say
            return [], None
        raise T.ToolError(err)
    hits = [(s, v) for s, v in r["rows"] if _passes(rule, v)]
    if rule.get("conditions") and hits:                       # a metric + conditions: both must hold
        ok = {x for x, _f in T.screen(ctx, rule["conditions"], only={h for h, _v in hits})[0]}
        hits = [(s, v) for s, v in hits if s in ok]
    return hits, r


def names_of(result):
    """Names in a list answer, in display order (stocks, or sectors for the sector ranking)."""
    out = []
    for x in (result or {}).get("rows") or []:
        if isinstance(x, (tuple, list)) and x:
            out.append(x[0])
        elif isinstance(x, dict) and (x.get("sym") or x.get("sector")):
            out.append(x.get("sym") or x.get("sector"))
    return out


def run_query(ctx, rule):
    return T.run(ctx, rule["tool"], dict(rule.get("args") or {}), all_rows=True)


_MEMBERS = {}                    # alert id -> (day, set of names in the list at the last check)


def new_entries(alert, hits, day):
    """'alert above result': only names that were NOT in the list at the previous check.
    The first check of a day (or after a restart) only records the list, it never pings —
    except on the creation day, where the list saved with the alert is the starting point."""
    names = [n for n, _v in hits]
    prev = _MEMBERS.get(alert["id"])
    if prev is None or prev[0] != day:
        base = set(alert.get("baseline") or []) if alert.get("day") == day else None
        _MEMBERS[alert["id"]] = (day, set(names))
        if base is None:
            return []
        prev = (day, base)
    _MEMBERS[alert["id"]] = (day, set(names))
    return [(n, v) for n, v in hits if n not in prev[1]]


def due(alert, hits, now):
    """Names to ping now: not pinged today (once) or not in the last 30 min (every)."""
    fired = alert.get("fired") or {}
    day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    if fired.get("_day") != day:
        fired = {}
    out = []
    for name, _v in hits:
        last = fired.get(name)
        if last is None or (alert.get("repeat") == "every" and now - float(last) >= GAP_REPEAT_S):
            out.append(name)
    return out


def ping_text(ctx, alert, names, result, now):
    head = "🔔 Alert #%d · %s · %s" % (alert["id"], datetime.fromtimestamp(now).strftime("%H:%M"), describe(alert))
    if alert["target"] == "index":
        lines = [head]
        for n in names:
            v, px, pct = result[n]
            lines.append("%s %s · day %s" % (n, T._px2(px), T._p(pct)) +
                         ("" if (spec_of(alert) or {}).get("event") else " · now %s%s" % (
                             ("%+.2f" % v) if alert["metric"] in ("pct", "move") else "%.2f" % v,
                             (spec_of(alert) or {}).get("unit", ""))))
        return "\n".join(lines)
    if alert["target"] == "query":
        order = names_of(result)
        keep = [x for x in result["rows"] if names_of({"rows": [x]})[0] in names]
        r = dict(result, rows=keep, notes=None, n=len(keep), unsupported=[], scope=None)
        body = T.format_reply(r).split("\n")
        ranks = " · ".join("%s #%d of %d" % (n, order.index(n) + 1, len(order)) for n in names if n in order)
        return head + "\nrank in the list now: " + ranks + "\n" + "\n".join(body[1:]).strip("\n") + "\nNot advice."
    r = dict(result, rows=[x for x in result["rows"] if x[0] in names], notes=None, n=len(names), unsupported=[])
    body = T.format_reply(r).split("\n")
    return head + "\n" + "\n".join(body[1:]).strip("\n") + "\nNot advice."


def check_alerts(ctx, store, send, now=None, day=None):
    """Run every active alert once. send(chat, text) -> True when Telegram took it.
    Never raises; a failed send leaves the alert un-fired, so the next minute retries."""
    now = now if now is not None else time.time()
    day = day or datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    sent = 0
    hm = datetime.fromtimestamp(now).hour * 60 + datetime.fromtimestamp(now).minute
    for a in store.active(day=day):
        if a.get("status") != "on":
            continue
        if a.get("metric") in ("near_low", "near_high") and hm < NEAR_FROM:
            continue
        if hm < NEAR_FROM and any(c["left"] in ("open", "high", "low") or c.get("right") in ("open", "high", "low")
                                  for c in a.get("conditions") or []):
            continue                                   # first 15 min: open / high / low are the same for most
        try:
            hits, res = hits_for(ctx, a)
        except T.ToolError:
            continue                                  # data not ready (restart / pre-open): try next minute
        except Exception as e:
            print("[AI alerts] #%s check error: %s" % (a.get("id"), e))
            continue
        if a["target"] == "query":
            hits = new_entries(a, hits, day)
        names = due(a, hits, now)
        if not names:
            continue
        more = max(0, len(names) - MAX_PING_ROWS)
        names = names[:MAX_PING_ROWS]                  # hits come ranked: the strongest 10
        try:
            txt = ping_text(ctx, a, names, res, now)
            if more:
                txt += "\n+%d more — narrow it (a sector / named stocks) with: alert %d ..." % (more, a["id"])
            ok = send(a["chat"], txt)
        except Exception as e:
            print("[AI alerts] #%s send error: %s" % (a.get("id"), e))
            ok = False
        if ok:
            store.mark_fired(a["id"], day, names, now)
            sent += 1
    return sent


def list_text(store, chat, day):
    al = store.active(chat, day)
    if not al:
        return "No active alerts.\nCreate one: alert me when HDFCBANK crosses above VWAP today", []
    lines, buttons = ["🔔 Your alerts (%d of max %d)" % (len(al), MAX_ACTIVE)], []
    for a in sorted(al, key=lambda x: x["id"]):
        st = "⏸ paused" if a.get("status") == "paused" else "on"
        lines.append("#%d %s\n   %s · %s · fired %d%s" % (
            a["id"], describe(a), when_text(a), st, a.get("n_fired", 0),
            (" (last %s)" % a["last_fired"]) if a.get("last_fired") else ""))
        buttons.append([("✏️ #%d" % a["id"], "a:chg:%d" % a["id"]),
                        (("▶️ Resume #%d" if a.get("status") == "paused" else "⏸ #%d") % a["id"],
                         ("a:res:%d" if a.get("status") == "paused" else "a:pau:%d") % a["id"]),
                        ("🗑 #%d" % a["id"], "a:del:%d" % a["id"])])
    return "\n".join(lines), buttons


# ═════════════════════════════════════════════════════════════════════════════
# The conversation: create -> ask what is missing (buttons / typed) -> confirm -> save.
# Returns replies as [(text, buttons)]; ai_chat only sends them.
# ═════════════════════════════════════════════════════════════════════════════
_NUM = re.compile(r"^\s*[-+]?\d+(\.\d+)?\s*(%|x|min|pts)?\s*$", re.I)


class AlertFlow:
    def __init__(self, ctx, router, store, err_text=None, recent=None):
        self.ctx, self.router, self.store = ctx, router, store
        self.recent = recent or (lambda chat: [])     # the bot's last answers (most recent first)
        self.err_text = err_text or (lambda kind: "Could not read that alert right now (%s)." % kind)
        self.drafts = {}                        # chat -> draft (RAM: a draft is 10 min of typing, not data)

    # ── entry points ──
    def on_text(self, chat, text):
        """Replies if the message belongs to the alert flow, else None (= a normal question)."""
        d = self._draft(chat)
        if d and d["missing"] and d["missing"][0] in ("value", "window_min") and _NUM.match(text or ""):
            return self._answer(chat, d["missing"][0], text)
        m = QUERY_PREFIX.match(text or "")
        if m:
            return self._from_answer(chat, m.group(1), text)
        m = THESE_PREFIX.match(text or "")
        if m:
            return self._these(chat, m.group(1), text)
        m = EDIT_PREFIX.match(text or "")
        if m:
            return self._edit(chat, int(m.group(1)), m.group(2).strip(), text)
        if CREATE_PREFIX.match(text or ""):
            return self._create(chat, CREATE_PREFIX.sub("", text, count=1), text)
        return None

    def on_button(self, chat, data):
        parts = (data or "").split(":")
        if len(parts) < 2 or parts[0] != "a":
            return [("Unknown button.", None)]
        act, arg = parts[1], (parts[2] if len(parts) > 2 else "")
        slot = {"val": "value", "win": "window_min", "rep": "repeat", "exp": "expiry"}.get(act)
        if slot:
            return self._answer(chat, slot, arg)
        if act == "ok":
            return self._save(chat)
        if act == "no":
            self.drafts.pop(chat, None)
            return [("Cancelled — nothing saved.", None)]
        aid = int(arg) if arg.isdigit() else None
        a = self.store.get(aid) if aid else None
        if not a or a["chat"] != chat:
            return [("Alert #%s not found." % arg, None)]
        if act == "chg":
            return [("Send the change in words, e.g.\nalert %d make it 1.5%%\nalert %d add ICICIBANK\n"
                     "alert %d everyday" % (aid, aid, aid), None)]
        if act == "pau":
            self.store.update(aid, status="paused")
            return [("⏸ Alert #%d paused. Resume from /alerts." % aid, None)]
        if act == "res":
            self.store.update(aid, status="on")
            return [("▶️ Alert #%d is on again." % aid, None)]
        if act == "del":
            return [("Delete alert #%d?\n%s" % (aid, describe(a)),
                     [[("🗑 Yes, delete #%d" % aid, "a:delok:%d" % aid), ("Keep it", "a:keep:%d" % aid)]])]
        if act == "delok":
            self.store.update(aid, status="deleted")
            return [("🗑 Alert #%d deleted (its number is never reused)." % aid, None)]
        if act == "keep":
            return [("Kept alert #%d." % aid, None)]
        return [("Unknown button.", None)]

    def _last_answer(self, chat):
        h = self.recent(chat)
        if not h:
            return None, [("No recent answer to watch — ask a question first, then write 'alert above result' "
                           "(answers are kept 15 min).", None)]
        return h[0], None

    def _from_answer(self, chat, rest, text):
        """'alert above result': watch the last answer's QUESTION; ping names that newly enter it."""
        last, err = self._last_answer(chat)
        if err:
            return err
        tool, args = last.get("tool"), dict(last.get("args") or {})
        if tool not in LIST_TOOLS:
            why = ("it is from past sessions and cannot change during the day" if tool in
                   ("top_symbols", "level_breaks_history", "sector_history", "symbol_history", "signal_performance")
                   else "it is not a list")
            return [("⚠️ Can't watch the last answer (%s): %s.\nAsk a live list question first, e.g. 'stocks "
                     "crossed above vwap last 5 min in bank'." % (last.get("q", "")[:60], why), None)]
        notes = []
        if args.pop("from_previous", None):
            notes.append("'from that list' (watching all stocks, not only the earlier list)")
        args.pop("limit", None)
        rule = {"target": "query", "metric": "query", "tool": tool, "args": args, "q": last.get("q", ""),
                "window_min": 0, "filters": []}
        r, e = run_query(self.ctx, rule)
        rule["baseline"] = names_of(r) if r else []
        exp, rep = words_from_text(rest)
        rule["expiry"], rule["repeat"] = exp or "today", rep or "once"
        self.drafts[chat] = {"mode": "create", "rule": rule, "missing": [], "notes": notes, "text": text,
                             "ts": time.time()}
        t, b = confirm_text(rule, notes)
        base = rule["baseline"]
        t += "\nAlready in it now (no ping): %s" % (
            (", ".join(base[:10]) + (" +%d more" % (len(base) - 10) if len(base) > 10 else "")) if base else "none")
        return [(t, b)]

    def _these(self, chat, rest, text):
        """'alert these when pace > 1.5': the condition for exactly the stocks of the last answer."""
        last, err = self._last_answer(chat)
        if err:
            return err
        syms = [x for x in (last.get("symbols") or []) if x]
        if not syms or last.get("tool") in ("live_sector_rank", "sector_history"):
            return [("⚠️ The last answer has no stocks to watch.", None)]
        if not rest.strip():
            return [("Write what to watch for these %d stocks, e.g.\nalert these when pace changes" % len(syms), None)]
        out = self._create(chat, rest, text)
        d = self.drafts.get(chat)
        if not d or d["rule"].get("target") != "stocks":
            return out
        d["rule"].update(symbols=sorted(syms), sector=None)
        return self._next(chat)

    def list(self, chat):
        return [list_text(self.store, chat, self.ctx.today)]

    # ── steps ──
    def _draft(self, chat):
        d = self.drafts.get(chat)
        if d and time.time() - d["ts"] > DRAFT_TTL_S:
            self.drafts.pop(chat, None)
            return None
        return d

    def _route(self, prompt):
        try:
            return self.router.route(prompt), None
        except Exception as e:
            return None, [(self.err_text(getattr(e, "kind", "unknown")), None)]

    def _create(self, chat, body, text):
        import ai_examples
        if not body.strip():
            return [("Write what to watch after 'alert me', e.g.\nalert me when HDFCBANK crosses above VWAP today",
                     None)]
        route, err = self._route(ai_examples.prompt_with_examples(body))
        if err:
            return err
        if route.name == "cannot_alert":
            return [("⚠️ I can't watch that from the live data: %s\nI can watch price %%, volume, fut vol, OI, "
                     "PACE, RVOL, VWAP, IV, score, crossings and 'near a level', for stocks, sectors and VIX."
                     % (route.args.get("reason") or "not in the data"), None)]
        if route.name == "update_alert":
            m = EDIT_PREFIX.match(text)
            return self._edit(chat, int(route.args.get("id") or (m.group(1) if m else 0)), body, text, route)
        try:
            rule, missing, notes = validate_rule(self.ctx, route.args, text)
        except T.ToolError as e:
            return [("⚠️ %s" % e, None)]
        blocking = [n for n in notes if not _ORDER_W.search(n)] + scope_problems(self.ctx, rule, text)
        if blocking:
            return [("⚠️ Can't watch: %s.\nNothing saved — every part of an alert must be watchable, and "
                     "saving only the rest would change its meaning." % "; ".join(blocking), None)]
        self.drafts[chat] = {"mode": "create", "rule": rule, "missing": missing, "notes": notes,
                             "text": text, "ts": time.time()}
        return self._next(chat)

    def _edit(self, chat, aid, rest, text, route=None):
        a = self.store.get(aid)
        if not a or a["chat"] != chat:
            return [("Alert #%d not found. /alerts lists yours." % aid, None)]
        if a.get("target") == "query" and rest.lower().strip(" .!") not in _CMD_WORDS and rest.strip() and \
                not (words_from_text(rest)[0] or words_from_text(rest)[1]):
            return [("Alert #%d follows a saved question. To change it, ask the new question and write "
                     "'alert above result' (then delete #%d)." % (aid, aid), None)]
        word = rest.lower().strip(" .!")
        if not word:
            return [("#%d %s\n%s" % (aid, describe(a), when_text(a)),
                     [[("✏️ Change", "a:chg:%d" % aid), ("⏸ Pause", "a:pau:%d" % aid),
                       ("🗑 Delete", "a:del:%d" % aid)]])]
        if word in _CMD_WORDS:
            act = _CMD_WORDS[word]
            return self.on_button(chat, "a:%s:%d" % ({"delete": "del", "pause": "pau", "resume": "res"}[act], aid))
        exp, rep = words_from_text(rest)
        left = _REPEAT_W.sub("", _ONCE_W.sub("", _EVERY_W.sub("", _TODAY_W.sub("", rest)))).strip(" .,!")
        if (exp or rep) and not left:                # only 'always' / 'today only' / 'every time' / 'once'
            rule = {k: a.get(k) for k in a if k not in ("id", "chat", "text", "day", "status", "created", "fired",
                                                         "history", "n_fired", "last_fired")}
            rule["expiry"], rule["repeat"] = exp or a.get("expiry"), rep or a.get("repeat")
            self.drafts[chat] = {"mode": "update", "id": aid, "old": a, "rule": rule, "missing": [],
                                 "notes": [], "text": text, "ts": time.time()}
            return self._next(chat)
        if route is None:
            route, err = self._route("alert #%d is: %s (%s). Change: %s" % (aid, describe(a), when_text(a), rest))
            if err:
                return err
        if route.name != "update_alert":
            return [("I could not read a change for alert #%d. Example: alert %d make it 1.5%%" % (aid, aid), None)]
        ch = {k: v for k, v in route.args.items() if k not in ("id", "add_symbols", "remove_symbols")
              and v not in (None, "", [])}
        raw = {k: a.get(k) for k in ("target", "metric", "window_min", "symbols", "sector", "op", "value",
                                     "filters", "unsupported")}
        if "metric" in ch and ch["metric"] != a["metric"]:
            raw.update(op=None, value=None, window_min=None)    # a new measure: its own number / window
        raw.update(ch)
        syms = set(raw.get("symbols") or [])
        syms |= {str(x).upper() for x in route.args.get("add_symbols") or []}
        syms -= {str(x).upper() for x in route.args.get("remove_symbols") or []}
        raw["symbols"] = sorted(syms)
        try:
            rule, missing, notes = validate_rule(self.ctx, raw, rest)
        except T.ToolError as e:
            return [("⚠️ %s" % e, None)]
        exp, rep = words_from_text(rest)
        rule["expiry"], rule["repeat"] = exp or a.get("expiry"), rep or a.get("repeat")
        missing = list(missing)
        self.drafts[chat] = {"mode": "update", "id": aid, "old": a, "rule": rule, "missing": missing,
                             "notes": notes, "text": text, "ts": time.time()}
        return self._next(chat)

    def _answer(self, chat, slot, raw):
        d = self._draft(chat)
        if not d:
            return [("That alert draft expired — send it again.", None)]
        if slot not in d["missing"]:
            return [("Already set. " + question(d["rule"], d["missing"][0])[0], None)] if d["missing"] \
                else self._next(chat)
        try:
            d["rule"] = fill(d["rule"], slot, raw)
        except (ValueError, TypeError):
            t, b = question(d["rule"], slot)
            return [("Please tap a button or type a number.\n" + t, b)]
        d["missing"].remove(slot)
        d["ts"] = time.time()
        return self._next(chat)

    def _next(self, chat):
        d = self.drafts[chat]
        if d["missing"]:
            return [question(d["rule"], d["missing"][0])]
        if d["mode"] == "update":
            return [confirm_text(d["rule"], d["notes"], old=d["old"], aid=d["id"])]
        return [confirm_text(d["rule"], d["notes"])]

    def _save(self, chat):
        d = self._draft(chat)
        if not d or d["missing"]:
            return [("Nothing to save — the draft expired. Send the alert again.", None)]
        self.drafts.pop(chat, None)
        rule = d["rule"]
        try:
            if d["mode"] == "update":
                self.store.update(d["id"], **dict(rule, fired={}))     # a changed rule starts fresh
                aid = d["id"]
                head = "✅ Alert #%d changed" % aid
            else:
                aid = self.store.add(chat, rule, d["text"], self.ctx.today)
                head = "✅ Saved alert #%d" % aid
        except T.ToolError as e:
            return [("⚠️ %s" % e, None)]
        w = rule.get("window_min") or 0
        ready = ""
        if w:
            ready = "\nNeeds %d min of live samples — checks start once they exist." % w
        return [("%s\n%s\n%s%s\n/alerts to change, pause or delete." % (head, describe(rule), when_text(rule),
                                                                         ready), None)]
