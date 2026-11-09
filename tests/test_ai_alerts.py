# -*- coding: utf-8 -*-
"""Standing alerts: the full conversation (incl. the user's NEXT reply after every bot
question), the file store, and the minute checker. No network: fake router + fake Telegram."""
import json
import re
import os

import pytest

import ai_alerts as A
import ai_chat as C
import ai_tools as T
from ai_llm import FakeRouter, RouterError
from test_ai_tools import (FakeTG, NOW, RecRouter, TODAY, _engine_data, _feed, _fresh_samples,  # noqa: F401
                           _tiny_sectors, ctx)

CHAT = "111"
NOWE = NOW.timestamp()


class AlertRouter(FakeRouter):
    """Maps the text after the prefix (or the edit prompt) to a fixed tool call."""
    def __init__(self, table):
        super().__init__(table)
        self.prompts = []

    def route(self, q):
        self.prompts.append(q)
        q = q.split("\nNEW: ", 1)[-1]                  # the solved examples in front are context, not the ask
        for key, v in self.table.items():
            if key in q:
                if isinstance(v, Exception):
                    raise v
                return type("R", (), {"name": v[0], "args": dict(v[1]), "model": "fake", "tokens": 0, "ms": 0})()
        return type("R", (), {"name": "cannot_alert", "args": {"reason": "unknown"}, "model": "fake"})()


@pytest.fixture
def store(tmp_path):
    return A.AlertStore(str(tmp_path / "data" / "ai_alerts.json"))


def _flow(ctx, store, table):
    return A.AlertFlow(ctx, AlertRouter(table), store, err_text=C.router_message)


def _btn_data(reply):
    return [c for row in (reply[1] or []) for _l, c in row]


VWAP_AAA = ("create_alert", {"target": "stocks", "metric": "vwap_up", "symbols": ["aaa"]})
VIX_MOVE = ("create_alert", {"target": "index", "metric": "move", "symbols": ["VIX"]})


# ── create ──────────────────────────────────────────────────────────────────────────
def test_complete_sentence_goes_straight_to_confirm(ctx, store):
    f = _flow(ctx, store, {"AAA crosses above VWAP": VWAP_AAA})
    (text, btn), = f.on_text(CHAT, "alert me when AAA crosses above VWAP once today")
    assert "Watch: AAA crosses ABOVE VWAP" in text and "today only · once per stock per day" in text
    assert _btn_data((text, btn)) == ["a:ok", "a:no"]
    (done, _), = f.on_button(CHAT, "a:ok")
    assert done.startswith("✅ Saved alert #1") and "Needs 5 min of live samples" in done
    a = store.get(1)
    assert a["metric"] == "vwap_up" and a["symbols"] == ["AAA"] and a["window_min"] == 5 and a["status"] == "on"


def test_missing_pieces_are_asked_one_by_one_with_buttons(ctx, store):
    """Only a missing NUMBER is asked. Today + once are the defaults (user 05-11)."""
    ctx.state.SECTOR_INTEL["data"]["INDIA VIX"]["price"] = 14.0
    f = _flow(ctx, store, {"VIX moves fast": VIX_MOVE})
    q1, = f.on_text(CHAT, "alert: VIX moves fast")
    assert q1[0].startswith("How much?") and _btn_data(q1) == ["a:val:0.3", "a:val:0.5", "a:val:1", "a:no"]
    q2, = f.on_button(CHAT, "a:val:1")                         # tap
    assert q2[0].startswith("In how many minutes?") and "a:no" in _btn_data(q2)
    conf, = f.on_text(CHAT, "5 min")                           # typed answer -> straight to confirm
    assert "Watch: INDIA VIX moves ±1% in 5 min" in conf[0]
    assert "today only · once per index per day" in conf[0]
    done, = f.on_button(CHAT, "a:ok")
    assert done[0].startswith("✅ Saved alert #1")
    a = store.get(1)
    assert (a["target"], a["symbols"], a["op"], a["value"], a["window_min"], a["expiry"], a["repeat"]) == \
        ("index", ["INDIA VIX"], "abs>=", 1.0, 5, "today", "once")

def test_next_reply_variants_while_a_question_is_open(ctx, store):
    f = _flow(ctx, store, {"VIX moves": VIX_MOVE})
    f.on_text(CHAT, "alert me VIX moves")
    assert f.on_text(CHAT, "top gainers today") is None          # a normal question: not swallowed
    again, = f.on_button(CHAT, "a:win:15")                        # an answer out of order is kept...
    assert again[0].startswith("How much?") and f.drafts[CHAT]["missing"] == ["value"]   # ...and we re-ask
    same, = f.on_button(CHAT, "a:win:30")                         # the same slot twice
    assert same[0].startswith("Already set. How much?")
    bad, = f.on_button(CHAT, "a:val:abc")
    assert bad[0].startswith("Please tap a button or type a number")
    f.on_text(CHAT, "-2")                                         # abs>=: a minus sign is ignored
    assert f.drafts[CHAT]["rule"]["value"] == 2
    cancel, = f.on_button(CHAT, "a:no")
    assert cancel[0].startswith("Cancelled") and store.active() == []
    late, = f.on_button(CHAT, "a:ok")
    assert "expired" in late[0]

def test_draft_expires(ctx, store, monkeypatch):
    f = _flow(ctx, store, {"VIX moves": VIX_MOVE})
    f.on_text(CHAT, "alert me VIX moves")
    f.drafts[CHAT]["ts"] -= A.DRAFT_TTL_S + 1
    assert "expired" in f.on_button(CHAT, "a:val:1")[0][0]


def test_words_today_everyday_once_every_are_read_by_code(ctx, store):
    f = _flow(ctx, store, {"AAA crosses": VWAP_AAA})
    (t, _), = f.on_text(CHAT, "alert me AAA crosses above vwap har vakhte always")
    assert "every trading day · every time (30-min gap)" in t
    (t, _), = f.on_text(CHAT, "alert me AAA crosses above vwap")
    assert "today only · once per stock per day" in t                       # defaults, nothing asked
    assert A.words_from_text("alert: x forever") == ("everyday", None)
    assert A.words_from_text("alert: x roj ek var") == ("everyday", "once")
    assert A.words_from_text("alert me any stock near the low of today or break low") == (None, None)

def test_cannot_alert_and_unsupported_parts(ctx, store):
    f = _flow(ctx, store, {"PCR": ("cannot_alert", {"reason": "PCR is not in the live data"})})
    (t, _), = f.on_text(CHAT, "alert me when nifty PCR above 1.2 today")
    assert t.startswith("⚠️ I can't watch that") and "PCR" in t
    f = _flow(ctx, store, {"news": ("create_alert", dict(VWAP_AAA[1], unsupported="news"))})
    (t, _), = f.on_text(CHAT, "alert me AAA crosses vwap or news once today")
    assert t.startswith("⚠️ Can't watch: news.\nNothing saved") and CHAT not in f.drafts      # AND: never half


def test_bad_metric_and_gemini_errors(ctx, store):
    f = _flow(ctx, store, {"x": ("create_alert", {"target": "stocks", "metric": "pcr"}),
                           "y": RouterError("quota_day", "429")})
    assert f.on_text(CHAT, "alert me x")[0][0].startswith("⚠️ Can't watch 'pcr' for stocks")
    assert "resets about 13:30 IST" in f.on_text(CHAT, "alert me y")[0][0]
    assert "Write what to watch" in f.on_text(CHAT, "alert me")[0][0]


def test_validate_rule_numbers_mean_one_thing(ctx):
    r, miss, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "pct", "op": "<=", "value": 2}, "x once today")
    assert (r["op"], r["value"], miss) == ("<=", -2, [])                    # 'falls 2%'
    r, _, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "near_low", "op": ">=", "value": -0.3}, "")
    assert (r["op"], r["value"]) == ("<=", 0.3)
    r, miss, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "fut_volume", "value": 3}, "")
    assert miss == ["window_min"] and (r["expiry"], r["repeat"]) == ("today", "once")
    r, _, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "pct", "window_min": 2, "value": 1}, "")
    assert r["window_min"] == 5
    r, _, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "fut_volume", "window_min": 90, "value": 3}, "")
    assert r["window_min"] == 40
    r, _, _ = A.validate_rule(ctx, {"target": "stocks", "metric": "vwap_up", "sector": "NIFTS BANK",
                                    "filters": [{"col": "rvol", "op": ">", "value": 2}]}, "")
    assert r["sector"] == "NIFTY BANK" and r["filters"] == [{"col": "rvol", "op": ">", "value": 2.0}]
    assert A.describe(dict(r, expiry="today", repeat="once")) == \
        "any BANK stock crosses ABOVE VWAP and rvol > 2x"
    with pytest.raises(T.ToolError):
        A.validate_rule(ctx, {"target": "index", "metric": "move", "symbols": ["CHEMICALS INDEX"]}, "")


# ── change / pause / delete ─────────────────────────────────────────────────────────
def _saved(ctx, store, sentence="alert me AAA crosses above vwap once today", table=None):
    f = _flow(ctx, store, table or {"AAA crosses": VWAP_AAA})
    f.on_text(CHAT, sentence)
    f.on_button(CHAT, "a:ok")
    return f


def test_change_shows_old_and_new_then_saves(ctx, store):
    f = _saved(ctx, store, table={"Change: make it": ("update_alert", {"id": 1, "metric": "near_vwap", "value": 0.2}),
                                  "Change: add BBB": ("update_alert", {"id": 1, "add_symbols": ["bbb"]}),
                                  "AAA crosses": VWAP_AAA})
    store.mark_fired(1, TODAY, ["AAA"], NOWE)
    (t, btn), = f.on_text(CHAT, "alert 1 make it near vwap 0.2%")
    assert t.startswith("🔔 Change alert #1") and "Was:  AAA crosses ABOVE VWAP" in t
    assert "Now:  AAA trades within 0.2% of VWAP" in t and "today only · once per stock per day" in t
    f.on_button(CHAT, "a:ok")
    a = store.get(1)
    assert a["metric"] == "near_vwap" and a["fired"] == {} and a["history"]
    f.on_text(CHAT, "alert 1 add BBB")
    f.on_button(CHAT, "a:ok")
    assert store.get(1)["symbols"] == ["AAA", "BBB"]


def test_pause_resume_delete_and_ids_never_reused(ctx, store):
    f = _saved(ctx, store)
    assert f.on_text(CHAT, "alert 1 pause")[0][0].startswith("⏸ Alert #1 paused")
    txt, btn = f.list(CHAT)[0]
    assert "⏸ paused" in txt and "a:res:1" in [c for row in btn for _l, c in row]
    f.on_button(CHAT, "a:res:1")
    (q, b), = f.on_text(CHAT, "alert 1 delete")
    assert q.startswith("Delete alert #1?") and _btn_data((q, b)) == ["a:delok:1", "a:keep:1"]
    f.on_button(CHAT, "a:delok:1")
    assert store.get(1) is None and f.list(CHAT)[0][0].startswith("No active alerts")
    f.on_text(CHAT, "alert me AAA crosses above vwap once today")
    assert f.on_button(CHAT, "a:ok")[0][0].startswith("✅ Saved alert #2")
    assert f.on_text("999", "alert 2 delete")[0][0].startswith("Alert #2 not found")      # not your alert
    assert f.on_text(CHAT, "alert 2")[0][1]                                               # menu with buttons


def test_max_active_alerts(ctx, store):
    f = _flow(ctx, store, {"AAA crosses": VWAP_AAA})
    for i in range(A.MAX_ACTIVE):
        f.on_text(CHAT, "alert me AAA crosses above vwap once today")
        f.on_button(CHAT, "a:ok")
    f.on_text(CHAT, "alert me AAA crosses above vwap once today")
    assert "max 10" in f.on_button(CHAT, "a:ok")[0][0]


def test_store_survives_restart_and_bad_file(ctx, tmp_path):
    p = str(tmp_path / "data" / "ai_alerts.json")
    s1 = A.AlertStore(p)
    s1.add(CHAT, {"target": "stocks", "metric": "vwap_up", "expiry": "everyday", "repeat": "once"}, "t", TODAY)
    s2 = A.AlertStore(p)                                         # 'restart'
    assert [a["id"] for a in s2.active()] == [1] and s2._d["next_id"] == 2
    open(p, "w").write("{broken")
    s3 = A.AlertStore(p)
    assert s3.active() == [] and os.path.exists(p + ".bad")


# ── the minute check ────────────────────────────────────────────────────────────────
def _sender(log, ok=True):
    def send(chat, text):
        log.append((chat, text))
        return ok
    return send


def test_vwap_cross_fires_once_per_stock_per_day(ctx, store):
    store.add(CHAT, {"target": "stocks", "metric": "vwap_up", "symbols": [], "window_min": 5, "filters": [],
                     "expiry": "today", "repeat": "once"}, "t", TODAY)
    _feed(ctx, 20, lambda s, k: {"AAA": 102.0 if k > 2 else 105.0}.get(s))
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1
    chat, text = out[0]
    assert text.startswith("🔔 Alert #1 · ") and "any F&O stock crosses ABOVE VWAP" in text
    assert "AAA" in text and "day +2.00%" in text and "price 102.00 → 105.00" in text and "Not advice." in text
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 0       # no spam
    assert store.get(1)["n_fired"] == 1


def test_repeat_every_waits_30_min(ctx, store):
    store.add(CHAT, {"target": "stocks", "metric": "near_low", "op": "<=", "value": 0.3, "symbols": ["BBB"],
                     "window_min": 0, "filters": [], "expiry": "today", "repeat": "every"}, "t", TODAY)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1             # BBB falling: at its low
    assert "BBB" in out[0][1] and "AT / BREAKING" in out[0][1]
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 600, day=TODAY) == 0
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 1801, day=TODAY) == 1


def test_near_low_waits_for_0930_and_caps_rows(ctx, store, monkeypatch):
    store.add(CHAT, {"target": "stocks", "metric": "near_low", "op": "<=", "value": 50, "symbols": [],
                     "window_min": 0, "filters": [], "expiry": "today", "repeat": "once"}, "t", TODAY)
    early = NOWE - (NOW.hour * 60 + NOW.minute - 9 * 60 - 20) * 60          # 09:20
    assert A.check_alerts(ctx, store, _sender([]), now=early, day=TODAY) == 0
    monkeypatch.setattr(A, "MAX_PING_ROWS", 2)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1
    assert "+2 more — narrow it" in out[0][1] and len(store.get(1)["fired"]) == 3   # _day + 2 names


def test_failed_send_retries_next_minute(ctx, store):
    store.add(CHAT, {"target": "stocks", "metric": "near_low", "op": "<=", "value": 0.3, "symbols": ["BBB"],
                     "window_min": 0, "filters": [], "expiry": "today", "repeat": "once"}, "t", TODAY)
    out = []
    assert A.check_alerts(ctx, store, _sender(out, ok=False), now=NOWE, day=TODAY) == 0
    assert store.get(1).get("n_fired") is None
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 1


def test_expiry_pause_and_new_day(ctx, store):
    rule = {"target": "stocks", "metric": "near_low", "op": "<=", "value": 0.3, "symbols": ["BBB"],
            "window_min": 0, "filters": [], "repeat": "once"}
    store.add(CHAT, dict(rule, expiry="today"), "t", "2026-11-03")             # yesterday's 'today' alert
    store.add(CHAT, dict(rule, expiry="everyday"), "t", "2026-11-03")
    store.add(CHAT, dict(rule, expiry="everyday"), "t", TODAY)
    store.update(3, status="paused")
    store.mark_fired(2, "2026-11-03", ["BBB"], NOWE - 86400)                     # fired yesterday
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1
    assert out[0][1].startswith("🔔 Alert #2")                                  # new day: fires again


def test_index_alerts_vix_move_and_any_sector_new_high(ctx, store):
    si = ctx.state.SECTOR_INTEL["data"]
    si["NIFTY IT"] = {"price": 100.0, "pct": 0.1, "day_high": 101.0, "day_low": 99.0, "vwap_v": 100.0}
    si["INDIA VIX"].update({"price": 14.0, "day_high": 14.2, "day_low": 13.8, "vwap_v": 14.0})
    si["NIFTY 50"].update({"price": 24000.0, "day_high": 24100.0, "day_low": 23900.0, "vwap_v": 24000.0})
    for k in range(10, -1, -1):
        si["INDIA VIX"]["price"] = 14.0 if k > 4 else 14.2                 # +1.4% in the last 5 min
        if k == 2:
            si["NIFTY IT"].update(price=101.5, day_high=101.5)              # new day high 2 min ago
            si["NIFTY 50"].update(price=24200.0, day_high=24200.0)          # NIFTY too, but it is not a sector
        T.sample_once(ctx, NOWE - 60 * k)
    store.add(CHAT, {"target": "index", "metric": "move", "symbols": ["INDIA VIX"], "op": "abs>=", "value": 1,
                     "window_min": 5, "expiry": "today", "repeat": "once"}, "t", TODAY)
    store.add(CHAT, {"target": "index", "metric": "day_high_up", "symbols": ["ANY"], "window_min": 5,
                     "expiry": "today", "repeat": "once"}, "t", TODAY)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 2
    vix, sec = out[0][1], out[1][1]
    assert "INDIA VIX moves ±1% in 5 min" in vix and "INDIA VIX 14.20" in vix and "now +1.43%" in vix
    assert "any sector index makes a new day high" in sec and "NIFTY IT 101.50" in sec
    assert "INDIA VIX" not in sec and "NIFTY 50" not in sec                    # not sectors


def test_checker_survives_missing_data(ctx, store):
    store.add(CHAT, {"target": "stocks", "metric": "vwap_up", "window_min": 5, "filters": [], "symbols": [],
                     "expiry": "today", "repeat": "once"}, "t", TODAY)          # no samples yet
    store.add(CHAT, {"target": "index", "metric": "move", "symbols": ["INDIA VIX"], "op": "abs>=", "value": 1,
                     "window_min": 5, "expiry": "today", "repeat": "once"}, "t", TODAY)
    assert A.check_alerts(ctx, store, _sender([]), now=NOWE, day=TODAY) == 0


def test_every_stock_metric_can_be_watched(ctx, store):
    """The alert table and the question engine must never drift apart."""
    _engine_data(ctx)
    _feed(ctx, 20, lambda s, k: None, lambda s, k: (100 + 0.1 * (20 - k), 100))
    for m, spec in A.STOCK_SPEC.items():
        raw = {"target": "stocks", "metric": m, "value": (spec.get("choices") or [100])[0], "window_min": 15}
        if m == "screen":
            raw["conditions"] = [{"left": "price", "op": ">", "value": 0}]
        rule, missing, _ = A.validate_rule(ctx, raw, "once today")
        assert missing == [], (m, missing)
        try:
            hits, res = A.hits_for(ctx, rule)
        except T.ToolError as e:
            assert str(e).startswith(("No ", "Collecting")), (m, e)
        A.describe(rule)


# ── through the bot (Telegram updates) ──────────────────────────────────────────────
def _bot(ctx, tmp_path, store, table):
    return C.Bot(ctx, RecRouter({}), FakeTG(), [CHAT], str(tmp_path / "logs"), alerts=store,
                 alert_router=AlertRouter(table))


def test_bot_buttons_end_to_end(ctx, tmp_path, store):
    b = _bot(ctx, tmp_path, store, {"AAA crosses": VWAP_AAA})
    b.handle({"update_id": 1, "message": {"chat": {"id": 111}, "text": "alert me AAA crosses above vwap"}})
    assert b.tg.sent[-1][1].startswith("🔔 New alert") and b.tg.buttons[-1]
    for i, data in enumerate(("a:ok",)):
        b.handle({"update_id": 2 + i, "callback_query": {"id": "cq%d" % i, "data": data,
                                                          "message": {"chat": {"id": 111}}}})
    assert b.tg.sent[-1][1].startswith("✅ Saved alert #1") and b.tg.callbacks == ["cq0"]
    assert b.router.prompts == []                                    # question router never used
    reply, rec = b.answer("/alerts", CHAT)
    assert reply.startswith("🔔 Your alerts (1 of max 10)") and rec["buttons"]
    b.handle({"update_id": 9, "callback_query": {"id": "x", "data": "a:ok", "message": {"chat": {"id": 222}}}})
    assert len(b.tg.sent) == 2                                       # other chat: ignored


def test_bot_normal_questions_are_not_alerts(ctx, tmp_path, store):
    b = _bot(ctx, tmp_path, store, {})
    b.router.table["today's alerts for AAA"] = ("recent_alerts", {"symbol": "AAA"})
    b.handle({"update_id": 1, "message": {"chat": {"id": 111}, "text": "today's alerts for AAA"}})
    assert b.router.prompts and "alerts today" in b.tg.sent[-1][1]


# ── the 50 user-first cases (evals/alert_cases.jsonl), through the FULL flow ─────────
CASES = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "evals", "alert_cases.jsonl"), encoding="utf-8") if l.strip()]
NAMES = {"NYKAA": 349.0, "RELIANCE": 2950.0, "RELIANCE": 2950.0, "TCS": 4150.0, "HDFCBANK": 1650.0, "ICICIBANK": 1200.0, "SBIN": 800.0,
         "INFY": 1500.0, "TATAMOTORS": 700.0, "TATASTEEL": 152.0}


def _market(ctx):
    st = ctx.state
    for sym, px in NAMES.items():
        st.FNO_SYMBOLS.append(sym)
        st.SECTOR_MAP[sym] = {"HDFCBANK": "NIFTY BANK", "ICICIBANK": "NIFTY BANK", "SBIN": "NIFTY BANK",
                              "NYKAA": "NIFTY FMCG", "TATASTEEL": "NIFTY METAL"}.get(sym, "NIFTY IT")
        st.SESSION_RESULTS[sym] = {"price": px, "intra_pct": 0.5, "vwap": px, "day_open": px, "pdc": px}
    si = st.SECTOR_INTEL["data"]
    si["NIFTY IT"] = {"price": 40000.0, "pct": 0.2, "day_high": 40100.0, "day_low": 39900.0, "vwap_v": 40000.0}
    si["NIFTY FMCG"] = {"price": 55000.0, "pct": 0.1, "day_high": 55100.0, "day_low": 54900.0, "vwap_v": 55000.0}
    return ctx


def _same(want, got):
    if isinstance(want, list) and want and isinstance(want[0], dict) and "left" in want[0]:
        k = lambda c: (c["left"], c["op"], c.get("right"), None if c.get("right") else float(c.get("value") or 0),  # noqa
                       float(c.get("pct") or 0))
        return sorted(map(k, want)) == sorted(map(k, got or []))
    if isinstance(want, list) and want and isinstance(want[0], dict):
        key = lambda fl: sorted((f["col"], f["op"], float(f["value"])) for f in fl or [])  # noqa: E731
        return key(want) == key(got)
    if isinstance(want, list):
        return sorted(str(x).upper() for x in want) == sorted(str(x).upper() for x in (got or []))
    if isinstance(want, float) or isinstance(want, int):
        return got is not None and abs(float(got) - float(want)) < 1e-9
    return want == got


@pytest.mark.parametrize("case", [c for c in CASES if c["tool"] == "create_alert"], ids=lambda c: c["q"][:40])
def test_case_full_flow(ctx, store, case):
    """Correct model output -> the flow must: never ask today / how often, ask for a number ONLY
    when the sentence has none, confirm with the right rule, and save it."""
    _market(ctx)
    body = A.CREATE_PREFIX.sub("", case["q"], count=1)
    f = _flow(ctx, store, {body: ("create_alert", case.get("model") or case["args"])})
    reply = f.on_text(CHAT, case["q"])
    seen = []
    for _ in range(3):                                         # answer at most 3 questions
        (text, btn), = reply
        seen.append(text)
        assert not text.startswith(("How often", "How long")), (case["q"], text)
        if text.startswith("🔔 New alert"):
            break
        assert case.get("asks"), ("asked although the sentence had it", case["q"], text)
        reply = f.on_text(CHAT, "0.3") if text.startswith(("How much", "Which price")) else f.on_text(CHAT, "15")
    assert seen[-1].startswith("🔔 New alert"), (case["q"], seen)
    rule = f.drafts[CHAT]["rule"]
    for k, v in case["args"].items():
        assert _same(v, rule.get(k)), (case["q"], k, v, rule.get(k))
    assert rule["expiry"] == case["args"].get("expiry", "today") and rule["repeat"] == case["args"].get("repeat", "once")
    if case.get("unsupported"):
        assert "⚠️ Can't watch yet: placing orders" in seen[-1]
    assert f.on_button(CHAT, "a:ok")[0][0].startswith("✅ Saved alert #")


def test_case_edits_and_commands(ctx, store):
    _market(ctx)
    table = {"nykaa goes below": ("create_alert", {"target": "stocks", "metric": "price", "symbols": ["NYKAA"],
                                                   "op": "<=", "value": 348}),
             "Change: make it 345": ("update_alert", {"id": 3, "value": 345}),
             "Change: add ICICIBANK": ("update_alert", {"id": 3, "add_symbols": ["ICICIBANK"]})}
    f = _flow(ctx, store, table)
    for _ in range(3):
        f.on_text(CHAT, "alert me when nykaa goes below 348")
        f.on_button(CHAT, "a:ok")
    (t, _), = f.on_text(CHAT, "alert 3 make it 345")
    assert "Was:  NYKAA price ≤ 348" in t and "Now:  NYKAA price ≤ 345" in t
    f.on_button(CHAT, "a:ok")
    f.on_text(CHAT, "alert 3 add ICICIBANK")
    f.on_button(CHAT, "a:ok")
    assert store.get(3)["symbols"] == ["ICICIBANK", "NYKAA"] and store.get(3)["value"] == 345
    n_prompts = len(f.router.prompts)
    (t, _), = f.on_text(CHAT, "alert 3 always")                 # code only
    assert "Now:  ICICIBANK, NYKAA price ≤ 345" in t and "every trading day" in t
    f.on_button(CHAT, "a:ok")
    assert store.get(3)["expiry"] == "everyday"
    assert f.on_text(CHAT, "alert 3 pause")[0][0].startswith("⏸")
    assert f.on_text(CHAT, "alert 3 delete")[0][0].startswith("Delete alert #3?")
    assert len(f.router.prompts) == n_prompts                   # none of these called Gemini


def test_case_refusal(ctx, store):
    f = _flow(ctx, store, {"PCR": ("cannot_alert", {"reason": "PCR is not in the live data"})})
    assert f.on_text(CHAT, "alert me nifty PCR above 1.3")[0][0].startswith("⚠️ I can't watch that")


def test_nykaa_regression_fires_as_a_price_alert(ctx, store):
    """05-11 13:54: model sent pct + filter price < 348; the bot asked 'How much?'. Now: price <= 348."""
    _market(ctx)
    f = _flow(ctx, store, {"nykaa goes below 348": ("create_alert", CASES[0]["model"])})
    (t, _), = f.on_text(CHAT, "alert me when nykaa goes below 348")
    assert "Watch: NYKAA price ≤ 348" in t and "today only · once per stock per day" in t
    f.on_button(CHAT, "a:ok")
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 0      # 349: not yet
    ctx.state.SESSION_RESULTS["NYKAA"].update(price=347.6, intra_pct=-0.4)
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 1
    assert "NYKAA" in out[0][1] and "347.60" in out[0][1] and "day -0.40%" in out[0][1]
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 120, day=TODAY) == 0      # once


def test_index_price_and_between(ctx, store):
    _market(ctx)
    si = ctx.state.SECTOR_INTEL["data"]
    si["NIFTY 50"].update(price=23990.0, day_high=24100.0, day_low=23950.0, vwap_v=24000.0)
    T.sample_once(ctx, NOWE)
    store.add(CHAT, {"target": "index", "metric": "price", "symbols": ["NIFTY 50"], "op": "<=", "value": 24000,
                     "window_min": 0, "expiry": "today", "repeat": "once"}, "t", TODAY)
    store.add(CHAT, {"target": "stocks", "metric": "price", "symbols": ["HDFCBANK"], "op": "between", "value": 1640,
                     "value2": 1660, "window_min": 0, "filters": [], "expiry": "today", "repeat": "once"}, "t", TODAY)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 2
    assert "NIFTY 50 23990.0" in out[0][1] and "HDFCBANK price between 1640 and 1660" in out[1][1]


def test_cross_direction_comes_from_the_live_price(ctx):
    _market(ctx)
    up, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "price", "symbols": ["TCS"], "op": "cross",
                                       "value": 4200}, "")
    down, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "price", "symbols": ["TCS"], "op": "cross",
                                         "value": 4100}, "")
    assert (up["op"], down["op"]) == (">=", "<=")


def test_price_without_number_asks_which_price_with_cancel(ctx, store):
    _market(ctx)
    f = _flow(ctx, store, {"nykaa price": ("create_alert", {"target": "stocks", "metric": "price", "symbols": ["NYKAA"],
                                                            "op": "<="})})
    q, = f.on_text(CHAT, "alert me nykaa price falls")
    assert q[0].startswith("Which price? Type the number") and _btn_data(q) == ["a:no"]
    (t, _), = f.on_text(CHAT, "348")
    assert "Watch: NYKAA price ≤ 348" in t


# ── v14: PACE value change (user: 1.25 since morning -> 1.27 = notify) ───────────────
@pytest.fixture(autouse=True)
def _fresh_members(monkeypatch):
    monkeypatch.setattr(A, "_MEMBERS", {})


def _pace_feed(ctx, steps):
    """steps: [(minutes_before_now, AAA day range)]; budget 3 -> PACE = range / 3."""
    st = ctx.state
    st.IMPLIED_BUDGET["AAA"] = 3.0
    for k, rng in steps:
        st.SESSION_HILO["AAA"] = {"hi": 100 + rng, "lo": 100}
        T.sample_once(ctx, NOWE - 60 * k)


FLAT_THEN_MOVE = [(k, 3.75) for k in range(20, 2, -1)] + [(2, 3.7512), (1, 3.81), (0, 3.81)]   # 1.25 ... 1.27


def test_pace_value_change_is_seen_and_noise_is_not(ctx):
    _pace_feed(ctx, FLAT_THEN_MOVE)
    h = T.pace_history(ctx, "AAA")
    assert h["n"] == 1 and h["events"] == [("09:33", 1.25, 1.27)]           # 1.2504 at k=2: same 2 decimals
    assert h["flat_min"] == 19 and T.pace_line(ctx, "AAA") == "PACE changed 1× · last 09:33 (flat 19 min before)"
    r, text = ok_(ctx, "live_top", metric="pace_change", window_min=5)
    assert [s for s, _ in r["rows"]] == ["AAA"]
    assert "1 stock PACE changed · newest first · last 5 min" in text and "1.25 → 1.27" in text
    assert "changed 09:33 · HIGH break 103.81" in text and "flat 19 min before" in text
    assert "price 105.00" in text and "day +2.00%" in text and text.count("flat") == 1      # no duplicate line


def test_pace_change_window_not_ready_after_restart(ctx):
    _pace_feed(ctx, [(3, 3.75), (2, 3.75), (1, 3.81), (0, 3.81)])
    r, err = T.run(ctx, "live_top", {"metric": "pace_change", "window_min": 15})
    assert err.startswith("Collecting PACE since 09:31 (app start) — a 15-min window is ready at")


def test_pace_history_survives_restart_and_names_the_gap(ctx, monkeypatch):
    _pace_feed(ctx, [(30, 3.0), (29, 3.0), (28, 3.3)])                         # 1.00 -> 1.10 at 09:06
    monkeypatch.setattr(T, "_PACE", {"day": None, "last": {}, "events": {}, "runs": [], "loaded": False, "hb": 0})
    monkeypatch.setattr(T, "_SAMPLE_META", {"since": None, "last": None, "n": 0})   # 'restart'
    _pace_feed(ctx, [(10, 3.3), (9, 3.6)] + [(k, 3.6) for k in range(8, -1, -1)])   # 1.10 -> 1.20 at 09:25
    h = T.pace_history(ctx, "AAA")
    assert [e[1:] for e in h["events"]] == [(1.0, 1.1), (1.1, 1.2)] and h["gaps"] == [("09:06", "09:24")]
    r, text = ok_(ctx, "live_symbol", symbol="AAA")
    assert "changed 2× today (live, since 09:04): 09:06 1.10 · 09:25 1.20" in text
    assert "app was off 09:06–09:24: changes then not seen" in text


def ok_(ctx, name, **args):
    r, err = T.run(ctx, name, args)
    assert err is None, err
    return r, T.format_reply(r)


def test_alert_change_pace_sentences(ctx, store):
    """Model reads 'change pace' as a PACE level (05-11 14:37): code turns it into the value change."""
    f = _flow(ctx, store, {"any stock change pace": ("create_alert", {"target": "stocks", "metric": "pace"}),
                           "pace should be > 1.4": ("create_alert", {"target": "stocks", "metric": "pace",
                                                                      "op": ">=", "value": 1.4})})
    (t, _), = f.on_text(CHAT, "alert me when any stock change pace")
    assert "Watch: any F&O stock PACE value changes (e.g. 1.25 → 1.27)" in t
    f.on_button(CHAT, "a:ok")
    (t, _), = f.on_text(CHAT, "alert me when pace changes and pace should be > 1.4")
    assert "Watch: any F&O stock PACE value changes (e.g. 1.25 → 1.27) and pace > 1.4" in t
    f.on_button(CHAT, "a:ok")
    _pace_feed(ctx, FLAT_THEN_MOVE)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1      # #2: 1.27 is not > 1.4
    assert out[0][1].startswith("🔔 Alert #1") and "1.25 → 1.27" in out[0][1] and "flat 19 min before" in out[0][1]
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 0       # once


# ── v14: alert above result ─────────────────────────────────────────────────────────
def _qa_bot(ctx, tmp_path, store, table):
    b = C.Bot(ctx, RecRouter(table), FakeTG(), [CHAT], str(tmp_path / "logs"), alerts=store,
              alert_router=AlertRouter({"pace changes": ("create_alert", {"target": "stocks",
                                                                          "metric": "pace_change"})}))
    return b


def _say(b, text):
    b.handle({"update_id": 1, "message": {"chat": {"id": 111}, "text": text}})
    return b.tg.sent[-1][1]


def _tap(b, data):
    b.handle({"update_id": 2, "callback_query": {"id": "c", "data": data, "message": {"chat": {"id": 111}}}})
    return b.tg.sent[-1][1]


def test_alert_above_result_pings_only_new_entries(ctx, tmp_path, store):
    _feed(ctx, 20, lambda s, k: {"AAA": 102.0 if k > 2 else 105.0, "CCC": 99.5}.get(s))   # AAA crossed, CCC below
    b = _qa_bot(ctx, tmp_path, store, {"crossed above vwap": ("live_top", {"metric": "vwap_up", "window_min": 5})})
    assert "crossed ABOVE VWAP" in _say(b, "crossed above vwap")
    t = _say(b, "alert above result")
    assert 'NEW names entering: "crossed above vwap"' in t and "Already in it now (no ping): AAA" in t
    assert _tap(b, "a:ok").startswith("✅ Saved alert #1")
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 0         # AAA was there already
    ctx.state.SESSION_RESULTS["CCC"]["price"] = 100.6                                 # CCC crosses now
    T.sample_once(ctx, NOWE + 60)
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 1
    assert out[0][1].startswith('🔔 Alert #1') and "rank in the list now: CCC #" in out[0][1]
    assert "CCC" in out[0][1] and "AAA" not in out[0][1].split("rank in the list now:")[1].split("\n")[0]
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 120, day=TODAY) == 0      # once


def test_alert_above_result_refuses_history_and_nothing(ctx, tmp_path, store):
    b = _qa_bot(ctx, tmp_path, store, {"top gainers 5 days": ("top_symbols", {"metric": "gain", "days": 5})})
    assert "ask a question first" in _say(b, "alert above result")
    _say(b, "top gainers 5 days")
    assert "from past sessions and cannot change during the day" in _say(b, "alert this")


def test_alert_these_uses_the_last_answers_stocks(ctx, tmp_path, store):
    b = _qa_bot(ctx, tmp_path, store, {"top gainers": ("live_top", {"metric": "pct", "limit": 2})})
    _say(b, "top gainers")
    syms = b.recent(CHAT)[0]["symbols"]
    t = _say(b, "alert these when pace changes")
    assert "Watch: %s PACE value changes" % ", ".join(sorted(syms)) in t


def test_query_alert_after_restart_first_check_is_baseline(ctx, store):
    _feed(ctx, 20, lambda s, k: {"AAA": 102.0 if k > 2 else 105.0}.get(s))
    store.add(CHAT, {"target": "query", "metric": "query", "tool": "live_top",
                     "args": {"metric": "vwap_up", "window_min": 5}, "q": "x", "baseline": [], "window_min": 0,
                     "filters": [], "expiry": "everyday", "repeat": "once"}, "t", "2026-11-03")   # made yesterday
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 0      # AAA: baseline, no flood
    ctx.state.SESSION_RESULTS["BBB"]["price"] = 88.5                                # BBB crosses above VWAP 88
    T.sample_once(ctx, NOWE + 60)
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE + 60, day=TODAY) == 1
    assert "BBB" in out[-1][1] and "rank in the list now: BBB" in out[-1][1]


# ── v15: which end moved (LOW / HIGH break) + the 3 kinds of volume support ──────────
def _low_feed(ctx, sym="AAA", steps=None, start=20):
    """Futures day range 103.75-100 (PACE 1.25, budget 3); the LOW breaks to 99.94 at k=1."""
    st = ctx.state
    st.IMPLIED_BUDGET[sym] = 3.0
    for k, lo in steps or ([(k, 100.0) for k in range(start, 1, -1)] + [(1, 99.94), (0, 99.94)]):
        st.SESSION_HILO[sym] = {"hi": 103.75, "lo": lo}
        T.sample_once(ctx, NOWE - 60 * k)


def _fut_ticks(ctx, ratios):
    """Futures ticks: last 5 min volume = ratio × the 5 min before."""
    import threading
    from collections import deque
    st = ctx.state
    st.LIVE_TICK_LOCK = threading.Lock()
    st.LIVE_VOL_HISTORY = {s: deque([(NOWE - 610, 1000), (NOWE - 300, 2000), (NOWE - 5, 2000 + 1000 * r)])
                           for s, r in ratios.items()}


def test_pace_change_is_tagged_low_or_high_break(ctx):
    _low_feed(ctx)
    h = T.pace_history(ctx, "AAA")
    assert h["sides"] == ["LOW"] and h["events"][0][1:] == (1.25, 1.27)
    r, text = ok_(ctx, "live_top", metric="pace_low", window_min=5)
    assert [s for s, _ in r["rows"]] == ["AAA"] and "LOW break 99.94" in text
    r, err = T.run(ctx, "live_top", {"metric": "pace_high", "window_min": 5})
    assert err == "No stock changed PACE by a new day high in the last 5 min."


def test_alert_pace_low_break_with_volume_support(ctx, store):
    """user 05-11 15:14: 'pace changes but only when low breaks with volume support'."""
    _fut_ticks(ctx, {"AAA": 3.0, "BBB": 0.5})
    f = _flow(ctx, store, {"low breaks": ("create_alert", {"target": "stocks", "metric": "pace_change"})})
    (t, _), = f.on_text(CHAT, "alert me when pace changes but only when low breaks with volume support")
    assert "Watch: any F&O stock PACE changes by a new day LOW (low break) and fut vol (5 min vs 5 before) >= 1.5x" in t
    f.on_button(CHAT, "a:ok")
    _low_feed(ctx)
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1
    ping = out[0][1]
    assert "AAA" in ping and "LOW break 99.94" in ping and "FUTVOL5m 3.00x" in ping and "day +2.00%" in ping


def test_alert_pace_low_break_without_volume_does_not_fire(ctx, store):
    _fut_ticks(ctx, {"AAA": 1.1})
    f = _flow(ctx, store, {"low breaks": ("create_alert", {"target": "stocks", "metric": "pace_change"})})
    f.on_text(CHAT, "alert me when pace changes but only when low breaks with volume support")
    f.on_button(CHAT, "a:ok")
    _low_feed(ctx)
    assert A.check_alerts(ctx, store, _sender([]), now=NOWE, day=TODAY) == 0


def test_three_volume_kinds_get_users_defaults(ctx):
    rule, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "pace_change"},
                                   "pace changes on low break with fut volume, rvol and stock volume")
    assert rule["metric"] == "pace_low"
    assert sorted((f["col"], f["op"], f["value"]) for f in rule["filters"]) == \
        [("fut_vol5", ">=", 1.5), ("rvol", ">=", 2.0), ("vol5", ">=", 2.0)]
    rule, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "pace_change", "filters": [
        {"col": "fut_vol5", "op": ">=", "value": 3}]}, "pace changes with fut vol 3x")
    assert rule["filters"] == [{"col": "fut_vol5", "op": ">=", "value": 3.0}]          # user's number wins
    rule, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "fut_volume", "window_min": 15,
                                         "value": 3}, "fut vol 3x in 15 min")
    assert rule["filters"] == []                                                     # the metric IS volume


def test_flat_time_is_honest_after_a_restart(ctx, store, monkeypatch):
    _low_feed(ctx, steps=[(k, 100.0) for k in range(12, 1, -1)] + [(1, 99.94), (0, 99.94)])   # bot from 09:22
    r, text = ok_(ctx, "live_top", metric="pace_change", window_min=5)
    assert "unchanged since 09:22 (bot started then — earlier not seen)" in text and "flat" not in text
    assert T.pace_line(ctx, "AAA") == "PACE changed 1× · last 09:33 (no change seen since 09:22, bot started then)"


def test_flat_time_is_real_when_watched_from_the_open(ctx):
    _low_feed(ctx, start=20)                                      # from 09:14: the whole day was watched
    r, text = ok_(ctx, "live_top", metric="pace_change", window_min=5)
    assert "flat 19 min before" in text


def test_flat_time_never_spans_an_app_off_gap(ctx, monkeypatch):
    _pace_feed(ctx, [(30, 3.0), (29, 3.0), (28, 3.3)])                  # change at 09:06 (run 1)
    monkeypatch.setattr(T, "_SAMPLE_META", {"since": None, "last": None, "n": 0})   # app off 09:06-09:24
    _pace_feed(ctx, [(k, 3.3) for k in range(10, 2, -1)] + [(2, 3.6), (1, 3.6), (0, 3.6)])   # change 09:32
    r, text = ok_(ctx, "live_top", metric="pace_change", window_min=5)
    assert "unchanged since 09:24 (bot started then — earlier not seen)" in text and "flat 26 min" not in text



# ── v16: the condition language (user 06-11: open = high, open = low, any price relation) ──
from test_ai_tools import _bars  # noqa: E402


def _oh_market(ctx):
    """OH1 open = high, now -0.6% from open · OH2 open = high, only -0.25% · OL1 open = low."""
    st = ctx.state
    for sym, op, closes in (("OH1", 200.0, [200.0 - 0.06 * i for i in range(21)]),
                            ("OH2", 200.0, [200.0 - 0.025 * i for i in range(21)]),
                            ("OL1", 100.0, [100.0 + 0.05 * i for i in range(21)])):
        st.FNO_SYMBOLS.append(sym)
        st.SECTOR_MAP[sym] = "NIFTY METAL"
        st._INTRADAY_CACHE[sym] = _bars(closes, [1000] * 21)
        st.SESSION_RESULTS[sym] = {"price": closes[-1], "intra_pct": (closes[-1] / op - 1) * 100, "day_open": op,
                                   "pdc": op, "vwap": op, "rvol": 1.0}
    return ctx


def test_screen_open_equals_high_and_low(ctx):
    _oh_market(ctx)
    r, text = ok_(ctx, "live_screen", conditions=[{"left": "open", "op": "=", "right": "high"}])
    assert {"OH1", "OH2"} <= {s for s, _ in r["rows"]} and "OL1" not in {s for s, _ in r["rows"]}
    assert "open = high (±0.05%)" in text.split("\n")[0] and "open 200.00 = high 200.00" in text
    r, text = ok_(ctx, "live_screen", conditions=[{"left": "open", "op": "=", "right": "high"},
                                                  {"left": "price", "op": "<=", "right": "open", "pct": -0.5}])
    assert [s for s, _ in r["rows"]] == ["OH1"]
    assert "price 198.80 ≤ open-0.5% 199.00" in text and "day -0.60%" in text
    r, _ = ok_(ctx, "live_screen", conditions=[{"left": "open", "op": "=", "right": "low"}])
    assert "OL1" in {s for s, _ in r["rows"]} and "OH1" not in {s for s, _ in r["rows"]}


def test_screen_says_when_it_is_too_early(ctx):
    _oh_market(ctx)
    ctx._now = NOW.replace(hour=9, minute=20)
    r, text = ok_(ctx, "live_screen", conditions=[{"left": "open", "op": "=", "right": "high"}])
    assert "Few minutes into the day" in text


def test_screen_refuses_unknown_fields(ctx):
    r, err = T.run(ctx, "live_screen", {"conditions": [{"left": "pcr", "op": ">", "value": 1}]})
    assert err.startswith("Can't watch 'pcr'")
    r, err = T.run(ctx, "live_screen", {"conditions": [{"left": "price", "op": ">"}]})
    assert "compare with a field or a number" in err


def test_alert_open_equals_high_and_down_half_percent(ctx, store):
    """user 06-11 09:12: 'alert me when stock open=high and decrease 0.5% after open'."""
    _oh_market(ctx)
    conds = [{"left": "open", "op": "=", "right": "high"}, {"left": "price", "op": "<=", "right": "open", "pct": -0.5}]
    f = _flow(ctx, store, {"open=high": ("create_alert", {"target": "stocks", "metric": "screen", "conditions": conds})})
    (t, _), = f.on_text(CHAT, "alert me when stock open=high and decrease 0.5% after open")
    assert "Watch: any F&O stock where open = high (±0.05%) and price ≤ open −0.5%" in t
    assert "every minute, 09:30–15:30" in t
    f.on_button(CHAT, "a:ok")
    out = []
    assert A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY) == 1
    assert "OH1" in out[0][1] and "OH2" not in out[0][1] and "open 200.00 = high 200.00" in out[0][1]
    early = NOWE - 15 * 60                                                    # 09:19: not yet
    store2 = A.AlertStore(store.path + "2")
    store2.add(CHAT, store.get(1), "t", TODAY)
    assert A.check_alerts(ctx, store2, _sender([]), now=early, day=TODAY) == 0


def test_the_0912_half_alert_is_now_refused(ctx, store):
    bad = {"target": "stocks", "metric": "pct", "op": "<=", "value": -0.5,
           "filters": [{"col": "open", "op": "<=", "value": 1}], "unsupported": "open=high condition"}
    f = _flow(ctx, store, {"open=high": ("create_alert", bad)})
    (t, btn), = f.on_text(CHAT, "alert me when stock open=high and decrease 0.5% after open")
    assert t.startswith("⚠️ Can't watch: open=high condition.\nNothing saved") and btn is None


def test_metric_plus_conditions_both_hold(ctx, store):
    _oh_market(ctx)
    rule, _m, _n = A.validate_rule(ctx, {"target": "stocks", "metric": "pct", "op": "<=", "value": -0.1,
                                         "conditions": [{"left": "open", "op": "=", "right": "high"}]}, "")
    hits, _r = A.hits_for(ctx, dict(rule, expiry="today", repeat="once"))
    assert {"OH1", "OH2"} <= {h for h, _v in hits} and "OL1" not in {h for h, _v in hits}


def test_open_equals_high_uses_the_005_percent_tolerance(ctx):
    st = ctx.state
    for sym, closes in (("OH3", [200.06] + [199.9 - 0.01 * i for i in range(20)]),   # high 0.03% above open: '='
                        ("OH5", [200.2] + [199.9 - 0.01 * i for i in range(20)])):   # 0.1% above: not '='
        st.FNO_SYMBOLS.append(sym)
        st.SECTOR_MAP[sym] = "NIFTY METAL"
        st._INTRADAY_CACHE[sym] = _bars(closes, [1000] * 21)
        st.SESSION_RESULTS[sym] = {"price": closes[-1], "intra_pct": -0.1, "day_open": 200.0, "pdc": 200.0,
                                   "vwap": 200.0}
    r, _ = ok_(ctx, "live_screen", conditions=[{"left": "open", "op": "=", "right": "high"}])
    names = {s for s, _ in r["rows"]}
    assert "OH3" in names and "OH5" not in names


def test_high_is_never_below_the_open(ctx):
    """Bars that start after the open (restart) must not make 'high' lower than the open."""
    st = ctx.state
    st.FNO_SYMBOLS.append("OH4")
    st.SECTOR_MAP["OH4"] = "NIFTY METAL"
    st._INTRADAY_CACHE["OH4"] = _bars([199.5 - 0.01 * i for i in range(21)], [1000] * 21)
    st.SESSION_RESULTS["OH4"] = {"price": 199.3, "intra_pct": -0.35, "day_open": 200.0, "pdc": 200.0, "vwap": 200.0}
    f = T.stock_fields(ctx, {"high", "low"}, only={"OH4"})["OH4"]
    assert f["high"] == 200.0 and f["low"] == pytest.approx(199.3)


# ── v17: 06-11 morning failures, each with the WRONG model output Gemini really gave ───
def test_0942_watchlist_word_is_kept(ctx, store):
    f = _flow(ctx, store, {"watchlist stock break": ("create_alert", {"target": "stocks", "metric": "new_low"})})
    (t, _), = f.on_text(CHAT, "alert me when watchlist stock break todays low")
    assert "Watch: any watchlist stock makes a new day low" in t


def test_0945_break_todays_low_is_a_break_not_near(ctx, store):
    _market(ctx)
    f = _flow(ctx, store, {"nykaa break": ("create_alert", {"target": "stocks", "metric": "near_low",
                                                           "symbols": ["NYKAA"]})})
    (t, _), = f.on_text(CHAT, "alert me when nykaa break todays low")
    assert "Watch: NYKAA makes a new day low" in t and not t.startswith("How much")


def test_0947_new_day_low_never_becomes_high(ctx, store):
    _market(ctx)
    f = _flow(ctx, store, {"new day low": ("create_alert", {"target": "stocks", "metric": "new_high",
                                                           "symbols": ["NYKAA"]})})
    (t, _), = f.on_text(CHAT, "alert me when nykaa makes new day low")
    assert "Watch: NYKAA makes a new day low" in t


@pytest.mark.parametrize("case", [c for c in CASES if c["tool"] == "create_alert" and
                                  c["args"].get("metric") in T.MIRROR], ids=lambda c: c["q"][:40])
def test_mirror_every_case_survives_a_flipped_model(ctx, case):
    """Gemini returns the OPPOSITE side for every high/low, up/down case: code must flip it back."""
    _market(ctx)
    wrong = dict(case.get("model") or case["args"], metric=T.MIRROR[case["args"]["metric"]])
    rule, _m, _n = A.validate_rule(ctx, wrong, case["q"])
    assert rule["metric"] == case["args"]["metric"], (case["q"], rule["metric"])


def test_both_directions_in_words_are_left_alone():
    assert T.direction_fix("near today's high or low", "near_low") == ("near_low", None)
    assert T.direction_fix("near today's high or low", "near_high") == ("near_high", None)
    assert T.direction_fix("goes above or below vwap", "vwap_up") == ("vwap_up", None)
    assert T.direction_fix("breaks today's low", "new_high")[0] == "new_low"
    assert T.direction_fix("crosses above vwap", "vwap_down")[0] == "vwap_up"
    assert T.direction_fix("vwap niche jay tyare", "vwap_up")[0] == "vwap_down"


def test_named_stock_missing_from_rule_saves_nothing(ctx, store):
    f = _flow(ctx, store, {"aaa and bbb": ("create_alert", {"target": "stocks", "metric": "new_low",
                                                           "symbols": ["AAA"]})})
    (t, btn), = f.on_text(CHAT, "alert me when aaa and bbb makes new day low")
    assert t.startswith("⚠️ Can't watch: you named BBB but the rule watches AAA") and btn is None
    assert store.active() == []


def test_watchlist_universe_in_the_checker(ctx, store):
    wl = A.T.universe_symbols(ctx, "watchlist")
    assert wl
    store.add(CHAT, {"target": "stocks", "metric": "near_low", "op": "<=", "value": 50, "symbols": [],
                     "universe": "watchlist", "window_min": 0, "filters": [], "expiry": "today",
                     "repeat": "once"}, "t", TODAY)
    out = []
    A.check_alerts(ctx, store, _sender(out), now=NOWE, day=TODAY)
    named = set(re.findall(r"^\s*\d+ (\S+)", out[0][1], re.M)) if out else set()
    assert named and named <= wl


def test_questions_get_the_same_guards(ctx):
    n, a, fx = T.fix_route(ctx, "which stocks made new day low in last 15 min", "live_top",
                           {"metric": "new_high", "window_min": 15})
    assert a["metric"] == "new_low" and fx
    n, a, fx = T.fix_route(ctx, "watchlist stocks where open = high", "live_screen",
                           {"conditions": [{"left": "open", "op": "=", "right": "high"}]})
    assert a["universe"] == "watchlist"


def test_example_retrieval_finds_the_pattern():
    import ai_examples as E
    top = E.similar("nykaa break todays low")
    assert top and any(a.get("metric") in ("new_low", "pdl_down", "day_low_down") for _b, _t, a in top)
    p = E.prompt_with_examples("nykaa break todays low", exclude="nykaa break todays low")
    assert p.startswith("SOLVED EXAMPLES") and p.endswith("NEW: nykaa break todays low")
    assert "'nykaa break todays low' ->" not in p                            # leave-one-out for evals
