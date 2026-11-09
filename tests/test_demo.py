# -*- coding: utf-8 -*-
"""The public demo runs the real bot code on fake_market. These tests keep it honest:
every tool answers, runs repeat exactly, and the offline tour shows a guard fixing a
real wrong model output and an alert firing from plain code."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import ai_tools as T                     # noqa: E402
import fake_market                       # noqa: E402

ONE_CALL_PER_TOOL = [
    ("top_symbols", {"metric": "gain", "days": 5}), ("level_breaks_history", {"kind": "prev_high"}),
    ("symbol_history", {"symbol": "SBIN", "days": 5}), ("sector_history", {"days": 5}),
    ("signal_performance", {"by": "stage"}), ("live_top", {"metric": "fut_volume", "window_min": 15}),
    ("live_screen", {"conditions": [{"left": "open", "op": "=", "right": "high"}]}),
    ("live_vwap", {"side": "above"}), ("live_level_breaks", {"kind": "prev_high"}),
    ("current_candidates", {"side": "LONG"}), ("watchlist_status", {}), ("recent_alerts", {}),
    ("live_sector_rank", {"metric": "price"}), ("live_value", {"symbols": ["NYKAA"], "fields": ["low"]}),
    ("live_symbol", {"symbol": "NYKAA"}), ("compare_symbols", {"symbols": ["SBIN", "PNB"]}),
    ("market_mood", {}), ("breadth_trend", {"window_min": 30}), ("cannot_answer", {"reason": "advice"}),
]


@pytest.fixture(autouse=True)
def _fresh_samples(monkeypatch):
    monkeypatch.setattr(T, "_SAMPLES", {})
    monkeypatch.setattr(T, "_ISAMPLES", {})
    monkeypatch.setattr(T, "_PACE", {"day": None, "last": {}, "events": {}, "runs": [], "loaded": False, "hb": 0})
    monkeypatch.setattr(T, "_SAMPLE_META", {"since": None, "last": None, "n": 0})


def test_every_tool_answers_on_the_fake_market(tmp_path):
    m = fake_market.Market(str(tmp_path))
    assert {n for n, _a in ONE_CALL_PER_TOOL} == T.TOOL_NAMES        # a new tool must be added here
    for name, args in ONE_CALL_PER_TOOL:
        r, err = T.run(m.ctx, name, args)
        assert err is None, (name, err)
        assert T.format_reply(r).strip()


def test_fake_market_repeats_exactly(tmp_path):
    a = fake_market.Market(str(tmp_path / "a"))
    pa = {s: r["price"] for s, r in a.state.SESSION_RESULTS.items()}
    b = fake_market.Market(str(tmp_path / "b"))
    assert pa == {s: r["price"] for s, r in b.state.SESSION_RESULTS.items()}


def test_empty_scope_has_no_empty_arrow_line():
    r = {"tool": "cannot_answer", "reason": "x", "scope": {"symbols": ["NYKAA"]}}
    assert "↳" not in T.format_reply(r)


def test_offline_tour_guard_and_alert():
    env = dict(os.environ, GEMINI_API_KEY="", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "demo.py"), "--tour"], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         timeout=120).stdout
    assert "Traceback" not in out
    # the recorded model output was "day % + price filter"; the guard turns it into a price alert
    assert "Watch: NYKAA price ≤ 348" in out and "Saved alert #1" in out
    assert out.count("🔔 Alert #1 ·") == 1                         # fires once, from code, no model call
    assert "I can't answer that from the data." in out             # no trade advice
