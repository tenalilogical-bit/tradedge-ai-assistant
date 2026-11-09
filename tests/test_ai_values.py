# -*- coding: utf-8 -*-
"""v18: every value the app has must be ASKABLE (generated from ai_tools.FIELDS), the stock card must
show every price field, and the alert examples must be found by meaning, not only by letters."""
import pytest

import ai_examples as X
import ai_tools as T
from test_ai_tools import (NOW, _engine_data, _feed, _fresh_samples, _tiny_sectors, ctx)  # noqa: F401


def _rich(ctx):
    """Every source a field can come from: bars, prev day, PACE, futures ticks, OI, IV, score."""
    _engine_data(ctx)
    _feed(ctx, 20, lambda s, k: None, lambda s, k: (100 + 0.1 * (20 - k), 100))
    st = ctx.state
    st.SESSION_RESULTS["AAA"]["pct_score"] = 72
    st.OI_CACHE.setdefault("AAA", {}).update(oi_pct=4.2)
    from collections import deque
    now = NOW.timestamp()
    st.LIVE_VOL_HISTORY["AAA"] = deque((now - 60 * k, 1000 + 100 * (20 - k) ** 2) for k in range(20, -1, -1))
    return ctx


def _val(ctx, sym, field):
    r, err = T.run(ctx, "live_value", {"symbols": [sym], "fields": [field]})
    assert err is None, (field, err)
    return r["rows"][0]["f"].get(field), T.format_reply(r)


@pytest.mark.parametrize("field", list(T.FIELDS))
def test_every_field_is_answerable(ctx, field):
    _rich(ctx)
    v, text = _val(ctx, "AAA", field)
    assert v is not None, (field, text)
    assert text.startswith("AAA  %s " % T.FIELD_LABEL[field]), text
    assert "— (not available)" not in text


def test_what_is_low_of_x(ctx):
    _rich(ctx)
    r, err = T.run(ctx, "live_value", {"symbols": ["aaa"], "fields": ["low"]})
    text = T.format_reply(r)
    assert text.startswith("AAA  low 100.00 (09:15) · price 105.00 · day +2.00%"), text
    r, err = T.run(ctx, "live_value", {"symbols": ["AAA"]})                 # no field = price, high, low
    assert "price 105.00" in T.format_reply(r) and "high 119.00" in T.format_reply(r)


def test_index_values_and_unknown_field(ctx):
    si = ctx.state.SECTOR_INTEL["data"]
    si["NIFTY 50"].update(price=24050.0, day_high=24120.0, day_low=23980.0, vwap_v=24040.0, pct=0.3)
    r, err = T.run(ctx, "live_value", {"symbols": ["nifty", "AAA"], "fields": ["high", "pdh"]})
    text = T.format_reply(r)
    assert "NIFTY 50  high 24120.0" in text and "pdh — (not for an index)" in text
    r, err = T.run(ctx, "live_value", {"symbols": ["AAA"], "fields": ["pcr"]})
    assert err.startswith("I don't have 'pcr'")


def test_card_shows_every_price_field_and_flat_since(ctx):
    _rich(ctx)
    r, err = T.run(ctx, "live_symbol", {"symbol": "AAA"})
    text = T.format_reply(r)
    assert "Day   high 119.00 (09:34) · low 100.00 (09:15)" in text
    assert "Prev  high" in text and "close" in text and "VWAP" in text
    assert "PACE" in text


# ── hybrid example retrieval ─────────────────────────────────────────────────────────
_CONCEPT = [{"break", "breaks", "broke", "slips", "under", "beneath", "drops", "below", "tute", "sinks", "plunges"},
            {"low", "bottom", "lows"}, {"high", "top", "highs"}, {"above", "over", "climbs", "jumps"},
            {"vwap"}, {"yesterday", "previous", "prev", "pdh", "pdl"}, {"near", "around", "within"}]


def _fake_embed(texts):
    out = []
    for t in texts:
        w = set(t.lower().replace("'", " ").split())
        out.append([1.0 if w & c else 0.0 for c in _CONCEPT] + [0.01])
    return out


@pytest.fixture
def embed(monkeypatch, tmp_path):
    calls = []

    def e(texts):
        calls.append(list(texts))
        return _fake_embed(texts)
    monkeypatch.setattr(X, "EMBED", e)
    monkeypatch.setattr(X, "EMB_CACHE", str(tmp_path / "emb.json"))
    monkeypatch.setattr(X, "_emb", {"key": None, "vecs": {}, "error": None})
    return calls


def test_meaning_side_finds_a_synonym_the_letters_miss(embed, monkeypatch):
    """'tcs plunges to the bottom': no example has these words (letters pull TCS *high* examples);
    meaning pulls the LOW pattern. Use a word no example has yet — adding real sentences will fix letters too."""
    q = "tcs plunges to the bottom"
    lows = ("new_low", "pdl_down", "day_low_down", "near_low", "pace_low")
    hybrid = [a.get("metric") for _b, _t, a in X.similar(q)[:3]]
    assert any(m in lows for m in hybrid), hybrid
    monkeypatch.setattr(X, "EMBED", None)
    letters = [a.get("metric") for _b, _t, a in X.similar(q)[:3]]
    assert not any(m in lows for m in letters), letters          # proves the meaning side is what found it
    monkeypatch.setattr(X, "EMBED", _fake_embed)

def test_examples_embedded_once_then_cached(embed):
    X.similar("nykaa break todays low")
    n_batch = len(embed[0])
    X.similar("tcs breaks today's high")
    assert len(embed) == 3 and len(embed[2]) == 1 and n_batch > 50      # 1 batch for examples, then 1 per query


def test_embedding_failure_falls_back_to_keywords(monkeypatch, tmp_path):
    def boom(texts):
        raise RuntimeError("quota")
    monkeypatch.setattr(X, "EMBED", boom)
    monkeypatch.setattr(X, "EMB_CACHE", str(tmp_path / "emb.json"))
    monkeypatch.setattr(X, "_emb", {"key": None, "vecs": {}, "error": None})
    top = X.similar("nykaa break todays low")
    assert top and X.status().startswith("keyword only — embedding error: RuntimeError")


def test_synonym_direction_words():
    assert T.direction_fix("nykaa slips beneath today's low", "new_high")[0] == "new_low"
    assert T.direction_fix("sbin goes beneath vwap", "vwap_up")[0] == "vwap_down"
    assert T.direction_fix("infy climbs over vwap", "vwap_down")[0] == "vwap_up"
