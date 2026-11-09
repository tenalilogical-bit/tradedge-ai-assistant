# -*- coding: utf-8 -*-
"""GeminiRouter with the REAL google-genai types and a fake client (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
types = pytest.importorskip("google.genai.types")
from google.genai import errors                                  # noqa: E402

import ai_llm                                                    # noqa: E402
import ai_tools                                                  # noqa: E402


def _resp(name, args, tokens=100):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name=name, args=args))]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=tokens))


def _err(code):
    return errors.APIError(code, {"error": {"message": "x", "status": "X"}})


class _Models:
    def __init__(self, plan):
        self.plan, self.calls = list(plan), []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        x = self.plan.pop(0)
        if isinstance(x, Exception):
            raise x
        return x


@pytest.fixture
def router(monkeypatch):
    monkeypatch.setattr(ai_llm.time, "sleep", lambda s: None)
    return ai_llm.GeminiRouter(ai_tools.TOOLS, {"BANK", "IT"}, api_key="dummy",
                               model="m1", fallback="m2")


def _use(router, plan):
    m = _Models(plan)
    router.client = type("C", (), {})()
    router.client.models = m
    return m


def test_all_declarations_build(router):
    assert len(router._config().tools[0].function_declarations) == len(ai_tools.TOOLS)


def test_retry_transient_then_success(router):
    m = _use(router, [_err(429), _err(503), _resp("top_symbols", {"metric": "gain", "days": 5}, 321)])
    r = router.route("top gainers")
    assert (r.name, r.args, r.model, r.tokens) == ("top_symbols", {"metric": "gain", "days": 5}, "m1", 321)
    assert m.calls == ["m1", "m1", "m1"]


def test_fallback_model_after_retries(router):
    m = _use(router, [_err(503)] * 3 + [_resp("market_breadth", {})])
    assert router.route("x").model == "m2" and m.calls == ["m1"] * 3 + ["m2"]


def _err_msg(code, msg):
    return errors.APIError(code, {"error": {"message": msg, "status": "X"}})


def test_bad_request_tries_fallback_once(router):
    m = _use(router, [_err(400), _err(400)])
    with pytest.raises(ai_llm.RouterError) as e:
        router.route("x")
    assert e.value.kind == "bad_request" and m.calls == ["m1", "m2"]


def test_daily_quota_skips_retries_and_uses_fallback(router):
    """05-11 12:16: free tier 500/day used up -> 1 s / 3 s retries are pointless."""
    day = _err_msg(429, "Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    m = _use(router, [day, _resp("market_mood", {})])
    r = router.route("x")
    assert r.model == "m2" and m.calls == ["m1", "m2"]
    assert router.last_error[1:3] == ("m1", "quota_day")


def test_missing_fallback_model_is_marked_dead(router):
    day = _err_msg(429, "PerDay")
    m = _use(router, [day, _err(404), day])
    with pytest.raises(ai_llm.RouterError) as e:
        router.route("x")
    assert e.value.kind == "model_missing" and "m2" in router.dead
    with pytest.raises(ai_llm.RouterError) as e:
        router.route("x")
    assert e.value.kind == "quota_day" and m.calls == ["m1", "m2", "m1"]   # m2 never called again


def test_rejected_key_stops_at_once(router):
    m = _use(router, [_err(403)])
    with pytest.raises(ai_llm.RouterError) as e:
        router.route("x")
    assert e.value.kind == "key" and m.calls == ["m1"]


def test_text_reply_without_tool_is_rejected(router):
    _use(router, [types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text="Buy RELIANCE!")]))])])
    with pytest.raises(ai_llm.RouterError) as e:
        router.route("x")
    assert e.value.kind == "no_call"


def test_missing_key_is_config_error(monkeypatch):
    monkeypatch.setattr(ai_llm, "_cfg", lambda n, d="": "")
    with pytest.raises(ai_llm.RouterError) as e:
        ai_llm.GeminiRouter(ai_tools.TOOLS, set())
    assert e.value.kind == "config"


def test_alert_tools_build_with_their_own_prompt():
    import ai_alerts
    r = ai_llm.GeminiRouter(ai_alerts.ALERT_TOOLS, {"NIFTY BANK"}, api_key="dummy", model="m1", fallback="",
                            system=ai_alerts.system_prompt)
    cfg = r._config()
    assert [d.name for d in cfg.tools[0].function_declarations] == ["create_alert", "update_alert", "cannot_alert"]
    assert cfg.system_instruction.startswith("You turn ONE alert sentence")
