# -*- coding: utf-8 -*-
"""
ai_llm.py -- the LLM as a ROUTER only.  v1, 04-11-2026.

One job: turn a question ("top gainers last 5 days?") into ONE tool call
(name + arguments). The model is forced to call a tool (mode=ANY), so it
cannot reply with free text -- and therefore cannot write a number.
`cannot_answer` is the tool it picks for advice, chit-chat, news.

Same call discipline as AI BOaT (integrations/gemini.py):
  * 20 s timeout per call, 30 s total budget per question
  * retry ONLY transient errors (429 / 500 / 503 / 504), waits 1 s then 3 s
  * optional fallback model when the main one stays unavailable
  * temperature 0 -- picking a tool is classification, not writing

Config (config.py first, then environment):
  GEMINI_API_KEY          required
  AI_MODEL                default gemini-3.5-flash-lite (same as AI BOaT)
  AI_FALLBACK_MODEL       optional
"""
import json
import os
import time
from datetime import datetime

CALL_TIMEOUT_S = 20
BUDGET_S = 30
RETRY_WAITS_S = (1, 3)
RETRYABLE = {429, 500, 503, 504}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Used only when the main model stays overloaded (503) after retries. Both are on the
# free tier. Set AI_FALLBACK_MODEL = "" in config.py to switch the fallback off.
DEFAULT_FALLBACK = "gemini-2.5-flash-lite"


def _cfg(name, default=""):
    try:
        import config
        v = getattr(config, name, None)
        if v:
            return str(v).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def _cfg_raw(name, default):
    """Like _cfg, but an explicit empty string in config.py means 'off', not 'default'."""
    try:
        import config
        if hasattr(config, name):
            return str(getattr(config, name) or "").strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def system_prompt(sectors, today=None):
    """All shared rules live HERE once (tool schemas only point to them), so the prompt
    stays small: every tool's text is sent with every question."""
    today = today or datetime.now().strftime("%Y-%m-%d (%A) %H:%M IST")
    return (
        "You route questions from one trader about Indian NSE F&O stocks to exactly ONE tool.\n"
        "Now is %s. Questions may be in English, Gujarati or Hinglish.\n"
        "Rules:\n"
        "1. TIME decides live vs history. 'now', 'today', 'aaje', 'hmna', 'abhi', 'right now', "
        "'last N min' -> a live tool. 'today' NEVER means history: days=1 means YESTERDAY only.\n"
        "2. 'last N days', 'this week', 'last week', 'past', a date -> a history tool. 'this/last week' "
        "= 5 sessions, 'yesterday' = 1, 'last month' = 20. A named past date -> end_day.\n"
        "3. 'yesterday's high/low', 'previous day high/low' is a PRICE LEVEL, not the time: 'aaje "
        "yesterday low break', 'broke yesterday's high today' -> live_level_breaks. Only a past day "
        "('on 3rd Nov', 'yesterday, which broke...') -> level_breaks_history.\n"
        "4. No time given: weekday 09:15-15:40 -> live tool; otherwise history with 5 sessions.\n"
        "5. COLUMNS: words after 'with', 'and', 'saathe', 'pan batavo', 'show' naming price, rvol, "
        "volume, oi, vwap, pace, state, sector, streak, score, prox, open, gap go into `show` (max 4). "
        "'stocks above vwap now with volume and rvol' -> live_vwap(side=above, show=[volume, rvol]). "
        "Columns never change the ranking metric.\n"
        "6. live_top metrics: 'after opening' / 'since open' / 'opening thi' -> open; 'gap up / gap "
        "down' -> gap; 'new day high in last N min' -> new_high (new_low); 'fut vol', 'futures volume' -> "
        "fut_volume; 'pace band crossed / pace todyu' -> pace_cross; 'pace changed / pace badlayu' (the "
        "value moved) -> pace_change (on a low / high break -> pace_low / pace_high); 'pace increased "
        "most' -> pace; 'with volume support' -> filter fut_vol5 >= 1.5, 'stock volume' -> vol5; 'iv' -> iv; 'far from vwap' -> vwap; 'highest score' -> score; 'today' alone -> "
        "pct. SECTORS the same way: 'sector from open' -> live_sector_rank(metric=open). JUST CROSSED "
        "('just broke', 'just crossed', 'abhi todyu', 'crossed ... in last N min'): VWAP up -> vwap_up, "
        "VWAP down / just below VWAP -> vwap_down, yesterday high -> pdh_up, yesterday low -> pdl_down, "
        "open -> open_up / open_down. Without 'just' or minutes, 'above VWAP' stays live_vwap and "
        "'broke yesterday high today' stays live_level_breaks.\n"
        "7. 'last 15 min', 'chhela 10 minute' -> window_min. 'jumped', 'moved', 'rose', 'fell most' with "
        "minutes -> metric=pct + window_min (NOT pace). 'volume spike' -> volume + window_min.\n"
        "8. Buy/sell advice, 'should I buy', targets, predictions, news, FII/DII, PCR, option strikes "
        "-> cannot_answer. 'best stocks right now' / 'any setups' -> current_candidates (scanner "
        "flags, not advice). 'market kevu che', 'mood', 'overall market' -> market_mood. Two or three "
        "stocks to compare ('X vs Y', 'compare X and Y') -> compare_symbols. One stock -> live_symbol.\n"
        "9. Symbols are NSE tickers in capitals (RELIANCE, HDFCBANK, TATAMOTORS): 'hdfc bank' -> "
        "HDFCBANK, 'sbi' -> SBIN, 'infosys' -> INFY, 'nykaa' -> NYKAA.\n"
        "10. Known sectors (use EXACTLY these strings, only if the user names a sector): %s.\n"
        "11. The message may start with 'PREVIOUS ANSWERS' (1 = most recent). Three kinds of NEW message:\n"
        "   a) CONTINUES an answer ('and in IT?', 'only shorts', 'last 30 min instead', 'top 3 only'): "
        "same tool + same arguments, change only what the user changed. 'with X' / 'X pan batavo' = "
        "same tool + same arguments + X added to show. NOT from_previous.\n"
        "   b) Uses the STOCKS of an answer ('from that', 'of these', 'among them', 'aama thi', "
        "'emathi'): set from_previous (usually 1). 'which one has highest X' -> live_top(metric=X, "
        "from_previous=1). 'from that, which are above vwap' -> live_vwap(side=above, from_previous=1).\n"
        "   c) A FRESH question: ignore PREVIOUS ANSWERS completely.\n"
        "   Numeric conditions go into `filters`: 'rvol above 2' -> {col: rvol, op: '>', value: 2}; "
        "'price below 500' -> {col: price, op: '<', value: 500}; 'only gainers' -> {col: pct, op: '>', "
        "value: 0}. pct = day %%, open = %% since open, vwap = %% from VWAP, oi = %% OI change.\n"
        "15. ONE VALUE of a stock / index ('what is low of nykaa', 'nykaa high', 'prev close of tcs', 'pace of "
        "reliance', 'nifty high', 'vix now') -> live_value(symbols, fields). 'how is nykaa' / 'nykaa details' -> "
        "live_symbol. Fields: price, open, high, low, vwap, pdc (prev close), pdh, pdl, pct, open_pct, gap, "
        "range_pct, pace, rvol, fut_vol5, vol5, oi, iv, score.\n"
        "14. PRICE CONDITIONS -> live_screen(conditions): fields price, open, high, low (today), vwap, pdc / pdh "
        "/ pdl (previous close / high / low), pct, open_pct, gap, range_pct, pace, rvol, fut_vol5, vol5, oi, iv, "
        "score. 'open = high' -> {left: open, op: '=', right: high}; 'down 0.5%% from open' -> {left: price, "
        "op: '<=', right: open, pct: -0.5}; 'above previous high' -> {left: price, op: '>', right: pdh}; 'range "
        "above 2%%' -> {left: range_pct, op: '>', value: 2}. 'between A and B' = two conditions. A single "
        "metric ranking ('top gainers') stays live_top.\n"
        "12. Never invent arguments the user did not imply; leave optional ones empty. 'top 5', 'best 3', "
        "'only 2' -> limit on ANY tool. 'lowest', 'weakest', 'fallen most', 'closest' -> order=bottom.\n"
        "13. If the best tool cannot do part of the question (a time base, a measure, an option it does not "
        "have), STILL call it, and write that part in `unsupported` in a few words (e.g. 'since open for "
        "sectors'). Never quietly answer a different question."
    ) % (today, ", ".join(sorted(sectors)) or "unknown")


def context_prompt(history, text):
    """history: recent answers, most recent first: [{"time", "q", "tool", "args", "n"}].
    Shared by the bot and the evals so both send the model exactly the same framing."""
    if not history:
        return text
    lines = ["PREVIOUS ANSWERS (1 = most recent):"]
    for i, h in enumerate(history, 1):
        args = {k: v for k, v in (h.get("args") or {}).items() if v not in (None, "", [], False, 0)}
        n = h.get("n")
        lines.append("%d. [%s] %r -> %s %s%s" % (i, h.get("time", ""), h.get("q", ""), h.get("tool", ""),
                                               json.dumps(args, ensure_ascii=False),
                                               (" (%d stocks)" % n) if n else ""))
    return "\n".join(lines) + "\nNEW: " + text


class Route:
    def __init__(self, name, args, model="", tokens=None, ms=0):
        self.name, self.args, self.model, self.tokens, self.ms = name, dict(args or {}), model, tokens, ms


class RouterError(Exception):
    """kind: config (no key) | key (rejected) | model_missing | bad_request | quota (per minute) |
    quota_day (free daily limit) | timeout | overloaded | no_call | unknown"""
    def __init__(self, kind, msg):
        super().__init__(msg)
        self.kind = kind


class GeminiRouter:
    def __init__(self, tools, sectors, api_key=None, model=None, fallback=None, system=None):
        from google import genai
        from google.genai import types
        self._types = types
        key = api_key or _cfg("GEMINI_API_KEY")
        if not key:
            raise RouterError("config", "GEMINI_API_KEY is not set in config.py or environment")
        self.client = genai.Client(api_key=key,
                                   http_options=types.HttpOptions(timeout=CALL_TIMEOUT_S * 1000))
        self.model = model or _cfg("AI_MODEL", DEFAULT_MODEL)
        fb = fallback if fallback is not None else _cfg_raw("AI_FALLBACK_MODEL", DEFAULT_FALLBACK)
        self.models = [self.model] + ([fb] if fb and fb != self.model else [])
        self.tools = tools
        self.sectors = sectors
        self.system = system or system_prompt     # alert mode passes its own rules
        self.dead = set()                 # models that answered 404 this run: never retried
        self.last_error = None            # (HH:MM, model, kind, short message) for /check

    def _config(self):
        t = self._types
        decls = [t.FunctionDeclaration(name=x["name"], description=x["description"],
                                       parameters=x["parameters"]) for x in self.tools]
        return t.GenerateContentConfig(
            system_instruction=self.system(self.sectors),
            tools=[t.Tool(function_declarations=decls)],
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(mode="ANY")),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            temperature=0)

    @staticmethod
    def _kind(e):
        code = getattr(e, "code", None)
        name = type(e).__name__.lower()
        text = str(e)
        if code == 504 or "timeout" in name:
            return "timeout"
        if code == 429:
            # The free tier has a per-minute AND a per-day limit. Retrying in 1-3 s only helps
            # the per-minute one; the daily one resets at midnight US Pacific (13:30 IST).
            return "quota_day" if ("PerDay" in text or "per day" in text.lower()) else "quota"
        if code in (500, 502, 503):
            return "overloaded"
        if code == 404:
            return "model_missing"
        if code in (401, 403):
            return "key"
        if code == 400:
            return "bad_request"
        return "unknown"

    def route(self, question):
        """Per model: retry the short-lived errors (per-minute quota, 5xx, timeout) with waits;
        move to the next model at once on errors that will not clear in seconds (daily quota,
        model not found, request rejected). A rejected key stops everything."""
        t0 = time.time()
        deadline = t0 + BUDGET_S
        cfg = self._config()
        last, last_kind = None, None
        for model in [m for m in self.models if m not in self.dead]:
            for attempt in range(len(RETRY_WAITS_S) + 1):
                if time.time() + CALL_TIMEOUT_S > deadline and last is not None:
                    raise RouterError(last_kind or self._kind(last), "time budget used up: %s" % last)
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=question, config=cfg)
                except Exception as e:
                    last, last_kind = e, self._kind(e)
                    self.last_error = (datetime.now().strftime("%H:%M"), model, last_kind, str(e)[:160])
                    if last_kind == "key":
                        raise RouterError("key", str(e))
                    if last_kind == "model_missing":
                        self.dead.add(model)
                        break
                    if last_kind in ("quota_day", "bad_request"):
                        break
                    if getattr(e, "code", None) not in RETRYABLE and last_kind != "timeout":
                        raise RouterError(last_kind, str(e))
                    if attempt < len(RETRY_WAITS_S):
                        time.sleep(RETRY_WAITS_S[attempt])
                    continue
                calls = getattr(resp, "function_calls", None) or []
                if not calls:
                    raise RouterError("no_call", "model returned no tool call")
                fc = calls[0]
                um = getattr(resp, "usage_metadata", None)
                return Route(fc.name, dict(fc.args or {}), model,
                             getattr(um, "total_token_count", None), int((time.time() - t0) * 1000))
        if last is None:
            raise RouterError("model_missing", "no usable model (all returned 404: %s)" % ", ".join(sorted(self.dead)))
        raise RouterError(last_kind or "unknown", str(last))


class FakeRouter:
    """For tests: returns a fixed route per question (or raises)."""
    def __init__(self, table):
        self.table = table

    def route(self, question):
        v = self.table.get(question)
        if isinstance(v, Exception):
            raise v
        if v is None:
            return Route("cannot_answer", {"reason": "fake router: unknown question"}, "fake")
        return Route(v[0], v[1], "fake")
