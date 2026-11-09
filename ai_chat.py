# -*- coding: utf-8 -*-
"""
ai_chat.py -- Telegram Q&A + scheduled digests for TradEdge SAGA.  v2, 04-11-2026.

FLOW (daemon threads, started from app.py)
    listener: Telegram getUpdates (long poll, 25 s) -> allowlist (your chat id only)
      -> "/help" answered by code (no LLM) -> rate limit
      -> ai_llm router picks ONE tool (+ previous question for follow-ups, 5 min)
      -> ai_tools validates + runs it -> ai_tools.format_reply builds text
      -> sendMessage inside <pre> (monospace, columns line up)
      -> one JSONL line per question in logs/ai_chat/ai_chat_YYYY-MM-DD.jsonl
    digests: at fixed IST times on weekdays, market_mood (+ EOD extras) is sent
      with NO LLM call. Skipped silently when there is no live data (holiday).

WHY LONG POLLING
    The app runs on a laptop with no public URL, so Telegram cannot push to a
    webhook. getUpdates needs only outbound HTTPS. Telegram allows ONE reader
    per bot: telegram_alerts.detect_chat_id() also reads getUpdates, so use
    /tg/chatid only while this thread is off (AI_CHAT_ENABLED = False).

NEVER BLOCKS TRADING
    Own threads, no engine locks held across network calls (ai_tools copies
    under lock then releases), every failure caught and logged. Digests and
    answers do not go through notify.py, so they never use the alert caps.

Config (config.py first, then environment):
    AI_CHAT_ENABLED    default True
    AI_TG_TOKEN        optional separate bot; default = TG_BOT_TOKEN used for alerts
    AI_TG_CHAT_IDS     optional comma list; default = TG_CHAT_ID (fail-closed if empty)
    AI_DIGEST_TIMES    default "09:20,11:00,13:00,15:35"; "" = digests off
"""
import html
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import ai_tools

POLL_TIMEOUT_S = 25
HTTP_TIMEOUT_S = POLL_TIMEOUT_S + 10
MAX_PER_MIN = 6                  # protects the Gemini free-tier quota
MAX_QUESTION_CHARS = 300
FOLLOWUP_S = 900                 # answers stay usable for follow-ups for 15 min
MEMORY_N = 5                     # ... the last 5 of them
BACKOFF_S = (5, 15, 30, 60)
DEFAULT_DIGESTS = "09:20,11:00,13:00,15:35"
EOD_DIGEST_AFTER = "15:30"       # digests at/after this time add the day's movers + sectors

_started = False
_start_lock = threading.Lock()


def _cfg(name, default=""):
    try:
        import config
        v = getattr(config, name, None)
        if v is not None:
            return v
    except Exception:
        pass
    return os.environ.get(name, default)


class Telegram:
    def __init__(self, token):
        import requests
        self._s = requests.Session()
        self._base = "https://api.telegram.org/bot%s/" % token

    def get_updates(self, offset):
        r = self._s.get(self._base + "getUpdates",
                        params={"timeout": POLL_TIMEOUT_S, "offset": offset,
                                "allowed_updates": json.dumps(["message", "callback_query"])},
                        timeout=HTTP_TIMEOUT_S)
        d = r.json()
        if not d.get("ok"):
            raise RuntimeError("getUpdates: %s %s" % (d.get("error_code"), d.get("description")))
        return d.get("result") or []

    def post(self, chat_id, text, parse_mode=None, markup=None):
        data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
        if parse_mode:
            data["parse_mode"] = parse_mode
        if markup:
            data["reply_markup"] = json.dumps(markup)
        return self._s.post(self._base + "sendMessage", data=data, timeout=15).json()

    def send(self, chat_id, text, buttons=None):
        """True when Telegram accepted every part. buttons = [[(label, callback_data)]] on the last part."""
        markup = {"inline_keyboard": [[{"text": l, "callback_data": c} for l, c in row] for row in buttons]} \
            if buttons else None
        parts, ok = chunks(text), True
        for i, chunk in enumerate(parts):
            mk = markup if i == len(parts) - 1 else None
            d = self.post(chat_id, "<pre>%s</pre>" % html.escape(chunk, quote=False), "HTML", mk)
            if not d.get("ok"):                     # HTML rejected: never lose the answer
                d = self.post(chat_id, chunk, None, mk)
            ok = ok and bool(d.get("ok"))
        return ok

    def answer_callback(self, cq_id, text=""):
        try:
            self._s.post(self._base + "answerCallbackQuery", data={"callback_query_id": cq_id, "text": text},
                         timeout=10)
        except Exception:
            pass                                    # only stops the button spinner


def chunks(text, size=3500):
    """Split on line breaks so a <pre> block is never cut mid-line (Telegram max 4096)."""
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > size and cur:
            out.append(cur)
            cur = ""
        cur = (cur + "\n" + line) if cur else line[:size]
    if cur:
        out.append(cur)
    return out or [""]


ROUTER_MESSAGES = {
    "quota": "Gemini per-minute limit hit — try again in a minute.",
    "quota_day": "Gemini free daily limit is used up — it resets about 13:30 IST. "
                 "/check, /help, /alerts and saved alerts still work.",
    "timeout": "Gemini is slow right now — try again.",
    "overloaded": "Gemini is overloaded — try again in a minute.",
    "model_missing": "Gemini model not found — check AI_MODEL / AI_FALLBACK_MODEL in config.py.",
    "key": "Gemini rejected the API key — check GEMINI_API_KEY in config.py.",
    "bad_request": "Gemini rejected this request (logged as a bug) — try rephrasing.",
    "config": "AI is not configured: GEMINI_API_KEY is missing in config.py."}


def router_message(kind):
    return ROUTER_MESSAGES.get(kind, "Could not understand that right now — try rephrasing.")


class Bot:
    """All logic, no thread. Tests drive handle() / digest() directly with fakes."""

    def __init__(self, ctx, router, tg, allowed, log_dir, alerts=None, alert_router=None):
        self.ctx, self.router, self.tg = ctx, router, tg
        self.alerts = alerts                           # ai_alerts.AlertStore or None
        self.flow = None
        if alerts is not None and alert_router is not None:
            import ai_alerts
            self.flow = ai_alerts.AlertFlow(ctx, alert_router, alerts, err_text=router_message, recent=self.recent)
        self.allowed = {str(x).strip() for x in allowed if str(x).strip()}
        self.log_dir = log_dir
        self._hits = {}
        self._mem = {}             # chat -> deque of recent answers (RAM only, lost on restart)

    def _rate_ok(self, chat):
        q = self._hits.setdefault(chat, deque())
        now = time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= MAX_PER_MIN:
            return False
        q.append(now)
        return True

    def _log(self, rec):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            p = os.path.join(self.log_dir, "ai_chat_%s.jsonl" % datetime.now().strftime("%Y-%m-%d"))
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print("[AI chat] log write failed: %s" % e)

    def recent(self, chat):
        """Fresh answers for this chat, most recent first."""
        now = time.time()
        return [m for m in reversed(self._mem.get(chat, ())) if now - m["ts"] <= FOLLOWUP_S]

    def answer(self, text, chat=""):
        """Question -> (reply, log record). Never raises."""
        rec = {"ts": datetime.now().strftime("%H:%M:%S"), "q": text}
        t = text.strip()
        if t.lower() in ("help",) or t.startswith("/"):        # commands: code, never the LLM
            cmd = t.split()[0].lower().split("@")[0] if t.startswith("/") else "/help"
            rec["tool"] = cmd
            if cmd in ("/help", "/start"):
                return ai_tools.HELP_TEXT, rec
            if cmd == "/alerts":
                if not self.flow:
                    return "Alerts are off (no alert store).", rec
                txt, btn = self.flow.list(chat)[0]
                rec["buttons"] = btn
                return txt, rec
            if cmd == "/check":
                try:
                    ar = getattr(self.flow, "router", None)
                    am = " · alerts use %s" % ", ".join(m for m in getattr(ar, "models", []) if m not in
                                                       getattr(ar, "dead", ())) if ar is not None else ""
                    le = getattr(self.router, "last_error", None)
                    dead = sorted(getattr(self.router, "dead", ()) or ())
                    try:
                        import ai_examples
                        am += " · examples: " + ai_examples.status()
                    except Exception:
                        pass
                    g = "\n\nGEMINI" + am + "\n" + ("✅ no errors since start" if not le else
                                          "⚠️ last error %s · %s · %s" % (le[0], le[1], le[2]))
                    if dead:
                        g += "\n❌ not found (skipped): %s" % ", ".join(dead)
                    return ai_tools.check_report(self.ctx) + g, rec
                except Exception as e:
                    rec["error"] = "check: %s" % e
                    return "Data check failed: %s" % e, rec
            return "Unknown command %s — send /help (or ask in plain words)." % cmd, rec
        if len(t) > MAX_QUESTION_CHARS:
            rec["error"] = "too_long"
            return "Please keep the question under %d characters." % MAX_QUESTION_CHARS, rec
        import ai_llm
        hist = self.recent(chat)
        prompt = ai_llm.context_prompt(hist, t)
        if hist:
            rec["history_n"] = len(hist)
        try:
            route = self.router.route(prompt)
        except Exception as e:
            kind = getattr(e, "kind", "unknown")
            rec.update({"error": "router_%s" % kind, "detail": str(e)[:300]})
            return router_message(kind), rec
        rec.update({"tool": route.name, "args": route.args, "model": route.model,
                    "tokens": route.tokens, "llm_ms": route.ms})
        name, args, fixes = ai_tools.fix_route(self.ctx, t, route.name, route.args, hist)
        if fixes:
            rec.update({"fixes": fixes, "tool": name})
        result, err = ai_tools.run(self.ctx, name, args, prev=hist)
        if err:
            rec["error"] = err
            return err, rec
        rec["tool_ms"] = result.get("_ms")
        rec["clean_args"] = result.get("_args")
        rec["rows"] = len(result.get("rows") or [])
        if result.get("unsupported"):
            rec["unsupported"] = result["unsupported"]
        if name != "cannot_answer":
            syms = ai_tools.result_symbols(result)
            self._mem.setdefault(chat, deque(maxlen=MEMORY_N)).append({
                "ts": time.time(), "time": datetime.now().strftime("%H:%M"), "q": t,
                "tool": name, "args": result.get("_args"), "symbols": syms,
                "n": len(syms) if syms else None})
        try:
            reply = ai_tools.format_reply(result)
            rec["reply_head"] = reply.split("\n", 1)[0][:120]     # title only: lets a log review spot wrong answers
            return reply, rec
        except Exception as e:
            rec["error"] = "format: %s" % e
            return "Got the data but could not format it (bug logged).", rec

    def _reply(self, chat, text, buttons=None):
        try:
            return self.tg.send(chat, text, buttons) if buttons else self.tg.send(chat, text)
        except Exception as e:
            print("[AI chat] send failed: %s" % e)
            return False

    def handle_button(self, cq):
        chat = str(((cq.get("message") or {}).get("chat") or {}).get("id", ""))
        if chat not in self.allowed or not self.flow:
            return
        try:
            self.tg.answer_callback(cq.get("id"))
        except Exception:
            pass
        data = str(cq.get("data") or "")
        try:
            replies = self.flow.on_button(chat, data)
        except Exception as e:
            replies = [("Alert step failed (bug logged): %s" % e, None)]
        self._log({"ts": datetime.now().strftime("%H:%M:%S"), "button": data, "reply_head":
                   (replies[0][0].split("\n", 1)[0][:120] if replies else "")})
        for text, btn in replies:
            self._reply(chat, text, btn)

    def handle(self, update):
        if update.get("callback_query"):
            return self.handle_button(update["callback_query"])
        msg = update.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = msg.get("text")
        if not chat or not text:
            return
        if chat not in self.allowed:                          # fail closed, silent
            print("[AI chat] blocked chat %s" % chat)
            return
        if not self._rate_ok(chat):
            self.tg.send(chat, "Slow down — max %d questions a minute." % MAX_PER_MIN)
            return
        if self.flow and not text.strip().startswith("/"):
            try:
                replies = self.flow.on_text(chat, text)
            except Exception as e:
                replies = [("Alert step failed (bug logged): %s" % e, None)]
            if replies is not None:                # alert mode: 'alert me ...', 'alert 3 ...', a typed answer
                self._log({"ts": datetime.now().strftime("%H:%M:%S"), "q": text, "alert": True,
                           "reply_head": replies[0][0].split("\n", 1)[0][:120] if replies else ""})
                for t, btn in replies:
                    self._reply(chat, t, btn)
                return
        reply, rec = self.answer(text, chat)
        btn = rec.pop("buttons", None)
        self._log(rec)
        self._reply(chat, reply, btn)

    # ── scheduled digests (no LLM) ───────────────────────────────────────────
    def digest_text(self, slot):
        parts = []
        r, err = ai_tools.run(self.ctx, "market_mood", {})
        if err:
            return None                                       # no live data: holiday / app just started
        parts.append("⏰ %s digest\n%s" % (slot, ai_tools.format_reply(r)))
        if slot >= EOD_DIGEST_AFTER:
            for name, args in (("live_top", {"metric": "pct", "limit": 5, "show": ["price", "rvol"]}),
                               ("live_top", {"metric": "pct", "order": "bottom", "limit": 5,
                                             "show": ["price", "rvol"]}),
                               ("live_sector_rank", {"metric": "price"})):
                x, e = ai_tools.run(self.ctx, name, args)
                if x:
                    parts.append(ai_tools.format_reply(x))
        return "\n\n".join(parts)

    def digest(self, slot):
        text = self.digest_text(slot)
        self._log({"ts": datetime.now().strftime("%H:%M:%S"), "digest": slot, "sent": bool(text)})
        if not text:
            return False
        for chat in self.allowed:
            try:
                self.tg.send(chat, text)
            except Exception as e:
                print("[AI chat] digest send failed: %s" % e)
        return True


def _loop(bot):
    offset, fails = None, 0
    # Skip anything sent while the app was off: answering a stale "now" question
    # with live data would be wrong.
    try:
        old = bot.tg.get_updates(-1)
        if old:
            offset = old[-1]["update_id"] + 1
    except Exception as e:
        print("[AI chat] first poll failed: %s" % e)
    print("[AI chat] listening (allowed chats: %d)" % len(bot.allowed))
    while True:
        try:
            ups = bot.tg.get_updates(offset)
            fails = 0
            for u in ups:
                offset = u["update_id"] + 1
                try:
                    bot.handle(u)
                except Exception as e:                    # one message must not kill the thread
                    print("[AI chat] handle error: %s" % e)
        except Exception as e:
            wait = BACKOFF_S[min(fails, len(BACKOFF_S) - 1)]
            fails += 1
            print("[AI chat] poll error (%s) — retry in %ds" % (str(e)[:120], wait))
            time.sleep(wait)


def parse_slots(raw):
    out = []
    for x in str(raw or "").split(","):
        x = x.strip()
        try:
            datetime.strptime(x, "%H:%M")
            out.append(x)
        except ValueError:
            if x:
                print("[AI chat] ignoring bad digest time %r" % x)
    return sorted(set(out))


def due_slots(slots, now, sent):
    """Slots due right now: weekday, time reached within the last 10 minutes, not yet sent today.
    The 10-minute window means a restart at 11:04 still sends 11:00, but 11:30 does not."""
    if now.weekday() >= 5:
        return []
    hm, day = now.strftime("%H:%M"), now.strftime("%Y-%m-%d")
    out = []
    for s in slots:
        h, m = int(s[:2]), int(s[3:])
        mins_late = (now.hour * 60 + now.minute) - (h * 60 + m)
        if 0 <= mins_late <= 10 and (day, s) not in sent and hm >= s:
            out.append(s)
    return out


def _digest_loop(bot, slots):
    sent = set()
    print("[AI chat] digests at %s (weekdays)" % ", ".join(slots))
    while True:
        try:
            now = datetime.now()
            for s in due_slots(slots, now, sent):
                sent.add((now.strftime("%Y-%m-%d"), s))
                bot.digest(s)
        except Exception as e:
            print("[AI chat] digest error: %s" % e)
        time.sleep(20)


KITE_FILL_AT = "09:25"           # after the 09:20 re-prefetch, so only real gaps remain


def _kite_fill_loop(ctx):
    """Once per weekday (and once ~60 s after start): fill yesterday's missing daily bars
    from Kite. Read-only; failures are logged, never raised."""
    time.sleep(60)
    done = ""
    while True:
        try:
            now = datetime.now()
            if now.weekday() < 5 and done != now.strftime("%Y-%m-%d") and \
                    (now.strftime("%H:%M") >= KITE_FILL_AT or not done):
                st = ai_tools.kite_fill(ctx)
                if st.get("day") == ctx.today:
                    done = now.strftime("%Y-%m-%d")
        except Exception as e:
            print("[AI chat] kite fill error: %s" % e)
        time.sleep(120)


def _sample_loop(ctx, bot=None):
    """Every 60 s in market hours: copy price / VWAP / PACE of every stock and the indices
    (ai_tools.sample_once), then run the saved alerts (code only, no LLM)."""
    import ai_alerts
    while True:
        try:
            now = datetime.now()
            hm = now.hour * 60 + now.minute
            if now.weekday() < 5 and SAMPLE_FROM <= hm <= SAMPLE_TO:
                ai_tools.sample_once(ctx)
                if bot is not None and bot.alerts is not None and hm <= ALERT_TO:
                    ai_alerts.check_alerts(ctx, bot.alerts, lambda chat, text: bot._reply(chat, text) is not False)
        except Exception as e:
            print("[AI chat] sampler error: %s" % e)
        time.sleep(60 - time.time() % 60 + 1)       # on the minute, +1 s after the scan writes


SAMPLE_FROM, SAMPLE_TO = 9 * 60 + 15, 15 * 60 + 40
ALERT_TO = 15 * 60 + 30            # equity cash ends 15:15; no pings after 15:30


def start():
    """Called once from app.py. Never raises; logs why it did not start."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    try:
        import state
        enabled = _cfg("AI_CHAT_ENABLED", True)
        if str(enabled).lower() in ("0", "false", "no", "off"):
            print("[AI chat] disabled (AI_CHAT_ENABLED)")
            return
        token = _cfg("AI_TG_TOKEN") or getattr(state, "_CFG_TG_TOKEN", "")
        chats = str(_cfg("AI_TG_CHAT_IDS") or getattr(state, "_CFG_TG_CHAT", "") or "").split(",")
        if not token or not any(c.strip() for c in chats):
            print("[AI chat] not started: Telegram token or chat id missing")
            return
        import ai_llm
        sectors = {str(v).upper() for v in (getattr(state, "SECTOR_MAP", {}) or {}).values() if v}
        router = ai_llm.GeminiRouter(ai_tools.TOOLS, sectors)
        base = os.path.dirname(os.path.abspath(__file__))
        ctx = ai_tools.Ctx(state)
        import ai_alerts
        store = ai_alerts.AlertStore(os.path.join(getattr(state, "DATA_DIR", os.path.join(base, "data")),
                                                  "ai_alerts.json"))
        # Alerts are a few a day and a wrong rule costs a missed trade: the stronger model reads them;
        # the question model is the fallback (also when flash is not on this key -> 404 -> skipped).
        alert_model = ai_llm._cfg("AI_ALERT_MODEL", "gemini-3.5-flash")
        alert_router = ai_llm.GeminiRouter(ai_alerts.ALERT_TOOLS, sectors, model=alert_model,
                                           fallback=ai_llm._cfg("AI_MODEL", ai_llm.DEFAULT_MODEL),
                                           system=ai_alerts.system_prompt)
        try:                                   # hybrid example retrieval: the meaning side
            import ai_examples
            ai_examples.EMBED = ai_examples.gemini_embedder(alert_router.client)
        except Exception as e:
            print("[AI chat] embeddings off (keyword-only examples): %s" % e)
        bot = Bot(ctx, router, Telegram(token), chats, os.path.join(base, "logs", "ai_chat"),
                  alerts=store, alert_router=alert_router)
        threading.Thread(target=_loop, args=(bot,), name="ai_chat", daemon=True).start()
        threading.Thread(target=_kite_fill_loop, args=(ctx,), name="ai_kite_fill", daemon=True).start()
        threading.Thread(target=_sample_loop, args=(ctx, bot), name="ai_sampler", daemon=True).start()
        try:
            print("[AI chat] startup data check:\n" + ai_tools.check_report(ctx, live=False))
        except Exception as e:
            print("[AI chat] startup data check failed: %s" % e)
        slots = parse_slots(_cfg("AI_DIGEST_TIMES", DEFAULT_DIGESTS))
        if slots:
            threading.Thread(target=_digest_loop, args=(bot, slots), name="ai_digest",
                             daemon=True).start()
    except Exception as e:
        print("[AI chat] not started: %s" % e)
