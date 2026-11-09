# -*- coding: utf-8 -*-
"""
demo.py -- talk to the TradEdge AI assistant in your terminal, on a FAKE market.

    python demo.py              real Gemini routes your questions (needs GEMINI_API_KEY)
    python demo.py --offline    no key: replays recorded model outputs from the eval cases
    python demo.py --tour       non-interactive walk-through (offline; used by CI)

It runs the SAME code as the Telegram bot (ai_chat.Bot, ai_alerts.AlertFlow, ai_tools),
only Telegram is swapped for the console and the market is fake_market (a seeded random
walk, 24 stocks). Type a question, or start with "alert me" to create an alert.

Commands
    /tick 5    the fake clock moves 5 minutes; saved alerts are checked every minute
    /alerts    your alerts          /help    what you can ask
    /try       sample questions     /quit
    1, 2, ...  press the numbered button under the last reply
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ai_alerts                          # noqa: E402
import ai_chat                            # noqa: E402
import ai_llm                             # noqa: E402
import ai_tools                           # noqa: E402
from fake_market import TODAY, Market     # noqa: E402

CHAT = "demo"

# Offline mode knows exactly these sentences (and every other one in evals/*.jsonl).
TRY = ["top gainers today with price and rvol", "what is low of nykaa", "stocks where open = high",
       "fut vol increase highest in last 15 min in fmcg", "which stocks made new day low in last 15 min",
       "top 3 sectors right now", "top gainers last 5 days", "compare sbi, icici bank and axis bank",
       "overall market mood right now", "aaje open = low vala stock batavo", "should I buy TATAMOTORS",
       "alert me when nykaa goes below 348", "alert me when nykaa slips under today's low"]

TOUR = [("q", "top gainers today with price and rvol"), ("q", "what is low of nykaa"),
        ("q", "stocks where open = high"), ("q", "top gainers last 5 days"), ("q", "should I buy TATAMOTORS"),
        ("q", "alert me when nykaa goes below 348"), ("btn", 1), ("tick", 8), ("q", "/alerts")]


class Console:
    """Stands in for ai_chat.Telegram: prints replies, remembers the last buttons."""
    def __init__(self):
        self.buttons = []

    def send(self, chat, text, buttons=None):
        print("\n" + "─" * 60 + "\n" + text)
        self.buttons = [b for row in (buttons or []) for b in row]
        if self.buttons:
            print("   " + "   ".join("[%d] %s" % (i + 1, lbl) for i, (lbl, _d) in enumerate(self.buttons)))
        return True

    def answer_callback(self, cq_id, text=""):
        pass


class Replay(ai_llm.FakeRouter):
    """Offline router: answers with what the model returned for that sentence in the eval
    cases ("model" = the model's real, sometimes wrong, output; else the expected route).
    Guards and validation still run on it -- that is the part worth seeing."""
    def __init__(self, path, miss="cannot_answer"):
        self.miss = miss
        table = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                if c.get("tool") != "code":
                    v = (c["tool"], c.get("model") or c["args"])
                    q = c["q"].strip().lower()
                    table[q] = v
                    table.setdefault(ai_alerts.CREATE_PREFIX.sub("", q, count=1).strip(), v)  # alert body
        super().__init__(table)

    def route(self, question):
        q = question.split("\nNEW: ")[-1].strip().lower()
        v = self.table.get(q)
        if v is None:
            return ai_llm.Route(self.miss, {"reason": "offline mode only knows the sentences in evals/*.jsonl "
                                                      "(try /try); set GEMINI_API_KEY for any sentence"}, "replay")
        return ai_llm.Route(v[0], dict(v[1]), "replay")


def routers(offline):
    sectors = {v.upper() for v in __import__("fake_market").SECTOR_MAP.values()}
    if offline:
        return (Replay(os.path.join(HERE, "evals", "ai_router_cases.jsonl")),
                Replay(os.path.join(HERE, "evals", "alert_cases.jsonl"), miss="cannot_alert"))
    q = ai_llm.GeminiRouter(ai_tools.TOOLS, sectors)
    a = ai_llm.GeminiRouter(ai_alerts.ALERT_TOOLS, sectors, model=ai_llm._cfg("AI_ALERT_MODEL", "gemini-3.5-flash"),
                            fallback=ai_llm._cfg("AI_MODEL", ai_llm.DEFAULT_MODEL), system=ai_alerts.system_prompt)
    try:
        import ai_examples
        ai_examples.EMBED = ai_examples.gemini_embedder(a.client)
    except Exception as e:
        print("(embeddings off, keyword-only examples: %s)" % e)
    return q, a


def tick(market, bot, minutes):
    for _ in range(max(1, minutes)):
        market.step()
        ai_alerts.check_alerts(market.ctx, bot.alerts, lambda chat, text: bot._reply(chat, text),
                               now=market.now.timestamp(), day=TODAY)
    print("\n⏱  fake clock %s" % market.now.strftime("%H:%M"))


def handle(market, bot, tg, line):
    line = line.strip()
    if not line:
        return True
    if line in ("/quit", "/exit", "q"):
        return False
    if line == "/try":
        print("\n".join("  • " + t for t in TRY))
        return True
    if line.startswith("/tick"):
        parts = line.split()
        tick(market, bot, int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1)
        return True
    if line.isdigit() and tg.buttons:
        i = int(line) - 1
        if 0 <= i < len(tg.buttons):
            bot.handle({"callback_query": {"id": "x", "data": tg.buttons[i][1],
                                           "message": {"chat": {"id": CHAT}}}})
            return True
    bot.handle({"update_id": 1, "message": {"chat": {"id": CHAT}, "text": line}})
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no API key: replay recorded model outputs")
    ap.add_argument("--tour", action="store_true", help="scripted walk-through (offline)")
    a = ap.parse_args()
    offline = a.offline or a.tour or not ai_llm._cfg("GEMINI_API_KEY")
    work = tempfile.mkdtemp(prefix="tradedge_demo_")
    market = Market(work)
    ai_chat.MAX_PER_MIN = 60                  # one local user; the 6/min limit protects the free quota
    tg = Console()
    qr, ar = routers(offline)
    bot = ai_chat.Bot(market.ctx, qr, tg, [CHAT], os.path.join(work, "logs"),
                      alerts=ai_alerts.AlertStore(os.path.join(work, "data", "ai_alerts.json")), alert_router=ar)
    print("TradEdge AI assistant · fake market %s %s · %s" % (
        TODAY, market.now.strftime("%H:%M"), "OFFLINE replay (no key)" if offline else "Gemini " + qr.model))
    if a.tour:
        for kind, val in TOUR:
            if kind == "q":
                print("\n\n>>> " + val)
                handle(market, bot, tg, val)
            elif kind == "btn":
                print("\n\n>>> [press %d]" % val)
                handle(market, bot, tg, str(val))
            else:
                print("\n\n>>> /tick %d" % val)
                tick(market, bot, val)
        return 0
    print("Type a question (/try for ideas, /tick 5 to move the clock, /quit).")
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not handle(market, bot, tg, line):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
