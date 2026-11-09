# -*- coding: utf-8 -*-
"""
Router evals: does Gemini pick the right tool + arguments?  READ-ONLY.

Calls the REAL model once per case in ai_router_cases.jsonl (175 cases), then
checks the result the same way the bot does (ai_tools.validate). Nothing is
executed against your data; no Telegram message is sent.

Run from the TradEdge folder (needs GEMINI_API_KEY in config.py or env):
    python evals/run_ai_evals.py
    python evals/run_ai_evals.py --model gemini-3.5-flash     # compare a model

Pass = right tool AND every expected argument matches after validation.
Arguments the case does not list are not checked.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import ai_llm      # noqa: E402
import ai_tools    # noqa: E402


def _norm(v):
    return v.strip().upper() if isinstance(v, str) else v


def _cond_key(c):
    c = dict(c or {})
    return (c.get("left"), {"==": "="}.get(c.get("op"), c.get("op")), c.get("right") or None,
            None if c.get("right") else float(c.get("value") or 0), float(c.get("pct") or 0))


def _match(key, want, got):
    if key == "conditions":                # every expected condition present (extra ones allowed)
        return {_cond_key(c) for c in want} <= {_cond_key(c) for c in got or []}
    if isinstance(want, list) and key not in ("show", "filters"):     # e.g. symbols: same set, any case
        return {_norm(x) for x in want} == {_norm(x) for x in (got or [])}
    if key == "show":                      # expected columns must be present; extras (default price) ok
        return set(want) <= set(got or [])
    if key == "filters":                   # same conditions, any order
        norm = lambda fl: sorted((f["col"], f["op"], float(f["value"])) for f in fl or [])  # noqa: E731
        return norm(want) == norm(got)
    return _norm(got) == _norm(want)


def _check(ctx, r, want):
    """{} when every expected argument matches after validation, else the mismatches."""
    if not want:
        return {}
    try:
        clean = ai_tools.validate(ctx, r.name, r.args)
        sc = ai_tools.scope_args(ctx, r.name, r.args, _FAKE_PREV) or {}
        clean["from_previous"] = sc.get("from", 0)
        clean["filters"] = sc.get("filters", [])
        bad = {k: clean.get(k) for k, v in want.items() if k != "unsupported" and not _match(k, v, clean.get(k))}
        if want.get("unsupported") and not str(r.args.get("unsupported") or "").strip():
            bad["unsupported"] = "model did not say what it can't do"      # honesty guard (rule 13)
        return bad
    except ai_tools.ToolError as e:
        return {"validation": str(e)}


def _prev_list(c):
    """The bot's memory shape for this case's earlier answers (fix_route + scope_args read it)."""
    prev = c.get("prev")
    if not prev:
        return None
    if isinstance(prev[0], str):
        prev = [prev]
    out = []
    for i, x in enumerate(prev):
        try:
            args = ai_tools.validate(_CTX[0], x[1], x[2])
        except Exception:
            args = x[2]
        out.append({"tool": x[1], "args": args, "symbols": ["X"], "time": "14:%02d" % (40 - i), "q": x[0]})
    return out + _FAKE_PREV[len(out):]


_CTX = [None]
_FAKE_PREV = [{"symbols": ["X"], "time": "14:40", "q": "x"}] * 5   # lets scope_args accept any #1-5


def _prompt(c):
    """Same follow-up framing the bot uses (ai_llm.context_prompt). `prev` = list of
    earlier answers, most recent first: [[question, tool, args, n_stocks], ...]
    (a single [question, tool, args] is accepted too)."""
    prev = c.get("prev")
    if not prev:
        return c["q"]
    if isinstance(prev[0], str):
        prev = [prev]
    hist = [{"time": "14:%02d" % (40 - i), "q": p[0], "tool": p[1], "args": p[2],
             "n": p[3] if len(p) > 3 else None} for i, p in enumerate(prev)]
    return ai_llm.context_prompt(hist, c["q"])


def run_alert_evals(ctx, sectors, key, args):
    """Alert mode: right tool + every expected field after ai_alerts.validate_rule (the same path
    the bot uses). 'code' cases (alert 3 delete / always) never reach the model: skipped here."""
    import ai_alerts
    router = ai_llm.GeminiRouter(ai_alerts.ALERT_TOOLS, sectors, api_key=key or None,
                                 model=args.model or ai_llm._cfg("AI_ALERT_MODEL", "gemini-3.5-flash"),
                                 fallback="", system=ai_alerts.system_prompt)
    import ai_examples                     # the same hybrid retrieval as the bot (cache in a separate file)
    ai_examples.EMBED = ai_examples.gemini_embedder(router.client)
    ai_examples.EMB_CACHE = os.path.join(HERE, "eval_examples_emb.json")
    cases = [json.loads(l) for l in open(os.path.join(HERE, "alert_cases.jsonl"), encoding="utf-8") if l.strip()]
    cases = [c for c in cases if c["tool"] != "code"]
    passed = 0
    for i, c in enumerate(cases, 1):
        bad = None
        body = ai_alerts.CREATE_PREFIX.sub("", c["q"], count=1)
        try:
            import ai_examples                     # same retrieval as the bot, leaving this case out
            r = router.route(ai_examples.prompt_with_examples(body, exclude=body)
                             if c["tool"] == "create_alert" else body)
            ok = r.name == c["tool"] or any(r.name == a[0] for a in c.get("alt", []))
            if ok and r.name == "create_alert":
                rule, _m, _n = ai_alerts.validate_rule(ctx, r.args, c["q"])
                bad = {k: rule.get(k) for k, v in c["args"].items() if not _match(k, v, rule.get(k))}
                sp = ai_alerts.scope_problems(ctx, rule, c["q"])
                if sp and not c.get("refuse"):
                    bad["scope"] = sp
                if c.get("unsupported") and not rule.get("unsupported"):
                    bad["unsupported"] = "model did not say what it can't do"
            elif ok and r.name == "update_alert":
                bad = {k: r.args.get(k) for k, v in c["args"].items() if not _match(k, v, r.args.get(k))}
            ok = ok and not bad
            got = r.name
        except Exception as e:
            ok, got = False, "error: %s" % e
        passed += ok
        print("%2d %s %-50s -> %s %s" % (i, "PASS" if ok else "FAIL", c["q"][:50], got, bad or ""))
        time.sleep(args.sleep)
    print("\nalerts | model %s | %d/%d passed (%.0f%%)" % (router.model, passed, len(cases), 100.0 * passed / len(cases)))
    return 0 if passed == len(cases) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between calls (free-tier RPM)")
    ap.add_argument("--failed-only", action="store_true", help="re-run only the last run's failures")
    ap.add_argument("--alerts", action="store_true", help="evaluate the ALERT sentences (alert_cases.jsonl)")
    ap.add_argument("--use-bot-key", action="store_true",
                    help="allow the bot's GEMINI_API_KEY (uses the bot's 500/day free quota)")
    args = ap.parse_args()
    # The free daily quota is per Google Cloud project. Evals on the bot's key eat the bot's
    # questions for the next 24 h (the day resets 13:30 IST), so a separate project key is the default.
    key = ai_llm._cfg("GEMINI_EVAL_API_KEY")
    if not key and not args.use_bot_key:
        print("Set GEMINI_EVAL_API_KEY in config.py (a key from a SEPARATE Google Cloud project),\n"
              "or pass --use-bot-key to spend the bot's daily quota.")
        return 2

    try:
        import state
        sectors = {str(v).upper() for v in (getattr(state, "SECTOR_MAP", {}) or {}).values() if v}
    except Exception as e:
        print("[evals] state.py not importable (%s) -- validating without symbol/sector lists" % e)
        state = type("S", (), {"FNO_SYMBOLS": [], "SECTOR_MAP": {}})()
        sectors = set()
    ctx = ai_tools.Ctx(state)
    _CTX[0] = ctx
    if args.alerts:
        return run_alert_evals(ctx, sectors, key, args)
    router = ai_llm.GeminiRouter(ai_tools.TOOLS, sectors, api_key=key or None, model=args.model, fallback="")

    cases = [json.loads(l) for l in open(os.path.join(HERE, "ai_router_cases.jsonl"), encoding="utf-8")
             if l.strip()]
    if args.failed_only:
        import glob
        runs = sorted(glob.glob(os.path.join(HERE, "results_*.jsonl")))
        if not runs:
            print("No earlier results_*.jsonl — run without --failed-only first.")
            return 2
        failed = {json.loads(l)["q"] for l in open(runs[-1], encoding="utf-8") if l.strip() and
                  not json.loads(l).get("pass")}
        cases = [c for c in cases if c["q"] in failed]
        print("re-running %d failed cases from %s" % (len(cases), os.path.basename(runs[-1])))
        if not cases:
            return 0
    out_path = os.path.join(HERE, "results_%s.jsonl" % datetime.now().strftime("%Y-%m-%d_%H%M"))
    passed, ms, tok = 0, [], []
    with open(out_path, "w", encoding="utf-8") as out:
        for i, c in enumerate(cases, 1):
            rec = {"q": c["q"], "want_tool": c["tool"], "want_args": c["args"]}
            try:
                r = router.route(_prompt(c))
                rec.update({"model_tool": r.name, "model_args": r.args, "ms": r.ms, "tokens": r.tokens})
                r.name, r.args, fixes = ai_tools.fix_route(ctx, c["q"], r.name, r.args, _prev_list(c))
                rec.update({"tool": r.name, "args": r.args})
                if fixes:
                    rec["fixes"] = fixes
                ms.append(r.ms)
                if r.tokens:
                    tok.append(r.tokens)
                # A case may list other acceptable answers: "alt": [[tool, args], ...]
                ok, bad = False, None
                for tool, want in [(c["tool"], c["args"])] + [tuple(a) for a in c.get("alt", [])]:
                    if r.name != tool:
                        continue
                    bad = _check(ctx, r, want)
                    if not bad:
                        ok = True
                        break
                if not ok and bad:
                    rec["mismatch"] = bad
                rec["pass"] = ok
            except ai_llm.RouterError as e:
                rec.update({"pass": False, "error": "%s: %s" % (e.kind, e)})
            passed += rec["pass"]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print("%2d %s %-55s -> %s %s" % (i, "PASS" if rec["pass"] else "FAIL", c["q"][:55],
                                           rec.get("tool", rec.get("error", "")), rec.get("mismatch", "")))
            time.sleep(args.sleep)
    n = len(cases)
    print("\nmodel %s | %d/%d passed (%.0f%%) | median latency %s ms | avg tokens %s"
          % (router.model, passed, n, 100.0 * passed / n,
             sorted(ms)[len(ms) // 2] if ms else "-", int(sum(tok) / len(tok)) if tok else "-"))
    print("details: %s" % out_path)
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
