# -*- coding: utf-8 -*-
"""
predeploy.py -- run BEFORE replacing the live files. Prints GO or NO-GO.

    python predeploy.py           tests + alert evals + question evals (about 15 min)
    python predeploy.py --quick   tests + alert evals + only the question cases that failed last time

Why: unit tests use a FAKE Gemini that answers what we expect, so they cannot catch how the
real model reads a sentence (05-11: "nykaa goes below 348" came back as day % + price filter).
Only the eval run on the real model catches that, so it gates the deploy.

Needs GEMINI_EVAL_API_KEY in config.py or the environment (a key from a SEPARATE Google Cloud project, so the
bot's own 500 questions/day are never used). Read-only: no Telegram message, no data change.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_PASS = 90          # % of eval cases; a real model varies a little run to run


def _run(cmd):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", errors="replace")
    out = p.stdout or ""
    fails = [l for l in out.splitlines() if " FAIL " in l or l.startswith("FAILED")]
    for l in fails[:25]:
        print("   " + l)
    tail = [l for l in out.splitlines() if l.strip()][-2:]
    for l in tail:
        print("   " + l)
    return p.returncode, out


def _pct(out):
    m = re.findall(r"(\d+)/(\d+) passed", out)
    if not m:
        return None
    a, b = map(int, m[-1])
    return 100.0 * a / b if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    results = []

    code, out = _run([py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
    results.append(("unit tests", code == 0, "all passed" if code == 0 else "FAILED"))

    code, out = _run([py, os.path.join("evals", "run_ai_evals.py"), "--alerts"])
    pct = _pct(out)
    if code == 2 or pct is None:
        results.append(("alert evals", False, "did not run (GEMINI_EVAL_API_KEY set?)"))
    else:
        results.append(("alert evals", pct >= MIN_PASS, "%.0f%% (need %d%%)" % (pct, MIN_PASS)))

    qcmd = [py, os.path.join("evals", "run_ai_evals.py")] + (["--failed-only"] if a.quick else [])
    code, out = _run(qcmd)
    pct = _pct(out)
    if "re-running 0 failed" in out:
        results.append(("question evals", True, "nothing failed last time"))
    elif code == 2 or pct is None:
        results.append(("question evals", False, "did not run (GEMINI_EVAL_API_KEY set?)"))
    else:
        need = 50 if a.quick else MIN_PASS     # quick = only hard cases: half fixed is the bar
        results.append(("question evals", pct >= need,
                        ("%.0f%% of last run's failures now pass" % pct) if a.quick else
                        "%.0f%% (need %d%%)" % (pct, MIN_PASS)))

    print("\n" + "=" * 46)
    for name, ok, why in results:
        print("%s %-15s %s" % ("✅" if ok else "❌", name, why))
    go = all(ok for _n, ok, _w in results)
    print("=" * 46)
    print("GO — safe to deploy (after market hours)" if go else "NO-GO — send the FAIL lines above, do not deploy")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
