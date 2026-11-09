# -*- coding: utf-8 -*-
"""
state.py -- public stand-in for the private app's state module.

In the real app, state.py holds the live caches (scan results, bars, OI) and the F&O
universe. ai_chat.start() and evals/run_ai_evals.py import it for the symbol list and
sector map only, so this file gives them the fake market's universe. The live caches
come from fake_market.Market (see demo.py).
"""
from fake_market import FNO_SYMBOLS, SECTOR_MAP  # noqa: F401
