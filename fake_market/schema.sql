-- Public schema: ONLY the columns the AI tools read.
-- The private app's tables have more columns; they are not needed here and not published.

CREATE TABLE IF NOT EXISTS symbol_day (     -- one row per stock per session
  day TEXT NOT NULL, sym TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL,
  day_pct REAL,       -- day % change
  oi_sess REAL,       -- futures OI % change over the session
  max_pace REAL,      -- highest PACE seen that day
  rvol_max REAL,      -- highest relative volume seen that day
  PRIMARY KEY (day, sym)
);

CREATE TABLE IF NOT EXISTS signals (        -- scanner signals (names only, no rules)
  day TEXT NOT NULL, t TEXT NOT NULL, sym TEXT NOT NULL,
  stage TEXT NOT NULL, side TEXT,
  PRIMARY KEY (day, sym, stage, t)
);

CREATE TABLE IF NOT EXISTS outcomes (       -- what happened after a signal
  day TEXT NOT NULL, sym TEXT NOT NULL, stage TEXT NOT NULL, t TEXT NOT NULL,
  side TEXT, m15 REAL, m30 REAL, m60 REAL,  -- % move after 15 / 30 / 60 min
  res_10 TEXT,                              -- WIN | LOSS | FLAT
  src TEXT, bars_n INTEGER,                 -- bars_n = 0 means no data after the signal
  PRIMARY KEY (day, sym, stage, t)
);

-- One move can log many rows; count each stock once per day (first signal of the day).
CREATE VIEW IF NOT EXISTS v_signals_1pd AS
SELECT s.* FROM signals s
WHERE s.t = (SELECT MIN(x.t) FROM signals x WHERE x.day = s.day AND x.sym = s.sym);
