"""Append today's schedule lines to a running history so line movement accumulates run by run.
History: data/line_history.csv  (game_id, captured_at, spread_line, total_line, home_ml, away_ml)
Seeds Wednesday's build (lines_2026_wk4.json) as the first snapshot if the file is new."""
import json, os, datetime as dt
import pandas as pd
H = "data/line_history.csv"
s = pd.read_parquet("data/sched.parquet")
s = s[(s.season == 2026) & s.spread_line.notna()]
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
snap = s[["game_id", "spread_line", "total_line", "home_moneyline", "away_moneyline"]].assign(captured_at=now)
if os.path.exists(H):
    h = pd.read_csv(H)
else:
    h = pd.DataFrame(columns=snap.columns)
# only append when a game's line actually changed since the last capture (or is new)
last = h.sort_values("captured_at").groupby("game_id").tail(1).set_index("game_id")
rows = []
for _, r in snap.iterrows():
    if r.game_id not in last.index or last.loc[r.game_id, "spread_line"] != r.spread_line or last.loc[r.game_id, "total_line"] != r.total_line:
        rows.append(r)
h = pd.concat([h, pd.DataFrame(rows)]) if rows else h
h.to_csv(H, index=False)
print(f"history: {len(h)} rows, {h.game_id.nunique()} games; appended {len(rows)} changes at {now}")
