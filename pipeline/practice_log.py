"""Practice-trend collector (Nov 8, Clinton: does the Wed/Thu/Fri pattern matter?). nflverse's injury file keeps only each player's
LATEST practice status, so the week's pattern is lost unless we snapshot it. Every daily run appends this week's skill-player rows
to data/practice_log.csv with the capture time; a row is added only when a player's practice or game status changed since his last
snapshot that week. Morning runs see the previous day's practice (Thu run = Wed practice, Fri = Thu, Sat = Fri final).
Test once ~4 weeks are logged: does the sequence (e.g. DNP-DNP-Limited vs Full-Full-Limited) move the chance beyond
injuries.p_play_q? Until then it is a log only, not a model input. Usage: python practice_log.py W"""
import os, sys, pandas as pd
W = int(sys.argv[1]); SEASON = int(os.environ.get("NFL_SEASON", 2026))
f = "data/practice_log.csv"
inj = pd.read_parquet(f"data/injuries_{SEASON}.parquet")
cur = inj[(inj.week == W) & inj.position.isin(["QB", "RB", "WR", "TE"])][["season", "week", "gsis_id", "full_name", "team", "position", "practice_status", "report_status"]].copy()
cur["captured_et"] = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M")
cur = cur.fillna({"practice_status": "", "report_status": ""})
old = pd.read_csv(f, dtype=str).fillna("") if os.path.exists(f) else cur.iloc[:0].astype(str)
key = ["season", "week", "gsis_id"]
if len(old):
    last = old.sort_values("captured_et").drop_duplicates(key, keep="last").set_index(key)[["practice_status", "report_status"]]
    c = cur.astype({"season": str, "week": str}).set_index(key)
    same = c.index.isin(last.index)
    unchanged = same & (c.practice_status.values == last.reindex(c.index).practice_status.values) & (c.report_status.values == last.reindex(c.index).report_status.values)
    cur = cur[~unchanged]
out = pd.concat([old, cur.astype(str)])
out.to_csv(f, index=False)
print(f"practice_log: week {W}, {len(cur)} new/changed rows, {len(out)} total")
