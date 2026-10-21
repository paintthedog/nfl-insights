"""Build ol_missing_wk<W>.csv: how many regular offensive-line starters each team is missing this week.
A regular = 50%+ of offensive snaps at an OL position over the team's last four games. Missing = on injured reserve /
retired / cut per the roster file, Out or Doubtful on this week's report, DNP on the latest practice report, or named
in news_overrides_wk<W>.json under "ol_out" (retirements and trades never reach the injury report)."""
import json, os, sys
import pandas as pd
W = int(os.environ.get("NFL_WEEK", sys.argv[1] if len(sys.argv) > 1 else 5)); SEASON = 2026
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"]).drop_duplicates("gsis_id")
OL = {"T", "G", "C", "OT", "OG", "OL"}
snaps = pd.read_parquet("data/snaps.parquet", columns=["game_id", "gsis_id", "team", "offense_pct"]).drop_duplicates(["game_id", "gsis_id"])
if snaps.offense_pct.max() > 1.5: snaps["offense_pct"] = snaps.offense_pct / 100
sched = pd.read_parquet("data/sched.parquet")
sc26 = sched[(sched.season == SEASON) & sched.result.notna()][["game_id", "gameday"]]
snaps = snaps.merge(sc26, on="game_id").merge(pl, on="gsis_id")
snaps = snaps[snaps.position.isin(OL)]
ros = pd.read_parquet(f"data/rosters_{SEASON}.parquet")
ros = ros[ros.week <= W].sort_values("week").drop_duplicates("gsis_id", keep="last")
gone_ros = set(ros[ros.status.isin(["RES", "RET", "CUT", "EXE"])].gsis_id)
inj = pd.read_parquet(f"data/injuries_{SEASON}.parquet"); inj = inj[inj.week == W]
out_rep = set(inj[inj.report_status.isin(["Out", "Doubtful"])].gsis_id)
dnp = set(inj[inj.practice_status.astype(str).str.contains("Did Not", na=False) & inj.report_status.isna()].gsis_id)
news_out = set()
try:
    nw = json.load(open(f"news_overrides_wk{W}.json"))
    keys = {str(n).lower().replace("'", "").replace(".", "") for n in nw.get("ol_out", [])}
    news_out = set(pl[pl.display_name.str.lower().str.replace("'", "").str.replace(".", "", regex=False).isin(keys)].gsis_id)
except FileNotFoundError: pass
rows = []
for tm, d in snaps.groupby("team"):
    last4 = d.drop_duplicates("game_id").sort_values("gameday").game_id.tolist()[-4:]
    reg = d[d.game_id.isin(last4)].groupby("gsis_id").offense_pct.mean(); reg = reg[reg >= 0.5]
    miss = [p for p in reg.index if p in gone_ros or p in out_rep or p in news_out or p in dnp]
    rows.append(dict(team=tm, ol_missing_pred=len(miss),
                     who="; ".join(pl.set_index("gsis_id").display_name.get(p, p) for p in miss)))
f = pd.DataFrame(rows)
f.to_csv(f"ol_missing_wk{W}.csv", index=False)
print(f[f.ol_missing_pred > 0].sort_values("ol_missing_pred", ascending=False).to_string(index=False))
