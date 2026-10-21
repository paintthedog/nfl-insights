"""Pull everything that changes week to week from nflverse: play-by-play, FTN charting, schedule/lines, snap counts,
weekly rosters (incl. game-day inactives once posted), injury reports, player ids. Run before any projection script."""
import time, subprocess, sys
import pandas as pd, nflreadpy as nfl
SEASON = 2026
def get(fn, *a):
    for i in range(5):
        try: return fn(*a).to_pandas()
        except Exception as e: print("retry", fn.__name__, i, str(e)[:80]); time.sleep(5)
    raise SystemExit(f"could not load {fn.__name__}")
subprocess.run([sys.executable, "get_data.py"], check=True)
import os
if os.path.exists("spread_fill.json"): os.remove("spread_fill.json")
if os.path.exists("total_fill.json"): os.remove("total_fill.json")      # fresh schedule: any lent spreads are gone
# Retractable-roof stadiums (ARI, ATL, DAL, HOU, IND) have no roof status until game day; they are closed in 80-93% of
# games since 2021, so unplayed games there are treated as closed
_s = pd.read_parquet("data/sched.parquet"); _m = _s.roof.isna() & _s.result.isna()
_s.loc[_m, "roof"] = "closed"; _s.to_parquet("data/sched.parquet"); print("retractable-roof games set to closed:", int(_m.sum()))
pl = get(nfl.load_players); pl.to_parquet("data/players.parquet")
ros = get(nfl.load_rosters_weekly, [SEASON]); ros.to_parquet(f"data/rosters_{SEASON}.parquet")
_sd = get(nfl.load_snap_counts, [SEASON]); _sd = _sd[[c for c in ["game_id","season","week","game_type","team","opponent","pfr_player_id","player","position","defense_snaps","defense_pct","offense_snaps","offense_pct"] if c in _sd.columns]]
_old = pd.read_parquet("data/snaps_def.parquet"); pd.concat([_old[_old.season != SEASON], _sd]).to_parquet("data/snaps_def.parquet")
inj = get(nfl.load_injuries, [SEASON]); inj.to_parquet(f"data/injuries_{SEASON}.parquet")
try:   # team depth charts (updated daily): latest snapshot per team, for next-man-up candidate order
    dc = get(nfl.load_depth_charts, [SEASON]); dc = dc.copy(); dc["dt"] = pd.to_datetime(dc.dt, errors="coerce")
    dc = dc.sort_values("dt").groupby(["team", "pos_abb", "gsis_id"], as_index=False).tail(1)
    latest = dc.groupby("team").dt.transform("max"); dc[dc.dt >= latest - pd.Timedelta(days=10)].to_parquet("data/depth_charts.parquet")
    print("depth charts saved:", dc.dt.max())
except Exception as e: print("depth charts skipped:", e)
allinj = pd.read_parquet("data/injuries_all.parquet")
pd.concat([allinj[allinj.season != SEASON], inj]).to_parquet("data/injuries_all.parquet")
sn = get(nfl.load_snap_counts, [SEASON])
pfr = pl.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
sn["gsis_id"] = sn.pfr_player_id.map(pfr)
sn = sn.dropna(subset=["gsis_id"])[["game_id", "season", "week", "team", "gsis_id", "position", "offense_snaps", "offense_pct"]]
old = pd.read_parquet("data/snaps.parquet")
pd.concat([old[old.season != SEASON], sn.astype(old.dtypes.to_dict(), errors="ignore")]).to_parquet("data/snaps.parquet")
pbp = pd.read_parquet(f"data/pbp_{SEASON}.parquet", columns=["game_id", "week"])
print("pbp games by week:", pbp.groupby("week").game_id.nunique().to_dict())
print("rosters weeks:", sorted(ros.week.unique()), "| injury weeks:", sorted(inj.week.dropna().unique()), "| snaps weeks:", sorted(sn.week.unique()))
