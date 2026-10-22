"""Ingest the extra markets from a game-morning pull into market_log.csv (one row per priced outcome).
Raw files: odds_raw/wk<W>/*_x.txt with lines  market|subject|side|point|price
  - player_pass_interceptions | QB name | Over/Under | 0.5 | price
  - player_receptions_alternate / player_rush_yds_alternate | player | Over | rung | price
  - team_totals | full team name | Over/Under | pts | price
  - game_h2h | full team name | ML | | price ;  game_totals | Over/Under | Over/Under | pts | price ;  game_spreads | full team name | | pts | price
Usage: python ingest_more.py 6 "2026-10-18T11:30Z" """
import sys, glob, os, pandas as pd
W, pulled = int(sys.argv[1]), sys.argv[2]
sch = pd.read_parquet("data/sched.parquet"); sch = sch[(sch.season == 2026) & (sch.week == W)]
TEAMS = pd.read_parquet("data/sched.parquet")[["home_team"]].drop_duplicates().home_team
NICK = {}   # 'Denver Broncos' -> DEN via nickname (last word), with the two LA/NY teams resolved by full city
from team_names import FULL
rows = []
for f in sorted(glob.glob(f"odds_raw/wk{W}/*_x.txt")):
    g = os.path.basename(f).replace("_x.txt", ""); away, home = g.split("_")[:2]; gid = f"2026_{W:02d}_{away}_{home}"
    for line in open(f):
        x = line.strip().split("|")
        if len(x) != 5 or x[0] == "LAST_UPDATE": continue
        mk, subj, side, pt, pr = x
        if mk in ("team_totals", "game_h2h", "game_spreads"): subj = FULL.get(subj, subj)
        rows.append(dict(week=W, game_id=gid, market=mk, subject=subj, side=side or ("ML" if mk == "game_h2h" else ""), point=float(pt) if pt else None,
                         price=float(pr), pulled_at=pulled))
new = pd.DataFrame(rows)
old = pd.read_csv("market_log.csv") if os.path.exists("market_log.csv") else pd.DataFrame(columns=new.columns)
pd.concat([old, new]).drop_duplicates().to_csv("market_log.csv", index=False)
print(f"week {W}: {len(new)} outcomes ->", new.groupby("market").size().to_dict() if len(new) else {})
