"""FanDuel prices from the same pull (Oct 21) -> fd_log.csv, for line shopping and the DK-vs-FanDuel board check.
Add bookmakers=draftkings,fanduel to the URLs Clinton already pastes (as far as we know the Odds API charges per market, not per book;
confirm the credit count on the first pull). Save FanDuel's outcomes to odds_raw/wk<W>/<AWAY>_<HOME>_fd.txt, one per line, in the
same 5-field format as the DraftKings files:  market|subject|side|point|price
  player_anytime_td | player | Yes | | price      player_receptions | player | Over/Under | pts | price
  *_alternate | player | Over | rung | price       game_h2h | full team name | ML | | price
  game_spreads | full team name | | pts | price    game_totals | Over/Under | Over/Under | pts | price    team_totals | full team name | Over/Under | pts | price
Usage: python ingest_fd.py 7 "2026-10-25T11:30Z" """
import sys, glob, os, pandas as pd
from team_names import FULL
W, pulled = int(sys.argv[1]), sys.argv[2]
rows = []
for f in sorted(glob.glob(f"odds_raw/wk{W}/*_fd.txt")):
    away, home = os.path.basename(f).split("_")[:2]; gid = f"2026_{W:02d}_{away}_{home}"
    for line in open(f):
        x = line.strip().split("|")
        if len(x) != 5 or x[0] == "LAST_UPDATE": continue
        mk, subj, side, pt, pr = x
        if mk in ("team_totals", "game_h2h", "game_spreads"): subj = FULL.get(subj, subj)
        rows.append(dict(week=W, game_id=gid, market=mk, subject=subj, side=side or ("ML" if mk == "game_h2h" else ""), point=float(pt) if pt else None,
                         price=float(pr), pulled_at=pulled))
new = pd.DataFrame(rows, columns=["week", "game_id", "market", "subject", "side", "point", "price", "pulled_at"])
old = pd.read_csv("fd_log.csv") if os.path.exists("fd_log.csv") else new.iloc[0:0]
pd.concat([old, new]).drop_duplicates().to_csv("fd_log.csv", index=False)
print(f"week {W}: {len(new)} FanDuel outcomes ->", new.groupby("market").size().to_dict() if len(new) else {})
