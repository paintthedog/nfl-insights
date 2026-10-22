"""Offensive-line continuity from snap counts: for each team-game, how many of the team's 5 'regular' starters
(the OL with the most snaps over the prior 4 games) are missing or playing reduced snaps. Prior games only."""
import numpy as np, pandas as pd
sn = pd.read_parquet("data/snaps.parquet")
sn = sn[sn.position.isin(["T", "G", "C", "OL", "OT", "OG"]) & (sn.offense_snaps > 0)].copy()
sched = pd.read_parquet("data/sched.parquet")[["game_id", "gameday", "season", "week"]]
sn = sn.merge(sched, on=["game_id", "season", "week"]).sort_values(["team", "gameday"])
rows = []
for team, g in sn.groupby("team"):
    games = g.drop_duplicates("game_id").sort_values("gameday")[["game_id", "gameday", "season"]].to_numpy()
    for i, (gid, gd, seas) in enumerate(games):
        prev = [x for x in games[max(0, i - 4):i] if x[2] == seas or i < 4]   # last 4 games (same season preferred)
        if len(prev) < 2:
            continue
        pg_ = g[g.game_id.isin([x[0] for x in prev])]
        reg = pg_.groupby("gsis_id").offense_pct.mean().sort_values(ascending=False).head(5)   # regular 5
        today = g[g.game_id == gid].set_index("gsis_id").offense_pct
        pct = today.reindex(reg.index).fillna(0.0)
        rows.append(dict(game_id=gid, posteam=team, ol_missing=float((pct < 0.5).sum()), ol_cont=float(pct.mean()), ol_n_prev=len(prev)))
ol = pd.DataFrame(rows)
ol.to_parquet("data/ol_continuity.parquet")
print(len(ol), "team-games |", ol.ol_missing.value_counts().sort_index().to_dict())
print("avg starters missing:", ol.ol_missing.mean().round(2))
