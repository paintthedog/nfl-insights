import numpy as np, pandas as pd, td_model as T
pbp, sched, players = T.load(); pg, team, ctx = T.build_tables(pbp, sched, players)
pg["season"] = pg.season if "season" in pg else pg.game_id.str[:4].astype(int)
V = {"current": (1.0, 1.0), "old team x0.3": (1.0, 0.3),
     "WR x0.5 + old team x0.3": ({"WR": 0.5, "RB": 1.0, "QB": 1.0, "TE": 1.0}, 0.3),
     "all x0.6 + old team x0.3": (0.6, 0.3)}
res = {}
for name, (sc, tc) in V.items():
    T.TD_SEASON_CARRY, T.TD_TEAM_CARRY = sc, tc
    d = T.features(pg, team, ctx); d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()]
    ps = []
    for s in range(2021, 2027):
        tmm = T.fit_team(team[team.season < s]); mr, mp = T.fit_players(d[d.season < s], tmm)
        ps.append(T.predict(d[d.season == s], tmm, mr, mp))
    a = pd.concat(ps); a["hit"] = ((a.rtd + a.ptd) >= 1).astype(int)
    res[name] = {"all": np.mean((a.p_td - a.hit) ** 2), "weeks 1-4": np.mean(((a.p_td - a.hit) ** 2)[a.week <= 4]),
                 **{g: np.mean(((a.p_td - a.hit) ** 2)[a.pos == g]) for g in ["QB", "RB", "WR", "TE"]}}
r = pd.DataFrame(res).T
print(((r / r.loc["current"] - 1) * 100).round(2).to_string())
