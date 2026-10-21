import numpy as np, pandas as pd
exec(open("tests/t_td_totals.py").read().split("# (b) history")[0].split('print("(a)')[0])
s5 = s[(s.season == 2026) & (s.week == 5)].sort_values("total_line", ascending=False)
print(s5[["game_id", "total_line", "spread_line", "away_score", "home_score", "total"]].head(7).to_string(index=False))
x = a[(a.week == 5) & a.top5].copy()
pos = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).set_index("gsis_id").position
x["pos"] = x.pid.map(pos)
# team of player: from props_log if present
print(x.columns.tolist()[:20])
ro = pd.read_parquet("data/rosters_2026.parquet"); tm = ro[ro.week == 5].drop_duplicates("gsis_id").set_index("gsis_id").team
x["team"] = x.pid.map(tm)
g5 = s5.head(5).game_id.tolist()
# all TD scorers in those games (incl. unpriced)
pb = pbp[pbp.game_id.isin(g5)]
sc = pb[pb.touchdown.eq(1) & pb.td_player_id.notna()].groupby(["game_id", "td_team", "td_player_name"]).size().reset_index(name="tds")
for gid in g5:
    r = s5[s5.game_id == gid].iloc[0]; y = x[x.game_id == gid].sort_values("price")
    print(f"\n=== {gid}  total {r.total_line}  spread(home) {r.spread_line}  final {r.away_team} {int(r.away_score)} - {r.home_team} {int(r.home_score)}  | priced players: {len(y)}")
    for t in y.itertuples():
        if t.price <= 6.0: print(f"   {'✔' if t.hit else ' '} {t.player:24s} {t.team!s:4s} {t.pos!s:3s} {t.price:5.2f}  model {t.model_p:.2f}")
    print("   scorers:", ", ".join(f"{z.td_player_name} ({z.td_team}{' x'+str(z.tds) if z.tds>1 else ''})" for z in sc[sc.game_id == gid].itertuples()))
