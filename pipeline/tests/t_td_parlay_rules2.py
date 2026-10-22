import itertools, numpy as np, pandas as pd
exec(open("tests/t_td_totals.py").read().split("# (b) history")[0].split('print("(a)')[0])
pos = pd.read_parquet("data/players.parquet", columns=["gsis_id", "position"]).set_index("gsis_id").position
ro = pd.read_parquet("data/rosters_2026.parquet")
a["pos"] = a.pid.map(pos)
rows = []
def rec(wk, type_, combo):
    hit = all(l["hit"] for l in combo); pay = float(np.prod([l["price"] for l in combo]))
    rows.append(dict(week=wk, type=type_, legs=" + ".join(l["player"] for l in combo), price=pay, dk=float(np.prod([1 / l["price"] for l in combo])), hit=hit, pnl=pay - 1 if hit else -1.0))
for wk, x in a.groupby("week"):
    tm = ro[ro.week == wk].drop_duplicates("gsis_id").set_index("gsis_id").team; x = x.assign(team=x.pid.map(tm))
    sw = s[(s.season == 2026) & (s.week == wk)].set_index("game_id")
    pool = x[(x.price >= 1.15) & (x.price <= 3.0)].sort_values("price").drop_duplicates("game_id").to_dict("records")
    for l in pool: rec(wk, "singles (likeliest per game)", [l])
    for i in range(0, len(pool) - 1, 2):
        if i // 2 < 4: rec(wk, "all games, 2 legs", pool[i:i + 2])
    for i in range(0, len(pool) - 2, 3):
        if i // 3 < 3: rec(wk, "all games, 3 legs", pool[i:i + 3])
    tp = [l for l in pool if l["top5"]]
    for c in itertools.combinations(tp, 2): rec(wk, "top-5 totals, 2 legs", c)
    for c in itertools.combinations(tp, 3): rec(wk, "top-5 totals, 3 legs", c)
    fr = []
    for gid in x[x.top5].game_id.unique():
        g = sw.loc[gid]; fav = g.home_team if g.spread_line > 0 else g.away_team
        c = x[(x.game_id == gid) & (x.team == fav) & (x.pos == "RB") & x.price.between(1.15, 3.0)].sort_values("price")
        if len(c): fr.append(c.iloc[0].to_dict())
    for k in (2, 3):
        for c in itertools.combinations(fr, k): rec(wk, "top-5 favorites' lead RB", c)
    hg = x.sort_values("total_line", ascending=False).game_id.iloc[0]; g = sw.loc[hg]; both = []
    for t in (g.away_team, g.home_team):
        c = x[(x.game_id == hg) & (x.team == t) & x.price.between(1.15, 3.0)].sort_values("price")
        if len(c): both.append(c.iloc[0].to_dict())
    if len(both) == 2: rec(wk, "top-total game, both teams' shortest", both)
r = pd.DataFrame(rows)
print(r[r.week == 5].groupby("type").agg(n=("hit", "size"), won=("hit", "sum"), avg_price=("price", "mean"), dk_exp=("dk", "sum"), units=("pnl", "sum")).round(2).to_string())
print(r[(r.week == 5) & r.type.isin(["top-5 favorites' lead RB", "top-total game, both teams' shortest"])][["type", "legs", "price", "hit"]].round(2).to_string(index=False))
