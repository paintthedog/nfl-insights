"""Backtest the paper-slate TD parlay rules on the priced weeks (props_log): likeliest scorer per game (1.25-3.00),
parlayed across games; all-games pairs/triples in price order, and top-5-total combos."""
import itertools, numpy as np, pandas as pd
exec(open("tests/t_td_totals.py").read().split("# (b) history")[0].split('print("(a)')[0])
rows = []
for wk, x in a.groupby("week"):
    x = x[(x.price >= 1.25) & (x.price <= 3.0)].sort_values("price").drop_duplicates("game_id")   # shortest per game
    pool = x.to_dict("records")
    def rec(type_, combo):
        hit = all(l["hit"] for l in combo); pay = float(np.prod([l["price"] for l in combo]))
        rows.append(dict(week=wk, type=type_, legs=" + ".join(l["player"] for l in combo), price=pay, model=float(np.prod([l["model_p"] for l in combo])),
                         dk=float(np.prod([1 / l["price"] for l in combo])), hit=hit, pnl=pay - 1 if hit else -1.0))
    for l in pool: rec("singles", [l])
    for i in range(0, len(pool) - 1, 2):
        if i // 2 < 4: rec("all games, 2 legs", pool[i:i + 2])
    for i in range(0, len(pool) - 2, 3):
        if i // 3 < 3: rec("all games, 3 legs", pool[i:i + 3])
    tp = [l for l in pool if l["top5"]]
    for c in itertools.combinations(tp, 2): rec("top-5 totals, 2 legs", c)
    for c in itertools.combinations(tp, 3): rec("top-5 totals, 3 legs", c)
r = pd.DataFrame(rows)
print(r.groupby(["type"]).agg(n=("hit", "size"), won=("hit", "sum"), avg_price=("price", "mean"), dk_exp=("dk", "sum"), model_exp=("model", "sum"), units=("pnl", "sum")).round(2).to_string())
print(); print(r.groupby(["type", "week"]).agg(n=("hit", "size"), won=("hit", "sum"), units=("pnl", "sum")).round(2).to_string())
print(); print(r[r.type != "singles"].sort_values(["type", "week"])[["week", "type", "legs", "price", "hit", "pnl"]].round(2).to_string(index=False))
