import sys, re, numpy as np, pandas as pd; sys.path.insert(0, ".")
pbp = pd.read_parquet("data/pbp_2026.parquet"); pbp = pbp[pbp.two_point_attempt.ne(1)]
done = set(pbp.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
NM = {}; [NM.setdefault(norm(n), []).append(g) for g, n in zip(pl.gsis_id, pl.display_name)]
touched = set(pbp.passer_player_id.dropna()) | set(pbp.rusher_player_id.dropna()) | set(pbp.receiver_player_id.dropna()) | set(pbp.td_player_id.dropna())
def pid(n, gid):
    c = NM.get(norm(n), [])
    if len(c) == 1: return c[0]
    gm = pbp[pbp.game_id == gid]; ids = set(gm.passer_player_id.dropna()) | set(gm.rusher_player_id.dropna()) | set(gm.receiver_player_id.dropna())
    h = [x for x in c if x in ids]; return h[0] if h else (c[0] if c else None)
tds = pbp[pbp.touchdown.eq(1) & pbp.td_player_id.notna()].groupby(["game_id", "td_player_id"]).size()
a = pd.read_csv("props_log.csv"); a = a[a.model_p.notna()].sort_values("pulled_at").drop_duplicates(["week", "player", "game_id"], keep="last")
a = a[a.game_id.isin(done)].copy(); a["pid"] = [pid(n, g) for n, g in zip(a.player, a.game_id)]; a = a[a.pid.isin(touched)]
a["hit"] = [int(tds.get((g, p), 0) > 0) for g, p in zip(a.game_id, a.pid)]
a["ev"] = a.model_p * a.price - 1; a["pnl"] = np.where(a.hit == 1, a.price - 1, -1.0)
b = a[a.price < 2.5].copy()
print(f"ATD priced under 2.5, games final, weeks {sorted(b.week.unique())}: n={len(b)}  scorers {b.hit.sum()}  model expected {b.model_p.sum():.1f}  DK implied (raw) {(1/b.price).sum():.1f}")
print(f"  betting ALL of them: {b.pnl.sum():+.2f}u ({b.pnl.mean():+.1%}/bet) — the book's cut on this band")
b["grp"] = pd.cut(b.ev, [-1, -0.10, -0.03, 0.03, 0.10, 1], labels=["model dislikes (<-10%)", "slight dislike", "neutral (±3%)", "slight like (+3-10%)", "model likes (>+10%)"])
t = b.groupby("grp", observed=True).agg(n=("hit", "size"), scored=("hit", "sum"), model_exp=("model_p", "sum"), dk_exp=("price", lambda x: (1/x).sum()), units=("pnl", "sum"))
t["per_bet"] = t.units / t.n
print(t.round(2).to_string())
lk = b[b.ev >= 0.03]
print(f"\n'Model likes it' (edge +3% or more): {len(lk)} bets, {lk.hit.sum()} scored, {lk.pnl.sum():+.2f}u  | by week: " + ", ".join(f"wk{w}: {x.hit.sum()}/{len(x)} {x.pnl.sum():+.1f}u" for w, x in lk.groupby("week")))
print("\nmodel-liked bets in band:")
for _, r in lk.sort_values("ev", ascending=False).iterrows(): print(f"  wk{r.week} {r.player:<24} @{r.price:<5} model {r.model_p:.0%} edge {r.ev:+.0%}  {'SCORED' if r.hit else ''}")
print("\nby price:"); b["pb"] = pd.cut(b.price, [1, 1.6, 2.0, 2.5])
print(b.groupby("pb", observed=True).agg(n=("hit", "size"), scored=("hit", "sum"), model=("model_p", "sum"), dk=("price", lambda s: (1 / s).sum())).round(1))
