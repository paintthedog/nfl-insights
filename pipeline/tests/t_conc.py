"""Concentration: does sharpening the split of each team's expected TDs toward its top options fix the stars' under-prediction?
lam_i' = sum(lam) * lam_i^a / sum(lam^a) within a team-game; a chosen on earlier seasons only (walk-forward)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, td_model as T
T.RS_K = 30
pbp, sched, players = T.load(); pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx); d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()]; d["hit"] = ((d.rtd + d.ptd) >= 1).astype(int)
P = []
for s in range(2021, 2027):
    tmm = T.fit_team(team[team.season < s]); mr, mp = T.fit_players(d[d.season < s], tmm); P.append(T.predict(d[d.season == s], tmm, mr, mp))
p = pd.concat(P); p = p[p.lam > 0].copy()
def sharpen(x, a, by=("game_id", "posteam")):
    la = x.lam ** a; return 1 - np.exp(-x.lam.groupby([x[c] for c in by]).transform("sum") * la / la.groupby([x[c] for c in by]).transform("sum"))
ll = lambda q, y: -np.mean(y * np.log(q.clip(1e-4, 1 - 1e-4)) + (1 - y) * np.log((1 - q).clip(1e-4)))
print("check base p vs 1-exp(-lam):", np.abs(p.p_td - (1 - np.exp(-p.lam))).max().round(4))
grid = np.arange(0.9, 1.41, 0.05); out = []
for s in range(2022, 2027):
    tr = p[p.season < s]; a = grid[np.argmin([ll(sharpen(tr, g), tr.hit) for g in grid])]
    te = p[p.season == s].copy(); te["q"] = sharpen(te, a); te["a"] = a; out.append(te)
o = pd.concat(out); y = o.hit
print("chosen a by season:", o.groupby("season").a.first().to_dict())
print(f"logloss base {ll(o.p_td, y):.5f} sharpened {ll(o.q, y):.5f} | brier {np.mean((o.p_td-y)**2):.5f} -> {np.mean((o.q-y)**2):.5f}")
o["rk"] = o.groupby(["game_id", "posteam"]).p_td.rank(ascending=False, method="first")
print(o.groupby(o.rk.clip(upper=5)).agg(n=("hit", "size"), base=("p_td", "mean"), new=("q", "mean"), actual=("hit", "mean")).round(3))
rb = o[(o.pos == "RB") & (o.pr_car_sh >= .45)]; print("lead RBs base", rb.p_td.mean().round(3), "new", rb.q.mean().round(3), "act", rb.hit.mean().round(3))
o["band"] = pd.cut(o.q, [0, .05, .1, .2, .3, .4, .5, .7, 1]); print(o.groupby("band", observed=True).agg(n=("hit", "size"), pred=("q", "mean"), actual=("hit", "mean")).round(3))
# logit recalibration q = sigmoid(a + b*logit(p) + c*top), fitted on earlier seasons
import statsmodels.api as sm
lg = lambda q: np.log(q.clip(1e-4, 1 - 1e-4) / (1 - q.clip(1e-4, 1 - 1e-4)))
p["rk1"] = (p.groupby(["game_id", "posteam"]).p_td.rank(ascending=False, method="first") == 1).astype(int)
for feats in (["lp"], ["lp", "rk1"]):
    out = []
    for s in range(2022, 2027):
        tr = p[p.season < s]; te = p[p.season == s].copy()
        X = lambda z: sm.add_constant(pd.DataFrame({"lp": lg(z.p_td), "rk1": z.rk1})[feats], has_constant="add")
        m = sm.GLM(tr.hit, X(tr), family=sm.families.Binomial()).fit(); te["q"] = m.predict(X(te)); out.append(te)
    o = pd.concat(out); y = o.hit
    print(feats, m.params.round(3).to_dict(), f"logloss {ll(o.p_td, y):.5f} -> {ll(o.q, y):.5f}  brier {np.mean((o.p_td-y)**2):.5f} -> {np.mean((o.q-y)**2):.5f}")
    o["rk"] = o.groupby(["game_id", "posteam"]).p_td.rank(ascending=False, method="first")
    print(o.groupby(o.rk.clip(upper=4)).agg(base=("p_td", "mean"), new=("q", "mean"), actual=("hit", "mean")).round(3).T.to_string())
    rb = o[(o.pos == "RB") & (o.pr_car_sh >= .45)]; print("lead RBs base", rb.p_td.mean().round(3), "new", rb.q.mean().round(3), "act", rb.hit.mean().round(3))
