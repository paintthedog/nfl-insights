"""Recalibrate anytime-TD probabilities: logit(y) ~ a + b*logit(p). Leave-one-season-out evaluation, 2021-2026."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm; sys.path.insert(0, ".")
import td_model as T
tpbp, tsched, tplayers = T.load(); tpg, tteam, tctx = T.build_tables(tpbp, tsched, tplayers)
tf = T.features(tpg, tteam, tctx); tf = tf[(tf.pr_n >= 1) & tf.implied.notna() & tf.rush_share.notna()]
o = []
for s in range(2021, 2027):
    tmm = T.fit_team(tteam[tteam.season < s]); mr, mp = T.fit_players(tf[tf.season < s], tmm)
    o.append(T.predict(tf[tf.season == s].copy(), tmm, mr, mp))
o = pd.concat(o); o["y"] = ((o.rtd.fillna(0) + o.ptd.fillna(0)) > 0).astype(int)
o = o[o.pos != "QB"]; o["p"] = (1 - np.exp(-o.lam)).clip(1e-4, 1 - 1e-4); o["lp"] = np.log(o.p / (1 - o.p))
ll = lambda y, p: -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
res = []
for s in range(2021, 2027):
    tr, te = o[o.season != s], o[o.season == s]
    m = sm.Logit(tr.y, sm.add_constant(tr[["lp"]])).fit(disp=0)
    pc = m.predict(sm.add_constant(te[["lp"]], has_constant="add"))
    mid = te.p.between(.26, .40)
    res.append(dict(season=s, ll_raw=ll(te.y, te.p), ll_cal=ll(te.y, pc), mid_raw=te.p[mid].mean(), mid_cal=pc[mid].mean(), mid_act=te.y[mid].mean()))
r = pd.DataFrame(res); print(r.round(4).to_string(index=False))
print(f"\nlog-loss improved in {(r.ll_cal < r.ll_raw).sum()}/6 held-out seasons; avg {r.ll_raw.mean():.5f} -> {r.ll_cal.mean():.5f}")
m = sm.Logit(o.y, sm.add_constant(o[["lp"]])).fit(disp=0); a, b = float(m.params.iloc[0]), float(m.params.iloc[1])
print(f"final recalibration: a={a:.4f} b={b:.4f}")
for p in [.08, .15, .22, .30, .36, .45, .55]:
    l = np.log(p / (1 - p)); print(f"  model {p:.0%} -> {1/(1+np.exp(-(a + b*l))):.1%}")
json.dump(dict(a=a, b=b, fit="2021-2026 walk-forward non-QB"), open("data/td_recal.json", "w"))
