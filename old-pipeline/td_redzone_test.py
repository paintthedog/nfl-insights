"""Should goal-line carries and red-zone targets count for more?"""
import numpy as np, pandas as pd, statsmodels.api as sm
import td_model as T

pbp, sched, players = T.load()
pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx)
# long-memory scoring-zone shares: decayed COUNTS / decayed team counts (more games, less noise)
long = T.decayed_mean(pg.copy(), "pid", ["i10", "tm_i10", "i5", "tm_i5", "rz", "tm_rz", "ez", "tm_ez", "car", "tm_car", "tgt", "tm_tgt"], 0.95)
for a in ["i10", "i5", "rz", "ez", "car", "tgt"]:
    long[f"L_{a}"] = np.log((long[f"pr_{a}"] / long[f"pr_tm_{a}"]).fillna(0).clip(0, 1) + T.EPS)
d = d.merge(long[["game_id", "pid"] + [c for c in long.columns if c.startswith("L_")]], on=["game_id", "pid"], how="left")
d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()].copy()
d["grp"] = np.select([d.pos == "QB", d.pos.isin(["RB", "FB"]), d.pos == "WR", d.pos == "TE"], ["QB", "RB", "WR", "TE"], "OTH")
d = d[d.grp != "OTH"]; d["hit"] = ((d.rtd + d.ptd) >= 1).astype(int)

def run(rush_x, rec_x, boost=1.0, qb_x=None):
    qb_x = qb_x or [c for c in rush_x if c != "pos_RB"]
    ps, coefs = [], None
    for s in range(2021, 2027):
        tr, te = d[d.season < s], d[d.season == s].copy()
        tmm = T.fit_team(team[team.season < s])
        def off(df, side): return np.log(T.team_exp_td(tmm, df.implied) * (df.rush_share if side == "r" else 1 - df.rush_share))
        lam = {}
        for name, mask_tr, mask_te, xs, y, side in [("r", tr.grp != "QB", te.grp != "QB", rush_x, "rtd", "r"),
                                                     ("q", tr.grp == "QB", te.grp == "QB", qb_x, "rtd", "r"),
                                                     ("c", tr.grp.notna(), te.grp.notna(), rec_x, "ptd", "c")]:
            mt, me = mask_tr.to_numpy(), mask_te.to_numpy()
            m = sm.GLM(tr[y][mt], sm.add_constant(tr.loc[mt, xs]), family=sm.families.Poisson(), offset=off(tr, side)[mt]).fit()
            p = m.params.copy()
            for c in p.index:   # force scoring-zone terms bigger than the data chose
                if any(k in c for k in ["i10", "i5", "rz", "ez"]): p[c] *= boost
            X = sm.add_constant(te.loc[me, xs], has_constant="add")
            v = np.zeros(len(te)); v[me] = np.exp(X @ p + off(te, side)[me]); lam[name] = v
            if s == 2026 and name == "r": coefs = {k: round(v_, 2) for k, v_ in m.params.items() if k != "const"}
            if s == 2026 and name == "c": coefs.update({k: round(v_, 2) for k, v_ in m.params.items() if k != "const"})
        te["p"] = 1 - np.exp(-(lam["r"] + lam["q"] + lam["c"])); ps.append(te[["grp", "hit", "p"]])
    p = pd.concat(ps)
    out = {"all": np.mean((p.p - p.hit) ** 2)}
    for g in ["QB", "RB", "WR", "TE"]:
        r = p[p.grp == g]; out[g] = np.mean((r.p - r.hit) ** 2)
    return out, coefs

R = ["l_car_sh", "l_i10_sh", "l_rtd_sh", "pos_RB"]; C = ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"]
V = {
    "current":                                   (R, C, 1.0),
    "scoring-zone terms forced 2x":              (R, C, 2.0),
    "scoring-zone terms forced 3x":              (R, C, 3.0),
    "goal-line / red-zone ONLY (no overall share)": (["l_i10_sh", "l_rtd_sh", "pos_RB"], ["l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"], 1.0),
    "longer memory for scoring-zone usage":      (["l_car_sh", "L_i10", "l_rtd_sh", "pos_RB"], ["l_tgt_sh", "L_rz", "l_ptd_sh", "pos_TE", "pos_RB"], 1.0),
    "longer memory + inside-5 + end-zone":       (["l_car_sh", "L_i10", "L_i5", "l_rtd_sh", "pos_RB"], ["l_tgt_sh", "L_rz", "L_ez", "l_ptd_sh", "pos_TE", "pos_RB"], 1.0),
    "longer memory for everything":              (["L_car", "L_i10", "l_rtd_sh", "pos_RB"], ["L_tgt", "L_rz", "l_ptd_sh", "pos_TE", "pos_RB"], 1.0),
}
rows = []
for name, (rx, cx, b) in V.items():
    out, coefs = run(rx, cx, b); rows.append(dict(variant=name, **out))
    if name in ("current", "goal-line / red-zone ONLY (no overall share)", "longer memory + inside-5 + end-zone"):
        print(f"{name}: learned weights {coefs}")
r = pd.DataFrame(rows).set_index("variant")
pd.set_option("display.width", 200)
print("\nBrier (lower = better)\n", r.round(5).to_string())
print("\n% vs current (negative = better)\n", ((r / r.loc["current"] - 1) * 100).round(2).to_string())
