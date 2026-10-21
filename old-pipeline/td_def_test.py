"""Should the TD model use the opponent's rushing / receiving TDs allowed directly?"""
import numpy as np, pandas as pd, statsmodels.api as sm
import td_model as T, proj_model as P

pbp, sched, players = T.load(); pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx)
team = team.copy(); team["one"] = 1.0
lg_r, lg_p = team.tm_rtd.mean(), team.tm_ptd.mean()

def def_feats(carry, K, decay=0.90):
    t = team.rename(columns={"opp": "defteam"})[["game_id", "defteam", "season", "gameday", "tm_rtd", "tm_ptd", "one"]]
    t = P.decayed_carry(t, "defteam", ["tm_rtd", "tm_ptd", "one"], decay, season_carry=carry, mode="sum")
    t["l_def_rtd"] = np.log(((t.ds_tm_rtd + K * lg_r) / (t.ds_one + K)) / lg_r)
    t["l_def_ptd"] = np.log(((t.ds_tm_ptd + K * lg_p) / (t.ds_one + K)) / lg_p)
    return t[["game_id", "defteam", "l_def_rtd", "l_def_ptd"]]

opp = ctx[["game_id", "team", "opp"]].rename(columns={"team": "posteam"})
base = d.merge(opp, on=["game_id", "posteam"], how="left")
base = base[(base.pr_n >= 1) & base.implied.notna() & base.rush_share.notna()]
base["hit"] = ((base.rtd + base.ptd) >= 1).astype(int)

def evaluate(x, add_r, add_c):
    rx, cx, qx = T.RUSH_X + add_r, T.REC_X + add_c, T.QB_RUSH_X + add_r
    ps = []
    for s in range(2021, 2027):
        tr, te = x[x.season < s], x[x.season == s].copy()
        tmm = T.fit_team(team[team.season < s])
        off = lambda df, side: np.log(T.team_exp_td(tmm, df.implied) * (df.rush_share if side == "r" else 1 - df.rush_share))
        lam = np.zeros(len(te))
        for mtr, mte, xs, y, side in [(tr.is_qb == 0, te.is_qb == 0, rx, "rtd", "r"), (tr.is_qb == 1, te.is_qb == 1, qx, "rtd", "r"),
                                      (tr.pos.notna(), te.pos.notna(), cx, "ptd", "c")]:
            mt, me = mtr.to_numpy(), mte.to_numpy()
            m = sm.GLM(tr[y][mt], sm.add_constant(tr.loc[mt, xs]), family=sm.families.Poisson(), offset=off(tr, side)[mt]).fit()
            p = m.params.copy()
            if side == "r" and "l_i10_sh" in p.index and xs is rx: p["l_i10_sh"] *= T.GL_BOOST
            lam[me] += np.exp(sm.add_constant(te.loc[me, xs], has_constant="add") @ p + off(te, side)[me])
        te["p"] = 1 - np.exp(-lam); ps.append(te)
    a = pd.concat(ps)
    out = {"all": np.mean((a.p - a.hit) ** 2), "wk 1-4": np.mean(((a.p - a.hit) ** 2)[a.week <= 4])}
    for g in ["QB", "RB", "WR", "TE"]: out[g] = np.mean(((a.p - a.hit) ** 2)[a.pos == g])
    return out, (m.params.get("l_def_ptd") if add_c else None)

res = {"current (no defense term)": evaluate(base, [], [])[0]}
for carry in [1.0, 0.4]:
    for K in [4, 10]:
        x = base.merge(def_feats(carry, K).rename(columns={"defteam": "opp"}), on=["game_id", "opp"], how="left").fillna({"l_def_rtd": 0, "l_def_ptd": 0})
        out, coef = evaluate(x, ["l_def_rtd"], ["l_def_ptd"])
        res[f"opp TDs allowed (last season x{carry}, averaging {K} games)"] = out
        print(f"carry {carry} K {K}: learned rec-TD-allowed effect {coef:.2f}", flush=True)
r = pd.DataFrame(res).T
print(((r / r.iloc[0] - 1) * 100).round(2).to_string())
