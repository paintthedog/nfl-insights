"""Season-to-date TDs allowed BY POSITION as a TD-model input. Walk-forward 2021-2026."""
import numpy as np, pandas as pd, statsmodels.api as sm
import td_model as T, proj_model as P

pbp, sched, players = T.load(); pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx)
grp = lambda pos: np.select([pos == "QB", pos.isin(["RB", "FB"]), pos == "WR", pos == "TE"], ["QB", "RB", "WR", "TE"], "OTH")
opp = ctx[["game_id", "team", "opp"]].rename(columns={"team": "posteam"})

# TDs scored by each position group against each defense, per game (QB = rushing TDs only)
g = pg.merge(opp, on=["game_id", "posteam"]).assign(grp=lambda x: grp(x.pos))
g["td"] = np.where(g.grp == "QB", g.rtd, g.rtd + g.ptd)
alw = g[g.grp != "OTH"].groupby(["game_id", "opp", "grp", "gameday", "season"], as_index=False).td.sum()
# make sure every defense-game has a row for every position (zero TDs allowed counts)
dg = alw[["game_id", "opp", "gameday", "season"]].drop_duplicates()
full = dg.merge(pd.DataFrame({"grp": ["QB", "RB", "WR", "TE"]}), how="cross").merge(alw, how="left").fillna({"td": 0})
full["one"] = 1.0; full["key"] = full.opp + "|" + full.grp
lg = full.groupby("grp").td.mean()

def feats(decay, carry, K):
    t = P.decayed_carry(full.rename(columns={"key": "k"}).assign(posteam=""), "k", ["td", "one"], decay, season_carry=carry, mode="sum")
    t["l_dvp_td"] = np.log(((t.ds_td + K * t.grp.map(lg)) / (t.ds_one + K)) / t.grp.map(lg))
    return t[["game_id", "opp", "grp", "l_dvp_td"]]

base = d.merge(opp, on=["game_id", "posteam"], how="left").assign(grp=lambda x: grp(x.pos))
base = base[(base.pr_n >= 1) & base.implied.notna() & base.rush_share.notna() & (base.grp != "OTH")]
base["hit"] = ((base.rtd + base.ptd) >= 1).astype(int)

def evaluate(x, extra):
    rx, cx, qx = T.RUSH_X + extra, T.REC_X + extra, T.QB_RUSH_X + extra
    ps, coefs = [], {}
    for s in range(2021, 2027):
        tr, te = x[x.season < s], x[x.season == s].copy()
        tmm = T.fit_team(team[team.season < s])
        off = lambda df, side: np.log(T.team_exp_td(tmm, df.implied) * (df.rush_share if side == "r" else 1 - df.rush_share))
        lam = np.zeros(len(te))
        for nm, mtr, mte, xs, y, side in [("rush", tr.is_qb == 0, te.is_qb == 0, rx, "rtd", "r"), ("qb", tr.is_qb == 1, te.is_qb == 1, qx, "rtd", "r"),
                                          ("rec", tr.pos.notna(), te.pos.notna(), cx, "ptd", "c")]:
            mt, me = mtr.to_numpy(), mte.to_numpy()
            m = sm.GLM(tr[y][mt], sm.add_constant(tr.loc[mt, xs]), family=sm.families.Poisson(), offset=off(tr, side)[mt]).fit()
            p = m.params.copy()
            if nm == "rush": p["l_i10_sh"] *= T.GL_BOOST
            lam[me] += np.exp(sm.add_constant(te.loc[me, xs], has_constant="add") @ p + off(te, side)[me])
            if s == 2026 and extra: coefs[nm] = round(float(m.params["l_dvp_td"]), 2)
        te["p"] = 1 - np.exp(-lam); ps.append(te)
    a = pd.concat(ps)
    out = {"all": np.mean((a.p - a.hit) ** 2), "wk 1-4": np.mean(((a.p - a.hit) ** 2)[a.week <= 4]), "wk 5+": np.mean(((a.p - a.hit) ** 2)[a.week >= 5])}
    for gname in ["QB", "RB", "WR", "TE"]: out[gname] = np.mean(((a.p - a.hit) ** 2)[a.grp == gname])
    return out, coefs

res = {"current (no position defense term)": evaluate(base, [])[0]}
for label, (decay, carry, K) in {"season-to-date only, light averaging (3 games)": (1.0, 0.0, 3),
                                 "season-to-date only, heavier averaging (8 games)": (1.0, 0.0, 8),
                                 "season-to-date + last season x0.3 (8 games)": (1.0, 0.3, 8),
                                 "recent-weighted + last season x0.3 (8 games)": (0.9, 0.3, 8)}.items():
    x = base.merge(feats(decay, carry, K), on=["game_id", "opp", "grp"], how="left").fillna({"l_dvp_td": 0})
    out, coefs = evaluate(x, ["l_dvp_td"]); res[label] = out
    print(f"{label}: learned effect {coefs}", flush=True)
r = pd.DataFrame(res).T
print("\n% change in Brier vs current (negative = better)\n", ((r / r.iloc[0] - 1) * 100).round(2).to_string())

# ---- what the data says the effect is worth, in plain terms
x = base.merge(feats(0.9, 0.3, 8), on=["game_id", "opp", "grp"], how="left").fillna({"l_dvp_td": 0})
r = x[x.grp == "RB"]
lo, hi = np.quantile(np.exp(r.l_dvp_td), [0.1, 0.9])
print(f"\nRB: toughest-10% vs softest-10% run defenses by TDs allowed = {lo:.2f}x to {hi:.2f}x league avg;")
print(f"    learned rushing-TD effect 0.24 -> swing in a back's TD rate: {(hi/lo)**0.24*100-100:+.0f}% (soft vs tough)")
# how predictive are early-season TDs allowed by position of the rest of the season?
f26 = full[full.season <= 2025]
early = f26[f26.game_id.str[5:7].astype(int) <= 4].groupby(["season", "opp", "grp"]).td.mean().rename("early")
late = f26[f26.game_id.str[5:7].astype(int) >= 5].groupby(["season", "opp", "grp"]).td.mean().rename("late")
c = early.to_frame().join(late).dropna().reset_index()
print("\nCorrelation: TDs allowed per game to a position in weeks 1-4 vs weeks 5+ (same season, 2016-2025):")
print(c.groupby("grp").apply(lambda z: round(z.early.corr(z.late), 2)).to_dict())
