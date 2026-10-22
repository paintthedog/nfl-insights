"""Walk-forward test: position-specific TD models vs the current pooled model."""
import numpy as np, pandas as pd, statsmodels.api as sm
import td_model as T

pbp, sched, players = T.load()
pg, team, ctx = T.build_tables(pbp, sched, players)
d = T.features(pg, team, ctx)
d = d[(d.pr_n >= 1) & d.implied.notna() & d.rush_share.notna()].copy()
d["grp"] = np.select([d.pos == "QB", d.pos.isin(["RB", "FB"]), d.pos == "WR", d.pos == "TE"], ["QB", "RB", "WR", "TE"], "OTH")
d = d[d.grp != "OTH"]
d["hit"] = ((d.rtd + d.ptd) >= 1).astype(int)

def fit_pred(tr, te, tmm, rush_x, rec_x, split):
    """split: dict side -> list of group-lists that get their own model (others pooled)."""
    out = {}
    for side, xs, y, share in [("rush", rush_x, "rtd", "rush_share"), ("rec", rec_x, "ptd", None)]:
        tr_off = np.log(T.team_exp_td(tmm, tr.implied) * (tr.rush_share if side == "rush" else 1 - tr.rush_share))
        te_off = np.log(T.team_exp_td(tmm, te.implied) * (te.rush_share if side == "rush" else 1 - te.rush_share))
        lam = np.zeros(len(te))
        groups = split.get(side) or [["QB", "RB", "WR", "TE"]]
        covered = set(sum(groups, []))
        if covered != {"QB", "RB", "WR", "TE"}:
            groups = groups + [[g for g in ["QB", "RB", "WR", "TE"] if g not in covered]]
        for gl in groups:
            mtr, mte = tr.grp.isin(gl).to_numpy(), te.grp.isin(gl).to_numpy()
            if mte.sum() == 0: continue
            cols = [c for c in xs if tr.loc[mtr, c].std() > 0]      # drop constant dummies within group
            m = sm.GLM(tr.loc[mtr, y], sm.add_constant(tr.loc[mtr, cols]), family=sm.families.Poisson(), offset=tr_off[mtr]).fit()
            lam[mte] = m.predict(sm.add_constant(te.loc[mte, cols], has_constant="add"), offset=te_off[mte])
        out[side] = lam
    return out["rush"], out["rec"]

POS = ["is_qb", "pos_TE", "pos_RB"]
V = {
    "current (pooled)":                 (["l_car_sh", "l_i10_sh", "l_rtd_sh", "is_qb"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"], {}),
    "current (pooled) + QB fix":        (["l_car_sh", "l_i10_sh", "l_rtd_sh", "is_qb"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"], {}),
    "QB rushing split out only + QB fix": (["l_car_sh", "l_i10_sh", "l_rtd_sh", "pos_RB"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"],
                                         {"rush": [["QB"]]}),
    "split by position":                (["l_car_sh", "l_i10_sh", "l_rtd_sh"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh"],
                                         {"rush": [["QB"], ["RB"]], "rec": [["WR"], ["TE"], ["RB"]]}),
    "split + inside-5 / end-zone":      (["l_car_sh", "l_i10_sh", "l_i5_sh", "l_rtd_sh"], ["l_tgt_sh", "l_rz_sh", "l_ez_sh", "l_ptd_sh"],
                                         {"rush": [["QB"], ["RB"]], "rec": [["WR"], ["TE"], ["RB"]]}),
    "pooled + inside-5 / end-zone":     (["l_car_sh", "l_i10_sh", "l_i5_sh", "l_rtd_sh", "is_qb"], ["l_tgt_sh", "l_rz_sh", "l_ez_sh", "l_ptd_sh", "pos_TE", "pos_RB"], {}),
    "QB rushing split out only":        (["l_car_sh", "l_i10_sh", "l_rtd_sh", "pos_RB"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"],
                                         {"rush": [["QB"]]}),
    "QB split out + inside-5 for QB":   (["l_car_sh", "l_i10_sh", "l_i5_sh", "l_rtd_sh", "pos_RB"], ["l_tgt_sh", "l_rz_sh", "l_ptd_sh", "pos_TE", "pos_RB"],
                                         {"rush": [["QB"]]}),
    "split, WR+TE share one model":     (["l_car_sh", "l_i10_sh", "l_i5_sh", "l_rtd_sh"], ["l_tgt_sh", "l_rz_sh", "l_ez_sh", "l_ptd_sh", "pos_TE"],
                                         {"rush": [["QB"], ["RB"]], "rec": [["WR", "TE"], ["RB"]]}),
}
rows, preds = [], {}
for name, (rx, cx, split) in V.items():
    ps = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s], d[d.season == s].copy()
        tmm = T.fit_team(team[team.season < s])
        lr, lc = fit_pred(tr, te, tmm, rx, cx, split)
        if name.endswith("+ QB fix"):      # production's correction: scale QB rates so training P(1+) matches actual
            tlr, tlc = fit_pred(tr, tr, tmm, rx, cx, split)
            q = (tr.grp == "QB").to_numpy(); hit = tr.hit.to_numpy()[q]; lam_tr = (tlr + tlc)[q]
            grid = np.linspace(0.5, 1.2, 141)
            sc = grid[np.argmin([abs((1 - np.exp(-c * lam_tr)).mean() - hit.mean()) for c in grid])]
            qt = (te.grp == "QB").to_numpy(); lr = np.where(qt, lr * sc, lr); lc = np.where(qt, lc * sc, lc)
        te["p"] = 1 - np.exp(-(lr + lc)); ps.append(te[["game_id", "pid", "grp", "hit", "p"]])
    p = pd.concat(ps); preds[name] = p
    row = dict(variant=name, all=np.mean((p.p - p.hit) ** 2))
    for g in ["QB", "RB", "WR", "TE"]:
        r = p[p.grp == g]; row[g] = np.mean((r.p - r.hit) ** 2)
        row[g + " cal"] = f"{r.p.mean():.3f}/{r.hit.mean():.3f}"
    rows.append(row); print("done:", name, flush=True)
r = pd.DataFrame(rows).set_index("variant")
num = ["all", "QB", "RB", "WR", "TE"]
pd.set_option("display.width", 220)
print("\nBrier score (lower = better)\n", r[num].round(5).to_string())
print("\n% vs current (negative = better)\n", ((r[num] / r.loc["current (pooled) + QB fix", num] - 1) * 100).round(2).to_string())
print("\nMean predicted / actual TD rate\n", r[[g + " cal" for g in ["QB", "RB", "WR", "TE"]]].to_string())
c = pd.read_pickle if False else None
const = np.mean((preds["current (pooled)"].hit.mean() - preds["current (pooled)"].hit) ** 2)
print("\nSkill vs constant:", {k: f"{1 - np.mean((v.p - v.hit) ** 2) / const:.2%}" for k, v in preds.items()})
