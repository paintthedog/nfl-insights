"""Pressure mismatch: does the opponent's pressure-creation rate (sacks + QB hits per dropback, decayed, shrunk) and the
offense's own pressure-allowed rate improve QB passing projections beyond what's already in the calibration
(raw projection, defense yards ratio, wind/cold, missing OL starters)? Walk-forward 2021-2026, QB starts only."""
import sys, glob; sys.path.insert(0, ".")
import numpy as np, pandas as pd, statsmodels.api as sm
import proj_model as P
pbp = P.load(); sched = pd.read_parquet("data/sched.parquet")
players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
d = P.build(pbp, sched, players); pm = P.position_means(d); d = P.efficiency(d, pm)
# pressure rates per game from the full pbp (need qb_hit, not in P.load's column set)
full = pd.concat([pd.read_parquet(f, columns=["game_id","season","season_type","posteam","defteam","qb_dropback","sack","qb_hit"]) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
full = full[(full.season_type == "REG") & (full.qb_dropback == 1)]
full["prs"] = ((full.sack + full.qb_hit) > 0).astype(float)
gd = full.groupby(["game_id","defteam"]).agg(db=("prs","size"), pr=("prs","sum")).reset_index()
go = full.groupby(["game_id","posteam"]).agg(db=("prs","size"), pr=("prs","sum")).reset_index()
sc = sched[["game_id","gameday"]]; K = 300; LG = full.prs.mean()
def rate(df, key):
    df = df.merge(sc, on="game_id"); df["gameday"] = pd.to_datetime(df.gameday); df = df.sort_values("gameday")
    out = []
    for t, g in df.groupby(key):
        cdb = cpr = 0.0
        for r in g.itertuples():
            out.append((r.game_id, t, (cpr + K*LG) / (cdb + K)))
            cdb = cdb*0.95 + r.db; cpr = cpr*0.95 + r.pr
    return pd.DataFrame(out, columns=["game_id", key, "r"])
rd = rate(gd, "defteam").rename(columns={"r": "opp_prs"})   # defense's pressure created entering this game
ro = rate(go, "posteam").rename(columns={"r": "own_prs"})   # offense's pressure allowed entering this game
d = d.merge(rd.rename(columns={"defteam": "opp"}), on=["game_id","opp"], how="left")
d = d.merge(ro, on=["game_id","posteam"], how="left")
d["opp_prs"] = (d.opp_prs.fillna(LG) - LG) * 10; d["own_prs"] = (d.own_prs.fillna(LG) - LG) * 10
q = d[(d.pos == "QB") & d.qb_start & (d.pr_n >= 1) & d.implied.notna()].copy()
STATS = ["pass_yds", "cmp", "sacks", "ints", "pass_1d"]
def run(extra):
    res = {}
    for s in range(2021, 2027):
        tr0 = d[d.season < s].copy(); te0 = d[d.season == s].copy()
        teams = tr0.drop_duplicates(["game_id","posteam"]); va, vc = P.fit_volume(teams)
        tr0 = P.raw_projection(tr0, va, vc); te0 = P.raw_projection(te0, va, vc)
        tr = tr0[(tr0.pos=="QB") & tr0.qb_start & (tr0.pr_n>=1)]; te = te0[(te0.pos=="QB") & te0.qb_start & (te0.pr_n>=1) & te0.implied.notna()]
        for st in STATS:
            raw = {"pass_yds":"r_pass_yds","cmp":"r_cmp","sacks":"r_sacks","ints":"r_ints","pass_1d":"r_pass_1d"}[st]
            rows = tr[tr[raw] > 0.05]
            X = pd.DataFrame({"lraw": np.log(rows[raw])})
            if st in ("pass_yds","pass_1d"): X["ldef"] = np.log(rows.def_pass_ratio.fillna(1.0))
            X["wind_eff"] = rows.wind_eff.fillna(0); X["cold"] = rows.cold.fillna(0); X["ol_missing"] = rows.ol_missing.fillna(0.9)
            if extra: X["opp_prs"] = rows.opp_prs; X["own_prs"] = rows.own_prs
            m = sm.GLM(rows[st], sm.add_constant(X), family=sm.families.Poisson()).fit()
            sel = te[te[raw] > 0.05]
            Xt = pd.DataFrame({"lraw": np.log(sel[raw])})
            if st in ("pass_yds","pass_1d"): Xt["ldef"] = np.log(sel.def_pass_ratio.fillna(1.0))
            Xt["wind_eff"] = sel.wind_eff.fillna(0); Xt["cold"] = sel.cold.fillna(0); Xt["ol_missing"] = sel.ol_missing.fillna(0.9)
            if extra: Xt["opp_prs"] = sel.opp_prs; Xt["own_prs"] = sel.own_prs
            pr = m.predict(sm.add_constant(Xt, has_constant="add"))
            res.setdefault(st, []).append(pd.DataFrame({"a": sel[st].values, "p": pr.values, "opp_prs": sel.opp_prs.values, "coef": (m.params.get("opp_prs", np.nan))}))
    return {st: pd.concat(v) for st, v in res.items()}
A = run(False); B = run(True)
for st in STATS:
    a, b = A[st], B[st]
    hi = b.opp_prs.abs() >= b.opp_prs.abs().quantile(0.8)
    print(f"{st:9} MAE {np.abs(a.a-a.p).mean():8.3f} -> {np.abs(b.a-b.p).mean():8.3f} | extreme-pressure games {np.abs(a.a-a.p)[hi.values].mean():8.3f} -> {np.abs(b.a-b.p)[hi].mean():8.3f} | opp_prs coef {b.coef.iloc[-1]:+.3f}", flush=True)

# ---- team level: does the pressure mismatch (my defense's rate + your line's allowed rate) move margins or totals
# beyond the EPA ratings? Walk-forward on the lines backtest frame.
import lines_model as L
pbp2, sched2 = L.load()
played = sched2[sched2.result.notna()]
g = L.build(pbp2, played)
g = g.merge(rd.rename(columns={"defteam": "home_team", "opp_prs": "h_dprs"}), on=["game_id","home_team"], how="left")
g = g.merge(rd.rename(columns={"defteam": "away_team", "opp_prs": "a_dprs"}), on=["game_id","away_team"], how="left")
g = g.merge(ro.rename(columns={"posteam": "home_team", "own_prs": "h_oprs"}), on=["game_id","home_team"], how="left")
g = g.merge(ro.rename(columns={"posteam": "away_team", "own_prs": "a_oprs"}), on=["game_id","away_team"], how="left")
for c in ["h_dprs","a_dprs","h_oprs","a_oprs"]: g[c] = (g[c].fillna(LG) - LG) * 10
g["prs_gap"] = (g.h_dprs + g.a_oprs) - (g.a_dprs + g.h_oprs)   # positive = home pass rush meets a leaky away line
g["prs_sum"] = g.h_dprs + g.a_dprs + g.h_oprs + g.a_oprs
h = g[~g._pred & g.result.notna() & g.spread_line.notna() & (g.min_games >= 3) & (g.season >= 2019)]
out = {0: [], 1: []}
for s in sorted(h.season.unique()):
    tr, te = h[h.season < s], h[h.season == s]
    if len(tr) < 400: continue
    for k, Xs in ((0, L.MARGIN_X), (1, L.MARGIN_X + ["prs_gap"])):
        mm = sm.OLS(tr.result, sm.add_constant(tr[Xs])).fit()
        out[k].append(np.abs(te.result - mm.predict(sm.add_constant(te[Xs], has_constant="add"))))
    for k, Xs in ((0, L.TOTAL_X), (1, L.TOTAL_X + ["prs_sum"])):
        mt = sm.OLS(tr.total, sm.add_constant(tr[Xs])).fit()
        out.setdefault(10+k, []).append(np.abs(te.total - mt.predict(sm.add_constant(te[Xs], has_constant="add"))))
print(f"team margin MAE {pd.concat(out[0]).mean():.3f} -> {pd.concat(out[1]).mean():.3f} with prs_gap")
print(f"team total  MAE {pd.concat(out[10]).mean():.3f} -> {pd.concat(out[11]).mean():.3f} with prs_sum")
