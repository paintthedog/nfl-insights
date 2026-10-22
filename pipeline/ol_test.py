import numpy as np, pandas as pd, int_model as M, proj_model as P
# ---- INT model
pbp, sched = M.load(); qb, tg, ctx = M.build_tables(pbp, sched)
df = M.prep(M.build_features(qb, tg, ctx)); df = df[(df.prior_games >= 1) & df.att.notna() & (df.att >= 10)]
base = ["log_proj_att", "log_qb_risk", "log_qb_pd", "qb_adot", "qb_exp_low", "off_press_rate", "def_press_rate", "log_def_rate", "wind", "exp_margin"]
for xs, label in [(base, "INT: current"), (base + ["ol_missing"], "INT: + OL starters missing")]:
    M.INT_X = xs; tr = df[df.season <= 2023]; te = df[df.season >= 2024].dropna(subset=M.ATT_X + xs[1:])
    am = M.fit_attempts(tr); im = M.fit_ints(tr, am); t = M.predict(te, am, im)
    y = (t.ints >= 1).astype(int); bc = np.mean((y.mean() - y) ** 2); b = np.mean((t.p_int - y) ** 2)
    print(f"{label:30s} skill {1 - b / bc:.2%}" + (f"   coef {im.params['ol_missing']:+.3f}" if "ol_missing" in xs else ""))
# ---- projections (QB passing)
pbp2 = P.load(); players = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name", "position"])
rows = []
for use in [False, True]:
    P.USE_OL = use
    d = P.build(pbp2, sched, players); d = P.efficiency(d, P.position_means(d)); outs = []
    for s in range(2021, 2027):
        tr, te = d[d.season < s].copy(), d[d.season == s].copy()
        va, vc = P.fit_volume(tr.drop_duplicates(["game_id", "posteam"])); tr = P.raw_projection(tr, va, vc); te = P.raw_projection(te, va, vc)
        cal = P.fit_calibration(tr[tr.pr_n >= 1]); outs.append(P.apply_calibration(te, cal))
        if s == 2026 and use: print("  QB sacks OL coef", round(cal[("sacks", "QB")].params["ol_missing"], 3), "| pass yds", round(cal[("pass_yds", "QB")].params["ol_missing"], 4))
    o = pd.concat(outs); q = o[(o.pos == "QB") & o.qb_start & (o.pr_n >= 1) & o.implied.notna()]
    rows.append(dict(variant="proj: + OL" if use else "proj: current", pass_yds=np.mean(np.abs(q.pass_yds - q.p_pass_yds)), sacks=np.mean(np.abs(q.sacks - q.p_sacks)),
                     cmp=np.mean(np.abs(q.cmp - q.p_cmp)), ol2plus_pass_yds=np.mean(np.abs(q[q.ol_missing >= 2].pass_yds - q[q.ol_missing >= 2].p_pass_yds))))
r = pd.DataFrame(rows).set_index("variant"); print(r.round(3).to_string()); print(((r / r.iloc[0] - 1) * 100).round(2).to_string())
