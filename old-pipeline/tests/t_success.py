"""Success rate (share of plays with EPA > 0) added to team ratings: walk-forward 2019-2026, current vs + success rate.
Same decay/shrinkage as the EPA ratings. Scores margin/total error, ATS/OU at 3+ gaps vs close, blend weight vs the line."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import lines_model as L
import os; L.OPP_ADJ = float(os.environ.get("SOS", "0"))
pbp, sched = L.load(); g0 = L.build(pbp, sched)
p = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & pbp.epa.notna() & (pbp.two_point_attempt != 1) & pbp.play_type.isin(["pass", "run"])].copy()
p["succ"] = (p.epa > 0).astype(int); p["ps"] = p.succ * (p.play_type == "pass"); p["rs"] = p.succ * (p.play_type == "run")
p["pn"] = (p.play_type == "pass").astype(int); p["rn"] = 1 - p.pn
t = p.groupby(["game_id", "posteam", "defteam"]).agg(s=("succ", "sum"), n=("succ", "size"), ps=("ps", "sum"), pn=("pn", "sum"), rs=("rs", "sum"), rn=("rn", "sum")).reset_index()
d = t.rename(columns={"defteam": "team", "s": "ds", "n": "dn", "ps": "dps", "pn": "dpn", "rs": "drs", "rn": "drn"}).drop(columns="posteam")
o = t.rename(columns={"posteam": "team"}).drop(columns="defteam")
s = g0[["game_id", "season", "gameday", "home_team", "away_team"]].copy(); s["gameday"] = pd.to_datetime(s.gameday)
long = pd.concat([s.assign(team=s.home_team, home=1), s.assign(team=s.away_team, home=0)])[["game_id", "season", "gameday", "team", "home"]]
long = long.merge(o, on=["game_id", "team"], how="left").merge(d, on=["game_id", "team"], how="left")
cols = ["s", "n", "ps", "pn", "rs", "rn", "ds", "dn", "dps", "dpn", "drs", "drn"]
long = L.decayed(long, "team", cols, L.DECAY, L.SEASON_CARRY)
lg = p.succ.mean(); lgp = p.ps.sum() / p.pn.sum(); lgr = p.rs.sum() / p.rn.sum(); K = L.K_RATE
long["sr_off"] = (long.ds_s + K * lg) / (long.ds_n + K); long["sr_def"] = (long.ds_ds + K * lg) / (long.ds_dn + K)
long["sr_pass"] = (long.ds_ps + K * lgp) / (long.ds_pn + K); long["sr_dpass"] = (long.ds_dps + K * lgp) / (long.ds_dpn + K)
long["sr_rush"] = (long.ds_rs + K * lgr) / (long.ds_rn + K); long["sr_drush"] = (long.ds_drs + K * lgr) / (long.ds_drn + K)
k = ["game_id", "sr_off", "sr_def", "sr_pass", "sr_dpass", "sr_rush", "sr_drush"]
h = long[long.home == 1][k].add_prefix("h_").rename(columns={"h_game_id": "game_id"}); a = long[long.home == 0][k].add_prefix("a_").rename(columns={"a_game_id": "game_id"})
g = g0.merge(h, on="game_id").merge(a, on="game_id")
g["sr_off_gap"] = g.h_sr_off - g.a_sr_off; g["sr_def_gap"] = g.a_sr_def - g.h_sr_def
g["sr_off_sum"] = g.h_sr_off + g.a_sr_off; g["sr_def_sum"] = g.h_sr_def + g.a_sr_def
g["sr_pass_gap"] = (g.h_sr_pass - g.a_sr_dpass) - (g.a_sr_pass - g.h_sr_dpass)
g = g[g.result.notna() & g.spread_line.notna() & g.total_line.notna() & (g.min_games >= 3)].copy()
base_m, base_t = list(L.MARGIN_X), list(L.TOTAL_X)
variants = {"current": (base_m, base_t),
            "+ success rate": (base_m + ["sr_off_gap", "sr_def_gap"], base_t + ["sr_off_sum", "sr_def_sum"]),
            "+ SR incl pass matchup": (base_m + ["sr_off_gap", "sr_def_gap", "sr_pass_gap"], base_t + ["sr_off_sum", "sr_def_sum"]),
            "SR replaces EPA": ([c for c in base_m if c not in ("off_gap", "def_gap")] + ["sr_off_gap", "sr_def_gap"], [c for c in base_t if c not in ("off_sum", "def_sum")] + ["sr_off_sum", "sr_def_sum"])}
rows = []
for lab, (mx, tx) in variants.items():
    L.MARGIN_X, L.TOTAL_X = mx, tx
    outs = []
    for yr in range(2019, 2027):
        mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
    oo_all = pd.concat(outs)
    for part, oo in (("all", oo_all), ("wk2-8", oo_all[oo_all.week <= 8]), ("2026", oo_all[oo_all.season == 2026])):
        hc, oh = np.sign(oo.result - oo.spread_line), np.sign(oo.total - oo.total_line)
        r = dict(variant=lab, part=part, n=len(oo), margin_mae=np.mean(np.abs(oo.result - oo.m_margin)), total_mae=np.mean(np.abs(oo.total - oo.m_total)))
        pk = np.sign(oo.m_margin - oo.spread_line); m = (np.abs(oo.m_margin - oo.spread_line) >= 3) & (hc != 0); r["ATS 3+"] = f"{(pk[m] == hc[m]).mean()*100:.1f}% ({m.sum()})"
        pk = np.sign(oo.m_total - oo.total_line); m = (np.abs(oo.m_total - oo.total_line) >= 3) & (oh != 0); r["OU 3+"] = f"{(pk[m] == oh[m]).mean()*100:.1f}% ({m.sum()})"
        ws = np.linspace(0, 1, 21)
        r["blend_w_margin"] = ws[np.argmin([np.mean(np.abs(oo.result - (w * oo.m_margin + (1 - w) * oo.spread_line))) for w in ws])]
        r["blend_w_total"] = ws[np.argmin([np.mean(np.abs(oo.total - (w * oo.m_total + (1 - w) * oo.total_line))) for w in ws])]
        rows.append(r)
pd.set_option("display.width", 250); out = pd.DataFrame(rows).round(3).sort_values(["part", "variant"])
print(out.to_string(index=False)); out.to_csv("tests/success_rate_results.csv", index=False)
# paired check on margin error: current vs + success rate, plus season-by-season
res = {}
for lab in ("current", "+ success rate"):
    L.MARGIN_X, L.TOTAL_X = variants[lab]; outs = []
    for yr in range(2019, 2027):
        mm, mt = L.fit(g[g.season < yr]); outs.append(L.predict(g[g.season == yr].copy(), mm, mt))
    res[lab] = pd.concat(outs)
a, b = res["current"], res["+ success rate"]
dm = np.abs(a.result - a.m_margin) - np.abs(b.result - b.m_margin); dt = np.abs(a.total - a.m_total) - np.abs(b.total - b.m_total)
print(f"\nmargin MAE gain {dm.mean():.3f} +/- {dm.std()/np.sqrt(len(dm)):.3f} (se); total gain {dt.mean():.3f} +/- {dt.std()/np.sqrt(len(dt)):.3f}")
print("by season (margin gain):", dm.groupby(a.season).mean().round(3).to_dict())
