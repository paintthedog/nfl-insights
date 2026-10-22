import numpy as np, pandas as pd, statsmodels.api as sm
import lines_model as L
pbp, sched = L.load()
def run(label, decay=0.92, carry=0.5, K=300, extra_m=[], mkt=False):
    L.DECAY, L.SEASON_CARRY, L.K_RATE = decay, carry, K
    g = L.build(pbp, sched); g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy()
    g["pass_x_dpass"] = g.pass_gap + g.dpass_gap; g["rush_x_drush"] = g.rush_gap + g.drush_gap
    MX = L.MARGIN_X + extra_m + (["spread_line"] if mkt else [])
    TX = L.TOTAL_X + (["total_line"] if mkt else [])
    outs = []
    for s in range(2019, 2027):
        tr, te = g[g.season < s], g[g.season == s].copy()
        mm = sm.OLS(tr.result, sm.add_constant(tr[MX])).fit(); mt = sm.OLS(tr.total, sm.add_constant(tr[TX])).fit()
        te["m_margin"] = mm.predict(sm.add_constant(te[MX].fillna(0), has_constant="add"))
        te["m_total"] = mt.predict(sm.add_constant(te[TX].fillna(te[TX].median()), has_constant="add"))
        outs.append(te)
    o = pd.concat(outs)
    hc, oh = np.sign(o.result - o.spread_line), np.sign(o.total - o.total_line)
    res = dict(variant=label, margin_mae=np.mean(np.abs(o.result - o.m_margin)), total_mae=np.mean(np.abs(o.total - o.m_total)))
    for x in [2, 3]:
        pk = np.sign(o.m_margin - o.spread_line); m = (np.abs(o.m_margin - o.spread_line) >= x) & (hc != 0)
        res[f"ATS {x}+"] = f"{(pk[m] == hc[m]).mean()*100:.1f}% ({m.sum()})"
        pk = np.sign(o.m_total - o.total_line); m = (np.abs(o.m_total - o.total_line) >= x) & (oh != 0)
        res[f"OU {x}+"] = f"{(pk[m] == oh[m]).mean()*100:.1f}% ({m.sum()})"
    return res
rows = [run("current"), run("longer memory (0.95)", decay=0.95), run("shorter memory (0.88)", decay=0.88),
        run("less averaging (K=120)", K=120), run("+ pass/rush matchup terms", extra_m=["pass_x_dpass", "rush_x_drush"]),
        run("model + market line as input", mkt=True)]
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(2).to_string(index=False))
