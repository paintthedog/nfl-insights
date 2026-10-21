"""Strength-of-schedule adjustment for team ratings: walk-forward 2019-2026, current vs adjusted at several strengths.
Scores margin/total error, ATS and O/U at 3+ point gaps vs the closing line, overall and for weeks 2-8 (when schedules are lumpiest)."""
import numpy as np, pandas as pd, statsmodels.api as sm
import lines_model as L
pbp, sched = L.load()
def run(label, adj):
    L.OPP_ADJ = adj
    g = L.build(pbp, sched); g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy()
    outs = []
    for s in range(2019, 2027):
        tr, te = g[g.season < s], g[g.season == s].copy(); mm, mt = L.fit(tr); outs.append(L.predict(te, mm, mt))
    o = pd.concat(outs); rows = []
    for part, oo in (("all", o), ("wk2-8", o[o.week <= 8]), ("wk9+", o[o.week > 8])):
        hc, oh = np.sign(oo.result - oo.spread_line), np.sign(oo.total - oo.total_line)
        r = dict(variant=label, part=part, n=len(oo), margin_mae=np.mean(np.abs(oo.result - oo.m_margin)), total_mae=np.mean(np.abs(oo.total - oo.m_total)),
                 line_mae=np.mean(np.abs(oo.result - oo.spread_line)))
        pk = np.sign(oo.m_margin - oo.spread_line); m = (np.abs(oo.m_margin - oo.spread_line) >= 3) & (hc != 0); r["ATS 3+"] = f"{(pk[m] == hc[m]).mean()*100:.1f}% ({m.sum()})"
        pk = np.sign(oo.m_total - oo.total_line); m = (np.abs(oo.m_total - oo.total_line) >= 3) & (oh != 0); r["OU 3+"] = f"{(pk[m] == oh[m]).mean()*100:.1f}% ({m.sum()})"
        # does the model add anything to the line? best blend weight on margin
        ws = np.linspace(0, 1, 21); r["blend_w"] = ws[np.argmin([np.mean(np.abs(oo.result - (w * oo.m_margin + (1 - w) * oo.spread_line))) for w in ws])]
        rows.append(r)
    return rows
rows = []
for lab, a in (("current", 0.0), ("SOS 0.5", 0.5), ("SOS 1.0", 1.0)): rows += run(lab, a); print(lab, "done", flush=True)
pd.set_option("display.width", 220); print(pd.DataFrame(rows).round(3).sort_values(["part", "variant"]).to_string(index=False))
