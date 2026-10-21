"""Retro strength-of-schedule: at each week's cutoff, re-grade every past game against the opponent's rating AS OF THAT CUTOFF
(DVOA-style, iterated to settle), using the same decay/season-carry/shrinkage as the live ratings. Then the same walk-forward test."""
import numpy as np, pandas as pd
import lines_model as L
pbp, sched = L.load(); L.OPP_ADJ = 0.0
g0 = L.build(pbp, sched)
tg = L.team_games(pbp)
s = sched.copy(); s["gameday"] = pd.to_datetime(s.gameday)
lg = pd.concat([s.assign(team=s.home_team, opp=s.away_team), s.assign(team=s.away_team, opp=s.home_team)])[["game_id", "season", "week", "gameday", "team", "opp"]]
lg = lg.merge(tg.rename(columns={"posteam": "team"})[["game_id", "team", "off_epa", "plays", "def_epa_allowed", "def_plays"]], on=["game_id", "team"])
lg = lg.dropna(subset=["off_epa"]).sort_values(["team", "gameday"]).reset_index(drop=True)
lg["k"] = lg.groupby("team").cumcount()
LGO = tg.off_epa.sum() / tg.plays.sum()
teams = sorted(set(lg.team) | set(lg.opp)); ti = {t: i for i, t in enumerate(teams)}
T = lg.team.map(ti).to_numpy(int); O = lg.opp.map(ti).to_numpy(int)
off, pl, dfa, dpl = (lg[c].to_numpy(float) for c in ("off_epa", "plays", "def_epa_allowed", "def_plays"))
seas, wk, kk = lg.season.to_numpy(), lg.week.to_numpy(), lg.k.to_numpy()
def ratings(cut_s, cut_w, adj, iters=4):
    m = (seas < cut_s) | ((seas == cut_s) & (wk < cut_w))
    if not m.any(): return None
    t, o = T[m], O[m]
    last = np.full(len(teams), -1); np.maximum.at(last, t, kk[m])
    w = L.DECAY ** (last[t] - kk[m]) * L.SEASON_CARRY ** (cut_s - seas[m])
    Ro = np.full(len(teams), LGO); Rd = np.full(len(teams), LGO)
    for _ in range(iters if adj else 1):
        offa = off[m] - adj * pl[m] * (Rd[o] - LGO); dfaa = dfa[m] - adj * dpl[m] * (Ro[o] - LGO)
        num_o = np.bincount(t, w * offa, len(teams)); den_o = np.bincount(t, w * pl[m], len(teams))
        num_d = np.bincount(t, w * dfaa, len(teams)); den_d = np.bincount(t, w * dpl[m], len(teams))
        Ro, Rd = (num_o + L.K_RATE * LGO) / (den_o + L.K_RATE), (num_d + L.K_RATE * LGO) / (den_d + L.K_RATE)
    return Ro, Rd
def feats(adj):
    g = g0.copy(); cache = {}
    for (cs, cw), idx in g.groupby(["season", "week"]).groups.items():
        r = ratings(cs, cw, adj)
        if r is None: continue
        Ro, Rd = r; h = g.loc[idx, "home_team"].map(ti); a = g.loc[idx, "away_team"].map(ti)
        ok = h.notna() & a.notna(); h, a = h[ok].astype(int).to_numpy(), a[ok].astype(int).to_numpy(); ii = idx[ok.to_numpy()]
        g.loc[ii, "off_gap"] = Ro[h] - Ro[a]; g.loc[ii, "def_gap"] = Rd[a] - Rd[h]; g.loc[ii, "off_sum"] = Ro[h] + Ro[a]; g.loc[ii, "def_sum"] = Rd[h] + Rd[a]
    return g
def score(label, g):
    g = g[g.result.notna() & g.spread_line.notna() & (g.min_games >= 3)].copy(); outs = []
    for yr in range(2019, 2027):
        tr, te = g[g.season < yr], g[g.season == yr].copy(); mm, mt = L.fit(tr); outs.append(L.predict(te, mm, mt))
    o = pd.concat(outs); rows = []
    for part, oo in (("all", o), ("wk2-8", o[o.week <= 8]), ("wk9+", o[o.week > 8])):
        hc, oh = np.sign(oo.result - oo.spread_line), np.sign(oo.total - oo.total_line)
        r = dict(variant=label, part=part, n=len(oo), margin_mae=np.mean(np.abs(oo.result - oo.m_margin)), total_mae=np.mean(np.abs(oo.total - oo.m_total)))
        pk = np.sign(oo.m_margin - oo.spread_line); m = (np.abs(oo.m_margin - oo.spread_line) >= 3) & (hc != 0); r["ATS 3+"] = f"{(pk[m] == hc[m]).mean()*100:.1f}% ({m.sum()})"
        pk = np.sign(oo.m_total - oo.total_line); m = (np.abs(oo.m_total - oo.total_line) >= 3) & (oh != 0); r["OU 3+"] = f"{(pk[m] == oh[m]).mean()*100:.1f}% ({m.sum()})"
        ws = np.linspace(0, 1, 21); r["blend_w"] = ws[np.argmin([np.mean(np.abs(oo.result - (w * oo.m_margin + (1 - w) * oo.spread_line))) for w in ws])]
        r["tot_blend_w"] = ws[np.argmin([np.mean(np.abs(oo.total - (w * oo.m_total + (1 - w) * oo.total_line))) for w in ws])]
        rows.append(r)
    return rows
rows = score("current (live)", g0) + score("rebuilt, no SOS", feats(0.0))
for a in (0.5, 1.0): rows += score(f"retro SOS {a}", feats(a))
pd.set_option("display.width", 220); print(pd.DataFrame(rows).round(3).sort_values(["part", "variant"]).to_string(index=False))
# how much does it move this year's ratings?
Ro0, Rd0 = ratings(2026, 6, 0.0); Ro1, Rd1 = ratings(2026, 6, 1.0)
d = pd.DataFrame(dict(team=teams, off_raw=Ro0, off_sos=Ro1, def_raw=Rd0, def_sos=Rd1)); d["net_move"] = (d.off_sos - d.def_sos) - (d.off_raw - d.def_raw)
print(d.sort_values("net_move").round(3).to_string(index=False))
