"""SEASON section: projected wins for every team (remaining schedule simulated 20,000 times) and projected season stat lines for
players, with ranges, milestone chances and stat-leader races. Writes season.json. Model and tests live in season_model.py."""
import json, numpy as np, pandas as pd, season_model as M
W, YR, N = M.THRU, M.SEASON, 20000
TAU = 2.5      # points of uncertainty about each team's true strength; chosen so the 80% range held the result ~80% of the time on 2017-2025
rng = np.random.default_rng(11)
# ---------------------------------------------------------------- teams
rows = []
for yr in range(M.FIRST + 1, YR):   # the test: each season predicted by a model that never saw it
    g, sd, _ = M.remaining(yr, W, exclude=yr); done = M.S[(M.S.season == yr) & (M.S.week <= W)]; y = {}
    for h, a, r in zip(done.home_team, done.away_team, done.result):
        for t, mm in ((h, r), (a, -r)): o = y.setdefault(t, [0, 0.0]); o[0] += 1; o[1] += (mm > 0) + .5 * (mm == 0)
    ew = {t: v[1] for t, v in y.items()}
    for h, a, p in zip(g.home_team, g.away_team, g.p_home): ew[h] += p; ew[a] += 1 - p
    for t in ew: n, w = y[t]; tot = M.fin_tm.tg[(yr, t)]; rows.append(dict(fin=M.fin_tm.w[(yr, t)], model=ew[t], pace=w / n * tot, half=w + (tot - n) * (.5 * w / n + .25)))
tv = pd.DataFrame(rows); cov = []
for yr in range(M.FIRST + 1, YR):
    tms, wn, _, _ = M.simulate(yr, W, n=3000, tau=TAU, exclude=yr); lo, hi = np.percentile(wn, [10, 90], axis=0); cov += [lo[i] <= M.fin_tm.w[(yr, t)] <= hi[i] for i, t in enumerate(tms)]
team_val = dict(n=len(tv), seasons=f"{M.FIRST + 1}–{YR - 1}", model=round(float((tv.model - tv.fin).abs().mean()), 2), pace=round(float((tv.pace - tv.fin).abs().mean()), 2), half=round(float((tv.half - tv.fin).abs().mean()), 2), cover=round(float(np.mean(cov)) * 100))
teams, wins, g, sd = M.simulate(YR, W, n=N, tau=TAU); gm, _ = M.game_model(W); f = M.team_feats(YR, W); ti = {t: i for i, t in enumerate(teams)}
rating = {t: float(gm.coef_ @ np.array(f[t])) for t in teams}; avg = np.mean(list(rating.values())); rating = {t: v - avg for t, v in rating.items()}
# playoffs: four division winners and three wild cards per conference, ties broken at random
jit = wins + rng.random(wins.shape) * 0.01; conf = np.array([M.DIV[t].split()[0] for t in teams]); divs = np.array([M.DIV[t] for t in teams])
p_div = np.zeros(len(teams)); p_po = np.zeros(len(teams)); p_one = np.zeros(len(teams))
for c in ("AFC", "NFC"):
    ci = np.where(conf == c)[0]; dw = np.zeros((N, len(ci)), bool)
    for dv in sorted(set(divs[ci])):
        di = np.where(divs[ci] == dv)[0]; top = jit[:, ci[di]].argmax(1); dw[np.arange(N), di[top]] = True
    p_div[ci] = dw.mean(0); seed1 = np.where(dw, jit[:, ci], -1).argmax(1); one = np.zeros_like(dw); one[np.arange(N), seed1] = True; p_one[ci] = one.mean(0)
    wc = np.where(dw, -1, jit[:, ci]); kth = np.sort(wc, 1)[:, -3][:, None]; p_po[ci] = (dw | (wc >= kth)).mean(0)
# ---- playoffs played out inside each simulated season: seeds from that sim's standings, each game won with the
# probability its rating gap implies (ratings carry that sim's true-strength draw; home field for the better seed)
eff = M.simulate.eff; sd1 = float(np.sqrt(max(sd ** 2 - 2 * TAU ** 2, 100.0))); HFA_P = float(gm.intercept_)
ncdf = lambda z: 0.5 * (1 + np.vectorize(__import__("math").erf)(z / np.sqrt(2)))
import math
_ncdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
strength = np.array([[rating[t] for t in teams]]) + eff          # (N, 32)
p_conf = np.zeros(len(teams)); p_sb = np.zeros(len(teams)); rng_po = np.random.default_rng(11)
U = rng_po.random((N, 13 * 2))                                   # enough uniform draws per sim
champ = {"AFC": np.zeros(N, int), "NFC": np.zeros(N, int)}
for c in ("AFC", "NFC"):
    ci = np.where(conf == c)[0]
    order = np.argsort(-jit[:, ci], axis=1)                      # within-conference order by wins (jittered)
    dvs = divs[ci]
    uq = 0 if c == "AFC" else 13
    for s_ in range(N):
        row = jit[s_, ci]; isdw = np.zeros(len(ci), bool)
        for dv in set(dvs):
            di = np.where(dvs == dv)[0]; isdw[di[row[di].argmax()]] = True
        dwi = sorted(np.where(isdw)[0], key=lambda i: -row[i]); wci = sorted(np.where(~isdw)[0], key=lambda i: -row[i])[:3]
        seeds = [ci[i] for i in dwi + wci]                       # global team indices, seeds 1..7
        st = strength[s_]; u = U[s_]
        def beat(a, b, k, home):                                  # does seeds-index a beat b
            ph = _ncdf(((st[a] - st[b]) + (HFA_P if home else 0.0)) / sd1); return u[uq + k] < ph
        w47 = [seeds[1] if beat(seeds[1], seeds[6], 0, True) else seeds[6],
               seeds[2] if beat(seeds[2], seeds[5], 1, True) else seeds[5],
               seeds[3] if beat(seeds[3], seeds[4], 2, True) else seeds[4]]
        rk = {t: i for i, t in enumerate(seeds)}
        rem = sorted(w47, key=lambda t: rk[t])
        d1 = seeds[0] if beat(seeds[0], rem[-1], 3, True) else rem[-1]
        d2 = rem[0] if beat(rem[0], rem[1], 4, True) else rem[1]
        hi_, lo_ = (d1, d2) if rk[d1] < rk[d2] else (d2, d1)
        champ[c][s_] = hi_ if beat(hi_, lo_, 5, True) else lo_
for s_ in range(N):
    a, b = champ["AFC"][s_], champ["NFC"][s_]
    sbw = a if U[s_, 6] < _ncdf((strength[s_, a] - strength[s_, b]) / sd1) else b
    p_sb[sbw] += 1
for c in ("AFC", "NFC"):
    np.add.at(p_conf, champ[c], 1)
p_conf /= N; p_sb /= N
names = {}
try:
    L = json.load(open(f"site/data_wk{W + 1}.json"))["lines"]; names = {x["team"]: x["name"] for x in L["teams"]}
except Exception: pass
done = M.S[(M.S.season == YR) & (M.S.week <= W) & M.S.result.notna()]; rec = {t: [0, 0, 0] for t in teams}
for h, a, r in zip(done.home_team, done.away_team, done.result):
    rec[h][0 if r > 0 else 1 if r < 0 else 2] += 1; rec[a][0 if r < 0 else 1 if r > 0 else 2] += 1
opp = {t: [] for t in teams}
for h, a in zip(g.home_team, g.away_team): opp[h].append(rating[a]); opp[a].append(rating[h])
sos = {t: float(np.mean(v)) for t, v in opp.items()}; sos_rank = {t: i + 1 for i, t in enumerate(sorted(sos, key=lambda t: -sos[t]))}
T = []
for t in teams:
    w = wins[:, ti[t]]; mean = float(w.mean()); line = np.floor(mean) + .5; dist = [round(float((np.round(w) == k).mean()) * 100, 1) for k in range(18)]
    T.append(dict(team=t, name=names.get(t, t), div=M.DIV[t], rec="-".join(str(x) for x in (rec[t] if rec[t][2] else rec[t][:2])), w=round(mean, 1), lo=int(np.percentile(w, 10)), hi=int(np.percentile(w, 90)),
                  rating=round(rating[t], 1), mk=round(f[t][0], 1), pf=round(f[t][1], 1), sos=round(sos[t], 1), sos_rank=sos_rank[t], p_div=round(float(p_div[ti[t]]), 3), p_po=round(float(p_po[ti[t]]), 3), p_one=round(float(p_one[ti[t]]), 3),
                  p_conf=round(float(p_conf[ti[t]]), 4), p_sb=round(float(p_sb[ti[t]]), 4),
                  dist=dist, over=[round(float((w > k + .5).mean()), 3) for k in range(17)], line=line))
print("TEAMS", team_val, "home edge", round(float(gm.intercept_), 2), "weights", gm.coef_.round(2))
for r in sorted(T, key=lambda r: -r["w"])[:8]: print("  ", r["team"], r["rec"], r["w"], f"{r['lo']}-{r['hi']}", "rating", r["rating"], "div", r["p_div"], "playoffs", r["p_po"], "1 seed", r["p_one"], "sos", r["sos_rank"])
# ---------------------------------------------------------------- players
d = M.off_oof(W); off, _, _ = M.project(W); cur = off[off.season == YR].copy(); hist = d[(d.season < YR)]
cur["name"] = cur.pid.map(M.PL.display_name); cur["pos"] = cur.pid.map(M.PL.position_group); cur = cur[cur.pos.isin(["QB", "RB", "WR", "TE"]) & cur.name.notna()]
cur["opp"] = (cur.ytd_db + cur.ytd_car + cur.ytd_tgt); cur = cur[cur.opp >= 5 * W].reset_index(drop=True)
ST = ["pass_yds", "pass_td", "ints", "rush_yds", "rush_td", "rec", "rec_yds", "rec_td", "car", "tgt"]
# spread of outcomes: for each player and stat, how the rest of the season turned out for the 250 past players the model expected most similarly of
NS = 4000; U = rng.random((len(cur), NS)); sim = {}
for k in ST:
    sim[k] = np.zeros((len(cur), NS))
    for q in (True, False):
        hh = hist[(hist.qb == q) & hist["t_" + k].notna() & hist["o_" + k].notna()]; pr = (hh["o_" + k].clip(lower=0) * hh.rem).values; ac = (hh["t_" + k] * hh.rem).values; o = np.argsort(pr); pr, ac = pr[o], ac[o]
        for i in np.where((cur.pos == "QB").values == q)[0]:
            x = max(float(cur["r_" + k].iloc[i]), 0) * cur.rem.iloc[i]; j = np.searchsorted(pr, x); a, b = max(0, j - 125), min(len(pr), j + 125)
            res = np.sort(ac[a:b] - pr[a:b]); sim[k][i] = cur["ytd_" + k].iloc[i] + np.clip(x + res[(U[i] * len(res)).astype(int).clip(max=len(res) - 1)], 0, None)
sim["scrim"] = sim["rush_yds"] + sim["rec_yds"]; sim["td"] = sim["rush_td"] + sim["rec_td"]; sim["ppr"] = sim["pass_yds"] / 25 + sim["pass_td"] * 4 - sim["ints"] * 2 + sim["scrim"] / 10 + sim["td"] * 6 + sim["rec"]
y = lambda k: cur["ytd_" + k].values
ytd = {**{k: y(k) for k in ST}, "scrim": y("rush_yds") + y("rec_yds"), "td": y("rush_td") + y("rec_td")}; ytd["ppr"] = ytd["pass_yds"] / 25 + ytd["pass_td"] * 4 - ytd["ints"] * 2 + ytd["scrim"] / 10 + ytd["td"] * 6 + ytd["rec"]
mean = {**{k: cur[k].values for k in ST}}; mean["scrim"] = mean["rush_yds"] + mean["rec_yds"]; mean["td"] = mean["rush_td"] + mean["rec_td"]; mean["ppr"] = mean["pass_yds"] / 25 + mean["pass_td"] * 4 - mean["ints"] * 2 + mean["scrim"] / 10 + mean["td"] * 6 + mean["rec"]
MILE = {"QB": [("pass_yds", 4000, "4,000 pass yds"), ("pass_td", 30, "30 pass TDs")], "RB": [("rush_yds", 1000, "1,000 rush yds"), ("td", 10, "10 TDs")], "WR": [("rec_yds", 1000, "1,000 rec yds"), ("rec", 100, "100 catches")], "TE": [("rec_yds", 800, "800 rec yds"), ("td", 8, "8 TDs")]}
RACE = [("pass_yds", "Passing yards"), ("pass_td", "Passing touchdowns"), ("rush_yds", "Rushing yards"), ("rec_yds", "Receiving yards"), ("rec", "Receptions"), ("td", "Rushing + receiving touchdowns"), ("scrim", "Yards from scrimmage")]
lead = {}
for k, _ in RACE: top = sim[k].argmax(0); lead[k] = np.bincount(top, minlength=len(cur)) / NS
KEYS = ST + ["scrim", "td", "ppr"]; dec = lambda k: 0
PLY = []
for i, r in cur.iterrows():
    PLY.append(dict(name=r["name"], team=r.team, pos=r.pos, gp=int(r.gp), rem=int(r.rem),
                    s={k: round(float(mean[k][i]), 1) for k in KEYS}, y={k: round(float(ytd[k][i]), 1) for k in KEYS},
                    lo={k: round(float(np.percentile(sim[k][i], 10))) for k in KEYS}, hi={k: round(float(np.percentile(sim[k][i], 90))) for k in KEYS},
                    mile=[[lab, round(float((sim[k][i] >= th).mean()), 3)] for k, th, lab in MILE[r.pos]], lead={k: round(float(lead[k][i]), 3) for k, _ in RACE if lead[k][i] >= .005}))
races = [dict(key=k, label=lab, rows=[dict(name=cur["name"].iloc[i], team=cur.team.iloc[i], pos=cur.pos.iloc[i], p=round(float(lead[k][i]), 3), proj=round(float(mean[k][i])), now=round(float(ytd[k][i])), lo=round(float(np.percentile(sim[k][i], 10))), hi=round(float(np.percentile(sim[k][i], 90))))
                                      for i in np.argsort(-lead[k])[:10] if lead[k][i] >= .01]) for k, lab in RACE]
# the test for players: final totals after week W, model against simply extending the pace
h = hist; pv = []
for k, mask, lab in (("pass_yds", h.qb & (h.y_db >= 20), "Quarterback passing yards"), ("pass_td", h.qb & (h.y_db >= 20), "Quarterback passing touchdowns"), ("rush_yds", (h.y_car >= 8), "Running back rushing yards"), ("rec_yds", (h.y_tgt >= 5), "Receiving yards"), ("rec", (h.y_tgt >= 5), "Receptions")):
    m_ = h[mask & h["t_" + k].notna() & h["o_" + k].notna()]; fin = m_[k] + m_["t_" + k] * m_.rem; pm = m_[k] + m_["o_" + k].clip(lower=0) * m_.rem
    pv.append(dict(label=lab, n=len(m_), model=round(float((pm - fin).abs().mean()), 1), pace=round(float((m_[k] / m_.tg * (m_.tg + m_.rem) - fin).abs().mean()), 1), last=round(float((m_["p_" + k] * 17 - fin).abs().mean()), 1), avg=round(float(fin.mean()))))
print("PLAYERS", len(PLY)); [print("  ", v) for v in pv]
for rc in races: print("  ", rc["label"], [(x["name"], x["p"], x["proj"]) for x in rc["rows"][:4]])
for p in sorted(PLY, key=lambda p: -p["s"]["ppr"])[:6]: print("  ", p["name"], p["pos"], p["s"]["ppr"], p["lo"]["ppr"], p["hi"]["ppr"], p["mile"])
json.dump(dict(built=pd.Timestamp.now().strftime("%b %-d, %Y"), thru=W, teams=T, team_val=team_val, hfa=round(float(gm.intercept_), 1), wts=[round(float(x), 2) for x in gm.coef_], players=PLY, races=races, player_val=pv), open("season.json", "w"), allow_nan=False)
