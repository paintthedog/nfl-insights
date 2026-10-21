"""Award profiles. What the last ten winners of each AP award had in common, a ranking of this year's players learned from the top five
in each year's voting, how that ranking did at this point in past seasons, and how it has moved week to week. Counting stats are
projected to a full season (start-of-season numbers blended with last year's, fitted on 2016-2025), not stretched to 17 games.
Writes awards.json."""
import json, glob, os, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
warnings.filterwarnings("ignore")
SEASON, FIRST = 2026, 2016
VOTES = json.load(open("award_votes.json")); VOTES.pop("_notes", None)
WIN = {k: {int(y): v[0] for y, v in d.items()} for k, d in VOTES.items()}
SCORE = [1.0, .5, .35, .25, .15]
ALIAS = {"Shaquille Leonard": ["Shaquille Leonard", "Darius Leonard"], "Sauce Gardner": ["Sauce Gardner", "Ahmad Gardner"], "Will Anderson Jr.": ["Will Anderson Jr.", "Will Anderson"],
         "Patrick Surtain II": ["Patrick Surtain II", "Pat Surtain II", "Pat Surtain"], "Josh Hines-Allen": ["Josh Hines-Allen", "Josh Allen"], "Brian Thomas Jr.": ["Brian Thomas Jr.", "Brian Thomas"],
         "James Pearce Jr.": ["James Pearce Jr.", "James Pearce"], "Marvin Harrison Jr.": ["Marvin Harrison Jr.", "Marvin Harrison"], "Kenneth Walker III": ["Kenneth Walker III", "Kenneth Walker"],
         "Michael Pittman Jr.": ["Michael Pittman Jr.", "Michael Pittman"], "Odell Beckham Jr.": ["Odell Beckham Jr.", "Odell Beckham"], "Travis Etienne Jr.": ["Travis Etienne Jr.", "Travis Etienne"]}
# average share the model gave its #1 at this point, leave-one-season-out 2016-2025 (tests/t_award_calib.py)
TOPSHARE = {"mvp": 29, "opoy": 21, "dpoy": 23, "oroy": 36, "droy": 37, "coy": 21}
ODDS = json.load(open("award_odds_dk.json")) if os.path.exists("award_odds_dk.json") else {}
from season_model import PL, S, THRU, P, OSTAT, DSTAT, build, FULL, fin_tm, prev_wp, prev_coach, TOT_G, project

def derive(off):
    off = off.copy(); off["scrim"] = off.rush_yds + off.rec_yds; off["td"] = off.rush_td + off.rec_td; off["tot_td"] = off.td + off.pass_td; off["tot_yds"] = off.pass_yds + off.rush_yds
    off["epa_db"] = off.epa / off.db.clip(lower=1); off["int_rate"] = off.ints / off.db.clip(lower=1); return off
# what a defender does, from his career numbers to date, because the listed position calls edge rushers linebackers
CAR = FULL[1].sort_values("season").copy()
for k in ("sacks", "qbh", "tkl", "pd"): CAR["c_" + k] = CAR.groupby("pid")[k].cumsum() - CAR[k]
CARI = CAR.set_index(["season", "pid"])
RUSH = ["Edge rusher", "Interior lineman"]
def roles(d):
    """role from the charted position (edge, interior, linebacker, corner, safety); career numbers decide when that is missing"""
    pos = d.pid.map(PL.position); pff = d.pid.map(PL.pff_position); cs = lambda k: np.array([CARI["c_" + k].get((s, p), 0) for s, p in zip(d.season, d.pid)]) + d[k].values
    rush = (2 * cs("sacks") + cs("qbh")) / np.maximum(cs("tkl"), 1)
    fb = np.where(pos.isin(["DT", "NT"]), "Interior lineman", np.where(pos.isin(["DE", "DL"]), "Edge rusher", np.where(pos.isin(["CB"]), "Cornerback", np.where(pos.isin(["S", "SS", "FS", "SAF", "DB"]), "Safety", np.where(rush >= .35, "Edge rusher", "Linebacker")))))
    return np.where(pff.isin(["ED", "DI", "LB", "CB", "S"]), pff.map({"ED": "Edge rusher", "DI": "Interior lineman", "LB": "Linebacker", "CB": "Cornerback", "S": "Safety"}), fb)

def frame(kind, tabs, wide):
    off, de, tm = tabs
    if kind == "coy":
        d = tm.copy(); d["name"] = d.coach; d["pid"] = d.coach; d["prev"] = [prev_wp.get((s, tt), np.nan) for s, tt in zip(d.season, d.team)]
        d["gain"] = d.wp - d.prev; d["newc"] = [float(prev_coach.get((s, tt)) != c) for s, tt, c in zip(d.season, d.team, d.coach)]; d["pos"] = "HC"; d["grp"] = "HC"; d["pick"] = 0
        return d[d.season >= FIRST].dropna(subset=["prev"])
    d = (de if kind in ("dpoy", "droy") else derive(off)).copy()
    d["name"] = d.pid.map(PL.display_name); d["pos"] = d.pid.map(PL.position); d["grp"] = d.pid.map(PL.position_group); d["rookie"] = d.pid.map(PL.rookie_season) == d.season; d["pick"] = d.pid.map(PL.draft_pick).fillna(260)
    d = d[d.name.notna()].merge(tm[["season", "team", "wp", "mpg", "tg"] + [c for c in ("ytd_w", "ytd_tg") if c in tm]], on=["season", "team"], how="left"); d = d[d.season >= FIRST]
    if kind == "mvp": d = d[(d.grp == "QB") & (d.db >= d.tg * 14)]
    if kind == "opoy": d["fp"] = d.pass_yds / 25 + d.pass_td * 4 + d.scrim / 10 + d.td * 6 + d.rec * .5; d = d[d.groupby("season").fp.rank(ascending=False) <= (100 if wide else 60)]; d["is_qb"] = (d.grp == "QB").astype(float)
    if kind == "oroy": d = d[d.rookie & d.grp.isin(["QB", "RB", "WR", "TE"])]; d["fp"] = d.pass_yds / 25 + d.pass_td * 4 + d.scrim / 10 + d.td * 6; d = d[d.groupby("season").fp.rank(ascending=False) <= (40 if wide else 25)]; d["is_qb"] = (d.grp == "QB").astype(float); d["lpick"] = -np.log(d.pick)
    if kind in ("dpoy", "droy"):
        d["role"] = roles(d); d["rusher"] = d.role.isin(RUSH).astype(float); d["cb"] = (d.role == "Cornerback").astype(float)
        d["imp"] = d.sacks * 2 + d.tfl + d.qbh * .5 + d.dint * 3 + d.pd + d.ff * 2 + d.tkl * .15 + d.dtd * 4
        if kind == "droy": d = d[d.rookie]; d["lpick"] = -np.log(d.pick)
        d = d[d.groupby("season").imp.rank(ascending=False) <= ((40 if wide else 25) if kind == "droy" else (200 if wide else 150))]
    return d
FEATS = {"mvp": ["wp", "pass_td", "tot_td", "tot_yds", "epa_db", "int_rate"], "opoy": ["scrim", "rec_yds", "rush_yds", "td", "pass_td", "pass_yds", "wp", "is_qb"],
         "dpoy": ["sacks", "tfl", "qbh", "dint", "pd", "ff", "wp", "rusher"], "oroy": ["pass_yds", "scrim", "tot_td", "wp", "is_qb", "lpick"], "droy": ["sacks", "tfl", "tkl", "dint", "pd", "lpick"],
         "coy": ["wp", "gain", "newc", "mpg"]}
def zed(d, feats):
    d = d.copy()
    for f in feats: g = d.groupby("season")[f]; d["z_" + f] = ((d[f] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0)
    return d
def label(kind, d):
    """won = 1 for the winner; vote = credit for a top-five finish in the voting"""
    d = d.copy(); d["won"] = 0; d["vote"] = 0.0; miss = []
    for yr, names in VOTES[kind].items():
        for i, nm in enumerate(names[:5]):
            m = d[(d.season == int(yr)) & d.name.isin(ALIAS.get(nm, [nm]))]
            if kind == "dpoy" and nm == "Josh Hines-Allen": m = m[m.grp != "QB"]
            if len(m): ix = m.sort_values(FEATS[kind][0], ascending=False).index[0]; d.loc[ix, "vote"] = SCORE[i]; d.loc[ix, "won"] = int(i == 0)
            elif i == 0: miss.append(int(yr))
    return d, miss
def fit(kind, tr, votes):
    f = ["z_" + x for x in FEATS[kind]]
    if not votes: return LogisticRegression(C=0.3, max_iter=500).fit(tr[f], tr.won)
    pos = tr[tr.vote > 0]; X = pd.concat([tr[f], pos[f]]); y = np.r_[np.zeros(len(tr)), np.ones(len(pos))]; w = np.r_[1 - tr.vote.values, pos.vote.values]
    return LogisticRegression(C=0.5, max_iter=500).fit(X, y, sample_weight=w)
def score(kind, m, te):
    te = te.copy(); s = m.decision_function(te[["z_" + x for x in FEATS[kind]]]); e = np.exp(s - s.max()); te["p"] = e / e.sum(); te["rk"] = te.p.rank(ascending=False, method="min"); return te
def loso(kind, d, votes):
    r = {}
    for yr in WIN[kind]:
        w = d[(d.season == yr) & (d.won == 1)]
        if not len(w): r[yr] = None; continue
        te = score(kind, fit(kind, d[(d.season != yr) & (d.season < SEASON)], votes), d[d.season == yr]); r[yr] = int(te[te.won == 1].rk.iloc[0])
    return r
hits = lambda r, yrs, k: sum((r.get(y) or 999) <= k for y in yrs)

LABEL = {"mvp": "Most Valuable Player", "opoy": "Offensive Player of the Year", "dpoy": "Defensive Player of the Year", "oroy": "Offensive Rookie of the Year", "droy": "Defensive Rookie of the Year", "coy": "Coach of the Year"}
NICE = {"wp": "projected team wins", "pass_td": "passing touchdowns", "tot_td": "total touchdowns", "tot_yds": "passing + rushing yards", "epa_db": "efficiency per dropback", "int_rate": "interception rate", "scrim": "yards from scrimmage",
        "rec_yds": "receiving yards", "rush_yds": "rushing yards", "td": "touchdowns", "pass_yds": "passing yards", "sacks": "sacks", "tfl": "tackles for loss", "qbh": "quarterback hits", "dint": "interceptions", "pd": "passes defensed",
        "ff": "forced fumbles", "tkl": "tackles", "gain": "improvement on last season", "mpg": "point margin per game", "is_qb": "being a quarterback", "lpick": "draft position", "newc": "first year with the team", "rusher": "being a pass rusher"}
# stat columns shown for each award: (key, label, decimals, value function)
W17 = lambda r: r.wp * 17
SHOW = {"mvp": [("wins", "Team wins", 1, W17), ("pass_yds", "Pass yds", 0, None), ("tot_td", "Total TDs", 0, None), ("ints", "INTs", 0, None), ("epa_db", "EPA per dropback", 2, None)],
        "opoy": [("scrim", "Scrimmage yds", 0, None), ("td", "Rush + rec TDs", 0, None), ("rec_yds", "Rec yds", 0, None), ("rush_yds", "Rush yds", 0, None), ("pass_yds", "Pass yds", 0, None), ("wins", "Team wins", 1, W17)],
        "dpoy": [("sacks", "Sacks", 1, None), ("qbh", "QB hits", 0, None), ("tfl", "Tackles for loss", 0, None), ("dint", "INTs", 0, None), ("pd", "Passes defensed", 0, None), ("tkl", "Tackles", 0, None)],
        "oroy": [("pick", "Draft pick", 0, lambda r: r.pick if r.pick < 260 else None), ("pass_yds", "Pass yds", 0, None), ("scrim", "Scrimmage yds", 0, None), ("tot_td", "Total TDs", 0, None), ("wins", "Team wins", 1, W17)],
        "droy": [("pick", "Draft pick", 0, lambda r: r.pick if r.pick < 260 else None), ("sacks", "Sacks", 1, None), ("tkl", "Tackles", 0, None), ("tfl", "Tackles for loss", 0, None), ("dint", "INTs", 0, None), ("pd", "Passes defensed", 0, None)],
        "coy": [("wins", "Wins", 1, W17), ("prevw", "Wins last year", 1, lambda r: r.prev * 17), ("gainw", "Wins gained", 1, lambda r: r.gain * 17), ("mpg", "Point margin / game", 1, None)]}
CHK = {"mvp": ["Team projected for 12+ wins", "Top 3 in projected passing TDs", "Top 3 in projected total TDs", "Top 3 in efficiency", "Projected for 40+ TDs", "Projected for 10 or fewer INTs"],
       "opoy": ["Not a quarterback", "Projected to lead the league in rushing, receiving or scrimmage yards", "Projected for 1,700+ scrimmage yards", "Projected for 12+ TDs", "Team projected for a winning record"],
       "dpoy": ["Pass rusher", "Projected for 12+ sacks", "Top 5 in projected sacks", "Projected for 25+ QB hits", "Team projected for a winning record", "First-round pick"],
       "oroy": ["First-round pick", "Top-12 pick", "QB projected to start all year, or 1,000+ scrimmage yards"], "droy": ["First-round pick", "Top-10 pick", "Top 3 among rookies in projected sacks or tackles", "Projected for 7+ sacks, 100+ tackles or 15+ passes defensed"],
       "coy": ["Projected to improve by 3+ wins", "Projected to improve by 5+ wins", "First year with the team", "Projected for 11+ wins", "Team had a losing record last year"]}
def checks(kind, r, pool):
    g = lambda c: float(r.get(c, 0) or 0); rk_ = lambda c: int((pool[c] > g(c)).sum()) + 1
    if kind == "mvp": return [g("wp") >= .70, rk_("pass_td") <= 3, rk_("tot_td") <= 3, rk_("epa_db") <= 3, g("tot_td") >= 40, g("ints") <= 10]
    if kind == "opoy": return [r.grp != "QB", min(rk_("rush_yds"), rk_("rec_yds"), rk_("scrim")) == 1, g("scrim") >= 1700, g("td") >= 12, g("wp") > .5]
    if kind == "dpoy": return [r.role in RUSH, g("sacks") >= 12, rk_("sacks") <= 5, g("qbh") >= 25, g("wp") > .5, r.pick <= 32]
    if kind == "oroy": return [r.pick <= 32, r.pick <= 12, (g("db") >= 450) if r.grp == "QB" else (g("scrim") >= 1000)]
    if kind == "droy": return [r.pick <= 32, r.pick <= 10, min(rk_("sacks"), rk_("tkl")) <= 3, g("sacks") >= 7 or g("tkl") >= 100 or g("pd") >= 15]
    return [g("gain") >= 3 / 17, g("gain") >= 5 / 17, g("newc") == 1, g("wp") >= 11 / 17 - .01, g("prev") < .5]
def val(kind, r, key, fn):
    v = fn(r) if fn else r.get(key); return None if v is None or pd.isna(v) else float(v)
def ytd(kind, r, key):
    if key == "wins" and "ytd_w" in r and not pd.isna(r.get("ytd_w")): return f"{r.ytd_w:g}-{r.ytd_tg - r.ytd_w:g} now"
    comp = {"scrim": ["rush_yds", "rec_yds"], "td": ["rush_td", "rec_td"], "tot_td": ["rush_td", "rec_td", "pass_td"]}.get(key, [key])
    if all(("ytd_" + c) in r for c in comp): v = sum(float(r["ytd_" + c]) for c in comp); return f"{v:g} now" if key == "sacks" else f"{v:,.0f} now"
    return None

PROJ = {w: project(w) for w in range(2, THRU + 1)}
out = {}
for kind in VOTES:
    dF, missF = label(kind, zed(frame(kind, FULL, False), FEATS[kind])); dP, missP = label(kind, zed(frame(kind, PROJ[THRU], True), FEATS[kind]))
    yrs = [y for y in WIN[kind] if y not in missF]
    rk = {v: loso(kind, dP, v) for v in (False, True)}; sc = {v: (hits(rk[v], yrs, 3), hits(rk[v], yrs, 1), hits(rk[v], yrs, 5)) for v in rk}
    use = sc[True] >= sc[False]; rkP = rk[use]; rkF = loso(kind, dF, use)
    print(f"\n== {LABEL[kind]}: at wk{THRU} top3/top1/top5 winner-only {sc[False]} vs top-five voting {sc[True]} of {len(yrs)} -> {'voting' if use else 'winner-only'} | season's end top1/3 {hits(rkF, yrs, 1)}/{hits(rkF, yrs, 3)}", "| unmatched", missF, missP)
    # ---- what winners had in common (final numbers)
    W = dF[dF.won == 1]; n = len(W); traits = []
    rnk = lambda col: dF.groupby("season")[col].rank(ascending=False, method="min")[W.index]
    def T(txt, k, nn=None): traits.append(dict(text=txt, k=int(k), n=int(nn if nn is not None else n)))
    if kind == "mvp":
        T("were quarterbacks", n); T("played for a team that won 12 or more games", (W.wp >= .70).sum()); T("finished top 3 in efficiency per dropback", (rnk("epa_db") <= 3).sum())
        T("finished top 3 in passing touchdowns", (rnk("pass_td") <= 3).sum()); T("finished top 3 in total touchdowns, passing plus rushing", (rnk("tot_td") <= 3).sum())
        T("accounted for 40 or more touchdowns", (W.tot_td >= 40).sum()); T("threw 10 or fewer interceptions", (W.ints <= 10).sum())
    if kind == "opoy":
        nq = W[W.grp != "QB"]; T("were not quarterbacks", len(nq)); T("were running backs", (W.grp == "RB").sum()); T("were wide receivers", (W.grp == "WR").sum())
        T("led the league in rushing yards, receiving yards or yards from scrimmage", ((rnk("rush_yds") == 1) | (rnk("rec_yds") == 1) | (rnk("scrim") == 1)).sum())
        T("of the non-quarterbacks gained 1,700+ yards from scrimmage", (nq.scrim >= 1700).sum(), len(nq)); T("of the non-quarterbacks scored 12+ touchdowns", (nq.td >= 12).sum(), len(nq)); T("played for a team with a winning record", (W.wp > .5).sum())
    if kind == "dpoy":
        pr = W[W.role.isin(RUSH)]; T("were pass rushers", len(pr)); T("were edge rushers", (W.role == "Edge rusher").sum()); T("were interior linemen", (W.role == "Interior lineman").sum()); T("were cornerbacks", (W.role == "Cornerback").sum()); T("were safeties or off-ball linebackers", W.role.isin(["Safety", "Linebacker"]).sum())
        T("were first-round picks", (W.pick <= 32).sum()); T("of the pass rushers had 25+ quarterback hits", (pr.qbh >= 25).sum(), len(pr)); T("of the pass rushers had 12+ sacks", (pr.sacks >= 12).sum(), len(pr))
        T("of the pass rushers finished top 5 in sacks", (rnk("sacks")[pr.index] <= 5).sum(), len(pr)); T("played for a team with a winning record", (W.wp > .5).sum())
    if kind == "oroy":
        q = W[W.grp == "QB"]; s_ = W[W.grp != "QB"]; T("were quarterbacks", len(q)); T("were wide receivers", (W.grp == "WR").sum()); T("were running backs", (W.grp == "RB").sum()); T("were first-round picks", (W.pick <= 32).sum()); T("were top-12 picks", (W.pick <= 12).sum())
        T("of the quarterbacks started nearly all season (450+ dropbacks)", (q.db >= 450).sum(), len(q)); T("of the non-quarterbacks topped 1,000 yards from scrimmage", (s_.scrim >= 1000).sum(), len(s_))
    if kind == "droy":
        T("were edge rushers", (W.role == "Edge rusher").sum()); T("were off-ball linebackers", (W.role == "Linebacker").sum()); T("were cornerbacks", (W.role == "Cornerback").sum()); T("were first-round picks", (W.pick <= 32).sum()); T("were top-10 picks", (W.pick <= 10).sum())
        T("had 7+ sacks, 100+ tackles or 15+ passes defensed", ((W.sacks >= 7) | (W.tkl >= 100) | (W.pd >= 15)).sum()); T("finished top 3 among rookies in sacks or in tackles", ((rnk("sacks") <= 3) | (rnk("tkl") <= 3)).sum())
    if kind == "coy":
        T("improved the team by 3+ wins on the year before", (W.gain >= 3 / 17).sum()); T("won at least 11 games", (W.wp >= 11 / 17 - .01).sum()); T("took over or inherited a team with a losing record the year before", (W.prev < .5).sum())
        T("improved it by 5+ wins", (W.gain >= 5 / 17).sum()); T("were in their first year with the team", int(W.newc.sum()))
    # ---- this season, and how the ranking has moved since week 2
    track = {}
    for w in range(2, THRU + 1):
        dw, _ = label(kind, zed(frame(kind, PROJ[w], True), FEATS[kind])); te = score(kind, fit(kind, dw[dw.season < SEASON], use), dw[dw.season == SEASON])
        for r in te.itertuples(): track.setdefault(r.pid, []).append([w, int(r.rk)])
        if w == THRU: d26 = te; model = fit(kind, dw[dw.season < SEASON], use)
    wts = sorted(zip(FEATS[kind], model.coef_[0]), key=lambda x: -abs(x[1])); pool26 = d26.copy(); d26 = d26.sort_values("p", ascending=False).head(15); rows = []
    MK = ODDS.get(kind, {}); shown = set(d26.name)
    # market favourites the model leaves off its list: where it has them instead
    mkt_only = []
    for nm, pr in sorted(MK.items(), key=lambda x: x[1])[:8]:
        if nm in shown: continue
        m = pool26[pool26.name == nm]
        mkt_only.append(dict(name=nm, dk=pr, rank=int(m.rk.iloc[0]) if len(m) else None, fit=round(float(m.p.iloc[0]) * 100, 1) if len(m) else None))
    for r in d26.to_dict("records"):
        r = pd.Series(r); ck = checks(kind, r, pool26); h = track.get(r.pid, []); prev = next((x[1] for x in h if x[0] == THRU - 1), None)
        rows.append(dict(name=r["name"], dk=MK.get(r["name"]), team=r.team, pos=(r.role if kind in ("dpoy", "droy") else "" if kind == "coy" else r.pos), fit=round(float(r.p) * 100, 1), rank=int(r.rk),
                         move=None if prev is None else int(prev - r.rk), hist=h, tr=dict(k=int(sum(ck)), n=len(ck), miss=[CHK[kind][i] for i, c in enumerate(ck) if not c]),
                         s={k: (None if val(kind, r, k, fn) is None else round(val(kind, r, k, fn), d)) for k, _, d, fn in SHOW[kind]}, y={k: ytd(kind, r, k) for k, _, _, _ in SHOW[kind]}))
    # ---- past winners: at this point (projected the same way) and at season's end
    Wp = dP[dP.won == 1]; ref = W if kind not in ("opoy", "dpoy") else (W[W.grp != "QB"] if kind == "opoy" else W[W.role.isin(RUSH)]); refp = Wp[Wp.pid.isin(ref.pid)]
    bench = {}
    for k, _, d, fn in SHOW[kind]:
        a = [v for v in (val(kind, r, k, fn) for _, r in refp.iterrows()) if v is not None]; b = [v for v in (val(kind, r, k, fn) for _, r in ref.iterrows()) if v is not None]
        if a and b: bench[k] = dict(now=round(float(np.mean(a)), d), lo=round(float(min(a)), d), hi=round(float(max(a)), d), final=round(float(np.mean(b)), d))
    hist = []
    for yr in sorted(WIN[kind], reverse=True):
        m = dF[(dF.season == yr) & (dF.won == 1)]
        if not len(m): hist.append(dict(season=str(yr), name=WIN[kind][yr], team="", after=None, end=None, s={}, also=VOTES[kind][str(yr)][1:5])); continue
        r = m.iloc[0]; hist.append(dict(season=str(yr), name=r["name"], team=r.team, after=rkP.get(yr), end=rkF.get(yr), s={k: (None if val(kind, r, k, fn) is None else round(val(kind, r, k, fn), d)) for k, _, d, fn in SHOW[kind]}, also=VOTES[kind][str(yr)][1:5]))
    out[kind] = dict(label=LABEL[kind], traits=traits, thru=THRU, val={k: hits(rkP, yrs, k) for k in (1, 3, 5, 10)}, val_n=len(yrs), val_full={k: hits(rkF, yrs, k) for k in (1, 3)}, votes=bool(use),
                     weights=[NICE.get(k, k) for k, v in wts[:3]], checks=CHK[kind], stats=[[k, l, d] for k, l, d, _ in SHOW[kind]], rows=rows, bench=bench, hist=hist,
                     mkt_only=mkt_only, top_hist=TOPSHARE.get(kind), bench_note={"opoy": "non-quarterback winners", "dpoy": "pass-rusher winners"}.get(kind, "past winners"))
    for t in traits: print(f"   {t['k']}/{t['n']} {t['text']}")
    print("   leans on:", [(k, round(float(v), 2)) for k, v in wts], "\n   rank at wk", THRU, rkP)
    for r in rows[:7]: print("  ", r["rank"], r["name"], r["team"], r["pos"], r["fit"], "move", r["move"], f"{r['tr']['k']}/{r['tr']['n']}", r["s"], [h[1] for h in r["hist"]])
    print("   bench:", bench)
json.dump(dict(built=pd.Timestamp.now().strftime("%b %-d, %Y"), thru=THRU, odds_asof=ODDS.get("asof"), odds_book=ODDS.get("book"), awards=out), open("awards.json", "w"), allow_nan=False, default=lambda o: None if pd.isna(o) else float(o))
