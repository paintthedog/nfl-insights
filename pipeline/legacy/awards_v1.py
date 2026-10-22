"""Award profiles: what the last ten winners of each AP award had in common, how well those traits picked the eventual winner
at this point in past seasons, and who fits best this year. Writes awards.json."""
import json, glob, os, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
SEASON, FIRST = 2026, 2016
WIN = {
 "mvp":  {2016: "Matt Ryan", 2017: "Tom Brady", 2018: "Patrick Mahomes", 2019: "Lamar Jackson", 2020: "Aaron Rodgers", 2021: "Aaron Rodgers", 2022: "Patrick Mahomes", 2023: "Lamar Jackson", 2024: "Josh Allen", 2025: "Matthew Stafford"},
 "opoy": {2016: "Matt Ryan", 2017: "Todd Gurley", 2018: "Patrick Mahomes", 2019: "Michael Thomas", 2020: "Derrick Henry", 2021: "Cooper Kupp", 2022: "Justin Jefferson", 2023: "Christian McCaffrey", 2024: "Saquon Barkley", 2025: "Jaxon Smith-Njigba"},
 "dpoy": {2016: "Khalil Mack", 2017: "Aaron Donald", 2018: "Aaron Donald", 2019: "Stephon Gilmore", 2020: "Aaron Donald", 2021: "T.J. Watt", 2022: "Nick Bosa", 2023: "Myles Garrett", 2024: "Patrick Surtain II", 2025: "Myles Garrett"},
 "oroy": {2016: "Dak Prescott", 2017: "Alvin Kamara", 2018: "Saquon Barkley", 2019: "Kyler Murray", 2020: "Justin Herbert", 2021: "Ja'Marr Chase", 2022: "Garrett Wilson", 2023: "C.J. Stroud", 2024: "Jayden Daniels", 2025: "Tetairoa McMillan"},
 "droy": {2016: "Joey Bosa", 2017: "Marshon Lattimore", 2018: "Shaquille Leonard", 2019: "Nick Bosa", 2020: "Chase Young", 2021: "Micah Parsons", 2022: "Sauce Gardner", 2023: "Will Anderson Jr.", 2024: "Jared Verse", 2025: "Carson Schwesinger"},
 "coy":  {2016: "Jason Garrett", 2017: "Sean McVay", 2018: "Matt Nagy", 2019: "John Harbaugh", 2020: "Kevin Stefanski", 2021: "Mike Vrabel", 2022: "Brian Daboll", 2023: "Kevin Stefanski", 2024: "Kevin O'Connell", 2025: "Mike Vrabel"}}
ALIAS = {"Shaquille Leonard": ["Shaquille Leonard", "Darius Leonard"], "Sauce Gardner": ["Sauce Gardner", "Ahmad Gardner"], "Will Anderson Jr.": ["Will Anderson Jr.", "Will Anderson"], "Patrick Surtain II": ["Patrick Surtain II", "Pat Surtain II", "Patrick Surtain", "Pat Surtain"]}
PL = pd.read_parquet("data/players.parquet").drop_duplicates("gsis_id").set_index("gsis_id")
S = pd.read_parquet("data/sched.parquet"); S = S[(S.game_type == "REG") & (S.season >= FIRST - 1)].copy()
if os.path.exists("coach_fix.json"):
    for yr, fx in json.load(open("coach_fix.json")).items():
        for tm, c in fx.items():
            S.loc[(S.season == int(yr)) & (S.home_team == tm), "home_coach"] = c; S.loc[(S.season == int(yr)) & (S.away_team == tm), "away_coach"] = c
THRU = int(S[(S.season == SEASON) & S.result.notna()].week.max())
C = ["season", "week", "season_type", "game_id", "posteam", "defteam", "play_type", "two_point_attempt", "passer_player_id", "pass_attempt", "sack", "passing_yards", "pass_touchdown", "interception", "qb_epa", "qb_dropback",
     "rusher_player_id", "rushing_yards", "rush_touchdown", "receiver_player_id", "receiving_yards", "complete_pass", "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
     "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id", "qb_hit_1_player_id", "qb_hit_2_player_id", "interception_player_id", "pass_defense_1_player_id", "pass_defense_2_player_id",
     "forced_fumble_player_1_player_id", "forced_fumble_player_2_player_id", "solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id", "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id",
     "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id", "td_player_id", "td_team", "touchdown"]
P = pd.concat([pd.read_parquet(f, columns=C) for f in sorted(glob.glob("data/pbp_20*.parquet"))])
P = P[(P.season_type == "REG") & (P.two_point_attempt != 1)]

def tally(df, col, name, w=1.0, team="defteam"):
    d = df[df[col].notna()]; return d.groupby(["season", col, team]).size().mul(w).rename(name).rename_axis(["season", "pid", "team"])
def build(maxw):
    """player-season and team-season tables from games through week maxw"""
    p = P[P.week <= maxw]
    pa = p[(p.qb_dropback == 1) & p.passer_player_id.notna()].groupby(["season", "passer_player_id", "posteam"]).agg(db=("qb_dropback", "sum"), att=("pass_attempt", lambda x: 0), pass_yds=("passing_yards", "sum"), pass_td=("pass_touchdown", "sum"),
            ints=("interception", "sum"), epa=("qb_epa", "sum"), gp=("game_id", "nunique")).rename_axis(["season", "pid", "team"])
    ru = p[p.rusher_player_id.notna()].groupby(["season", "rusher_player_id", "posteam"]).agg(car=("rushing_yards", "size"), rush_yds=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum"), gr=("game_id", "nunique")).rename_axis(["season", "pid", "team"])
    rc = p[p.receiver_player_id.notna() & (p.play_type == "pass")].groupby(["season", "receiver_player_id", "posteam"]).agg(tgt=("receiving_yards", "size"), rec=("complete_pass", "sum"), rec_yds=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum"), gc=("game_id", "nunique")).rename_axis(["season", "pid", "team"])
    off = pa.join(ru, how="outer").join(rc, how="outer").fillna(0).reset_index()
    off = off.groupby(["season", "pid"]).agg({**{c: "sum" for c in off.columns if c not in ("season", "pid", "team")}, "team": "last"}).reset_index()   # traded players: totals, last team
    off["g"] = off[["gp", "gr", "gc"]].max(axis=1)
    off["scrim"] = off.rush_yds + off.rec_yds; off["td"] = off.rush_td + off.rec_td; off["tot_td"] = off.td + off.pass_td; off["tot_yds"] = off.pass_yds + off.rush_yds
    off["epa_db"] = off.epa / off.db.clip(lower=1); off["int_rate"] = off.ints / off.db.clip(lower=1)
    d = [tally(p, "sack_player_id", "sacks"), tally(p, "half_sack_1_player_id", "sacks", .5), tally(p, "half_sack_2_player_id", "sacks", .5),
         tally(p, "tackle_for_loss_1_player_id", "tfl"), tally(p, "tackle_for_loss_2_player_id", "tfl"), tally(p, "qb_hit_1_player_id", "qbh"), tally(p, "qb_hit_2_player_id", "qbh"),
         tally(p, "interception_player_id", "dint"), tally(p, "pass_defense_1_player_id", "pd"), tally(p, "pass_defense_2_player_id", "pd"),
         tally(p, "forced_fumble_player_1_player_id", "ff"), tally(p, "forced_fumble_player_2_player_id", "ff")] + \
        [tally(p, f"solo_tackle_{i}_player_id", "tkl") for i in (1, 2)] + [tally(p, f"assist_tackle_{i}_player_id", "tkl") for i in (1, 2, 3, 4)] + [tally(p, f"tackle_with_assist_{i}_player_id", "tkl") for i in (1, 2)]
    dtd = p[(p.touchdown == 1) & p.td_player_id.notna() & (p.td_team == p.defteam)]
    d.append(dtd.groupby(["season", "td_player_id", "defteam"]).size().rename("dtd").rename_axis(["season", "pid", "team"]))
    de = pd.concat(d, axis=1).T.groupby(level=0).sum().T.fillna(0).reset_index()
    de = de.groupby(["season", "pid"]).agg({**{c: "sum" for c in ["sacks", "tfl", "qbh", "dint", "pd", "ff", "tkl", "dtd"]}, "team": "last"}).reset_index()
    de = de[~de.pid.map(PL.position_group).isin(["QB", "RB", "WR", "TE", "OL", "SPEC"])]
    de["pd"] = de.pd + 0  # passes defensed already include interceptions in this data only when charted; keep separate
    s = S[(S.week <= maxw) & S.result.notna()]
    tg = pd.concat([s.assign(team=s.home_team, m=s.result, coach=s.home_coach), s.assign(team=s.away_team, m=-s.result, coach=s.away_coach)])
    tm = tg.groupby(["season", "team"]).agg(tg=("m", "size"), w=("m", lambda m: (m > 0).sum() + .5 * (m == 0).sum()), pd_=("m", "sum"), coach=("coach", "last")).reset_index()
    tm["wp"] = tm.w / tm.tg; tm["mpg"] = tm.pd_ / tm.tg
    dep = p[p.play_type.isin(["run", "pass"])].groupby(["season", "defteam"]).qb_epa.mean().rename("def_epa").reset_index().rename(columns={"defteam": "team"})
    tm = tm.merge(dep, on=["season", "team"], how="left")
    return off, de, tm
FULL, PART = build(99), build(THRU)
fin = FULL[2].set_index(["season", "team"]); prev_wp = {(s + 1, t): v for (s, t), v in fin.wp.items()}
prev_coach = {(s + 1, t): v for (s, t), v in fin.coach.items()}

def info(df):
    df = df.copy(); df["name"] = df.pid.map(PL.display_name); df["pos"] = df.pid.map(PL.position); df["grp"] = df.pid.map(PL.position_group)
    df["rookie"] = df.pid.map(PL.rookie_season) == df.season; df["pick"] = df.pid.map(PL.draft_pick).fillna(260)
    return df[df.name.notna()]
def frame(kind, part):
    off, de, tm = PART if part else FULL
    t = tm.set_index(["season", "team"])
    if kind == "coy":
        d = tm.copy(); d["name"] = d.coach; d["pid"] = d.coach; d["prev"] = [prev_wp.get((s, tt), np.nan) for s, tt in zip(d.season, d.team)]
        d["gain"] = d.wp - d.prev; d["newc"] = [float(prev_coach.get((s, tt)) != c) for s, tt, c in zip(d.season, d.team, d.coach)]; d["pos"] = "HC"
        return d[d.season >= FIRST].dropna(subset=["prev"])
    d = info(de if kind in ("dpoy", "droy") else off)
    d = d.merge(tm[["season", "team", "wp", "mpg", "def_epa", "tg"]], on=["season", "team"], how="left"); d = d[d.season >= FIRST]
    if kind == "mvp": d = d[(d.grp == "QB") & (d.db >= d.tg * 14)]
    if kind == "opoy": d["fp"] = d.pass_yds / 25 + d.pass_td * 4 + d.scrim / 10 + d.td * 6 + d.rec * .5; d = d[d.groupby("season").fp.rank(ascending=False) <= (100 if part else 60)]; d["is_qb"] = (d.grp == "QB").astype(float)
    if kind == "oroy": d = d[d.rookie & d.grp.isin(["QB", "RB", "WR", "TE"])]; d["fp"] = d.pass_yds / 25 + d.pass_td * 4 + d.scrim / 10 + d.td * 6; d = d[d.groupby("season").fp.rank(ascending=False) <= (40 if part else 25)]; d["is_qb"] = (d.grp == "QB").astype(float); d["lpick"] = -np.log(d.pick)
    if kind in ("dpoy", "droy"):
        d["imp"] = d.sacks * 2 + d.tfl + d.qbh * .5 + d.dint * 3 + d.pd + d.ff * 2 + d.tkl * .15 + d.dtd * 4
        if kind == "droy": d = d[d.rookie]; d["lpick"] = -np.log(d.pick)
        d = d[d.groupby("season").imp.rank(ascending=False) <= ((40 if part else 25) if kind == "droy" else (200 if part else 150))]
        d["edge"] = d.pos.isin(["DE", "OLB", "DT", "NT", "DL", "EDGE"]).astype(float) if "pos" in d else 0.0
    return d
FEATS = {"mvp": ["wp", "pass_td", "tot_td", "tot_yds", "epa_db", "int_rate"], "opoy": ["scrim", "rec_yds", "rush_yds", "td", "pass_td", "pass_yds", "wp", "is_qb"],
         "dpoy": ["sacks", "tfl", "qbh", "dint", "pd", "ff", "wp", "def_epa"], "oroy": ["pass_yds", "scrim", "tot_td", "wp", "is_qb", "lpick"], "droy": ["sacks", "tfl", "tkl", "dint", "pd", "lpick"],
         "coy": ["wp", "gain", "newc", "mpg"]}
def zed(d, feats):
    d = d.copy()
    for f in feats: g = d.groupby("season")[f]; d["z_" + f] = ((d[f] - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0)
    return d
def winner_ids(kind, d):
    out = {}
    for yr, nm in WIN[kind].items():
        m = d[(d.season == yr) & d.name.isin(ALIAS.get(nm, [nm]))]
        if len(m): out[yr] = m.sort_values(FEATS[kind][0], ascending=False).pid.iloc[0]
        else: print('   unmatched', kind, yr, nm)
    return out
def fit_predict(kind, d, test_season):
    f = ["z_" + x for x in FEATS[kind]]; tr = d[(d.season != test_season) & (d.season < SEASON)]; te = d[d.season == test_season].copy()
    m = LogisticRegression(C=0.3, max_iter=500).fit(tr[f], tr.won); s = m.decision_function(te[f]); e = np.exp(s - s.max()); te["p"] = e / e.sum(); return te, m

LABEL = {"mvp": "Most Valuable Player", "opoy": "Offensive Player of the Year", "dpoy": "Defensive Player of the Year", "oroy": "Offensive Rookie of the Year", "droy": "Defensive Rookie of the Year", "coy": "Coach of the Year"}
NICE = {"wp": "team win rate", "pass_td": "passing touchdowns", "tot_td": "total touchdowns", "tot_yds": "passing + rushing yards", "epa_db": "efficiency per dropback", "int_rate": "interception rate", "scrim": "yards from scrimmage",
        "rec_yds": "receiving yards", "rush_yds": "rushing yards", "td": "touchdowns", "pass_yds": "passing yards", "sacks": "sacks", "tfl": "tackles for loss", "qbh": "quarterback hits", "dint": "interceptions", "pd": "passes defensed",
        "ff": "forced fumbles", "tkl": "tackles", "gain": "improvement on last season", "mpg": "point margin per game", "def_epa": "team defense"}
out = {}
for kind in WIN:
    res = {}
    for part in (False, True):
        d = zed(frame(kind, part), FEATS[kind]); wid = winner_ids(kind, d)
        d["won"] = [int(wid.get(s) == p) for s, p in zip(d.season, d.pid)]
        ranks = {}
        for yr in WIN[kind]:
            if yr not in wid: ranks[yr] = None; continue
            te, _ = fit_predict(kind, d, yr); te["rk"] = te.p.rank(ascending=False); ranks[yr] = int(te[te.won == 1].rk.iloc[0])
        res[part] = (d, wid, ranks)
    dF, widF, rkF = res[False]; dP, widP, rkP = res[True]
    # ---- what winners had in common (full season), measured as rank among that season's candidates
    W = dF[dF.won == 1]; traits = []; n = len(W)
    def rk(col, asc=False): return dF.groupby("season")[col].rank(ascending=asc, method="min")[W.index]
    def T(txt, k): traits.append(dict(text=txt, k=int(k), n=n))
    lr = lambda col: dF[col].where(dF.groupby("season")[col].rank(ascending=False, method="min") == 1).groupby(dF.season).max()
    if kind == "mvp":
        T("were quarterbacks", n); T("played for a team that won at least 75% of its games (13+ wins)", (W.wp >= .75).sum()); T("played for a team that won at least 70% (12+ wins)", (W.wp >= .70).sum())
        T("finished top 3 in passing touchdowns", (rk("pass_td") <= 3).sum()); T("finished top 3 in total touchdowns, passing plus rushing", (rk("tot_td") <= 3).sum()); T("finished top 3 in efficiency per dropback", (rk("epa_db") <= 3).sum())
        T("accounted for 40 or more touchdowns", (W.tot_td >= 40).sum()); T("threw 10 or fewer interceptions", (W.ints <= 10).sum())
    if kind == "opoy":
        T("were not quarterbacks", (W.grp != "QB").sum()); T("were running backs", (W.grp == "RB").sum()); T("were wide receivers", (W.grp == "WR").sum())
        T("led the league in rushing yards, receiving yards or yards from scrimmage", ((rk("rush_yds") == 1) | (rk("rec_yds") == 1) | (rk("scrim") == 1)).sum())
        nq = W[W.grp != "QB"]; T("of the non-quarterbacks gained 1,700+ yards from scrimmage", (nq.scrim >= 1700).sum()); traits[-1]["n"] = len(nq); T("of the non-quarterbacks scored 12+ touchdowns", (nq.td >= 12).sum()); traits[-1]["n"] = len(nq)
        T("played for a team with a winning record", (W.wp > .5).sum())
    if kind == "dpoy":
        T("were pass rushers (edge or interior line)", W.grp.isin(["DL", "LB"]).sum() - 0); T("were cornerbacks", (W.grp == "DB").sum())
        pr = W[W.grp != "DB"]; T("of the pass rushers had 12+ sacks", (pr.sacks >= 12).sum()); traits[-1]["n"] = len(pr); T("of the pass rushers finished top 5 in sacks", (rk("sacks")[pr.index] <= 5).sum()); traits[-1]["n"] = len(pr)
        T("of the pass rushers had 25+ quarterback hits", (pr.qbh >= 25).sum()); traits[-1]["n"] = len(pr)
        T("played for a team with a winning record", (W.wp > .5).sum()); T("were first-round picks", (W.pick <= 32).sum())
    if kind == "oroy":
        T("were quarterbacks", (W.grp == "QB").sum()); T("were wide receivers", (W.grp == "WR").sum()); T("were running backs", (W.grp == "RB").sum()); T("were first-round picks", (W.pick <= 32).sum()); T("were top-12 picks", (W.pick <= 12).sum())
        q = W[W.grp == "QB"]; T("of the quarterbacks started nearly all season (450+ dropbacks)", (q.db >= 450).sum()); traits[-1]["n"] = len(q)
        s_ = W[W.grp != "QB"]; T("of the non-quarterbacks led all rookies in yards from scrimmage", (dF[dF.grp != "QB"].groupby("season").scrim.rank(ascending=False)[s_.index] == 1).sum()); traits[-1]["n"] = len(s_)
        T("of the non-quarterbacks topped 1,000 yards from scrimmage", (s_.scrim >= 1000).sum()); traits[-1]["n"] = len(s_)
    if kind == "droy":
        T("were pass rushers or off-ball linebackers", W.grp.isin(["DL", "LB"]).sum()); T("were cornerbacks", (W.grp == "DB").sum()); T("were first-round picks", (W.pick <= 32).sum()); T("were top-10 picks", (W.pick <= 10).sum())
        T("finished top 3 among rookies in sacks or in tackles", ((rk("sacks") <= 3) | (rk("tkl") <= 3)).sum()); T("had 7+ sacks, 100+ tackles or 15+ passes defensed", ((W.sacks >= 7) | (W.tkl >= 100) | (W.pd >= 15)).sum())
    if kind == "coy":
        T("improved the team's win rate by 3+ wins on the year before", (W.gain >= 3 / 17).sum()); T("improved it by 5+ wins", (W.gain >= 5 / 17).sum()); T("were in their first year with the team", int(W.newc.sum()))
        T("won at least 11 games", (W.wp >= 11 / 17 - .01).sum()); T("took over or inherited a team that had a losing record the year before", (W.prev < .5).sum())
    # ---- how the same reading did at this point in past seasons
    seasons = [y for y in WIN[kind] if y in widF]; val = {k: sum((rkP.get(y) or 999) <= k for y in seasons) for k in (1, 3, 5, 10)}
    valF = {k: sum(rkF[y] <= k for y in WIN[kind] if rkF.get(y)) for k in (1, 3)}
    # ---- this season
    d26 = dP[dP.season == SEASON].copy(); f = ["z_" + x for x in FEATS[kind]]; tr = dP[dP.season < SEASON]
    m = LogisticRegression(C=0.3, max_iter=500).fit(tr[f], tr.won); s_ = m.decision_function(d26[f]); e = np.exp(s_ - s_.max()); d26["p"] = e / e.sum()
    wts = sorted(zip(FEATS[kind], m.coef_[0]), key=lambda x: -abs(x[1]))
    scale = 17 / THRU
    def checks(r, pool):
        g = lambda c: float(r.get(c, 0) or 0); rk_ = lambda c: int((pool[c] > g(c)).sum()) + 1; k = scale
        if kind == "mvp": return [g("wp") >= .70, rk_("pass_td") <= 3, rk_("tot_td") <= 3, rk_("epa_db") <= 3, g("tot_td") * k >= 40, g("ints") * k <= 10]
        if kind == "opoy": return [r.grp != "QB", min(rk_("rush_yds"), rk_("rec_yds"), rk_("scrim")) == 1, g("scrim") * k >= 1700, g("td") * k >= 12, g("wp") > .5]
        if kind == "dpoy": return [r.grp != "DB", g("sacks") * k >= 12, rk_("sacks") <= 5, g("qbh") * k >= 25, g("wp") > .5, r.pick <= 32]
        if kind == "oroy": return [r.pick <= 32, r.pick <= 12, (g("db") * k >= 450) if r.grp == "QB" else (g("scrim") * k >= 1000)]
        if kind == "droy": return [r.pick <= 32, r.pick <= 10, min(rk_("sacks"), rk_("tkl")) <= 3, g("sacks") * k >= 7 or g("tkl") * k >= 100 or g("pd") * k >= 15]
        return [g("gain") >= 3 / 17, g("gain") >= 5 / 17, g("newc") == 1, g("wp") >= 11 / 17 - .01, g("prev") < .5]
    CHK = {"mvp": ["Team winning 70%+", "Top 3 in passing TDs", "Top 3 in total TDs", "Top 3 in efficiency", "On pace for 40+ TDs", "On pace for 10 or fewer INTs"],
           "opoy": ["Not a quarterback", "Leads the league in rushing, receiving or scrimmage yards", "On pace for 1,700+ scrimmage yards", "On pace for 12+ TDs", "Team has a winning record"],
           "dpoy": ["Pass rusher", "On pace for 12+ sacks", "Top 5 in sacks", "On pace for 25+ QB hits", "Team has a winning record", "First-round pick"],
           "oroy": ["First-round pick", "Top-12 pick", "QB on pace to start all year, or 1,000+ scrimmage yards"], "droy": ["First-round pick", "Top-10 pick", "Top 3 among rookies in sacks or tackles", "On pace for 7+ sacks, 100+ tackles or 15+ passes defensed"],
           "coy": ["On pace to improve by 3+ wins", "On pace to improve by 5+ wins", "First year with the team", "On pace for 11+ wins", "Team had a losing record last year"]}
    pool26 = d26.copy()
    d26 = d26.sort_values("p", ascending=False).head(15)
    def row(r, pace=True):
        k = scale if pace else 1; g = lambda c: float(r.get(c, 0) or 0)
        if kind == "mvp": st = [f"{int(round(g('w')))}-{int(round(g('tg') - g('w')))}" if "w" in r else pct(g("wp")), round(g("pass_yds") * k), round(g("tot_td") * k), round(g("ints") * k), round(g("epa_db"), 2)]
        elif kind == "opoy": st = [round(g("scrim") * k), round(g("pass_yds") * k) if r.grp == "QB" else None, round((g("tot_td") if r.grp == "QB" else g("td")) * k), round(g("rec") * k) if r.grp != "QB" and r.grp != "RB" else None, pct(g("wp"))]
        elif kind == "dpoy": st = [round(g("sacks") * k, 1), round(g("tfl") * k), round(g("qbh") * k), round(g("dint") * k), round(g("pd") * k), round(g("tkl") * k)]
        elif kind == "oroy": st = [int(r.pick) if r.pick < 260 else None, round(g("pass_yds") * k) if r.grp == "QB" else None, round(g("scrim") * k), round(g("tot_td") * k), pct(g("wp"))]
        elif kind == "droy": st = [int(r.pick) if r.pick < 260 else None, round(g("sacks") * k, 1), round(g("tkl") * k), round(g("dint") * k), round(g("pd") * k)]
        else: st = [f"{g('w'):g}-{g('tg') - g('w'):g}", f"{prev_wp.get((int(r.season), r.team), 0) * 17:.0f}", f"{(g('wp') - g('prev')) * 17:+.1f}", "Yes" if g("newc") else "No", round(g("mpg"), 1)]
        return st
    pct = lambda v: f"{v * 100:.0f}%"
    tmrec = PART[2].set_index(["season", "team"])
    COLS = {"mvp": ["Team record", "Pass yds pace", "Total TD pace", "INT pace", "EPA per dropback"], "opoy": ["Scrimmage yds pace", "Pass yds pace", "TD pace", "Receptions pace", "Team win rate"],
            "dpoy": ["Sacks pace", "TFL pace", "QB hits pace", "INT pace", "Passes defensed pace", "Tackles pace"], "oroy": ["Draft pick", "Pass yds pace", "Scrimmage yds pace", "TD pace", "Team win rate"],
            "droy": ["Draft pick", "Sacks pace", "Tackles pace", "INT pace", "Passes defensed pace"], "coy": ["Record", "Wins last year", "Pace vs last year", "First year with team", "Point margin / game"]}
    rows = []
    for r in d26.itertuples(index=False):
        r = pd.Series(r._asdict())
        if kind == "mvp": r["w"] = tmrec.w.get((SEASON, r.team), 0); r["tg"] = tmrec.tg.get((SEASON, r.team), THRU)
        ck = checks(r, pool26); rows.append([r["name"], r.team, r.pos if kind != "coy" else "", round(float(r.p) * 100, 1), dict(k=int(sum(ck)), n=len(ck), miss=[CHK[kind][i] for i, c in enumerate(ck) if not c])] + row(r))
    # ---- winners table: final numbers and where the model had them at this point
    hist = []
    for yr in sorted(WIN[kind], reverse=True):
        if yr not in widF: hist.append([str(yr), WIN[kind][yr], "", None, None] + [None] * len(COLS[kind])); continue
        r = dF[(dF.season == yr) & (dF.pid == widF[yr])].iloc[0].copy()
        if kind == "mvp": r["w"] = fin.w.get((yr, r.team), 0); r["tg"] = fin.tg.get((yr, r.team), 17)
        hist.append([str(yr), r["name"], r.team, rkP.get(yr), rkF.get(yr)] + row(r, pace=False))
    out[kind] = dict(label=LABEL[kind], traits=traits, thru=THRU, val=val, val_n=len(seasons), val_full=valF, weights=[[NICE.get(k, k if k != "is_qb" else "being a quarterback") if k not in ("lpick", "newc", "edge") else {"lpick": "draft position", "newc": "first year with the team"}[k], round(float(v), 2)] for k, v in wts],
                     checks=CHK[kind], cols=["Player" if kind != "coy" else "Coach", "Team", "Pos", "Fit", "Traits matched"] + COLS[kind], rows=rows,
                     hist_cols=["Season", "Winner", "Team", f"Model rank after week {THRU}", "Model rank at season's end"] + [c.replace(" pace", "").replace("Pace vs last year", "Wins vs last year") for c in COLS[kind]], hist=hist)
    print(f"\n== {LABEL[kind]}  thru wk{THRU} top1/3/5/10: {val} of {len(seasons)} | full season top1/3: {valF}")
    for t in traits: print(f"   {t['k']}/{t['n']} {t['text']}")
    print("   weights:", [(k, round(v, 2)) for k, v in wts]); print("   ranks at wk", THRU, rkP)
    for r in rows[:6]: print("  ", r[:4], r[4]['k'], '/', r[4]['n'], r[5:])
json.dump(dict(built=pd.Timestamp.now().strftime("%b %-d, %Y"), thru=THRU, awards=out), open("awards.json", "w"), allow_nan=False, default=lambda o: None if pd.isna(o) else float(o))
