"""DraftKings Classic lineup for the Sunday 1pm + 4pm slate (Oct 16). Projections = site DK points (mean & 90th pct);
team implied totals from DraftKings game lines; one lineup max-mean (cash), one max-ceiling with QB stack (tournament)."""
import json, re, numpy as np, pandas as pd, pulp
nm = lambda s: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower()))
S = pd.read_csv("/root/.claude/uploads/61a3e72b-dbd0-5c34-8049-91e3f7bfb7b5/727a0c71-DKSalaries.csv", encoding="utf-8-sig")
S["time"] = S["Game Info"].str.extract(r"(\d\d:\d\dPM)")[0]; S["game"] = S["Game Info"].str.split(" ").str[0]
S = S[S.time.isin(["01:00PM", "04:05PM", "04:25PM"])].copy()
D = json.load(open("site/data_wk6.json")); P = {nm(p["name"]): p for p in D["proj"]["players"]}
# implied team totals from Friday DK lines (market_log)
m = pd.read_csv("market_log.csv"); m = m[m.week == 6]
FULL = {"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CLE":"Cleveland Browns",
 "IND":"Indianapolis Colts","KC":"Kansas City Chiefs","LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","NE":"New England Patriots","NO":"New Orleans Saints",
 "NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans"}
imp = {}
for g, x in m.groupby("game_id"):
    tot = x[(x.market == "game_totals")].point.dropna().iloc[0]
    for _, r in x[x.market == "game_spreads"].iterrows(): imp[r.subject] = (tot - r.point) / 2
AB = lambda t: "LA" if t == "LAR" else t
S["implied"] = S.TeamAbbrev.map(lambda t: imp.get(AB(t)))
rows = []
for r in S.itertuples():
    if r.Position == "DST":
        opp = [t for t in r.game.split("@") if t != r.TeamAbbrev][0]; oi = imp.get(AB(opp))
        if oi is None: continue
        pts = 5.0 + (21 - oi) * 0.35            # rough DST points: league-average ~5, better the lower the opponent's implied total
        rows.append(dict(name=r.Name, pos="DST", team=r.TeamAbbrev, game=r.game, sal=r.Salary, mean=pts, ceil=pts + 6, implied=r.implied, pplay=1.0)); continue
    p = P.get(nm(r.Name))
    if not p or not p.get("dk") or (p.get("p_play") or 1) < 0.85: continue   # skip uncertain players: a 0 kills a lineup
    rows.append(dict(name=r.Name, pos=r.Position, team=r.TeamAbbrev, game=r.game, sal=r.Salary, mean=p["dk"]["pts"], ceil=p["dk"]["ceil"], implied=r.implied, pplay=p.get("p_play") or 1))
X = pd.DataFrame(rows).reset_index(drop=True); X["value"] = X["mean"] / (X.sal / 1000)
def build(obj, stack=False, exclude=None):
    pr = pulp.LpProblem("dk", pulp.LpMaximize); v = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, obj] * v[i] for i in X.index)
    pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) <= 50000
    pr += pulp.lpSum(v.values()) == 9
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1
    pr += cnt("RB") >= 2; pr += cnt("WR") >= 3; pr += cnt("TE") >= 1
    pr += cnt("RB") + cnt("WR") + cnt("TE") == 7
    for t in X.team.unique():   # never roster a DST against your own players
        d = [i for i in X.index if X.loc[i, "pos"] == "DST" and X.loc[i, "team"] == t]
        opp_pl = [i for i in X.index if X.loc[i, "pos"] != "DST" and X.loc[i, "game"] == (X.loc[d[0], "game"] if d else None) and X.loc[i, "team"] != t]
        for di in d:
            for j in opp_pl: pr += v[di] + v[j] <= 1
    if stack:                   # QB plus at least one of his own pass catchers (the one correlation that held up)
        for i in X.index[X.pos == "QB"]:
            mates = [j for j in X.index if X.loc[j, "team"] == X.loc[i, "team"] and X.loc[j, "pos"] in ("WR", "TE")]
            pr += pulp.lpSum(v[j] for j in mates) >= v[i]
    pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == "PHI") <= 1   # Clinton: at most one Eagle
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 3
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal", pulp.LpStatus[st]
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy()
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}; sel["o"] = sel.pos.map(order); return sel.sort_values(["o", "sal"], ascending=[True, False])
pd.set_option("display.width", 200)
a = build("mean", stack=True); b = build("ceil", stack=True, exclude=list(a.index))
for lab, L in (("CASH (highest projection, QB stacked)", a), ("TOURNAMENT (highest ceiling, QB stacked, 3+ players different)", b)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling-sum {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "value", "implied"]].round(1).to_string(index=False))
print("\nimplied team totals:", {k.split()[-1]: round(v, 1) for k, v in sorted(imp.items(), key=lambda t: -t[1])})
print("\nTop value (pts per $1k), 1pm/4pm, $4,500+:"); print(X[(X.sal >= 4500) & (X.pos != "DST")].sort_values("value", ascending=False).head(12)[["pos", "name", "team", "sal", "mean", "value", "implied"]].round(2).to_string(index=False))

# ---- tournament build (Oct 18): max ceiling, QB + 2 of his pass catchers + one bring-back from the opponent, at most one Eagle
def gpp(exclude=None):
    pr = pulp.LpProblem("gpp", pulp.LpMaximize); v = {i: pulp.LpVariable(f"g{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, "ceil"] * v[i] for i in X.index)
    pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) <= 50000; pr += pulp.lpSum(v.values()) == 9
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1; pr += cnt("RB") >= 2; pr += cnt("WR") >= 3; pr += cnt("TE") >= 1; pr += cnt("RB") + cnt("WR") + cnt("TE") == 7
    for i in X.index[X.pos == "QB"]:
        t, g = X.loc[i, "team"], X.loc[i, "game"]
        mates = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] in ("WR", "TE")]
        opp = [j for j in X.index if X.loc[j, "game"] == g and X.loc[j, "team"] != t and X.loc[j, "pos"] in ("WR", "TE", "RB")]
        pr += pulp.lpSum(v[j] for j in mates) >= 2 * v[i]; pr += pulp.lpSum(v[j] for j in opp) >= v[i]
    for t in X.team.unique():
        d = [i for i in X.index if X.loc[i, "pos"] == "DST" and X.loc[i, "team"] == t]
        for di in d:
            for j in [i for i in X.index if X.loc[i, "pos"] != "DST" and X.loc[i, "game"] == X.loc[di, "game"] and X.loc[i, "team"] != t]: pr += v[di] + v[j] <= 1
    pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == "PHI") <= 1
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 3
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal"
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy(); sel["o"] = sel.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}); return sel.sort_values(["o", "sal"], ascending=[True, False])
g1 = gpp(); g2 = gpp(exclude=list(g1.index))
for lab, L in (("GPP 1 (max ceiling, QB + 2 catchers + bring-back)", g1), ("GPP 2 (3+ different)", g2)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling-sum {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "value", "implied"]].round(1).to_string(index=False))

# ---- Oct 18, Clinton: tilt to the highest team totals. Stack from a top-4 implied-total team; every skill player from a team with 24+ implied;
# no player whose projection leans on an injury redistribution (new role, no track record): require last_snap >= 0.5 of a game.
SNAP = {nm(p["name"]): (p.get("last_snap") or 0) for p in D["proj"]["players"]}
X["snap"] = X.name.map(lambda n: SNAP.get(nm(n), 1.0)); X.loc[X.pos == "DST", "snap"] = 1.0
top4 = sorted({AB(t): v for t, v in imp.items()}.items(), key=lambda t: -t[1])[:4]; top4 = {t for t, _ in top4}
def gpp_tt(exclude=None):
    pr = pulp.LpProblem("gpptt", pulp.LpMaximize); v = {i: pulp.LpVariable(f"h{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, "ceil"] * v[i] for i in X.index)
    pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) <= 50000; pr += pulp.lpSum(v.values()) == 9
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1; pr += cnt("RB") >= 2; pr += cnt("WR") >= 3; pr += cnt("TE") >= 1; pr += cnt("RB") + cnt("WR") + cnt("TE") == 7
    for i in X.index:
        if X.loc[i, "pos"] != "DST" and ((X.loc[i, "implied"] or 0) < 24.5 or X.loc[i, "snap"] < 0.5 or X.loc[i, "team"] == "NO"): pr += v[i] == 0
        if X.loc[i, "pos"] == "QB" and AB(X.loc[i, "team"]) not in top4: pr += v[i] == 0
    for i in X.index[X.pos == "QB"]:
        t, g = X.loc[i, "team"], X.loc[i, "game"]
        mates = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] in ("WR", "TE")]
        opp = [j for j in X.index if X.loc[j, "game"] == g and X.loc[j, "team"] != t and X.loc[j, "pos"] in ("WR", "TE", "RB")]
        pr += pulp.lpSum(v[j] for j in mates) >= 2 * v[i]; pr += pulp.lpSum(v[j] for j in opp) >= v[i]
    for t in X.team.unique():
        for di in [i for i in X.index if X.loc[i, "pos"] == "DST" and X.loc[i, "team"] == t]:
            for j in [i for i in X.index if X.loc[i, "pos"] != "DST" and X.loc[i, "game"] == X.loc[di, "game"] and X.loc[i, "team"] != t]: pr += v[di] + v[j] <= 1
    pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == "PHI") <= 1
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 3
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal", pulp.LpStatus[st]
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy(); sel["o"] = sel.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}); return sel.sort_values(["o", "sal"], ascending=[True, False])
t1 = gpp_tt(); t2 = gpp_tt(exclude=list(t1.index))
print("\ntop-4 implied:", top4)
for lab, L in (("GPP high-total 1", t1), ("GPP high-total 2 (3+ different)", t2)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling-sum {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "value", "implied", "snap"]].round(2).to_string(index=False))

# ---- Oct 18: Milly Maker rules (ETR/Stokastic studies): QB + exactly 2 own pass catchers, exactly 1 bring-back (WR preferred),
# full salary, 2 RB / 4 WR / 1 TE (WR in flex), max 4 per team, no NO, no injury-role players, cheap DST, at most one of the mega-chalk pair (Kyren/Puka),
# at most 2 "chalk" players overall (proxy: top-3 value at $6k+ or the obvious injury-value plays).
# Clinton's rule (Oct 18): nobody without a real season record. Pass catchers need 8+ catches, backs 25+ carries, this season.
_pb = pd.read_parquet("data/pbp_2026.parquet", columns=["receiver_player_name", "receiver_player_id", "rusher_player_id", "complete_pass", "rush_attempt"])
_rec = _pb.groupby("receiver_player_id").complete_pass.sum(); _car = _pb[_pb.rush_attempt == 1].groupby("rusher_player_id").size()
PID = {nm(p["name"]): p["pid"] for p in D["proj"]["players"]}
def has_record(name, pos):
    pid = PID.get(nm(name))
    if pos in ("WR", "TE"): return _rec.get(pid, 0) >= 8
    if pos == "RB": return _car.get(pid, 0) >= 25 or _rec.get(pid, 0) >= 8
    return True
X["record"] = [has_record(n, p) for n, p in zip(X.name, X.pos)]
CHALK = {nm(n) for n in ["Kyren Williams", "Puka Nacua", "Dameon Pierce", "Josh Allen", "Christian McCaffrey", "Bijan Robinson"]}
def milly(exclude=None, qb_team=None):
    pr = pulp.LpProblem("milly", pulp.LpMaximize); v = {i: pulp.LpVariable(f"m{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, "ceil"] * v[i] for i in X.index)
    pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) <= 50000; pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) >= 49500
    pr += pulp.lpSum(v.values()) == 9
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1; pr += cnt("RB") == 2; pr += cnt("WR") == 4; pr += cnt("TE") == 1
    for i in X.index:
        if X.loc[i, "pos"] != "DST" and (X.loc[i, "snap"] < 0.5 or not X.loc[i, "record"] or X.loc[i, "team"] == "NO" or nm(X.loc[i, "name"]) in {nm("Dameon Pierce"), nm("Daniel Jones")}): pr += v[i] == 0
        if X.loc[i, "pos"] == "DST" and X.loc[i, "sal"] > 3000: pr += v[i] == 0
        if qb_team and X.loc[i, "pos"] == "QB" and X.loc[i, "team"] != qb_team: pr += v[i] == 0
    # exact stack sizes need per-QB conditionals: mates == 2 when QB chosen, else no restriction beyond team cap
    for i in X.index[X.pos == "QB"]:
        t, g = X.loc[i, "team"], X.loc[i, "game"]
        mates = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] in ("WR", "TE")]
        oppw = [j for j in X.index if X.loc[j, "game"] == g and X.loc[j, "team"] != t and X.loc[j, "pos"] in ("WR", "TE")]
        pr += pulp.lpSum(v[j] for j in mates) >= 2 * v[i]; pr += pulp.lpSum(v[j] for j in mates) <= 2 + 5 * (1 - v[i])
        pr += pulp.lpSum(v[j] for j in oppw) >= v[i]; pr += pulp.lpSum(v[j] for j in oppw) <= 1 + 5 * (1 - v[i])
        rbs = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] == "RB"]
        pr += pulp.lpSum(v[j] for j in rbs) <= 5 * (1 - v[i])   # QB + exactly 2 catchers, no own RB: three from the QB's team, no more
    for t in X.team.unique():
        pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == t) <= 4
        for di in [i for i in X.index if X.loc[i, "pos"] == "DST" and X.loc[i, "team"] == t]:
            for j in [i for i in X.index if X.loc[i, "pos"] != "DST" and X.loc[i, "game"] == X.loc[di, "game"] and X.loc[i, "team"] != t]: pr += v[di] + v[j] <= 1
    pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == "PHI") <= 1
    pr += pulp.lpSum(v[i] for i in X.index if nm(X.loc[i, "name"]) in CHALK) <= 2
    pr += pulp.lpSum(v[i] for i in X.index if nm(X.loc[i, "name"]) in {nm("Kyren Williams"), nm("Puka Nacua")}) <= 1
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 3
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal", pulp.LpStatus[st]
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy(); sel["o"] = sel.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}); return sel.sort_values(["o", "sal"], ascending=[True, False])
m1 = milly(); m2 = milly(exclude=list(m1.index)); m3 = milly(exclude=list(m1.index) + list(m2.index))
for lab, L in (("MILLY 1", m1), ("MILLY 2 (3+ different)", m2), ("MILLY 3", m3)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling-sum {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "value", "implied", "snap"]].round(2).to_string(index=False))

# ---- Clinton's build: Bills stack, must include Kyren Williams and Dallas Goedert
def locked(qb_team, must, exclude=None):
    pr = pulp.LpProblem("lock", pulp.LpMaximize); v = {i: pulp.LpVariable(f"k{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, "ceil"] * v[i] for i in X.index)
    pr += pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index) <= 50000; pr += pulp.lpSum(v.values()) == 9
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1; pr += cnt("RB") >= 2; pr += cnt("WR") >= 3; pr += cnt("TE") >= 1; pr += cnt("RB") + cnt("WR") + cnt("TE") == 7
    for i in X.index:
        if X.loc[i, "pos"] != "DST" and (X.loc[i, "snap"] < 0.5 or not X.loc[i, "record"] or X.loc[i, "team"] == "NO" or nm(X.loc[i, "name"]) in {nm("Dameon Pierce"), nm("Daniel Jones")}): pr += v[i] == 0
        if X.loc[i, "pos"] == "QB" and X.loc[i, "team"] != qb_team: pr += v[i] == 0
        if X.loc[i, "pos"] == "DST" and X.loc[i, "sal"] > 3000: pr += v[i] == 0
        if nm(X.loc[i, "name"]) in {nm(m) for m in must}: pr += v[i] == 1
    for i in X.index[X.pos == "QB"]:
        t, g = X.loc[i, "team"], X.loc[i, "game"]
        mates = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] in ("WR", "TE")]
        oppw = [j for j in X.index if X.loc[j, "game"] == g and X.loc[j, "team"] != t and X.loc[j, "pos"] in ("WR", "TE")]
        rbs = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] == "RB"]
        pr += pulp.lpSum(v[j] for j in mates) >= 2 * v[i]; pr += pulp.lpSum(v[j] for j in mates) <= 2 + 5 * (1 - v[i])
        pr += pulp.lpSum(v[j] for j in oppw) >= v[i]; pr += pulp.lpSum(v[j] for j in oppw) <= 1 + 5 * (1 - v[i]); pr += pulp.lpSum(v[j] for j in rbs) <= 5 * (1 - v[i])
    for t in X.team.unique():
        pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == t) <= 4
        for di in [i for i in X.index if X.loc[i, "pos"] == "DST" and X.loc[i, "team"] == t]:
            for j in [i for i in X.index if X.loc[i, "pos"] != "DST" and X.loc[i, "game"] == X.loc[di, "game"] and X.loc[i, "team"] != t]: pr += v[di] + v[j] <= 1
    pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == "PHI") <= 1
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 2
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal", pulp.LpStatus[st]
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy(); sel["o"] = sel.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5}); return sel.sort_values(["o", "sal"], ascending=[True, False])
k1 = locked("BUF", ["Kyren Williams", "Dallas Goedert"]); k2 = locked("BUF", ["Kyren Williams", "Dallas Goedert"], exclude=list(k1.index))
for lab, L in (("BILLS + KYREN + GOEDERT 1", k1), ("BILLS + KYREN + GOEDERT 2", k2)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling-sum {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "value", "implied"]].round(1).to_string(index=False))
