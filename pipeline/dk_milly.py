"""Tournament (Milly Maker) DK Classic build for any week, same rules as dk_lineup.py (Oct 18). Needs pulp<3 (pip install "pulp==2.9.0").
Usage: python dk_milly.py <data_wkW.json> <DKSalaries.csv> W <main-slate date MM/DD>
Check every starting QB against the news before trusting a lineup: the projections follow whoever the model has starting."""
import json, re, sys, pandas as pd, pulp
sys.path.insert(0, "/home/claude/nfl-insights/pipeline")
nm = lambda s: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower()))
S = pd.read_csv(sys.argv[2], encoding="utf-8-sig"); W = int(sys.argv[3])
S["time"] = S["Game Info"].str.extract(r"(\d\d:\d\dPM)")[0]; S["game"] = S["Game Info"].str.split(" ").str[0]
S = S[S["Game Info"].str.contains(sys.argv[4]) & S.time.isin(["01:00PM", "04:05PM", "04:25PM"])].copy()
D = json.load(open(sys.argv[1])); P = {nm(p["name"]): p for p in D["proj"]["players"]}
AB = lambda t: "LA" if t == "LAR" else t
imp = {}
for g in D["lines"]["games"]:
    if g.get("spread") is None: continue
    imp[g["home"]] = (g["total"] + g["spread"]) / 2; imp[g["away"]] = (g["total"] - g["spread"]) / 2   # nflverse sign: + = home favored
pb = pd.read_parquet("/home/claude/nfl-insights/pipeline/data/pbp_2026.parquet", columns=["week", "receiver_player_id", "rusher_player_id", "complete_pass", "rush_attempt"]); pb = pb[pb.week < W]
REC = pb.groupby("receiver_player_id").complete_pass.sum(); CAR = pb[pb.rush_attempt == 1].groupby("rusher_player_id").size()
rows = []
for r in S.itertuples():
    if r.Position == "DST":
        opp = [t for t in r.game.split("@") if t != r.TeamAbbrev][0]; oi = imp.get(AB(opp))
        if oi is None or r.Salary > 3000: continue
        pts = 5.0 + (21 - oi) * 0.35; rows.append(dict(name=r.Name, pos="DST", team=AB(r.TeamAbbrev), game=r.game, sal=r.Salary, mean=pts, ceil=pts + 6)); continue
    p = P.get(nm(r.Name))
    if not p or not p.get("dk") or (p.get("p_play") or 1) < 0.85 or AB(r.TeamAbbrev) == "NO": continue
    if (p.get("snap") or 0) < 0.5: continue
    if r.Position in ("WR", "TE") and REC.get(p["pid"], 0) < 8: continue
    if r.Position == "RB" and CAR.get(p["pid"], 0) < 25 and REC.get(p["pid"], 0) < 8: continue
    rows.append(dict(name=r.Name, pos=r.Position, team=AB(r.TeamAbbrev), game=r.game, sal=r.Salary, mean=p["dk"]["pts"], ceil=p["dk"]["ceil"],
                     steady=(p.get("steady") or {}).get("score")))
X = pd.DataFrame(rows).reset_index(drop=True); X["value"] = X["mean"] / (X.sal / 1000)
# chalk proxy: top-3 value at $6k+ and the obvious injury-value plays this week
hi = X[(X.sal >= 6000) & X.pos.isin(["RB", "WR", "TE"])].sort_values("value", ascending=False).head(3).name.tolist()
CHALK = {nm(n) for n in hi + ["Tyler Higbee", "Colby Parkinson"]}
print("chalk proxy:", hi + ["Tyler Higbee", "Colby Parkinson"])
def milly(exclude=None):
    pr = pulp.LpProblem("m", pulp.LpMaximize); v = {i: pulp.LpVariable(f"m{i}", cat="Binary") for i in X.index}
    pr += pulp.lpSum(X.loc[i, "ceil"] * v[i] for i in X.index)
    sal = pulp.lpSum(X.loc[i, "sal"] * v[i] for i in X.index); pr += sal <= 50000; pr += sal >= 49500
    cnt = lambda pos: pulp.lpSum(v[i] for i in X.index if X.loc[i, "pos"] == pos)
    pr += cnt("QB") == 1; pr += cnt("DST") == 1; pr += cnt("RB") == 2; pr += cnt("WR") == 4; pr += cnt("TE") == 1
    for i in X.index[X.pos == "QB"]:
        t, g = X.loc[i, "team"], X.loc[i, "game"]
        mates = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] in ("WR", "TE")]
        oppw = [j for j in X.index if X.loc[j, "game"] == g and X.loc[j, "team"] != t and X.loc[j, "pos"] in ("WR", "TE")]
        rbs = [j for j in X.index if X.loc[j, "team"] == t and X.loc[j, "pos"] == "RB"]
        pr += pulp.lpSum(v[j] for j in mates) >= 2 * v[i]; pr += pulp.lpSum(v[j] for j in mates) <= 2 + 5 * (1 - v[i])
        pr += pulp.lpSum(v[j] for j in oppw) >= v[i]; pr += pulp.lpSum(v[j] for j in oppw) <= 1 + 5 * (1 - v[i])
        pr += pulp.lpSum(v[j] for j in rbs) <= 5 * (1 - v[i])
    for t in X.team.unique():
        pr += pulp.lpSum(v[i] for i in X.index if X.loc[i, "team"] == t) <= 4
        for di in X.index[(X.pos == "DST") & (X.team == t)]:
            for j in X.index[(X.pos != "DST") & (X.game == X.loc[di, "game"]) & (X.team != t)]: pr += v[di] + v[j] <= 1
    pr += pulp.lpSum(v[i] for i in X.index if nm(X.loc[i, "name"]) in CHALK) <= 2
    if exclude is not None: pr += pulp.lpSum(v[i] for i in exclude) <= len(exclude) - 3
    st = pr.solve(pulp.PULP_CBC_CMD(msg=0)); assert pulp.LpStatus[st] == "Optimal", pulp.LpStatus[st]
    sel = X.loc[[i for i in X.index if v[i].value() > 0.5]].copy(); sel["o"] = sel.pos.map({"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 5})
    return sel.sort_values(["o", "sal"], ascending=[True, False])
m1 = milly(); m2 = milly(list(m1.index))
for lab, L in (("LINEUP 1", m1), ("LINEUP 2 (3+ different)", m2)):
    print(f"\n{lab}: salary {L.sal.sum()}, projected {L['mean'].sum():.1f}, ceiling {L.ceil.sum():.1f}")
    print(L[["pos", "name", "team", "game", "sal", "mean", "ceil", "steady"]].round(1).to_string(index=False))
