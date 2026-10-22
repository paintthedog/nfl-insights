"""Grade a week's paper slates (bets_wk{W}*.csv) against nflverse pbp. Flat 1u; did-not-play = push (void); unfinished games = pending."""
import re, sys, pandas as pd, numpy as np, nflreadpy as nfl
W = int(sys.argv[1]); FILES = sys.argv[2:] or sorted(f for f in __import__("glob").glob(f"bets_wk{W}*.csv") if "_graded" not in f)
p = nfl.load_pbp([2026]).to_pandas(); p = p[p.week == W]
done = set(p.groupby("game_id").desc.last().loc[lambda s: s.str.contains("END GAME", na=False)].index)
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", str(s).lower().replace("'", "")))).strip()
NAME = {}
for g, n in zip(pl.gsis_id, pl.display_name): NAME.setdefault(norm(n), []).append(g)
np2 = p[p.two_point_attempt.ne(1)]
stat = {
 "pass_yds": np2.groupby("passer_player_id").passing_yards.sum(),
 "rush_yds": np2.groupby("rusher_player_id").rushing_yards.sum(),
 "rec_yds": np2.groupby("receiver_player_id").receiving_yards.sum(),
 "rec": np2.groupby("receiver_player_id").complete_pass.sum(),
}
tds = np2[np2.touchdown.eq(1) & np2.td_player_id.notna()].groupby("td_player_id").size()
try:
    sc = nfl.load_snap_counts([2026]).to_pandas(); sc = sc[sc.week == W]
    pfr = pd.read_parquet("data/players.parquet", columns=["gsis_id", "pfr_id"]).dropna()
    snapped = set(sc.merge(pfr, left_on="pfr_player_id", right_on="pfr_id").query("offense_snaps > 0 or st_snaps > 0").gsis_id)
except Exception as e:
    snapped = set()
touched = set(np2.passer_player_id.dropna()) | set(np2.rusher_player_id.dropna()) | set(np2.receiver_player_id.dropna()) | set(np2.td_player_id.dropna())
gid_of = lambda g: f"2026_{W:02d}_{g}"
def pid_for(name, gcode):
    c = NAME.get(norm(name), [])
    if len(c) == 1: return c[0]
    game = p[p.game_id == gid_of(gcode)] if gcode else p
    ids = set(game.passer_player_id.dropna()) | set(game.rusher_player_id.dropna()) | set(game.receiver_player_id.dropna())
    hit = [x for x in c if x in ids]
    return hit[0] if hit else (c[0] if c else None)
# final scores per game
fin = p.sort_values("play_id").groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"), hs=("total_home_score", "last"), as_=("total_away_score", "last"))
MK = {"reception yds": "rec_yds", "rec yds": "rec_yds", "receptions": "rec", "rec": "rec", "rush yds": "rush_yds", "pass yds": "pass_yds"}
def grade(r):
    b, cat = r["bet"], r["cat"]
    gcode = r["game"] if isinstance(r["game"], str) else None
    if cat in ("TD", "OU"):
        if gid_of(gcode) not in done: return "pending", None
        m = re.match(r"(.+?) anytime TD", b) if cat == "TD" else re.match(r"(.+?) (Over|Under) ([\d.]+) (.+)$", b)
        name = m.group(1); pid = pid_for(name, gcode)
        played = pid in touched or pid in snapped
        if not played: return "void", None
        if cat == "TD":
            n = int(tds.get(pid, 0)); return ("win" if n > 0 else "loss"), n
        side, line, mk = m.group(2), float(m.group(3)), MK[m.group(4).strip()]
        v = float(stat[mk].get(pid, 0))
        if v == line: return "void", v
        return ("win" if (v > line) == (side == "Over") else "loss"), v
    m = re.search(r"\((\w+)@(\w+)\)|(\w+) at (\w+)", b)
    if cat == "teaser leg":
        mm = re.match(r"(\w+) \+?([\d.]+) teased to \+([\d.]+) (vs|at) (\w+)", b)
        team, adj, opp = mm.group(1), float(mm.group(3)), mm.group(5)
        g = [x for x in fin.index if team in x.split("_")[2:] and opp in x.split("_")[2:]]
    else:
        a, h = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        g = [x for x in fin.index if x.endswith(f"{a}_{h}")]
    if not g or g[0] not in done: return "pending", None
    f = fin.loc[g[0]]; tot = f.hs + f.as_
    if cat in ("total", "prime under"):
        side, line = re.search(r"(Over|Under) ([\d.]+)", b).groups(); line = float(line)
        if tot == line: return "void", tot
        return ("win" if (tot > line) == (side == "Over") else "loss"), int(tot)
    if cat == "spread":
        team, ln = re.match(r"(\w+) \+?(-?[\d.]+)", b).groups(); ln = float(ln)
        mrg = (f.hs - f.as_) if team == f.home else (f.as_ - f.hs)
        if mrg + ln == 0: return "void", mrg
        return ("win" if mrg + ln > 0 else "loss"), int(mrg)
    if cat == "teaser leg":
        mrg = (f.hs - f.as_) if team == f.home else (f.as_ - f.hs)
        return ("win" if mrg + adj > 0 else ("void" if mrg + adj == 0 else "loss")), int(mrg)
for fn in FILES:
    d = pd.read_csv(fn)
    res = d.apply(lambda r: pd.Series(grade(r), index=["result", "actual"]), axis=1)
    d = pd.concat([d, res], axis=1)
    d["pnl"] = np.select([d.result == "win", d.result == "loss"], [d.price - 1, -1.0], 0.0)
    d.to_csv(fn.replace(".csv", "_graded.csv"), index=False)
    g = d[d.result.isin(["win", "loss"])]
    print(f"\n=== {fn}: graded {len(g)}, pending {(d.result=='pending').sum()}, void {(d.result=='void').sum()}")
    print(f"record {(g.result=='win').sum()}-{(g.result=='loss').sum()}  P&L {g.pnl.sum():+.2f}u on {len(g)}u  ROI {g.pnl.sum()/max(len(g),1):+.1%}  | expected wins {g.model_p.sum():.1f}")
    print(d.groupby("cat").apply(lambda x: pd.Series(dict(n=x.result.isin(['win','loss']).sum(), w=(x.result=='win').sum(), pnl=round(x.pnl.sum(),2), exp_w=round(x.loc[x.result.isin(['win','loss']),'model_p'].sum(),1)))).to_string())
    if "news" in d: print("by news flag:", g.groupby("news").pnl.agg(['count','sum']).round(2).to_dict('index'))
