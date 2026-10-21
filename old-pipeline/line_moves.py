"""Line movement vs the model (Oct 18). Usage: python line_moves.py  -> line_moves.json (feeds the Record report card)
A sportsbook cares where a line CLOSES more than who wins. The test here: when the model's spread or total differed from the line at
the time the model was first published for that week, did the line then move toward the model's number, or away from it?
Toward = the market later agreed with something the model saw first. Away = the market learned something the model didn't have.
Uses the earliest archived copy of each week's site (archive/data_wk<W>_<stamp>.json, Eastern time), the line captured at or
before that time (data/line_history.csv, UTC), and the last line captured before kickoff."""
import glob, json, re
import pandas as pd

h = pd.read_csv("data/line_history.csv"); h["t"] = pd.to_datetime(h.captured_at, utc=True)
sched = pd.read_parquet("data/sched.parquet"); sched = sched[sched.season == 2026].set_index("game_id")
kick = lambda gid: pd.Timestamp(f"{sched.loc[gid].gameday} {sched.loc[gid].gametime}").tz_localize("America/New_York").tz_convert("UTC")
rows = []
for wk in sorted({int(re.search(r"wk(\d+)_", f).group(1)) for f in glob.glob("archive/data_wk*_*.json")}):
    snaps = sorted(glob.glob(f"archive/data_wk{wk}_*.json"))
    first = snaps[0]; st = first.split("_")[-2] + first.split("_")[-1][:4]
    t0 = pd.Timestamp(f"{st[:4]}-{st[4:6]}-{st[6:8]} {st[8:10]}:{st[10:12]}").tz_localize("America/New_York").tz_convert("UTC")
    D = json.load(open(first))
    for g in D["lines"]["games"]:
        gid = g["game_id"]
        if gid not in sched.index: continue
        hh = h[h.game_id == gid].sort_values("t")
        if not len(hh): continue
        then = hh[hh.t <= t0]; then = then.iloc[-1] if len(then) else hh.iloc[0]
        close = hh[hh.t < kick(gid)]
        if not len(close): continue
        close = close.iloc[-1]
        if pd.Timestamp.now(tz="UTC") < kick(gid): continue                 # only games that have kicked off: the close is final
        for mk, mv, lk in (("spread", g["m_margin"], "spread_line"), ("total", g["m_total"], "total_line")):
            l0, l1 = float(then[lk]), float(close[lk])
            if pd.isna(l0) or pd.isna(l1): continue
            gap = mv - l0; move = l1 - l0
            if abs(gap) < 1 or abs(move) < 0.5: res = "no move" if abs(move) < 0.5 else "no gap"
            else: res = "toward" if (move > 0) == (gap > 0) else "away"
            rows.append(dict(week=wk, game_id=gid, market=mk, model=round(mv, 1), line_then=l0, close=l1, gap=round(gap, 1), move=round(move, 1), result=res))
R = pd.DataFrame(rows)
out = dict(rows=R.to_dict("records"))
for mk in ("spread", "total"):
    x = R[(R.market == mk) & (R.result.isin(["toward", "away", "no move"])) & (R.gap.abs() >= 1)] if len(R) else R
    n = int(len(x)); tw = int((x.result == "toward").sum()) if n else 0; aw = int((x.result == "away").sum()) if n else 0
    out[mk] = dict(n=n, toward=tw, away=aw, still=n - tw - aw)
json.dump(out, open("line_moves.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
if len(R): print(R[R.gap.abs() >= 1].to_string(index=False))
