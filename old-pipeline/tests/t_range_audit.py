"""Floors/ceilings audit (Oct 16): published projection ranges vs what happened. Usage: python tests/t_range_audit.py 6 [7 ...]
Uses the last archived pre-game snapshot (archive/data_wk<W>_<stamp>.json) taken before each game's kickoff day.
A calibrated range puts ~10% of actual results below the 10th percentile and ~10% above the 90th."""
import sys; sys.path.insert(0, ".")
import json, numpy as np, pandas as pd, proj_model as P
pbp = P.load(); pg = P.player_games(pbp)[0]
sched = pd.read_parquet("data/sched.parquet")
QI = {q: i for i, q in enumerate(P.QUANTS)}
rows = []
import glob
WEEKS = [int(x) for x in sys.argv[1:]] or [6]
for wk in WEEKS:
    sw = sched[(sched.season == 2026) & (sched.week == wk)]
    snaps = sorted(glob.glob(f"archive/data_wk{wk}_*.json"))
    for gday, gs in sw.groupby("gameday"):            # each game day graded off the last snapshot taken before that day
        pre = [f for f in snaps if f.split("_")[-2] < gday.replace("-", "")]
        if not pre: print("no pre-game snapshot for", wk, gday); continue
        gids = set(gs.game_id); act = pg[pg.game_id.isin(gids)].set_index("pid")
        for p in json.load(open(pre[-1]))["proj"]["players"]:
            if p["pid"] not in act.index: continue
            a = act.loc[p["pid"]]; a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
            for st in ("tgt", "rec", "rec_yds", "car", "rush_yds", "pass_yds", "cmp"):
                q = p.get("q", {}).get(st)
                if not q or st not in a or max(q) <= 0: continue
                if st in ("pass_yds", "cmp") and p["pos"] != "QB": continue
                if st in ("tgt", "rec", "rec_yds") and p["pos"] == "QB": continue
                v = float(a[st]); rows.append(dict(wk=wk, pos=p["pos"], stat=st, v=v, q10=q[QI[0.1]], q25=q[QI[0.25]], q50=q[QI[0.5]], q75=q[QI[0.75]], q90=q[QI[0.9]]))
r = pd.DataFrame(rows)
# ties at discrete values (e.g. 0 catches) count half, so a range bottoming out at 0 isn't penalised for it
f = lambda v, q, below: np.where(v < q, 1.0, np.where(v == q, 0.5, 0.0)) if below else np.where(v > q, 1.0, np.where(v == q, 0.5, 0.0))
r["b10"] = f(r.v, r.q10, True); r["a90"] = f(r.v, r.q90, False); r["in50"] = ((r.v >= r.q25) & (r.v <= r.q75)).astype(float); r["b50"] = f(r.v, r.q50, True)
out = r.groupby("stat").agg(n=("v", "size"), below_floor=("b10", "mean"), above_ceiling=("a90", "mean"), in_middle_half=("in50", "mean"), below_median=("b50", "mean")).round(3)
out2 = r.groupby("pos").agg(n=("v", "size"), below_floor=("b10", "mean"), above_ceiling=("a90", "mean"), in_middle_half=("in50", "mean"), below_median=("b50", "mean")).round(3)
pd.set_option("display.width", 200); print("target: below_floor 0.10, above_ceiling 0.10, in_middle_half 0.50, below_median 0.50\n"); print(out); print(); print(out2)
print("\nall:", r[["b10", "a90", "in50", "b50"]].mean().round(3).to_dict(), "n", len(r))
