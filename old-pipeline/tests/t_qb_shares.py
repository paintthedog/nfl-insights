"""Does knowing who is throwing improve receiver target-share predictions? For each team-game, predict each receiver's share of
targets from (a) all recent games (current method, decay 0.80) and (b) a blend that leans on recent games with THIS quarterback."""
import glob, numpy as np, pandas as pd
cols = ["game_id", "season", "week", "season_type", "posteam", "play_type", "receiver_player_id", "passer_player_id", "qb_dropback", "two_point_attempt"]
p = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/pbp_20*.parquet"))]); p = p[(p.season_type == "REG") & (p.two_point_attempt != 1)]
st = p[p.qb_dropback == 1].groupby(["game_id", "posteam", "passer_player_id"]).size().reset_index(name="db").sort_values("db").drop_duplicates(["game_id", "posteam"], keep="last")[["game_id", "posteam", "passer_player_id"]].rename(columns={"passer_player_id": "qb"})
t = p[(p.play_type == "pass") & p.receiver_player_id.notna()].groupby(["game_id", "season", "week", "posteam", "receiver_player_id"]).size().reset_index(name="tgt")
t["sh"] = t.tgt / t.groupby(["game_id", "posteam"]).tgt.transform("sum")
games = t[["game_id", "season", "week", "posteam"]].drop_duplicates().merge(st, on=["game_id", "posteam"]).sort_values(["posteam", "season", "week"])
D = 0.80; rows = []
for (tm, yr), gs in games.groupby(["posteam", "season"]):
    hist = []   # (qb, {receiver: share})
    for r in gs.itertuples():
        actual = t[(t.game_id == r.game_id) & (t.posteam == tm)].set_index("receiver_player_id").sh
        if len(hist) >= 2:
            recs = set().union(*[h[1].keys() for h in hist]) | set(actual.index)
            w_all = np.array([D ** (len(hist) - 1 - i) for i in range(len(hist))]); same = np.array([h[0] == r.qb for h in hist])
            n_same = (w_all * same).sum(); changed = hist[-1][0] != r.qb
            for rc in recs:
                v = np.array([h[1].get(rc, 0.0) for h in hist]); base = (w_all * v).sum() / w_all.sum()
                withq = (w_all * same * v).sum() / n_same if n_same > 0 else base
                rows.append((r.game_id, tm, rc, changed, n_same, base, withq, actual.get(rc, 0.0)))
        hist.append((r.qb, actual.to_dict()))
R = pd.DataFrame(rows, columns=["game_id", "team", "rec", "changed", "n_same", "base", "withq", "actual"])
R = R[R.base >= 0.03]
for K in (0.5, 1, 2, 3, 5):
    R["blend"] = (R.n_same * R.withq + K * R.base) / (R.n_same + K)
    a = R; c = R[R.changed & (R.n_same > 0)]; c0 = R[R.changed]
    print(f"K={K}: all games MAE base {np.abs(a.base - a.actual).mean():.4f} blend {np.abs(a.blend - a.actual).mean():.4f} | after a QB change, new QB has thrown for them before (n={len(c)}): base {np.abs(c.base - c.actual).mean():.4f} blend {np.abs(c.blend - c.actual).mean():.4f} | any QB change (n={len(c0)}) base {np.abs(c0.base - c0.actual).mean():.4f} blend {np.abs(c0.blend - c0.actual).mean():.4f}")
# does a QB change shift shares toward/away from the top targets (a concentration effect rather than a QB-specific one)?
for nm, sub in (("no change", R[~R.changed]), ("QB change", R[R.changed])):
    b = pd.cut(sub.base, [0.03, 0.10, 0.18, 0.25, 1]); print(nm, (sub.actual - sub.base).groupby(b, observed=True).agg(["mean", "size"]).round(4).to_dict("index"))
