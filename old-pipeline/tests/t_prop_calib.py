"""Prop probability calibration (Oct 16): model vs DraftKings on every logged prop (weeks 4-6 so far) against results.
Brier score (lower = better) and the best model/DK blend in log-odds. DK over/under is de-vigged from both sides;
anytime TD has one side, so DK's implied chance is scaled by the slate's average overround estimated from the fit."""
import sys; sys.path.insert(0, ".")
import re, numpy as np, pandas as pd, proj_model as P
pg = P.player_games(P.load())[0]
pl = pd.read_parquet("data/players.parquet", columns=["gsis_id", "display_name"])
nm = lambda s: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower()))
key = dict(zip(pl.display_name.map(nm), pl.gsis_id))
pg["td"] = pg.get("rush_td", 0) + pg.get("rec_td", 0)
A = pg.set_index(["game_id", "pid"])
def actual(r, col):
    pid = key.get(nm(r.player))
    if pid is None or (r.game_id, pid) not in A.index: return np.nan
    v = A.loc[(r.game_id, pid)]; v = v.iloc[0] if isinstance(v, pd.DataFrame) else v
    return float(v[col]) if col in v else np.nan
lo = lambda p: np.log(p / (1 - p)); sg = lambda x: 1 / (1 + np.exp(-x))
def blend_table(y, pm, pk, label):
    pm, pk = np.clip(pm, 0.01, 0.99), np.clip(pk, 0.01, 0.99); out = []
    for w in np.linspace(0, 1, 11):
        pb = sg(w * lo(pm) + (1 - w) * lo(pk)); out.append((w, np.mean((y - pb) ** 2)))
    best = min(out, key=lambda t: t[1])
    print(f"{label}: n={len(y)}  Brier model {np.mean((y-pm)**2):.4f} | DK {np.mean((y-pk)**2):.4f} | best blend w_model={best[0]:.1f} -> {best[1]:.4f}")
    for lo_, hi_ in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, 1)):
        m = (pm >= lo_) & (pm < hi_)
        if m.sum() >= 15: print(f"     model said {lo_:.0%}-{hi_:.0%}: avg {pm[m].mean():.2f}, DK avg {pk[m].mean():.2f}, happened {y[m].mean():.2f} (n={m.sum()})")
# ---- anytime TD
t = pd.read_csv("props_log.csv"); t = t[t.market == "player_anytime_td"].drop_duplicates(["game_id", "player"], keep="last").copy()
t["y"] = [actual(r, "td") for r in t.itertuples()]; t = t.dropna(subset=["y", "model_p"]); t["y"] = (t.y >= 1).astype(float)
raw = 1 / t.price; vig = raw.sum() / (t.model_p.sum())          # rough: DK implied total vs model total
t["dk"] = raw / max(raw.mean() / t.y.mean(), 1.0)                  # scale DK to the realised base rate (de-vig proxy)
print(f"ATD: DK implied avg {raw.mean():.3f}, happened {t.y.mean():.3f}, model avg {t.model_p.mean():.3f}")
blend_table(t.y.to_numpy(), t.model_p.to_numpy(), t.dk.to_numpy(), "Anytime TD")
# ---- over/unders
COL = {"player_reception_yds": "rec_yds", "player_receptions": "rec", "player_rush_yds": "rush_yds", "player_pass_yds": "pass_yds",
       "player_rush_attempts": "car", "player_pass_completions": "cmp", "player_pass_attempts": "att"}
u = pd.read_csv("props_log_ou.csv").drop_duplicates(["game_id", "player", "market"], keep="last")
u = u[u.market.isin(COL)].copy(); u["v"] = [actual(r, COL[r.market]) for r in u.itertuples()]
u = u.dropna(subset=["v", "model_p_over", "mkt_p_over"]); u = u[u.v != u.point]; u["y"] = (u.v > u.point).astype(float)
blend_table(u.y.to_numpy(), u.model_p_over.to_numpy(), u.mkt_p_over.to_numpy(), "All over/unders")
for mk, x in u.groupby("market"):
    if len(x) >= 40: blend_table(x.y.to_numpy(), x.model_p_over.to_numpy(), x.mkt_p_over.to_numpy(), f"  {mk}")
print(f"\nOver rate on DK lines: {u.y.mean():.3f} (n={len(u)})")
