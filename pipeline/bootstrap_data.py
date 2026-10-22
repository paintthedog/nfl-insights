"""Rebuild (only what is missing) the historical data files that were too big to export from the old chat (Oct 21).
Safe to run any time: only builds what's missing. Run once in a fresh workspace, before refresh_all.sh."""
import os, subprocess, sys, time
import pandas as pd, nflreadpy as nfl

FIRST, LAST = 2016, 2026
os.makedirs("data", exist_ok=True)

def get(fn, *a):
    for i in range(5):
        try: return fn(*a).to_pandas()
        except Exception as e: print("retry", fn.__name__, i, str(e)[:80]); time.sleep(5)
    raise SystemExit(f"could not load {fn.__name__}")

yrs = list(range(FIRST, LAST + 1))
if not os.path.exists("data/players.parquet"):   # team_adj.py needs it; refresh_data.py normally writes it later
    get(nfl.load_players).to_parquet("data/players.parquet")
if not (os.path.exists("data/snaps.parquet") and os.path.exists("data/snaps_def.parquet")):
    sn = get(nfl.load_snap_counts, yrs)
    keep = [c for c in ["game_id", "season", "week", "game_type", "team", "opponent", "pfr_player_id", "player", "position",
                        "defense_snaps", "defense_pct", "offense_snaps", "offense_pct"] if c in sn.columns]
    sn[keep].to_parquet("data/snaps_def.parquet")
    pl = get(nfl.load_players)
    pfr = pl.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    o = sn.copy(); o["gsis_id"] = o.pfr_player_id.map(pfr)
    o = o.dropna(subset=["gsis_id"])[["game_id", "season", "week", "team", "gsis_id", "position", "offense_snaps", "offense_pct"]]
    o.to_parquet("data/snaps.parquet"); print("snaps:", len(o), "| snaps_def:", len(sn))
if not os.path.exists("data/injuries_all.parquet"):
    parts = []
    for y in yrs:
        try: parts.append(get(nfl.load_injuries, [y]))
        except SystemExit: print("injuries missing for", y)
    pd.concat(parts).to_parquet("data/injuries_all.parquet"); print("injuries_all:", sum(len(p) for p in parts))

# derived files built by existing scripts, only if missing
for script, out in [("team_adj.py", "data/team_qd.parquet"), ("calib_lines.py", "data/line_calib.json"),
                    ("proj_backtest.py", "ratio_table.pkl"), ("tests/t_combo_lift.py", "tests/_wf_proj.pkl"),
                    ("parlay_joint.py fit", "data/joint_corr.json")]:
    if not os.path.exists(out):
        print("building", out, "via", script)
        subprocess.run([sys.executable, *script.split()], check=False)
        print("  ", "ok" if os.path.exists(out) else "STILL MISSING")
