"""Replay the live projection pipeline on a past week (Nov 2, Clinton: test changes on full past seasons, don't wait for
weeks of archives). Usage: python replay/run.py 2025 8 [9 ...]   (run from pipeline/)
For each (season, week) it builds a sandbox whose data stops where a game-morning run would have stopped:
  play-by-play, snaps, xTD, OL/QB tables: games before week W only; later seasons removed
  schedule: week W onward has no scores (lines kept; they are the closing lines, close to a morning number)
  injury report: through week W (the final game statuses, like a Saturday/Sunday morning run)
  rosters: through week W, with week W's game-day inactives (INA) turned back to active (a morning run doesn't have them)
then runs the same chain refresh_all.sh runs per week (lines -> INT/QB -> TD/pool -> OL -> projections -> lines 2nd pass)
with NFL_SEASON/NFL_WEEK set, and saves proj/td/lines outputs to replay/out/.
Differences from live, by design: no hand news (no news_overrides, qb_overrides, forecasts), the schedule's QB is the one
who actually started (live has to get that from the news), depth charts absent. Calibration tables fitted on all seasons
(ratio_table.pkl, line_calib.json) are reused as-is: a small look-ahead for ranges/probabilities, none for means."""
import glob, os, shutil, subprocess, sys, time
import pandas as pd
P = os.path.abspath(".")
OUT = os.path.join(P, "replay", "out"); os.makedirs(OUT, exist_ok=True)
EXTRA = os.path.join(P, "replay", "data_extra")

def build(S, W, sb):
    if os.path.exists(sb): shutil.rmtree(sb)
    os.makedirs(f"{sb}/data")
    for f in glob.glob(f"{P}/*.py"): shutil.copy(f, sb)
    for f in ["ratio_table.pkl", "coach_fix.json", "weights.json", "lines_backtest.pkl"]:
        if os.path.exists(f"{P}/{f}"): shutil.copy(f"{P}/{f}", sb)
    for f in ["line_calib.json", "joint_corr.json", "players.parquet"]:
        os.symlink(f"{P}/data/{f}", f"{sb}/data/{f}")
    for f in glob.glob(f"{P}/data/ftn_*.parquet") + glob.glob(f"{P}/data/pbp_*.parquet"):
        y = int(os.path.basename(f)[4:8])
        if y < S: os.symlink(f, f"{sb}/data/{os.path.basename(f)}")
    pb = pd.read_parquet(f"{P}/data/pbp_{S}.parquet"); pb = pb[pb.week < W]; pb.to_parquet(f"{sb}/data/pbp_{S}.parquet")
    fn = f"{P}/data/ftn_{S}.parquet"
    if os.path.exists(fn):
        ft = pd.read_parquet(fn); ft[ft.nflverse_game_id.isin(set(pb.game_id))].to_parquet(f"{sb}/data/ftn_{S}.parquet")
    played = set(pd.concat([pd.read_parquet(f, columns=["game_id"]) for f in glob.glob(f"{P}/data/pbp_*.parquet") if int(os.path.basename(f)[4:8]) < S]).game_id) | set(pb.game_id)
    sc = pd.read_parquet(f"{P}/data/sched.parquet"); sc = sc[sc.season <= S].copy()
    fut = (sc.season == S) & (sc.week >= W)
    for c in ["result", "home_score", "away_score", "total", "overtime"]:
        if c in sc: sc.loc[fut, c] = None
    sc.to_parquet(f"{sb}/data/sched.parquet")
    asof = lambda d: d[(d.season < S) | ((d.season == S) & (d.week < W))]
    for f in ["snaps", "snaps_def", "xtd"]:
        if os.path.exists(f"{P}/data/{f}.parquet"): asof(pd.read_parquet(f"{P}/data/{f}.parquet")).to_parquet(f"{sb}/data/{f}.parquet")
    for f in ["team_dmiss", "team_qd", "ol_continuity"]:
        if os.path.exists(f"{P}/data/{f}.parquet"):
            d = pd.read_parquet(f"{P}/data/{f}.parquet"); d[d.game_id.isin(played)].to_parquet(f"{sb}/data/{f}.parquet")
    ia = pd.read_parquet(f"{P}/data/injuries_all.parquet"); ia = ia[(ia.season < S) | ((ia.season == S) & (ia.week <= W))]
    ia.to_parquet(f"{sb}/data/injuries_all.parquet"); ia[ia.season == S].to_parquet(f"{sb}/data/injuries_{S}.parquet")
    rf = f"{EXTRA}/rosters_{S}.parquet" if S != 2026 else f"{P}/data/rosters_2026.parquet"
    if not os.path.exists(rf):                        # weekly rosters for a past season (not kept in the repo)
        import nflreadpy as nfl; os.makedirs(EXTRA, exist_ok=True); nfl.load_rosters_weekly([S]).to_pandas().to_parquet(rf)
    ro = pd.read_parquet(rf); ro = ro[ro.week <= W].copy()
    ro.loc[(ro.week == W) & (ro.status == "INA"), "status"] = "ACT"
    ro.to_parquet(f"{sb}/data/rosters_{S}.parquet")

def run(S, W, sb):
    env = dict(os.environ, NFL_SEASON=str(S), NFL_WEEK=str(W))
    tag = os.environ.get("REPLAY_TAG", "")
    steps = [["lines_project.py", str(W)], ["project_week.py"], ["td_project.py"], ["ol_week.py", str(W)], ["proj_project.py"], ["lines_project.py", str(W)]]
    log = open(f"{OUT}/{tag}log_{S}_wk{W}.txt", "w")
    for st in steps:
        r = subprocess.run([sys.executable] + st, cwd=sb, env=env, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode: log.close(); return f"failed at {st[0]} (see replay/out/log_{S}_wk{W}.txt)"
    for f in [f"proj_{S}_wk{W}.json", f"td_props_{S}_wk{W}.csv", f"lines_{S}_wk{W}.json", f"int_props_{S}_wk{W}.csv"]:
        if os.path.exists(f"{sb}/{f}"): shutil.copy(f"{sb}/{f}", os.path.join(OUT, tag + f))
    return "ok"

if __name__ == "__main__":
    S = int(sys.argv[1]); sb = f"/tmp/claude-0/replay_sb_{os.environ.get('REPLAY_TAG', '')}{S}"
    for W in map(int, sys.argv[2:]):
        t = time.time(); build(S, W, sb); print(S, W, run(S, W, sb), f"{time.time() - t:.0f}s", flush=True)
