"""Split fetched Odds API prop text (one line per outcome: bookmaker|market|player|side|point|price) into the per-game raw files.
Usage: python odds_split.py W fetched.json     fetched.json = {url: text}, from the session that ran the WebFetch calls.
Writes, per game (odds_raw/wk<W>/events.json maps AWAY_HOME -> event id):
  <AWAY>_<HOME>.txt     DraftKings anytime TD + main receptions        -> ingest_odds.py
  <AWAY>_<HOME>_x.txt   DraftKings alternate rungs + team totals        -> ingest_more.py   (game-line rows already there are kept)
  <AWAY>_<HOME>_fd.txt  every FanDuel outcome                           -> ingest_fd.py
Prop rows replace earlier prop rows for the same game and market group; game-line rows are left alone."""
import json, re, sys, os
W, src = int(sys.argv[1]), sys.argv[2]
E = json.load(open(f"odds_raw/wk{W}/events.json")); by_id = {v["id"]: k for k, v in E.items()}
MAIN = {"player_anytime_td", "player_receptions", "player_pass_interceptions"}
GAME = {"game_h2h", "game_spreads", "game_totals"}
for url, text in json.load(open(src)).items():
    m = re.search(r"events/([0-9a-f]{32})", url)
    if not m or m.group(1) not in by_id: continue
    g = by_id[m.group(1)]; mk_req = set(re.search(r"markets=([^&]+)", url).group(1).split(","))
    rows = {"dk": [], "dkx": [], "fd": []}
    for line in text.splitlines():
        x = [s.strip() for s in line.strip().strip("`").split("|")]
        if len(x) != 6 or x[1] not in mk_req: continue
        bk, mk, pl, side, pt, pr = x
        try: float(pr)
        except ValueError: continue
        if mk == "team_totals": pl = pl or side
        rec = f"{mk}|{pl}|{side}|{pt}|{pr}"
        rows["fd" if bk.lower() == "fanduel" else "dk" if mk in MAIN else "dkx"].append(rec)
    for key, suf in (("dk", ""), ("dkx", "_x"), ("fd", "_fd")):
        f = f"odds_raw/wk{W}/{g}{suf}.txt"
        old = [l.rstrip("\n") for l in open(f)] if os.path.exists(f) else []
        keep = [l for l in old if l.split("|")[0] not in mk_req]          # drop stale rows for the markets just re-pulled
        if rows[key] or keep != old: open(f, "w").write("\n".join(keep + rows[key]) + "\n")
    print(g, {k: len(v) for k, v in rows.items()})
