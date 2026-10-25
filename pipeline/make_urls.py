"""Print the game-morning Odds API URL block (Oct 25). Writing the block by hand dropped player_pass_yds_alternate and
team_totals from the Week 7 pull, so the QB yards + WR1 catches stack logged zero bets. This script is now the only source
of the block; UPDATE_CHECKLIST step 8 points here.
Usage: python make_urls.py W [--late] [--only MARKETS]
  --late         only games that have not kicked off yet (from events.json 't')
  --only a,b     a targeted re-pull of just these markets (e.g. player_pass_yds_alternate,team_totals)
The API key is never stored in this public repo: the block carries KEY, and Clinton pastes the URLs with his key filled in.
Cost: one credit per market per event call (both books are in the us region)."""
import json, sys, datetime as dt
W = int(sys.argv[1]); late = "--late" in sys.argv
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
E = json.load(open(f"odds_raw/wk{W}/events.json"))
A = ["player_anytime_td", "player_receptions", "player_pass_interceptions"]
B = ["player_receptions_alternate", "player_rush_yds_alternate", "player_pass_yds_alternate", "team_totals"]
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
Q = "regions=us&bookmakers=draftkings,fanduel&oddsFormat=decimal&apiKey=KEY"
now = dt.datetime.now(dt.timezone.utc); urls = []; cost = 0
for g, e in sorted(E.items(), key=lambda kv: kv[1]["t"]):
    if late and dt.datetime.fromisoformat(e["t"].replace("Z", "+00:00")) <= now: continue
    for grp in ([only] if only else [A, B]):
        urls.append(f"{BASE}/events/{e['id']}/odds?markets={','.join(grp)}&{Q}"); cost += len(grp)
if not only:
    urls.append(f"{BASE}/odds?markets=h2h,spreads,totals&{Q}"); cost += 3
for i, u in enumerate(urls):
    if i and i % 4 == 0: print()          # Clinton sends them in batches of about 4
    print(u)
print(f"\n# {len(urls)} URLs, about {cost} credits", file=sys.stderr)
