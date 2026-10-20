# NFL Insights — project notes

Read this first in every chat. This repo is the single source of truth. Chats come and go; anything worth keeping is committed here.

## What this is
NFL Insights (also called "Paydirt") is Clinton's NFL projection and betting-research site, published as a claude.ai artifact: https://claude.ai/artifact/MVrDv9jyZBGoEnfbcnYSZH. Other people see it, so his personal bet log never goes on the site.

**Goal:** the most informed, likelihood-based bets possible, plus visibility into trends. Beating DraftKings is NOT the goal. Don't frame the work as edge-hunting, but stay honest when the market is more accurate than the model.

## Status (Oct 20, 2026)
- The original pipeline was built in a chat that hit its length limit. Its code lives only in that chat's workspace, and the daily 8:52 a.m. ET update still runs there via a chained reminder. **Its code can't be copied out.**
- **Rebuild plan:** see `SPEC.md`. Full transcript recovery was declined as too costly (Oct 20); a pilot of turns 3–20 is in `recover/`.
- **Oct 20:** the old daily run fired but failed to publish; treat the old pipeline as dead. Site patched from here (v150: Week 6 Monday night graded via `tools/grade_pending.py`). Rebuild starts Oct 21.
- **This repo is the rebuild.** Until the rebuilt pipeline publishes, do NOT publish to the site URL from here, because the old daily run overwrites it every morning.
- `tracker/` (built Oct 19) snapshots the live site's projections before kickoff and grades them against nflverse stats. See `tracker/TICK.md`.

## Next session (agreed Oct 20)
1. **Daily results updater** (runs from this repo every morning): pull nflverse results, update records/standings, grade pending Record-tab bets (`tools/grade_pending.py`), verify site vs actual results and fix or alert, republish. Schedule it as the daily task.
2. **Season simulator**: rebuild win totals and division/playoff/#1 seed/conference/Super Bowl odds from the ratings, home-field value and schedule stored in the site's `season` data, plus actual results. Becomes the rebuild's season-sim piece.
Patched by hand on Oct 20: v150 (Week 6 MNF bets graded), v151 (SF 6-0, WAS 1-4-1 on Season tab; their sim odds still pre-MNF).

## Clinton's standing rules
- **Odds:** decimal only. **DraftKings** is the reference book (he also bets FanDuel; FanDuel won't allow INT props in a same-game parlay). Never scrape DK or FanDuel sites.
- **Odds pulls:** The Odds API, on a 500-credit account, so stay within budget. Pulls are manual on game mornings, and Clinton pastes the URLs. Claude can only fetch Odds API URLs that he pastes as plain text in a message; send them in batches of about 4.
- **Weather:** MySportsWeather is the primary source and Covers is the cross-check. Open-Meteo is blocked, so don't suggest it.
- **Props:** used mainly as conservative legs in 2–3-leg parlays targeting a 2x+ payout. Favor consistent players with the best matchups: the opponent must be top-8 in fantasy points allowed to that position AND allow at least the league-average TD rate.
- **Anytime TD:** focus on the 2.5–3.9 decimal range. 4.5+ is too random to bet.
- **INT props:** no longer bet. Still tracked.
- **Never** put a skill player in a lineup or bet without a real season record: 8+ catches for pass catchers, 25+ carries for backs.
- **DFS (DraftKings):** pull salaries early in the week.
  - No zero-catch players.
  - Stack = QB + exactly 2 of his pass catchers, and no more from that team.
  - At most one Eagle. No Saints while they're bad.
  - Follow established tournament practice; don't hand-tune an optimizer.
- **Test strategies tracked every week:**
  - "ML + top TD scorer, 2 legs" (started Week 6)
  - matchup props
  - same-game stacks
  - TD fades
  - a random-pick control
- When asked a raw question (e.g. "most TDs allowed to which position"), answer the raw count, not a relative-to-average number.
- Clinton bets throughout the week and may send screenshots of bets to judge against the projections.

## Recurring schedule (ET)
- **Daily 8:52 a.m.:** site update (old chat for now, moving here).
- **Thu, Sun 12:44 p.m., Mon, Tue ~10:41 a.m.:** tracker snapshots.
- **Tuesday:** grade the week and send the recap: what worked, what didn't, what we learned, season accuracy.

## Repo layout
- `tracker/` — projection snapshots and season-long accuracy grading
- `pipeline/` — (to rebuild) data pulls, model, site build
- `site/` — (to rebuild) the published page
- `bets/` — Clinton's private bet log (never published)

## How to work here
- Keep chats short and focused on one topic. Commit at the end of any change.
- Put notes and decisions in this file, not in chat.
