# NFL Insights — project notes

Read this first in every chat. This repo is the single source of truth. Chats come and go; anything worth keeping is committed here.

## What this is
NFL Insights (also called "Paydirt") is Clinton's NFL projection and betting-research site, published as a claude.ai artifact: https://claude.ai/artifact/MVrDv9jyZBGoEnfbcnYSZH. Other people see it, so his personal bet log never goes on the site.

**Goal:** the most informed, likelihood-based bets possible, plus visibility into trends. Beating DraftKings is NOT the goal. Don't frame the work as edge-hunting, but stay honest when the market is more accurate than the model.

## Status (Oct 21, 2026)
- **The original pipeline is back.** Exported from the old chat on Oct 21 into `pipeline/` (code, config, small data). Large data and model files were left out of the export; `pipeline/bootstrap_data.py` rebuilds them from nflverse.
- **One command updates the site from a fresh workspace:** `bash pipeline/daily.sh` (picks the current week, adds next week once its lines exist). Output: `pipeline/site/paydirt.html`. Takes ~25 min from scratch, ~10 once data is cached.
- Oct 21 evening: Week 7 and 8 rebuilt and published as v152.
- Lost in the export: pre-kickoff archives before Oct 21 (the scorecard keeps its last numbers until new weeks are graded) and line-movement history (restarts Oct 21).
- Run state that must persist between runs is committed: `pipeline/archive/`, `pipeline/data/line_history.csv`, record and bet CSVs, override JSONs. Commit after every run.
- `SPEC.md` and `recover/` are now reference only.
- `tracker/` is superseded by the pipeline's own scorecard; not scheduled.

## Daily update (what the scheduled task does)
Follow `pipeline/UPDATE_CHECKLIST.md` steps 1–7 (never the odds pull, which is manual). In short: run `bash pipeline/daily.sh`; read the role feeds and fix `news_overrides_wk{W}.json`; resolve `audit_wk{W}.txt`; rebuild; publish `pipeline/site/paydirt.html` to the site URL; commit and push; send a short digest.

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
- **Test strategies tracked every week** (pre-registered, mechanical; see the checklist for exact rules):
  - "ML + top TD scorer, 2 legs" (started Week 6; picks the winner by the line-anchored chance since Oct 21)
  - matchup props, same-game stacks (incl. QB yards + WR1 catches), TD fades, a random-pick control
  - added Oct 21: small-underdog teasers, wind and prime-time unders, steady-leg parlays
  - cut Oct 21: top-5-total TD parlays, favorites' lead-RB TD parlays, the TD 4.0–6.0 band
- **What Clinton wants from the data:** find which predictions the model is genuinely good at, and the bets closest to structural "hacks" of the book: boosts and promos, DK vs FanDuel price shopping, key-number teasers, same-game parlays priced below their fair (correlated) price, and DK's board disagreeing with itself. The model has not beaten DK's lines on any market yet; say so when it matters.
- **Odds budget:** the Odds API account has 500 credits for the season. No second weekly prop pull; closing-line value comes free from nflverse for game lines only.
- When asked a raw question (e.g. "most TDs allowed to which position"), answer the raw count, not a relative-to-average number.
- Clinton bets throughout the week and may send screenshots of bets to judge against the projections.

## Recurring schedule (ET)
- **Daily 8:52 a.m.:** "NFL Insights daily update" scheduled task. Starts a fresh session from this repo each run; nothing chains or reschedules itself. Its prompt is the "Daily update" section above.
- **Tuesday:** the same run also sends the weekly recap (what worked, what didn't, what we learned, season accuracy).
- Oct 21: the old self-rescheduling tasks (daily update chain, tracker snapshots) were disabled. Never rebuild schedules as send_later chains tied to one chat.

## Repo layout
- `pipeline/` — the whole site pipeline: data pulls, models, site build (`daily.sh` is the entry point)
- `pipeline/archive/` — pre-kickoff copies of every published week (feeds the scorecard; commit them)
- `tracker/`, `recover/`, `SPEC.md` — reference only
- Clinton's private bet log is NOT in this repo (the repo is public). Never add it.

## How to work here
- Keep chats short and focused on one topic. Commit at the end of any change.
- Put notes and decisions in this file, not in chat.
