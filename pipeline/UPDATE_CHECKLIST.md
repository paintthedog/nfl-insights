# NFL Insights — standard global update checklist

Run top to bottom. W = current week (`NFL_WEEK`). Steps marked ⚑ need Clinton; everything else runs unattended.
Cadence: daily Wed–Sat; Sunday morning = full run + odds pull; Tuesday = paper-bet grading (separate scheduled task).

## 1. Data refresh
- [ ] `python get_data.py` — nflverse pbp, schedules, injuries, weekly rosters, snap counts, depth charts
- [ ] Confirm pbp has the latest completed games (TNF shows up Friday; `pbp_2026.parquet` max week)

## 1b. ⚑ DFS salaries (Tue/Wed, once DK posts the Sunday main slate)
- [ ] Clinton exports the salary CSV from any DK Classic main-slate contest ("Export to CSV" / download player list) and attaches it — no scraping DK
- [ ] Convert to `dk_salaries_wk{W}.csv` (name,salary; DK's export also has position, ID, game info — keep name and salary, fix name mismatches vs our player names)
- [ ] The audit flags [DFS] until the file exists; mention it in the digest each day it's missing

## 2. Injuries & availability (official layer)
- [ ] Rebuild injuries from the refreshed data (part of per-week chain below)
- [ ] Diff the new report vs `news_overrides_wk{W}.json`: anyone newly Out/Doubtful/IR who's still in our pool? Anyone upgraded (full practice) we're still docking?

## 3. News scan (soft layer) — the overrides file is only as good as its timestamp
- [ ] ROLE FEEDS (added Oct 18 after the Allgeier miss): WebFetch BOTH every morning and read every item for role facts, not just availability:
      https://www.thefantasyfootballers.com/fantasy-football-news/  (short headlines: "set for lead role", "activated", "ruled out", "expected to start")
      https://fantasydata.com/nfl/fantasy-football-player-news  (one-line summaries with "lead-back role", "committee", "snap share")
      Map each fact to the override that carries it: ruled out -> out · questionable/expected -> p_play · "lead back"/"will start" among known backs -> lead_back · unknown back promoted -> next_man · activated/returning -> back. The scheduled prompt lists these URLs so a fresh session can fetch them.
- [ ] For every team with a back in `out`, confirm who the lead is (feeds above, or WebSearch "<team> running back depth chart") and set `lead_back`. An even committee split is the model's default and it is usually wrong.
- [ ] WebSearch: "<team> injury news" for teams with Questionable stars; beat-writer reports
- [ ] Start a new week's overrides by COPYING last week's file, then prune — never start blank (Week 6: starting blank silently dropped five OL absences)
- [ ] (Oct 22) `auto_overrides.py` runs in daily.sh before every rebuild: seeds a missing week from the last one (carries out/lead_back/qb_start/ol_out, drops chances), dates every entry in `news_dates_wk{W}.json`, lets official game statuses beat hand chances, and removes contradictions. The audit prints [STALE] for any chance/lead back/uncertain QB older than 3 days: re-verify it and either change it or list the name (or team code) under `"confirmed": [...]` in the overrides file to re-date it. Team notes older than 4 days stop showing on the site.
- [ ] (Oct 22) News activates players ahead of the roster file: anyone in `back`, or given a `p_play` above 0, joins his team's pool even while nflverse still lists him RES (IR/PUP). Traded players join their new team's pool automatically with their usage carried over.
- [ ] Update `news_overrides_wk{W}.json`: `out`, `p_play`, `qb_start`, `next_man`, `lead_back`, `ol_out`, `team_notes` — and bump `asof`
- [ ] `lead_back` (Oct 18): when a starter is out and the news names the replacement among KNOWN backs, set `"lead_back": {"TEAM": ["Name", 0.65]}` (his share of the team's backfield carries). Without it the model splits the work evenly across the committee; that is what every "DK sees a bigger role" ROLE flag on a running back has meant so far (Allgeier, R. White, McLaughlin). `next_man` is only for a back with no usage history.
- [ ] QB audit: compare the schedule feed's listed QB (home_qb_name/away_qb_name) for week W against each team's last starter; for any mismatch, confirm the real starter from news and set `qb_start` in the overrides (the lines model now honors it). The feed listed Drew Lock for SEA in Week 6 and the model leaned Denver for no reason.
- [ ] Freshness rule (the Bigsby lesson): any override older than 48h gets re-verified or removed
- [ ] Tier discipline: availability facts → model inputs; role facts → pool/seeds; narrative → `team_notes` only. Never hand-edit projections.

## 4. Weather (within 48h of kickoff; skip early-week)
- [ ] WebFetch mysportsweather.com/nfl (primary: meteorologist, kickoff-hour wind + gusts) + covers.com/sport/nfl/weather (cross-check); note any >3 mph disagreement in the source column. Open-Meteo is blocked to fetching (robots); NWS point forecasts only for a single genuinely windy/snowy game
- [ ] Rewrite `forecasts_wk{W}.csv` (team,wind,temp,precip,source) — outdoor stadiums only; note source disagreements
- [ ] Roof games (ARI, ATL, DAL, HOU, IND, LV, MIN, NO, DET, LAR/LAC): leave to sched roof field

## 5. Rebuild everything
- [ ] `python dvp_td.py` (season-level TDs allowed by position)
- [ ] `bash refresh_all.sh W [W+1]` (weeks are REQUIRED args; include W+1 only once it has betting lines, or proj_project crashes) — per week: lines_project → project_week → td_project → ol_week → proj_project → lines_project (2nd pass)
- [ ] `python build_site.py {W} && python build_site.py {W+1} && python assemble_site.py {W} {W+1}`

## 6. Sanity checks (before publishing)
- [ ] Read `audit_wk{W}.txt` (written by refresh_all.sh; or `python audit_stale.py W`). Resolve every item: QB mismatches → qb_start; ROSTER/TEAM/INJURY conflicts → out/p_play; CARRYOVER → confirm each of last week's outs (players AND ol_out) is still out or back; FRESHNESS/WEATHER → refetch. Mention anything left unresolved in the digest.
- [ ] ROLE items (from the audit): DraftKings implies a very different role than the model. Search the news for that team's depth chart; fix through out/p_play/next_man/qb_start if the news supports it, otherwise mention it in the digest as 'trust DK here'. Never edit projections to match DK.
- [ ] `node check_page.js site/paydirt.html` must print OK (data parses, scripts compile, no bet log). The old click-through tests (t2.js, nav.js) were lost in the Oct 21 export; after a layout change, open the page on a phone and click every tab.
- [ ] Spot-check 3 projections against today's news: newly-out players gone, next-man seeds fired where expected, OL counts match reports
- [ ] `ol_missing_wk{W}.csv` exists and is current (this file silently defaulting to 0 was a real bug once)
- [ ] Prices-asof stamp on the site matches the latest pull, not a stale one

## 7. Publish & digest
- [ ] Publish to the artifact (same URL)
- [ ] Post a short digest: new outs/returns, projection moves >1.5 pts or >15% ATD swing, forecast changes, line moves, anything that touches open paper bets

## 8. ⚑ Odds pull — game mornings only (Sun ~7am; Thu/Mon if betting those)
- [ ] Fetch events list (0 credits) → build the URL block (1 ATD call per game + 1 lines call)
- [ ] ⚑ Clinton pastes the URL block (provenance requirement — cannot be automated)
- [ ] Fetch all; write `odds_raw/wk{W}/*.txt` (archive the prior pull to `odds_raw/wk{W}_<day>/` first)
- [ ] Per game, TWO event URLs (each under the ~250-char fetch limit), bookmakers=draftkings:
      (a) markets=player_anytime_td,player_receptions,player_pass_interceptions
      (b) markets=player_receptions_alternate,player_rush_yds_alternate,player_pass_yds_alternate,team_totals   (pass-yds rungs added Oct 21 for the QB + WR1 stack: +1 credit/game)
      plus ONE sport-level URL: markets=h2h,spreads,totals&oddsFormat=decimal  → ~7 credits/game + 3 ≈ 100/week for a full slate
      Oct 21: use bookmakers=draftkings,fanduel on every URL (line shopping + the DK-vs-FanDuel check). Believed to cost no extra credits;
      CHECK the x-requests-used header on the first pull and update this line.
- [ ] Save ATD/receptions to `odds_raw/wk{W}/<AWAY>_<HOME>.txt` (market|player|side|point|price) → `python ingest_odds.py {W} "<ts>"`
- [ ] Save INTs, alt rungs, team totals, and game lines (game_h2h / game_totals / game_spreads rows) to `odds_raw/wk{W}/<AWAY>_<HOME>_x.txt` → `python ingest_more.py {W} "<ts>"`
- [ ] FanDuel's outcomes (all markets) to `odds_raw/wk{W}/<AWAY>_<HOME>_fd.txt`, same 5-field format → `python ingest_fd.py {W} "<ts>"`
- [ ] Rebuild (build_site + assemble), then `python paper_slate.py {W}` — MECHANICAL slate, never hand-edit; it is the pre-registered record. Types as of Oct 21:
      TD parlays (likeliest scorer per game priced 1.15–3.00, paired across games; both teams' shortest in the top-total game); steady-leg parlays (alt rungs 1.25–1.50, real season record, model agrees; 2–3 legs to ~2.0);
      teasers (underdogs +1.5 to +2.5 teased 6, two legs from different games; reference price 1.80 — check DK's real 2-team teaser price in the app and note it);
      trend unders (verified wind 10–19 mph; kickoff 7 p.m. ET or later). CUT Oct 21: top-5-total TD parlays, favorites' lead-RB TD parlays, the TD 4.0–6.0 band.
- [ ] `python market_consistency.py {W} --paper` (run by paper_slate) now also checks main line vs alternate ladder, team totals vs the game line (bettable, joins Board consistency), and DK vs FanDuel (price shopping; not a bet). Put the FanDuel-better list in the digest.
- [ ] Publish
- [ ] Report: edges in the 2x–4x range only (ignore longshot-tail "edges" — known model weakness), market-vs-model disagreements worth knowing even as passes, credit ledger (used this season / ~remaining of 500)

## Scorecard (every published number vs results) — Oct 18
- [ ] `python scorecard.py` runs inside refresh_all.sh -> scorecard.json (top of the Record tab). It grades each finished game off the last copy of site/data_wk{W}.json saved BEFORE that game's kickoff.
- [ ] The Record tab's report card (scorecard.json `card`) is generated by fixed rules in scorecard.py: never hand-edit its wording; change the rule and note it in TEST_LEDGER.md.
- [ ] Bet types roll up into 8 families (record.py FAMILY). A new paper-bet type must be added to a family or it falls into "Other".
- [ ] Those copies are saved by `assemble_site.py` (archive/data_wk{W}_YYYYMMDD_HHMM.json, Eastern time, only when the data changed). So: only run assemble_site on a build you intend to publish, and always publish after assembling on a game day.
- [ ] Weeks 4–5 have no pre-kickoff site copies; they are graded only from the odds-pull logs (TD chances, player lines). Say "odds pulls only" if asked.

## Tuesday calibration checks (after the week is final)
- [ ] `python tests/t_range_audit.py W` — published floors/ceilings vs results (target: ~10% below the floor, ~10% above the ceiling, ~50% each side of the median). Pool weeks as they accumulate; one week is noise.
- [ ] Game-line probabilities are calibrated (`calib_lines.py` → data/line_calib.json). Refit in the offseason only.

## Board consistency and line movement (Oct 18)
- [ ] `python market_consistency.py W --paper` runs from paper_slate.py on game mornings: DK's lines checked against each other (moneyline vs spread, team TD mass vs implied points, player TD price vs his usage ladders, lead back vs game script). Writes consistency_wk{W}.csv; bettable flags (moneyline, ATD Yes) join the record as "Board consistency". Flags that aren't bettable (a short TD price) are information for the ROLE audit, not bets.
- [ ] `python line_moves.py` runs in refresh_all.sh: for each kicked-off game, did the spread/total move toward the model's first published number or away? Feeds the "Line movement" report-card line. Needs 20+ cases before it says anything.

## Same-game stacks in the paper record (Oct 18)
- [ ] `paper_slate.py` now appends `sgp_slate.build(W)`: for every game not yet started, the DK moneyline favorite gets (A) win + lead RB rushing rung and (B) its top two pass catchers' receptions rungs, each rung priced nearest 1.40. Re-running `python sgp_slate.py W` before kickoff replaces only games not yet started.
- [ ] Graded on hit rate vs the joint chance (model_p) and vs the legs multiplied (indep_p). The price column is the legs multiplied, a reference only: no same-game prices in the Odds API, and DK pays less on same-game parlays.

## Matchup props (Clinton's standing rule, Oct 18)
- [ ] Prop bets favor players with the best matchups: paper_slate.py adds "Matchup props" (TD Yes at 1.5-4.0, and receptions overs where the model agrees) ONLY for RB/WR/TE whose opponent ranks top-8 in fantasy points allowed to their position AND allows at least a league-average TD rate to it. When Clinton asks for prop picks, lead with these; a strong model number against a bad matchup is a pass.

## ML + top TD scorer (Clinton's standing rule, Oct 18)
- [ ] `ml_td_slate.build(W)` runs inside paper_slate.py: per game, the projected winner by the site's line-anchored win chance (Oct 21; was the raw model margin) on the moneyline + the game's likeliest anytime-TD scorer (either team), one 2-leg parlay. Backfill or re-log with `python ml_td_slate.py W --paper`. Price = the two DK prices multiplied (reference only). Graded on hit rate vs model_p.

## Pre-registered main tests (Oct 22)
- [ ] `record.py` PRIMARY: ML + top TD scorer (100 bets), matchup props TD and receptions (100 each), same-game stacks (75 each), small-underdog teasers (60). Fixed before any had results: don't add to or reshuffle the list mid-season. Main tests earn trust at a profit plus 2 standard errors above break-even; every other rule is exploratory and needs 150 bets and 3 standard errors.

## Market vs model vs blend (Oct 22)
- [ ] `scorecard.py` scores three numbers on the same games and players: the market's, the model's raw number, and the blend the site shows (game margin, game total, anytime TD). Report-card lines "Off the market? ..." say which was closest once the sample is big enough (40 games, 300 TD prices). If the market keeps winning, the site's numbers should sit on the market's.

## Same-game parlay pricing
- [ ] `from parlay_joint import joint; joint([("QB pass_yds", p1), ("WR1 rec_yds", p2), ...])` gives P(all legs hit) using historical same-game correlations (data/joint_corr.json; refit with `python parlay_joint.py fit` in the offseason). Use DK no-vig or the site's blended chance for each p. Compare to the DK same-game price: payout x joint P - 1 = expected return.

## Private bet log
- [ ] /home/claude/private/my_bets.csv is PRIVATE: never publish, never copy into site/ or the artifact. Grade pending rows once their games are final (result, payout) and fill closing_price when a pull covers the market. Mention graded results in the digest only to Clinton.

## Standing rules
- DraftKings only. Decimal odds. No scraping DK/FD sites directly.
- O/U prop re-pulls (~4 credits/game) only for games Clinton names.
- Never adjust model numbers for narrative; narrative renders as News notes on cards.

## 9. ⚑ Week 6+ parlay-leg test (game mornings, alongside step 8)
- [ ] Also pull DK alternate markets per game: player_receptions_alternate (~1 credit/game) and player_rush_yds_alternate (~1 credit/game)
- [ ] Score every rung with the model's walk-forward-calibrated P(stat >= k) (model runs ~1-2 pts conservative; receptions is the only O/U market where it adds info beyond DK's line)
- [ ] Keep legs where model P exceeds DK implied by >= 3 pts; build 5-10 paper parlays of 2-3 legs targeting ~2.0-2.5x
- [ ] Log to parlays_wk{W}.csv (legs, prices, model joint P, payout) and grade with the other paper bets
- [ ] Yardage gaps vs DK main lines are NOT bets (DK subsumes the model there); consistency adds nothing reliable except a hint on rushing

## Closing-line value and fair same-game prices (Oct 21)
- [ ] `grade_paper.py` adds `clv` (moneylines: price taken / closing price − 1) and `clv_pts` (totals and teaser legs: points better than the close), from nflverse's closing lines. Free; props have no CLV because a second paid pull per week doesn't fit the 500-credit budget.
- [ ] Record tab shows "Beat the close" per rule. A rule that beats the close over 50+ bets is worth more than its win–loss record says, and vice versa.
- [ ] Same-game stacks and ML + top TD scorer carry `fair_price` (1 / joint chance). When Clinton asks about a same-game parlay, compare DK's same-game price in the app with the fair price: above it = the parlay pays more than it should.
