#!/bin/bash
# Full refresh: data, background tables, then every model for each week given. Usage: ./refresh_all.sh 4 5
set -e
# pre-game snapshots for grading are now saved by assemble_site.py each time the page is built (Oct 18)
python refresh_data.py
python ol_build.py            # offensive-line continuity history
python xtd.py                 # expected touchdowns
python lines_backtest.py      # spread/total model record (the LINES verdict box)
python line_history.py
python dvp_td.py
for W in "$@"; do
  export NFL_WEEK=$W
  python lines_project.py $W; python project_week.py; python td_project.py; python ol_week.py $W; python proj_project.py; python lines_project.py $W   # second pass: team model uses this week's QB chances
done
for W in "$@"; do python paper_lines.py $W; done   # Oct 25: automatic game-line paper bets from nflverse prices (no manual pulls)
for W in $(seq 1 ${1}); do ls bets_wk${W}*.csv >/dev/null 2>&1 && python grade_bets.py $W > /dev/null; done   # grade paper slates for finished games
for f in paper_wk*.csv; do [ -f "$f" ] || continue; case $f in *_graded*) ;; *) W=$(echo $f | sed 's/[^0-9]//g'); python grade_paper.py $W > /dev/null;; esac; done   # mechanical paper slates
python record.py > /dev/null                      # Record tab
python line_moves.py > /dev/null                  # did lines move toward the model's numbers?
python scorecard.py                   # scorecard: every published number vs results (top of the Record tab)
NFL_WEEK=${@: -1} python explain_weights.py
for W in "$@"; do python build_site.py $W; done
python trends.py
python season.py
python awards.py
python assemble_site.py "$@"
for W in "$@"; do python audit_stale.py $W | tee audit_wk$W.txt || true; done   # staleness audit: every line needs a look before publishing
