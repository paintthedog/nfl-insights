#!/bin/bash
# One-command daily update for a fresh workspace (Oct 21). Run from pipeline/:  bash daily.sh
# Picks the current week W (earliest week with unplayed games) and adds W+1 when its betting lines exist.
set -e
cd "$(dirname "$0")"
pip install -q --break-system-packages statsmodels nflreadpy pyarrow polars scikit-learn >/dev/null 2>&1 || true
python get_data.py > /dev/null
python bootstrap_data.py
read W NEXT < <(python - <<'P'
import pandas as pd
s = pd.read_parquet("data/sched.parquet"); s = s[(s.season == 2026) & (s.game_type == "REG")]
W = int(s[s.result.isna()].week.min())
nxt = s[s.week == W + 1]
print(W, W + 1 if len(nxt) and nxt.spread_line.notna().all() else "")
P
)
echo "week $W ${NEXT:+and $NEXT}"
python auto_overrides.py $W $NEXT     # seed a missing week, date every entry, let official statuses win, prune no-ops
bash refresh_all.sh $W $NEXT
echo "built site/paydirt.html"
