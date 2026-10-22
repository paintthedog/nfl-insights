"""Housekeeping for news_overrides_wk{W}.json, run by daily.sh before every rebuild (Oct 22).

The hand-curated overrides file is the weakest link: entries go stale silently and outlive the news that set them.
This script does the mechanical part so the daily run only has to add new facts:
  1. Seeds a missing week from the previous one (never blank): keeps out / lead_back / qb_start / ol_out / back,
     drops week-specific chances (p_play, qb_uncertain) and team notes older than NOTE_DAYS.
  2. Stamps every entry with the date it first appeared (news_dates_wk{W}.json, a sidecar) so age is per entry,
     not one asof for the whole file.
  3. Lets the official report win: for a team whose game statuses are in the nflverse feed this week, hand entries for
     players on that report are dropped (Out -> the feed removes them; Questionable -> the feed's own measured chance).
  4. Removes contradictions (a player listed out and also given a chance or listed back).
  5. Prints STALE lines for chances older than STALE_DAYS; the audit repeats them.
To re-date an entry without changing it, list the name (or team code) under "confirmed": [...] in the overrides file.
Team notes older than NOTE_DAYS are hidden from the site by build_site.py (they stay in the file for the record).
Usage: python auto_overrides.py W [W+1]"""
import json, os, sys, shutil, datetime as dt
import pandas as pd

STALE_DAYS, NOTE_DAYS = 3, 4
today = dt.date.today().isoformat()


def load(path, default):
    try: return json.load(open(path))
    except FileNotFoundError: return default


def entries(ov):
    """Every dated fact in the file, as 'section|key' strings."""
    keys = [f"out|{n}" for n in ov.get("out", [])] + [f"back|{n}" for n in ov.get("back", [])] + [f"ol_out|{n}" for n in ov.get("ol_out", [])]
    for sec in ("p_play", "qb_start", "lead_back", "next_man", "team_notes", "qb_uncertain"):
        for k, v in (ov.get(sec) or {}).items():
            keys.append(f"{sec}|{k}|{json.dumps(v, sort_keys=True)}")   # a changed value counts as a new fact
    return keys


def run(W):
    path, dpath = f"news_overrides_wk{W}.json", f"news_dates_wk{W}.json"
    if not os.path.exists(path):
        prev = f"news_overrides_wk{W-1}.json"
        if not os.path.exists(prev): print(f"auto_overrides: no {path} and no week {W-1} file to seed from"); return
        ov = load(prev, {})
        pdates = load(f"news_dates_wk{W-1}.json", {})
        ov["p_play"] = {}; ov["qb_uncertain"] = {}
        ov["team_notes"] = {t: n for t, n in (ov.get("team_notes") or {}).items()
                            if (dt.date.today() - dt.date.fromisoformat(pdates.get(f"team_notes|{t}|{json.dumps(n, sort_keys=True)}", today))).days <= NOTE_DAYS}
        ov["source"] = f"Seeded {today} from week {W-1} by auto_overrides.py (out/lead_back/qb_start/ol_out carried; chances dropped). Re-verify against week {W} reports."
        json.dump(ov, open(path, "w"), indent=1)
        if os.path.exists(f"news_dates_wk{W-1}.json"): shutil.copy(f"news_dates_wk{W-1}.json", dpath)
        print(f"auto_overrides: seeded {path} from week {W-1}")
    ov = load(path, {}); dates = load(dpath, {})
    changed = []
    # 3. the official report wins for teams that have game statuses this week
    try:
        inj = pd.read_parquet("data/injuries_2026.parquet"); iw = inj[(inj.week == W) & inj.report_status.notna()]
    except Exception:
        iw = pd.DataFrame(columns=["full_name", "team", "report_status"])
    official = {r.full_name: r.report_status for r in iw.itertuples()}
    for n in list((ov.get("p_play") or {})):
        if n in official:
            ov["p_play"].pop(n); changed.append(f"p_play {n}: official report says {official[n]}, feed decides")
    # 4. contradictions (p_play 1.0 is NOT a no-op: it clears a practice-report discount, so it stays)
    for n, p in list((ov.get("p_play") or {}).items()):
        if n in ov.get("out", []): ov["p_play"].pop(n); changed.append(f"p_play {n}: also listed out, out wins")
    for n in list(ov.get("back", [])):
        if n in ov.get("out", []): ov["back"].remove(n); changed.append(f"back {n}: also listed out, out wins")
    # 2. per-entry dates. "confirmed": [names or team codes] re-verified today -> their entries are re-dated, list emptied
    live = set(entries(ov))
    dates = {k: v for k, v in dates.items() if k in live}
    conf = set(ov.pop("confirmed", []) or [])
    for k in list(dates):
        if k.split("|")[1] in conf: dates[k] = today
    for k in live: dates.setdefault(k, today)
    json.dump(ov, open(path, "w"), indent=1); json.dump(dates, open(dpath, "w"), indent=0, sort_keys=True)
    for c in changed: print("auto_overrides:", c)
    # 5. stale chances
    for k, d in sorted(dates.items()):
        age = (dt.date.today() - dt.date.fromisoformat(d)).days
        sec = k.split("|")[0]
        if sec in ("p_play", "qb_uncertain", "lead_back") and age > STALE_DAYS:
            print(f"STALE: {k.split('|')[0]} {k.split('|')[1]} set {d} ({age} days) - re-verify or remove")
    print(f"auto_overrides: week {W} ok, {len(live)} entries, {len(changed)} cleaned")


def stale_lines(W):
    """For audit_stale.py: STALE items for week W."""
    out = []
    for k, d in load(f"news_dates_wk{W}.json", {}).items():
        age = (dt.date.today() - dt.date.fromisoformat(d)).days
        if k.split("|")[0] in ("p_play", "qb_uncertain", "lead_back") and age > STALE_DAYS:
            out.append(f"{k.split('|')[0]} {k.split('|')[1]} set {d} ({age} days ago) - re-verify or remove")
    return out


def fresh_notes(W, notes):
    """For build_site.py: team notes no older than NOTE_DAYS (undated notes count as new)."""
    dates = load(f"news_dates_wk{W}.json", {})
    keep = {}
    for t, n in (notes or {}).items():
        d = dates.get(f"team_notes|{t}|{json.dumps(n, sort_keys=True)}", today)
        if (dt.date.today() - dt.date.fromisoformat(d)).days <= NOTE_DAYS: keep[t] = n
    return keep


if __name__ == "__main__":
    for w in sys.argv[1:]: run(int(w))
