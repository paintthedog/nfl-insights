"""Combine the per-week data files into the page. Usage: python assemble_site.py 6 7   (first week listed is shown first)
Oct 18 streamline: the page only ships what it displays. Per-player simulation draws ('sim') never reach the page, and the
percentile tables ('q') keep only receptions (the one stat the page prices against a DraftKings line). The full data stays in
site/data_wk{W}.json for the scripts that need it (paper_slate, audits, lineup builder)."""
import json, sys, os
weeks = [int(w) for w in sys.argv[1:]] or [5]
KEEP_Q = {"rec"}


def rnd(x):
    if isinstance(x, float): return round(x, 3)
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, list): return [rnd(v) for v in x]
    return x


def slim(d):
    d = json.loads(json.dumps(d))
    for p in d.get("proj", {}).get("players", []):
        p.pop("sim", None)
        if isinstance(p.get("q"), dict): p["q"] = {k: v for k, v in p["q"].items() if k in KEEP_Q}
    return rnd(d)


allw = {str(w): slim(json.load(open(f"site/data_wk{w}.json"))) for w in weeks}
opt = lambda f: json.load(open(f)) if os.path.exists(f) else None
payload = dict(weeks=weeks, default=weeks[0], data=allw, trends=opt("trends.json"), awards=opt("awards.json"),
               season=opt("season.json"), record=opt("record.json"), scorecard=opt("scorecard.json"))
payload = rnd(payload)
html = open("site_template.html").read().replace("/*__DATA__*/null", json.dumps(payload, allow_nan=False, separators=(",", ":")))
open("site/paydirt.html", "w").write(html)
print(f"site/paydirt.html: weeks {weeks}, {len(html)/1024:.0f} KB")

# Oct 18: archive what was just published (only when it changed), so the scorecard grades each game off the numbers that were
# actually on the site at kickoff. Stamps are Eastern time, matching the schedule's kickoff times.
import filecmp, glob, shutil, datetime as _dt
os.makedirs("archive", exist_ok=True)
for w in weeks:
    src = f"site/data_wk{w}.json"; prev = sorted(glob.glob(f"archive/data_wk{w}_*.json"))
    if prev and filecmp.cmp(src, prev[-1], shallow=False): continue
    shutil.copy(src, f"archive/data_wk{w}_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}.json")
