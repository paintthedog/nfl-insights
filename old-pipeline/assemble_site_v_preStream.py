"""Combine the per-week data files into the page. Usage: python assemble_site.py 4 5   (first week listed is shown first)"""
import json, sys
weeks = [int(w) for w in sys.argv[1:]] or [5]
allw = {str(w): json.load(open(f"site/data_wk{w}.json")) for w in weeks}
import os
payload = dict(weeks=weeks, default=weeks[0], data=allw, trends=json.load(open('trends.json')) if os.path.exists('trends.json') else None, awards=json.load(open('awards.json')) if os.path.exists('awards.json') else None, season=json.load(open('season.json')) if os.path.exists('season.json') else None, record=json.load(open('record.json')) if os.path.exists('record.json') else None)
html = open("site_template.html").read().replace("/*__DATA__*/null", json.dumps(payload, allow_nan=False))
open("site/paydirt.html", "w").write(html)
print(f"site/paydirt.html: weeks {weeks}, {len(html)/1024:.0f} KB")
