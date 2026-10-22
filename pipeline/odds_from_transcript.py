"""Usage: python odds_from_transcript.py out.json [transcript.jsonl]   then: python odds_split.py W out.json
Pull Odds API WebFetch results out of this session's transcript: maps each fetched URL's event id + market group to its text."""
import json, sys, re
import glob, os
T = sys.argv[2] if len(sys.argv) > 2 else max(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")), key=os.path.getmtime)   # this session's transcript
calls, out = {}, {}
for line in open(T):
    try: m = json.loads(line)
    except Exception: continue
    msg = m.get("message") or {}
    for c in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
        if c.get("type") == "tool_use" and c.get("name") == "WebFetch" and "the-odds-api" in str(c.get("input", {}).get("url", "")):
            calls[c["id"]] = c["input"]["url"]
        if c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
            t = c.get("content"); t = "".join(x.get("text", "") for x in t) if isinstance(t, list) else str(t)
            out[calls[c["tool_use_id"]]] = t
json.dump(out, open(sys.argv[1], "w"))
for u, t in out.items():
    eid = re.search(r"events/([0-9a-f]{32})", u); mk = re.search(r"markets=([^&]+)", u)
    print(eid.group(1) if eid else "sport", mk.group(1)[:40] if mk else "", len(t.splitlines()))
