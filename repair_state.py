#!/usr/bin/env python3
"""One-off legacy-state repair for a story-tracker instance, using the
shared engine's own duplicate guards as the oracle.

Pass 1 (per story): drop timeline entries the engine's
_timeline_entry_duplicate flags against the kept prefix (repeated source
URL, or near-identical delta/update text).
Pass 2 (cross story): one URL belongs to one story. A URL carried by
multiple stories keeps its FIRST (chronological) home; later copies are
dropped as contamination.

Recomputes last_updated + trajectory via the engine, saves atomically.
Dry-run by default; --apply writes.
"""
import sys

sys.path.insert(0, "/opt/data/shared")
from story_tracker import engine as e

CONFIG, STATE = sys.argv[1], sys.argv[2]
APPLY = "--apply" in sys.argv
e.load_config(CONFIG)

stories = e.load_stories(STATE)
dropped = []

for s in stories:
    kept = []
    for entry in s.get("timeline", []):
        probe = {**s, "timeline": kept}
        if e._timeline_entry_duplicate(probe, entry):
            dropped.append((s["id"], entry["date"], entry.get("source_url"), "dup-within-story"))
        else:
            kept.append(entry)
    s["timeline"] = kept

# Pass 2: cross-story URL uniqueness (first chronological home wins).
home: dict[str, tuple[str, str]] = {}
for s in stories:
    for entry in s.get("timeline", []):
        u = e.norm_url(entry.get("source_url") or "")
        if not u:
            continue
        if u not in home:
            home[u] = (s["id"], entry["date"])
for s in stories:
    kept = []
    for entry in s.get("timeline", []):
        u = e.norm_url(entry.get("source_url") or "")
        if u and home.get(u, (None,))[0] != s["id"]:
            dropped.append((s["id"], entry["date"], entry.get("source_url"),
                            f"cross-story (home: {home[u][0]} @ {home[u][1]})"))
        else:
            kept.append(entry)
    s["timeline"] = kept
    if kept:
        s["last_updated"] = max(en["date"] for en in kept)

for s in stories:
    s["trajectory"] = e.compute_trajectory(s)

print("dropped entries:")
for sid, d, u, why in dropped:
    print(f"  - {sid}  {d}  {u}  [{why}]")
print(f"\n{len(stories)} stories, {sum(len(s['timeline']) for s in stories)} entries remain:")
for s in stories:
    print(f"  {s['trajectory']:<9} last={s['last_updated']}  {len(s['timeline'])} entries  {s['title'][:55]}")

if APPLY:
    e.save_stories(STATE, stories)
    print(f"\nAPPLIED -> {STATE}")
else:
    print("\ndry run (no writes); re-run with --apply to write.")
