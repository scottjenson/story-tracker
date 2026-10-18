#!/usr/bin/env python3
"""Self-contained test suite for the story-tracker engine.

No live instances, no network, no LLM: synthetic config + fixture stories.
Runs from any checkout:  python3 tests/test_engine.py
"""
import copy
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # <shared>/ story_tracker pkg
from story_tracker import engine as ba  # noqa: E402

# --- synthetic topic config (mirrors the AI example; lanes never touched) ---
CONFIG = {
    "name": "test-topic",
    "topic_description": "test stories",
    "card_title": "Test Tracker",
    "lookback_hours": 48,
    "lanes": {
        "hn": {"queries": ["test"], "min_points": 10},
        "gnews": {"queries": ["test"]},
        "arxiv": {"cats": None},
        "reddit": {"subreddits": []},
        "github": {"topics": []},
    },
    "relevance_terms": ["test"],
    "exclude_regex": None,
    "pool_caps": {"Hacker News": 24, "Google News": 16, "reddit": 12, "premium": 4},
}

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump(CONFIG, fh)
    CFG_PATH = fh.name
ba.load_config(CFG_PATH)

# --- fixture stories (the only state the tests ever see) --------------------
GEM = "developing-gemini-4-argon-responsibly"
BI = "https://www.businessinsider.com/sergey-brin-google-gemini-ai-argon-rsi-2026-10"

FIXTURES = [
    {
        "id": GEM, "bucket": "narrative",
        "title": "Developing Gemini 4 Argon Responsibly",
        "overview": "Google's Gemini 4 safety narrative.",
        "key_entities": ["Google", "Gemini 4"],
        "created": "2026-10-13", "last_updated": "2026-10-16",
        "trajectory": "cooling",
        "timeline": [
            {"date": "2026-10-13",
             "update": "Initial discussions on safety protocols for Gemini 4 have begun.",
             "delta": "Initial discussions on safety protocols for Gemini 4 have begun.",
             "source_url": "https://blog.google/gemini-4-safety-security/"},
            {"date": "2026-10-16",
             "update": "Google's Gemini 4 model is now seen as nearing recursive self-improvement.",
             "delta": "Sergey Brin has stated that Google's AI models, including Gemini 4, are approaching recursive self-improvement capabilities.",
             "source_url": BI},
        ],
    },
    {"id": "stepfun-preview", "bucket": "narrative",
     "title": "StepFun's Flagship Model for Agentic Work",
     "overview": "A model preview.", "key_entities": ["StepFun"],
     "created": "2026-10-15", "last_updated": "2026-10-15", "trajectory": "cooling",
     "timeline": [{"date": "2026-10-15", "update": "Preview released.",
                   "delta": "Preview released.",
                   "source_url": "https://huggingface.co/stepfun-ai/step-5-preview"}]},
    {"id": "thinking-leakage", "bucket": "narrative",
     "title": "The Thinking Leakage in Reinforcement Learning for LLMs",
     "overview": "A paper.", "key_entities": ["reinforcement learning"],
     "created": "2026-10-16", "last_updated": "2026-10-16", "trajectory": "cooling",
     "timeline": [{"date": "2026-10-16", "update": "Study introduces thinking leakage.",
                   "delta": "Study introduces thinking leakage.",
                   "source_url": "https://arxiv.org/abs/2610.14270"}]},
]

NOW = int(ba._utcnow().timestamp())
TODAY = ba._today_iso()

def fresh():
    return copy.deepcopy(FIXTURES)

passed, failed = [], []

def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not cond else ""))

def hot(title, url, pts=None, corr=None, ts=None, source="Hacker News"):
    it = ba.NewsItem(title, url, source, TODAY)
    it.score = pts
    it.ts = ts if ts is not None else NOW - 3600
    if corr:
        it.corroboration = corr
        it.communities = corr
    return it

# ============ dedupe guards ============

items = [ba.NewsItem("Brin RSI rehash", BI, "Google News", TODAY)]
stories = fresh()
gem = next(s for s in stories if s["id"] == GEM)
before = (len(gem["timeline"]), gem["last_updated"])
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [{"story_id": GEM, "delta": "A fresh angle on the RSI rollout timeline.",
                           "update": "x", "source_idx": 1}], "new_stories": []}, items)
check("G1 duplicate-URL match skipped, no timestamp bump",
      n == 0 and (len(gem["timeline"]), gem["last_updated"]) == before)

stories = fresh()
existing_delta = next(x for x in stories if x["id"] == GEM)["timeline"][-1]["delta"]
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [{"story_id": GEM, "delta": existing_delta,
                           "update": "x", "source_idx": 1}], "new_stories": []},
    [ba.NewsItem("Rehash via other outlet", "https://arstechnica.com/ai/brin-rsi", "Ars", TODAY)])
gem = next(x for x in stories if x["id"] == GEM)
check("G2 duplicate-delta (new URL) match skipped", n == 0 and len(gem["timeline"]) == 2)

stories = fresh()
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [{"story_id": GEM, "delta": "Google published its internal Gemini 4 RLHF audit report.",
                           "update": "audit published", "source_idx": 1}], "new_stories": []},
    [ba.NewsItem("Gemini 4 audit", "https://blog.google/gemini-4-rlhf-audit", "Google News", TODAY)])
gem = next(x for x in stories if x["id"] == GEM)
check("G3 genuine new delta appended + bumped",
      n == 1 and len(gem["timeline"]) == 3 and gem["last_updated"] == TODAY)

stories = fresh()
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [],
              "new_stories": [{"id": "dup-test",
                               "title": "Google's Gemini 4 Argon Raises Ethical Concerns",
                               "key_entities": ["Google", "Gemini 4"],
                               "update": "New: Gemini 4 RSI safety board convenes for first session.",
                               "source_idx": 1, "reason": "ongoing"}]},
    [ba.NewsItem("safety board", "https://example.com/safety-board", "Google News", TODAY)])
ids = {x["id"] for x in stories}
gem = next(x for x in stories if x["id"] == GEM)
check("G4 duplicate new story merged, no second card",
      nn == 0 and "dup-test" not in ids and GEM in matched and len(gem["timeline"]) == 3)

stories = fresh()
items5 = [ba.NewsItem("Mistral Ships On-Device Vision Model", "https://mistral.ai/vision", "Google News", TODAY),
          ba.NewsItem("Mistral Ships On-Device Vision Model", "https://hn.example/mistral", "Hacker News", TODAY)]
ba._corroborate(items5)
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [],
              "new_stories": [{"id": "mistral-vision", "title": "Mistral Ships On-Device Vision Model",
                               "key_entities": ["Mistral", "vision"],
                               "update": "first release", "source_idx": 1, "reason": "product line"}]},
    items5)
check("G5 corroborated (2 communities) LLM seed created",
      nn == 1 and "mistral-vision" in {x["id"] for x in stories})

stories = fresh()
items5b = [ba.NewsItem("DeepSeek R2 Tops Coding Benchmark", "https://example.com/deepseek-r2", "Hacker News", TODAY)]
ba._corroborate(items5b)
n, nn, matched, new = ba.apply_evaluation(
    stories, {"matches": [],
              "new_stories": [{"id": "deepseek-r2", "title": "DeepSeek R2 Tops Coding Benchmark",
                               "key_entities": ["DeepSeek"],
                               "update": "new benchmark result", "source_idx": 1, "reason": "model race"}]},
    items5b)
check("G6 single-community LLM seed dropped by gate-2 floor",
      nn == 0 and "deepseek-r2" not in {x["id"] for x in stories})

# URL normalization
check("G7 norm_url www/case/params/fragment",
      ba.norm_url("https://WWW.Example.com/a/b") == ba.norm_url("https://example.com/a/b")
      and ba.norm_url("https://example.com/a/b?utm_source=x&ref=y") == ba.norm_url("https://example.com/a/b")
      and ba.norm_url("https://example.com/a/b#frag") == ba.norm_url("https://example.com/a/b/")
      and ba.norm_url("https://example.com/a/b?page=2") != ba.norm_url("https://example.com/a/b"))

# ============ two-bucket machinery ============

items_b = [hot("Story Alpha Breaks", "https://a.com/alpha", pts=400),
           hot("Story Beta Breaks", "https://b.com/beta", pts=1000),
           hot("Story Gamma Breaks", "https://c.com/gamma", pts=600),
           hot("Story Delta Breaks", "https://d.com/delta", pts=350),
           hot("Underfloor Story", "https://e.com/eps", pts=100)]
sel = ba.select_breaking(fresh(), items_b, TODAY)
check("B1 breaking seeds top-3 by velocity",
      [s["title"] for s in sel] == ["Story Beta Breaks", "Story Gamma Breaks", "Story Alpha Breaks"])

check("B2 stale (36h) item not breaking-eligible",
      ba.select_breaking(fresh(), [hot("Old Hot", "https://old.com/hot", pts=900, ts=NOW - 36 * 3600)], TODAY) == [])

sel3 = ba.select_breaking(fresh(), [hot("Big Policy Story", "https://gn.com/policy", pts=None, corr=3)], TODAY)
check("B3 corr>=3 no-points item breaking-eligible", len(sel3) == 1)

check("B4 tracked URL not re-seeded",
      ba.select_breaking(fresh(), [hot("Tracked URL", BI, pts=500)], TODAY) == [])

check("B5 title-overlap not re-seeded",
      ba.select_breaking(fresh(), [hot("Developing Gemini 4 Argon Responsibly: Update", "https://x.com/g", pts=700)], TODAY) == [])

def mk_breaking(created=TODAY):
    return {"id": "test-breakout", "bucket": "breaking", "title": "Test Breakout",
            "overview": "Breakout: 800 Hacker News points in under a day.",
            "key_entities": [], "created": created, "last_updated": created,
            "heat": {"points": 800, "communities": None, "kind": "points"},
            "timeline": [{"date": created, "update": "Test Breakout",
                          "delta": "Test Breakout", "source_url": "https://anthropic.com/research/x"}]}

s6 = [mk_breaking()]
ba.promote_breaking(s6)
check("B6a single-publisher breakout not promoted", s6[0]["bucket"] == "breaking")
s6[0]["timeline"].append({"date": TODAY, "update": "Follow-up", "delta": "Follow-up",
                          "source_url": "https://www.theverge.com/follow-up"})
check("B6b two publishers -> promoted", ba.promote_breaking(s6) and s6[0]["bucket"] == "narrative")
s6c = [mk_breaking()]
s6c[0]["timeline"].append({"date": TODAY, "update": "Same-domain follow-up", "delta": "d",
                           "source_url": "https://anthropic.com/news/y"})
ba.promote_breaking(s6c)
check("B6c same-publisher follow-up not promoted", s6c[0]["bucket"] == "breaking")

s8 = fresh() + ba.select_breaking(fresh(), [hot("Shadow Story", "https://shadow.com/story", pts=500)], TODAY)
s8.append({"id": "shadow-llm", "bucket": "narrative", "title": "Shadow Story Revisited",
           "key_entities": [], "created": TODAY, "last_updated": TODAY,
           "timeline": [{"date": TODAY, "update": "u", "delta": "d",
                         "source_url": "https://shadow.com/story"}]})
ba._drop_url_shadows(s8)
check("B8 URL shadow dropped, earlier card kept",
      all(x["id"] != "shadow-llm" for x in s8) and any(x.get("bucket") == "breaking" for x in s8))

# ============ high-signal auto-seed ============

it_c1 = hot("Claude reveals ECDSA curve origins", "https://anthropic.com/research/ecdsa",
            pts=1041, corr=2, ts=NOW - 60 * 3600)
check("C1a big+corroborated but 2.5d old: not breaking",
      ba.select_breaking(fresh(), [it_c1], TODAY) == [])
sel_c1 = ba.select_narrative_autoseed(fresh(), [it_c1], TODAY)
check("C1b ... auto-seeded as narrative",
      len(sel_c1) == 1 and sel_c1[0]["bucket"] == "narrative")

check("C2 points-only (single community) not auto-seeded",
      ba.select_narrative_autoseed(fresh(), [hot("Giant Demo", "https://x.com/demo", pts=900)], TODAY) == [])

check("C3 sub-floor corroboration not auto-seeded",
      ba.select_narrative_autoseed(fresh(), [hot("Quiet Story", "https://y.com/q", pts=200, corr=3)], TODAY) == [])

check("C4 tracked overlap not re-seeded",
      ba.select_narrative_autoseed(fresh(), [hot("Developing Gemini 4 Argon Responsibly Update", "https://z.com/g", pts=700, corr=2)], TODAY) == [])

# ============ trajectory + retention ============

story = {"timeline": [{"date": d} for d in ("2026-10-16", "2026-10-17", TODAY)],
         "last_updated": TODAY}
check("T1 heating = updated today + >=3 entries in 7d", ba.compute_trajectory(story) == "heating")
story2 = {"timeline": [{"date": TODAY}], "last_updated": TODAY}
check("T2 steady = updated today, light pace", ba.compute_trajectory(story2) == "steady")
story3 = {"timeline": [{"date": "2026-10-16"}], "last_updated": "2026-10-16"}
check("T3 cooling = quiet 1-3 days", ba.compute_trajectory(story3, today=TODAY) in ("cooling", "fading"))

import datetime as _dt
old = {"id": "old", "bucket": "narrative", "title": "Old",
       "last_updated": (ba._utcnow() - _dt.timedelta(days=9)).date().isoformat(), "timeline": []}
brk = [mk_breaking()]
kept = ba.prune_stale(brk + [old])
check("T4 7-day prune hits narratives, breaking passes through",
      any(x["id"] == "test-breakout" for x in kept) and not any(x["id"] == "old" for x in kept))
narrs = [{"id": f"n{i}", "bucket": "narrative", "title": f"N{i}",
          "last_updated": TODAY, "created": TODAY} for i in range(9)]
capped = ba.cap_stories(narrs + brk)
check("T5 cap applies to narrative bucket only", len(capped) == 8)

print(f"\n{len(passed)} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
