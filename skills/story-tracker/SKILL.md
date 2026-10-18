---
name: story-tracker
description: "Story-tracker cards: shared engine + per-topic config."
version: 1.0.0
author: "Hermes Agent"
license: "MIT"
metadata:
  hermes:
    tags: [pipelines, television, trackers, news]
    category: devops
    related_skills: [data-pipeline-maintenance, ai-news-briefing]
---

# Story Tracker — topic narrative trackers (TV cards)

## When to Use

- Creating a new narrative-tracker TV card for a topic (e.g. "track crypto mining", "track open-source fonts").
- Debugging tracker output: duplicate stories/cards, inflated "heating" badges, missing big stories, stale "Today's delta" labels.
- Tuning gates, thresholds, lanes, or queries of an existing tracker.

Stateful daily trackers that maintain **running narratives** per topic: what
changed today (delta), trajectory (heating→fading), plus a **Breaking**
section for high-velocity stories of the last 24h. One shared engine,
per-topic config, one TV card each. Use for: creating a new tracker topic,
debugging duplicate/missing/inflated stories, tuning ranking/gates.

Live instances: `/opt/data/ai-daily-briefing` (AI Narrative Tracker),
`/opt/data/desktop-ux-briefing` (Desktop UX Tracker).

**Installing from the repo:** this skill ships inside the story-tracker repo
at `skills/story-tracker/`. Clone the repo so the package is importable as
`story_tracker` (default `/opt/data/shared/story_tracker`; elsewhere set the
`STORY_TRACKER_HOME` env var in the `run.py` shims), copy this SKILL.md into
your Hermes profile's skills directory, then follow the recipe below.
Self-contained test suite: `python3 tests/test_engine.py` (no network, no keys).

## Architecture

```
/opt/data/shared/story_tracker/          # THE engine — edit ONCE, all topics fixed
    engine.py     # ingest → bucket maintenance → LLM reconcile → guards → state
    card.py       # TV card renderer (sections, trajectory/heat badges)
    probe.py      # keyword signal-density probe (free, keyless) — use to pick queries
    repair_state.py  # one-off legacy-state cleanup using the engine's own guards
/opt/data/<topic>/                        # one dir per topic (~5 files each)
    config.json   # topic identity: lanes+queries, keywords, title, prompt line
    run.py        # 10-line shim: sys.path → engine.main()
    stories.json  # state (do not hand-edit; use repair_state.py)
    .env          # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (OpenAI-compatible)
```

Flow (engine `run()`): load state (bucket default "narrative") → ingest
(6 lanes, per-lane caps, relevance + exclude gates, cross-lane corroboration)
→ cross-run URL filter → **promote_breaking** (2+ distinct publishers →
narrative) → **expire_breaking** (2-day TTL) → **select_breaking** (top-3
velocity of last 24h, code-only) → **select_narrative_autoseed** (big AND
corroborated, code-only) → ONE LLM call (matches + ≤2 narrative seeds) →
`apply_evaluation` (all guards below) → **_drop_url_shadows** (one card per
URL across buckets) → prune (narratives 7d) → cap (narratives 7) →
recompute trajectories → save (atomic) → markdown → cron renders card.

## The two-bucket model (why it exists)

One list forces one metric — the compromise that produced the original bug
(big stories missing, stale stories inflated). Now:
- **Breaking** (≤3, 2-day TTL): code-seeded, velocity-ranked (pts/stars/communities),
  freshness-gated (≤24h). Answers "what just blew up". Card section hidden
  when empty (no fake BREAKING banners). Breakouts that earn 2+ distinct
  publishers **promote** to narrative; otherwise they age out. Velocity
  badge (★ stars / ▲ pts / N outlets), red accent, no trajectory badge.
- **Narrative** (≤7, 7-day retention): LLM-curated follow-up stories plus
  the code auto-seed floor. Trajectory badge + trajectory-sorted.

## Deterministic gates & guards (all in code, zero tokens, calibrated)

Thresholds are module constants in engine.py; per-topic override via config
`"thresholds": {"HEAT_POINTS_FLOOR": 500}` etc.

| Guard | Rule | Why (calibrated on real data) |
|---|---|---|
| Cross-run URL dedupe | normalized URL consumed by any story is filtered pre-LLM | 48h window re-ingests same URLs daily; normalization (tracking params, www, fragment, case) stops variant-URL escapes |
| Timeline dup guard | match skipped if same URL in that story OR delta/update ≥85% similar to any existing entry | identical repeats score 1.00, genuine deltas ≤0.27 — 0.85 sits between |
| **No bump on skip** | skipped matches do NOT update `last_updated` | the quiet clock is what cools a story; bumping = artificial "heating" (the original bug) |
| Cross-story merge | LLM new story sharing ≥50% entities (OR ≥25% title Jaccard) with a tracked story → folded in | duplicate-story cards; Jaccard 0.33 dup vs 0.11 distinct on real pair |
| Cross-story URL guard | match skipped if source URL belongs to ANOTHER story's timeline | LLM attaches wrong article = contamination |
| Gate 2 floor | LLM narrative seed must rest on item with ≥2 communities | LLM prompt gates are SUGGESTIONS — it missed a 1041-pt corroborated story and seeded a 245-pt single-source one in the same run |
| High-signal auto-seed | ≥500 pts AND ≥2 communities → narrative card in code | the "both-lanes hole": big+corroborated but no longer fresh (the ECDSA case). Points-only = breaking's job; corroboration-only = stock op-ed noise |
| Breaking eligibility | (≥300 pts OR ≥3 communities) AND published ≤24h | daily run cadence: 24h gate keeps "breaking" honest vs the AI-news card's 48h top list |
| URL shadow collapse | LLM seed reusing an earlier card's URL → narrative dropped | one story, one card, across buckets |
| Same-day guard | story already updated today → re-run matches skipped | idempotent cron re-runs |

**Trajectory** (computed, never LLM-judged): heating = updated today AND ≥3
timeline entries in 7d; steady = updated today; cooling = quiet 1-3d; fading
= quiet 4d+. Card sort: heating→steady→cooling→fading, recency tiebreak.
Delta label says "Today's delta" ONLY for same-day entries, else
"Last update · Oct 16" — a stale entry labeled "today" reads as broken.

## Config schema (config.json)

```jsonc
{
  "name": "my-topic",                        // logs, UA, briefing title
  "topic_description": "…for the prompt line…", // "You maintain a log of developing <<topic>>"
  "card_title": "My Topic Tracker",
  "card_subtitle": "Ongoing …",
  "lookback_hours": 48,
  "lanes": {
    "hn":     { "queries": ["…"], "min_points": 10 },
    "gnews":  { "queries": ["…"] },
    "arxiv":  { "cats": "cat:cs.AI OR …" },   // null/absent = disabled
    "reddit": { "subreddits": ["…"] },
    "github": { "topics": ["…"] }              // recently-pushed repos, star-ranked; [] = disabled
  },
  "relevance_terms": ["…"],                  // substring gate ("ai" is word-boundary special-cased in code)
  "exclude_regex": "…" | {"pattern": "…", "flags": "ignorecase"} | null,
  "pool_caps": {"Hacker News": 24, "Google News": 16, "GitHub": 6, "reddit": 12, "arXiv": 4, "premium": 4},
  "thresholds": {}                            // optional: any engine constant override
}
```

LLM call: OpenAI-compatible via .env (`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`;
OpenRouter recommended). One call/day/topic. Prompt sends items with integer
`idx` — the model never re-emits URLs (hallucination shield).

## Recipe: create a new tracker for topic X

1. **Probe first** (never guess queries): `python3 /opt/data/shared/story_tracker/probe.py "kw1" "kw2" … --days=7`. Keep queries with real hit counts; top-pts shows whether the floor (300) is reachable.
2. **Draft** `/opt/data/<x>/config.json` + copy a `run.py` from an existing instance. Topic description = what an editor would call the beat (feeds the prompt).
3. **Dry run**: `cd /opt/data/<x> && python3 run.py --state /tmp/x_state.json --output /tmp/x_brief.md` (fresh empty state; one LLM call). Inspect: pool size, breaking seeds, story quality.
4. **Cron** (`/etc/cron.d/<x>-briefing`, write via tee; user field `root`; inline `HERMES_TZ`; stagger ≥30 min from other trackers to avoid HN Algolia collisions):
   `MM HH * * * root cd /opt/data/<x> && HERMES_TZ=<tz> /usr/bin/python3 run.py --state <state> --output <brief.md> >> /tmp/<x>_cron.log 2>&1 && HERMES_TZ=<tz> /usr/bin/python3 /opt/data/shared/story_tracker/card.py --config config.json --state <state> --output /root/.television/artifacts/<x>.html >> /tmp/<x>_cron.log 2>&1`
5. **TV artifact** (once): `tv create-path-artifact --channel <news-channel> --title "<card_title>" --path /root/.television/artifacts/<x>.html --no-focus`
6. **Verify**: grep rendered HTML for `<h1>`, sections, badges; check cron log for `Loaded topic config`, `Ingested …`, `Reconciliation: …`.

## Lane quirks

- HN Algolia `/search` (relevance+popularity) beats `search_by_date`; soft 10-pt floor.
- Google News RSS: cluster by normalized title; corroboration = distinct publishers; cluster URLs change per publisher → normalize URLs for dedupe, never trust raw equality.
- Reddit: `.json` API is 403 on datacenter IPs → use `/.rss` hot feeds with backoff; expect 429s in logs (normal).
- GitHub: repo search `topic:<t> pushed:><window>`, sort stars; stargazers_count is the score; star-ranked repos are the news for project-centric topics.
- arXiv export API lags live listings ~3 days (built into its floor).

## Pitfalls (learned the hard way)

- **LLM gates are suggestions, not enforcement.** Anything you need to happen must also exist in code (the missed 1041-pt story and the seeded 245-pt demo were the same run). Code = floors; LLM = discretion above floors.
- **Unit tests must not read live state as their baseline.** Use fixtures for volatile stories — the live file drifts as the real pipeline runs (happened twice in one session).
- **`logging.basicConfig` before `load_config`** — config-load logs are otherwise silently dropped.
- **Prompt templates: use `<<TOPIC>>` + `str.replace`, not `.format`** — the prompt body is full of literal JSON braces.
- **Card render `--output` is per-instance** — the shared card.py default is the AI card's path; forgetting it silently overwrites the wrong artifact.
- GitHub-lane breaking entries carry `heat.kind="stars"` (★ badge); legacy entries may lack `kind` (falls back to pts label — fix with a small script).
- A story matching a *different* project in the same niche (e.g. tinywm into the Niri story) is NOT catchable by deterministic guards (different URL, different words) — it's LLM judgment; review weekly, accept the error rate.
- `save_stories` is atomic (tmp+replace); never edit stories.json by hand mid-run.
- Keep legacy monolith scripts in place (untouched) for rollback until 2 clean cron days on the engine, then delete.

## Repairing legacy/corrupted state

`python3 /opt/data/shared/story_tracker/repair_state.py <config.json> <state.json> [--apply]`
— dry-run first; it drops entries the engine's own guards flag (in-story dup
URL/text, cross-story URL contamination) and recomputes last_updated +
trajectory. Expect "heating" stories to become cooling/steady — that is the
fix, not a regression.

## Token economics

Engine is $0 (stdlib+requests). Exactly ONE LLM call per topic per day
(~1-2k tokens at gpt-4o-mini scale). Never convert to a Hermes agent cron.
