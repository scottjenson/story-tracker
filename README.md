# story-tracker

Stateful daily **narrative trackers** that maintain running stories per topic —
what changed today (delta), story trajectory (heating → fading), and a
**Breaking** section for high-velocity stories of the last 24h — rendered as
Television TV cards (or plain markdown briefings, if you don't have TV).

One shared engine, one `config.json` per topic, one TV card each. Free
keyless ingestion (Hacker News, Google News, Reddit RSS, GitHub, arXiv),
one small OpenAI-compatible LLM call per topic per day for narrative
reconciliation, and a stack of deterministic guards that keep the LLM from
producing duplicate stories, inflated "heating" badges, or missed
high-signal news.

Live reference instances (the configs in `examples/` come from these):

| Topic | What it tracks |
|---|---|
| AI Narrative Tracker | AI / LLM / open-source / reverse-engineering threads |
| Desktop UX Tracker | Desktop environments, window managers, canvas/spatial desktops |

## Install

```bash
# 1. Clone so the package is importable as `story_tracker`
mkdir -p /opt/data/shared && cd /opt/data/shared
git clone <your-remote-url> story_tracker

# 2. Create a topic instance (copy an example, edit config.json)
mkdir -p /opt/data/<my-topic>
cp -r examples/ai-daily-briefing/* /opt/data/<my-topic>/   # config.json + run.py
cd /opt/data/<my-topic> && $EDITOR config.json
cp /path/to/story_tracker/.env.example .env && $EDITOR .env  # your LLM key

# 3. Dry run (creates the state file, one LLM call)
python3 run.py --state /tmp/probe_state.json --output /tmp/probe_brief.md

# 4. Cron (Linux cron, $0 agent cost — one LLM call/day via your key)
#    see skills/story-tracker/SKILL.md for the exact cron line

# 5. TV card (only with the Television app installed)
tv create-path-artifact --channel <channel-id> --title "My Topic Tracker" \
   --path /root/.television/artifacts/<my-topic>.html --no-focus
```

For **Hermes** agents: also copy `skills/story-tracker/SKILL.md` into the
profile's skills directory (e.g. `/root/.hermes/skills/story-tracker/`) —
it documents the architecture, every gate, the new-topic recipe, and the
pitfalls. Requires Python 3.10+ and `requests` (the only dependency).

## Repo layout

```
engine.py          # the pipeline: ingest → buckets → LLM reconcile → guards → state
card.py            # TV card renderer (Breaking + Developing sections)
probe.py           # keyword signal-density probe — use BEFORE writing a config
repair_state.py    # cleanup for legacy/corrupted state (uses the engine's own guards)
examples/          # real working configs + run.py shims
tests/test_engine.py   # self-contained test suite (synthetic fixtures, no live paths)
skills/story-tracker/  # the Hermes skill that documents all of this
```

## What makes it not-fall-over

- **Two buckets, two metrics**: *Breaking* (≤3, code-seeded by velocity,
  2-day TTL, promoted to narrative on 2+ distinct publishers) and *Narrative*
  (≤7, LLM-curated follow-up, 7-day retention). One list forcing one metric is
  how big stories go missing.
- **Gates are code, not prompt**: cross-run URL dedupe, timeline
  duplicate guard (same URL or ≥85%-similar text → skipped *without* bumping
  the freshness timestamp), cross-story merge, cross-story URL
  contamination guard, a corroboration floor on LLM seeds, and a
  high-signal auto-seed so big+corroborated stories can't fall through
  both lanes. Full table in the skill.
- **Trajectories are computed** (timeline dates), never LLM-judged.

## Privacy / secrets

No secrets in this repo. Instances keep their own `.env`
(LLM key) and `stories.json` (state) outside of it — both gitignored by
convention. Topic configs are public-safe.
