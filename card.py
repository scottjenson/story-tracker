#!/usr/bin/env python3
"""Render the AI narrative tracker (stories.json) into a Television HTML card.

Stage 2 of the AI daily briefing pipeline. Stage 1 (briefing_agent.py) ingests
news and reconciles it into ``stories.json`` via a single bounded LLM call.
This stage is *pure templating* — it reads that structured state and emits a
scan-friendly Television card. No LLM, no network, no tokens.

Design goals:
  * one story = one card block: title, trajectory badge, today's delta,
    entities, source link, and a short date-stamped timeline
  * trajectory is colour-coded so "escalating vs cooling" scans at a glance
  * stories pruned by stage 1 (>7 days quiet) never appear; the card only
    shows live narratives

Env:
  HERMES_TZ   timezone (never hardcoded)
CLI:
  --state  path to stories.json (default ./stories.json)
  --output path to write HTML (default /root/.television/artifacts/ai-narrative-tracker.html)
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

APP_VERSION = "1.4.26"  # Television app version this card is authored against
DEFAULT_OUT = "/root/.television/artifacts/ai-narrative-tracker.html"

# Card identity (reference AI-Narrative defaults; load_card_config() overrides
# per instance from the topic's config.json).
CARD_TITLE = "AI Narrative Tracker"
CARD_SUBTITLE = "Ongoing AI, open-source & reverse-engineering threads"

# Computed trajectory states (see briefing_agent.compute_trajectory) ->
# (label colour, bg tint). Fallback catches unknown values.
TRAJECTORY_STYLE = {
    "heating": ("#15963d", "#e7f6ec"),    # updated today, sustained coverage
    "steady": ("#0f766e", "#e0f2f1"),     # updated today, lighter pace
    "cooling": ("#4a6fa5", "#e9eef7"),    # quiet 1-3 days
    "fading": ("#6c757d", "#f0f1f3"),     # quiet 4+ days, nearing prune
    "unchanged": ("#6c757d", "#f0f1f3"),  # legacy pre-computed value
}
DEFAULT_BADGE = ("#6c757d", "#f0f1f3")

# Card scan order: hottest trajectory on top. Keys match exactly the states
# briefing_agent.compute_trajectory emits (plus the legacy "unchanged" value);
# anything unknown sorts last. Within a trajectory, stories are ordered by
# recency (last_updated desc, then created desc, then title for stability).
TRAJECTORY_RANK = {"heating": 0, "steady": 1, "cooling": 2, "fading": 3, "unchanged": 4}


def tz_name() -> str:
    name = os.environ.get("HERMES_TZ")
    if not name:
        print("ERROR: HERMES_TZ not set", file=sys.stderr)
        sys.exit(1)
    return name


def esc(s: object) -> str:
    return html.escape("" if s is None else str(s))


def badge(text: str, style: tuple[str, str]) -> str:
    fg, bg = style
    return (
        f'<span class="badge" style="color:{fg};background:{bg}">{esc(text)}</span>'
    )


def heat_badge(s: dict) -> str:
    """Velocity badge for breaking-bucket cards: the lane's own signal
    (stars / points / outlet count) in place of a trajectory badge —
    breaking stories are ranked by speed, not by follow-up pace."""
    heat = s.get("heat") or {}
    kind = heat.get("kind")
    if kind == "stars":
        text = f"★ {heat.get('points', 0):,}"
    elif kind == "points" or heat.get("points"):
        text = f"▲ {heat['points']} pts"
    else:
        text = f"{heat.get('communities', '—')} outlets"
    return (f'<span class="badge" style="color:#b3261e;background:#fdecea">'
            f'{esc(text)}</span>')


def link(url: object, label: str) -> str:
    u = "" if url is None else str(url)
    if not u:
        return esc(label)
    return f'<a href="{esc(u)}" target="_blank" rel="noopener">{esc(label)}</a>'


def render_story(s: dict, is_new: bool, today_utc: str,
                 badge_html: Optional[str] = None, card_class: str = "") -> str:
    title = s.get("title") or "(untitled)"
    if badge_html is None:
        traj = (s.get("trajectory") or "unchanged").lower()
        traj_style = TRAJECTORY_STYLE.get(traj, DEFAULT_BADGE)
        head_badge = badge(traj, traj_style)
    else:
        head_badge = badge_html

    timeline = s.get("timeline") or []
    latest = timeline[-1] if timeline else {}
    delta = latest.get("delta") or ""
    src_url = latest.get("source_url") or ""
    latest_date = latest.get("date") or ""

    # Never label a stale entry "Today's delta" — for a story whose latest
    # entry is days old (cooling/fading), that claim made the card look
    # broken even though the trajectory badge was telling the truth.
    if latest_date == today_utc:
        delta_label = "Today&rsquo;s delta"
    else:
        try:
            d = datetime.fromisoformat(latest_date).strftime("%b %d")
            delta_label = f"Last update &middot; {esc(d)}"
        except ValueError:
            delta_label = "Last update"

    # Short timeline: last 3 entries, newest first, date + one-line update.
    rows = []
    for entry in reversed(timeline[-3:]):
        d = entry.get("date") or "—"
        upd = entry.get("update") or ""
        eu = entry.get("source_url") or ""
        upd_html = link(eu, upd) if eu else esc(upd)
        rows.append(
            f'<div class="tl-row"><span class="tl-date">{esc(d)}</span>'
            f'<span class="tl-upd">{upd_html}</span></div>'
        )
    tl_html = "".join(rows) if rows else ""

    ents = s.get("key_entities") or []
    chips = "".join(f'<span class="chip">{esc(e)}</span>' for e in ents)

    overview = s.get("overview") or ""
    extra = f" {card_class}" if card_class else ""
    return f"""
  <article class="story{(' new' if is_new else '')}{extra}">
    <div class="story-head">
      <h2>{esc(title)}</h2>
      <div class="story-badges">
        {head_badge}
      </div>
    </div>
    <div class="overview">{esc(overview)}</div>
    <div class="delta"><span class="delta-label">{delta_label}</span> {esc(delta)}</div>
    {('<div class="chips">' + chips + '</div>') if chips else ''}
    {('<div class="timeline">' + tl_html + '</div>') if tl_html else ''}
    {('<div class="src">' + link(src_url, "Read source →") + '</div>') if src_url else ''}
  </article>"""


def render(state_path: str, output_path: str) -> str:
    with open(state_path, encoding="utf-8") as fh:
        data = json.load(fh)
    stories = data.get("stories", data if isinstance(data, list) else [])
    if not isinstance(stories, list):
        raise ValueError(f"Unexpected structure in {state_path}")

    # A story is "new" if it was first created today (UTC, matching how
    # briefing_agent.py stamps `created`).
    today_utc = datetime.now(timezone.utc).date().isoformat()
    new_ids = {s["id"] for s in stories if s.get("created") == today_utc}

    def _bucket(s: dict) -> str:
        return (s.get("bucket") or "narrative").lower()

    # Two sections, two metrics: breaking is ranked by velocity (points,
    # then corroboration); developing by trajectory (heating top,
    # cooling/fading bottom, recency as tiebreak within a trajectory).
    breaking = sorted(
        (s for s in stories if _bucket(s) == "breaking"),
        key=lambda s: ((s.get("heat") or {}).get("points") or 0,
                       (s.get("heat") or {}).get("communities") or 0,
                       s.get("last_updated") or ""),
        reverse=True,
    )
    narrative = [s for s in stories if _bucket(s) != "breaking"]

    # Stable multi-pass sort — each pass refines the one before it.
    ordered = sorted(narrative, key=lambda s: s.get("title") or "")
    ordered = sorted(ordered, key=lambda s: s.get("created") or "", reverse=True)
    ordered = sorted(ordered, key=lambda s: s.get("last_updated") or "", reverse=True)
    ordered = sorted(
        ordered,
        key=lambda s: TRAJECTORY_RANK.get((s.get("trajectory") or "unchanged").lower(), 99),
    )

    breaking_html = "".join(
        render_story(s, s["id"] in new_ids, today_utc,
                     badge_html=heat_badge(s), card_class="breaking")
        for s in breaking
    )
    developing_html = "".join(
        render_story(s, s["id"] in new_ids, today_utc) for s in ordered
    )
    if not developing_html:
        developing_html = '<div class="empty"><p>No narratives tracked yet.</p></div>'

    # The breaking section only appears when there IS a breakout — an empty
    # breaking section would be the card equivalent of a "BREAKING NEWS" banner
    # with nothing breaking.
    sections = ""
    if breaking_html:
        sections += ('<h3 class="section">Breaking &middot; top velocity, last 24h</h3>\n'
                     + breaking_html + "\n")
    sections += ('<h3 class="section">Developing &middot; still going</h3>\n'
                 + developing_html + "\n")

    now = datetime.now(ZoneInfo(tz_name()))
    updated = now.strftime("%a %b %d, %Y · %I:%M %p %Z")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(CARD_TITLE)}</title>
  <link rel="stylesheet" href="/canonical/v2/styles.css?authoredForAppVersion={APP_VERSION}" />
  <style>
    :root {{
      --surface: #f8f9fa; --text: #212529; --muted: #6c757d;
      --border: #e5e7eb; --accent: #0f766e;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--surface); color: var(--text);
      margin: 0; padding: 28px 28px 48px; line-height: 1.5;
    }}
    header {{ margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid var(--border); }}
    h1 {{ font-size: 20px; font-weight: 650; margin: 0 0 4px; }}
    .sub {{ font-size: 12px; color: var(--muted); }}
    .story {{
      background: #fff; border: 1px solid var(--border); border-radius: 10px;
      padding: 16px 18px; margin-bottom: 14px;
    }}
    .story.new {{ border-left: 3px solid var(--accent); }}
    .story.breaking {{ border-left: 3px solid #c0392b; }}
    .section {{
      font-size: 11px; font-weight: 700; letter-spacing: .08em;
      text-transform: uppercase; color: var(--muted); margin: 24px 0 10px;
    }}
    .story-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
    h2 {{ font-size: 15px; font-weight: 650; margin: 0; }}
    .story-badges {{ display: flex; gap: 6px; flex: 0 0 auto; }}
    .badge {{
      font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
      letter-spacing: .03em; text-transform: uppercase;
    }}
    .overview {{ font-size: 12.5px; color: var(--muted); margin-top: 6px; line-height: 1.45; }}
    .delta {{ font-size: 13px; margin-top: 10px; }}
    .delta-label {{
      font-size: 10px; font-weight: 700; color: var(--accent);
      text-transform: uppercase; letter-spacing: .05em; margin-right: 6px;
    }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .chip {{ font-size: 11px; color: #334155; background: #eef2f7; padding: 2px 9px; border-radius: 6px; }}
    .timeline {{ margin-top: 12px; border-top: 1px solid var(--border); padding-top: 8px; }}
    .tl-row {{ display: flex; gap: 12px; font-size: 12px; padding: 3px 0; }}
    .tl-date {{ flex: 0 0 78px; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .tl-upd {{ color: #495057; }}
    .src {{ margin-top: 10px; font-size: 12px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .empty {{ text-align: center; padding: 40px; color: var(--muted); border: 1px dashed var(--border); border-radius: 8px; }}
    .footer {{ margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 11px; color: var(--muted); }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(CARD_TITLE)}</h1>
    <div class="sub">{esc(CARD_SUBTITLE)} &middot; updated {esc(updated)}</div>
  </header>
{sections}
  <div class="footer">{len(ordered)} narratives tracked &middot; {len(breaking)} breaking &middot; rendered from stories.json &middot; refreshes with the daily pipeline.</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    return output_path


def load_card_config(path: str) -> None:
    """Set CARD_TITLE / CARD_SUBTITLE from a topic config, when present."""
    global CARD_TITLE, CARD_SUBTITLE
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    CARD_TITLE = cfg.get("card_title", CARD_TITLE)
    CARD_SUBTITLE = cfg.get("card_subtitle", CARD_SUBTITLE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="topic config for card identity")
    parser.add_argument("--state", default="stories.json", help="path to stories.json")
    parser.add_argument("--output", default=DEFAULT_OUT, help="output HTML path")
    args = parser.parse_args()
    load_card_config(args.config)
    out = render(args.state, args.output)
    with open(args.state, encoding="utf-8") as fh:
        stories = json.load(fh).get("stories", [])
    print(f"wrote {out}: {len(stories)} narratives tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())