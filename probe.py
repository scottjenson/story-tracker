#!/usr/bin/env python3
"""Probe candidate search terms for a new story-tracker topic.

For each keyword: Hacker News Algolia hit counts + top point total, and
Google News RSS item counts, over the window. Free, keyless, no LLM —
use it so a new topic's config.json queries/relevance_terms are chosen
from real signal density, not guesses.

Usage:
  python3 probe.py "window manager" wayland tiling [--days 7]
"""
import sys
import time

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; StoryTrackerProbe/1.0)"}


def hn_stats(query: str, days: int) -> tuple[int, int]:
    since = int(time.time()) - days * 86400
    try:
        r = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": 100,
                    "numericFilters": f"created_at_i>{since}"},
            timeout=20,
        )
        hits = r.json().get("hits", [])
    except (requests.RequestException, ValueError):
        return 0, 0
    pts = [h.get("points") or 0 for h in hits]
    return len(pts), (max(pts) if pts else 0)


def gn_count(query: str) -> int:
    import xml.etree.ElementTree as ET
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=UA, timeout=20,
        )
        return len(ET.fromstring(r.content).findall(".//item"))
    except (requests.RequestException, ET.ParseError):
        return 0


def main() -> int:
    argv = sys.argv[1:]
    days = 7
    queries = []
    for a in argv:
        if a.startswith("--days="):
            days = int(a.split("=", 1)[1])
        else:
            queries.append(a)
    if not queries:
        print(__doc__)
        return 1
    print(f"window: last {days}d")
    print(f"{'keyword':<30} {'HN hits':>8} {'HN top pts':>11} {'GN items':>9}")
    for q in queries:
        hn, top = hn_stats(q, days)
        gn = gn_count(q)
        print(f"{q:<30} {hn:>8} {top:>11} {gn:>9}")
        time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
