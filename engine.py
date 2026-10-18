#!/usr/bin/env python3
"""Stateful daily AI narrative briefing agent.

Ingests fresh AI / reverse-engineering / developer-tools news, reconciles it
against a persistent narrative store (``stories.json``), emits the daily
"delta" per developing story, and writes an evolutionary Markdown briefing.

Design notes:
  * The LLM is *optional*. Without ``LLM_API_KEY`` the agent still fetches,
    deduplicates, prunes stale stories, and emits a raw "digest" briefing — it just
    skips narrative reconciliation. This keeps the pipeline from hard-failing
    on a rate limit or missing key.
  * The LLM client is OpenAI-compatible only (``base_url + key``), which covers
    OpenAI, OpenRouter, Nous-via-``hermes proxy``, and any OpenAI-compatible
    endpoint. Anthropic models are reachable through OpenRouter's compatible
    API, so no second client is maintained.
  * LLM output is validated and repaired in code — never trusted blind.

Environment variables:
  LLM_API_KEY           OpenAI-compatible API key (falls back to
                        OPENROUTER_API_KEY, then OPENAI_API_KEY).
  LLM_BASE_URL          Base URL (default https://api.openai.com/v1). For
                        OpenRouter: https://openrouter.ai/api/v1. For local
                        Hermes: http://127.0.0.1:<proxy-port>/v1.
  LLM_MODEL             Model id (default gpt-4o-mini).
  EXA_API_KEY           Optional Exa search key (preferred ingestion).
  TAVILY_API_KEY        Optional Tavily search key (preferred ingestion).
  DISCORD_WEBHOOK_URL   Optional Discord webhook URL.
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   Optional Telegram delivery.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DAYS_TO_RETAIN: int = 7          # a story stays tracked if updated within this many days
MAX_STORIES: int = 7             # hard ceiling; evict oldest-last_updated beyond this
DEFAULT_LOOKBACK_HOURS: int = 48
MAX_ITEMS_TO_EVALUATE: int = 60          # cap tokens sent to the LLM
MAX_TIMELINE_ITEMS_SENT: int = 3         # only recent timeline sent to the LLM
REQUEST_TIMEOUT: float = 20.0
RETRY_DELAY: float = 3.0

HN_QUERIES: list[str] = [
    "LLM", "AI model", "GPT", "Claude", "Gemini",
    "open source model", "machine learning", "AI",
    "AI agent", "inference", "artificial intelligence",
    "reverse engineer", "distillation", "coding assistant",
]
MIN_HN_POINTS: int = 10          # soft recall floor; drops show-and-tell noise
# Query params that never identify article content — stripped before URL
# dedupe so the same story carried with tracking params still collapses.
TRACKING_PARAMS: set[str] = {
    "ref", "ref_src", "ref_url", "source", "gclid", "fbclid", "cmpid",
    "ns_source", "icid", "mbid", "s_kwcid", "spm", "mkt_tok", "wbx",
    "utm_id", "wt.svc", "wt.mc_id",
}
# Duplicate-detection thresholds (calibrated on real 2026-10-16..18 data:
# identical/near-identical repeats score 1.00, genuinely new deltas <= 0.27;
# duplicate story titles score 0.33 Jaccard vs 0.11 for distinct stories).
DELTA_SIMILARITY_MIN: float = 0.85   # ratio at/above = duplicate text
TITLE_JACCARD_MIN: float = 0.25      # title content-word overlap for same narrative
ENTITY_COVERAGE_MIN: float = 0.5     # fraction of a new story's entities already tracked

# Two-bucket tracking model:
#   "breaking"  — the top high-velocity stories of the last ~24h. Seeded by
#                 CODE (no LLM), velocity-ranked, short TTL. Answers
#                 "what just blew up" — a breaking story reaches the card
#                 even when the follow-up gate never would.
#   "narrative" — stories with follow-up coverage, LLM-curated, 7-day
#                 retention. Answers "what's still going".
# Separate metrics (velocity vs trajectory), separate caps, separate
# lifecycles: one list no longer has to compromise between them. A breaking
# story that earns follow-up coverage (2+ distinct publishers) is promoted
# into the narrative bucket; one that doesn't ages out at the TTL.
MAX_BREAKING: int = 3                     # breaking-bucket cap (tunable)
BREAKING_TTL_DAYS: int = 2                # breakouts age out unless promoted
BREAKING_MAX_AGE_HOURS: int = 24          # eligibility: published within ~24h
HEAT_POINTS_FLOOR: int = 300              # velocity lane: HN points floor
BREAKING_CORROBORATION_FLOOR: int = 3     # velocity lane: mainstream-only breakout
GATE2_CORROBORATION_MIN: int = 2          # LLM narrative seeds: code-enforced floor
# High-signal auto-seed (narrative bucket, code-only, no LLM discretion): a
# story big AND corroborated cannot fall through both lanes — it is no longer
# fresh enough for breaking, but it is exactly the kind of story the card must
# carry. Points-only is breaking's job; corroboration-only (multi-outlet stock
# op-ed clusters) is financial noise, not a narrative thread.
AUTO_SEED_POINTS_FLOOR: int = 500
AUTO_SEED_COMMUNITIES_MIN: int = 2
GN_QUERIES: list[str] = [
    "artificial intelligence",
    "LLM",
    "OpenAI OR Anthropic OR DeepMind",
    "AI model release",
    "AI funding",
    "AI regulation",
    "Nvidia AI chips",
]
ARXIV_CATS: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
REDDIT_SUBREDDITS: list[str] = [
    "LocalLLaMA", "MachineLearning", "programming", "reverseengineering", "rust",
]
# Per-lane caps for the LLM evaluation pool (in signal-strength order).
# No single lane can monopolize the budget; total <= MAX_ITEMS_TO_EVALUATE.
POOL_CAPS: dict[str, int] = {
    "Hacker News": 24, "Google News": 16, "arXiv": 4, "reddit": 12, "premium": 4,
}
# Free relevance gate: protects the evaluation budget from off-topic items.
# "ai" is matched on word boundaries (bare substring hits "email", "chain");
# the rest are safe as substrings. Reverse-engineering terms are in scope for
# this card, so they pass the gate too.
RELEVANCE_TERMS: tuple[str, ...] = (
    "llm", "gpt", "claude", "gemini", "chatgpt", "model", "agent",
    "machine learning", "neural", "inference", "transformer", "diffusion",
    "anthropic", "openai", "deepseek", "mistral", "llama", "deepmind",
    "nvidia", "gpu", "copilot", "artificial intelligence", "grok",
    "reverse engineer", "decompil", "distillation", "jailbreak",
    "coding assistant", "developer tool",
)
# Optional hard-exclusion regex (topic config may set it): titles matching it
# are dropped before the LLM sees them, regardless of relevance.
EXCLUDE_RE: Optional["re.Pattern"] = None
# Optional GitHub topic lane (topic config may set it): recently-pushed repos
# searched by topic, star-ranked. None = lane disabled.
GITHUB_TOPICS: Optional[list[str]] = None
# Instance identity (set from config; used in logs and API user-agents).
_NAME: str = "story-tracker"

log = logging.getLogger("briefing")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today_iso() -> str:
    return _utcnow().date().isoformat()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

def _make_id(raw: str) -> str:
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or _make_id(text)


def safe_iso(ts: Any, created_utc: bool = False) -> str:
    """Coerce a numeric timestamp into an ISO date string."""
    try:
        epoch = float(ts)
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.date().isoformat()
    except (TypeError, ValueError):
        return _today_iso()


def days_since(iso_date: str) -> int:
    try:
        d = datetime.fromisoformat(iso_date).date()
        return (_utcnow().date() - d).days
    except ValueError:
        return 0


class NewsItem:
    def __init__(
        self,
        title: str,
        url: str,
        source: str,
        published: str,
        snippet: str = "",
        score: Optional[int] = None,
    ) -> None:
        self.title = (title or "").strip()
        self.url = (url or "").strip()
        self.source = source
        self.published = published
        self.snippet = (snippet or "").strip()
        self.score = score
        self.corroboration: Optional[int] = None  # distinct communities (2+)
        self.communities: Optional[int] = None    # community count incl. 0/1
        self.ts: Optional[int] = None             # epoch seconds, when known
        self.id = _make_id(self.url or self.title)

    def to_json(self) -> dict[str, Any]:
        rec = {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published": self.published,
            "snippet": self.snippet,
            "score": self.score,
        }
        if self.corroboration:
            rec["corroboration"] = self.corroboration
        return rec


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #

def load_stories(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        log.warning("No state file at %s; starting with an empty story set.", path)
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    stories = data.get("stories", data if isinstance(data, list) else [])
    if not isinstance(stories, list):
        raise ValueError(f"Unexpected structure in {path}: expected a list of stories")
    return stories


def save_stories(path: str, stories: list[dict[str, Any]]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"stories": stories}, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic write


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def _get_json(url: str, *, headers: Optional[dict[str, str]] = None,
              params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    for attempt in range(3):
        try:
            resp = requests.get(
                url, headers=headers, params=params, timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                log.warning("Rate limited (%d) on %s; %s", resp.status_code, url, resp.url)
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            if resp.status_code >= 400:
                log.warning("HTTP %d fetching %s", resp.status_code, resp.url)
                return None
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Request error fetching %s: %s", url, exc)
            time.sleep(RETRY_DELAY)
    return None


def _dedupe(items: list[NewsItem]) -> list[NewsItem]:
    seen: dict[str, NewsItem] = {}
    for it in items:
        if it.url and it.url not in seen:
            seen[it.url] = it
    return list(seen.values())


def fetch_exa(since_ts: int) -> list[NewsItem]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        return []
    out: list[NewsItem] = []
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    query = "AI OR LLM OR (reverse engineering) OR (developer tools)"
    try:
        resp = requests.post(
            "https://api.exa.ai/search",
            headers=headers,
            json={"query": query, "numResults": 20, "startPublishedDate": f"{since_ts}",
                  "type": "keyword"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            out.append(NewsItem(
                title=r.get("title") or "",
                url=r.get("url") or "",
                source="exa",
                published=r.get("publishedDate") or _today_iso(),
                snippet=(r.get("text") or "")[:300],
            ))
    except (requests.RequestException, ValueError) as exc:
        log.warning("Exa ingestion failed: %s", exc)
    return out


def fetch_tavily(days: int) -> list[NewsItem]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    out: list[NewsItem] = []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": "AI LLM reverse engineering developer tools news",
                  "topic": "news", "days": days,
                  "max_results": 20},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        for r in resp.json().get("results", []):
            out.append(NewsItem(
                title=r.get("title") or "",
                url=r.get("url") or "",
                source="tavily",
                published=_today_iso(),
                snippet=(r.get("content") or "")[:300],
                score=r.get("score"),
            ))
    except (requests.RequestException, ValueError) as exc:
        log.warning("Tavily ingestion failed: %s", exc)
    return out


def fetch_hn(since_ts: int) -> list[NewsItem]:
    """HN stories from the query set. /search (relevance+popularity ranking)
    surfaces strong stories better than search_by_date (raw date order); the
    soft points floor drops show-and-tell noise."""
    out: list[NewsItem] = []
    base = "https://hn.algolia.com/api/v1/search"
    for q in HN_QUERIES:
        data = _get_json(base, params={
            "query": q, "tags": "story", "hitsPerPage": 25,
            "numericFilters": f"created_at_i>{since_ts}",
        })
        if not data:
            continue
        for h in data.get("hits", []) or []:
            if not h.get("title"):
                continue
            if (h.get("points") or 0) < MIN_HN_POINTS:
                continue
            oid = h.get("objectID")
            url = h.get("url") or (f"https://news.ycombinator.com/item?id={oid}" if oid else "")
            it = NewsItem(
                title=h["title"], url=url, source="Hacker News",
                published=safe_iso(h.get("created_at_i")),
                score=h.get("points"),
            )
            it.ts = h.get("created_at_i")
            out.append(it)
    out.sort(key=lambda i: (i.score or 0), reverse=True)
    return out


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _norm_title(title: str) -> str:
    t = re.sub(r"\s+", " ", (title or "").lower())
    return re.sub(r"[^\w\s]", "", t).strip()


def _relevant(title: str) -> bool:
    """Free keyword gate — True if the title looks in-scope (AI/ML/LLM,
    reverse engineering, or dev tooling)."""
    t = title.lower()
    if re.search(r"\bai\b", t):
        return True
    return any(term in t for term in RELEVANCE_TERMS)


def _excluded(title: str) -> bool:
    """True if a title is off-topic per the hard exclusion regex (if any)."""
    return bool(EXCLUDE_RE and EXCLUDE_RE.search(title or ""))


def fetch_gnews(since_ts: int) -> list[NewsItem]:
    """Google News RSS: mainstream / funding / policy coverage that often
    never reaches HN or Reddit. No popularity score — items are clustered by
    normalized title and ranked by distinct-publisher corroboration."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    headers = {"User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)"}
    clusters: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for q in GN_QUERIES:
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers=headers, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except (requests.RequestException, ET.ParseError) as exc:
            log.warning("Google News query '%s' failed: %s", q, exc)
            continue
        for item in root.findall(".//item"):
            title = _strip_html(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            publisher = _strip_html(item.findtext("source") or "")
            if not title or not link:
                continue
            ts = 0
            pub = (item.findtext("pubDate") or "").strip()
            if pub:
                try:
                    dt = parsedate_to_datetime(pub)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts = int(dt.timestamp())
                    if ts < since_ts:
                        continue
                except (TypeError, ValueError):
                    pass
            if " - " in title:  # Google News embeds "Headline - Source"
                title = title.rsplit(" - ", 1)[0].strip()
            key = _norm_title(title)
            c = clusters.get(key)
            if c is None:
                c = {"title": title, "url": link, "ts": ts, "publishers": set()}
                clusters[key] = c
                order.append(key)
            c["publishers"].add(publisher.lower())
            if ts > c["ts"]:
                c["ts"], c["url"] = ts, link
    out: list[NewsItem] = []
    for key in order:
        c = clusters[key]
        it = NewsItem(
            title=c["title"], url=c["url"], source="Google News",
            published=safe_iso(c["ts"]) if c["ts"] else _today_iso(),
        )
        it.ts = c["ts"] or None
        if len(c["publishers"]) > 1:
            it.corroboration = len(c["publishers"])
        out.append(it)
    out.sort(key=lambda i: ((i.corroboration or 1), i.published), reverse=True)
    return out


def fetch_arxiv(since_ts: int) -> list[NewsItem]:
    """arXiv cs.AI/LG/CL newest submissions. Pre-categorized: no relevance
    gate. The abstract becomes the snippet so the LLM can judge oblique
    titles. The export API lags the live listing by up to ~3 days
    (announcement cycle), so the recency floor is the lookback window plus
    3 days of slack — the sortBy=submittedDate sort already orders by
    freshness."""
    import xml.etree.ElementTree as ET

    floor = since_ts - 3 * 86400
    try:
        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": ARXIV_CATS,
                "sortBy": "submittedDate", "sortOrder": "descending",
                "max_results": 20,
            },
            headers={"User-Agent": "Mozilla/5.0 (compatible; HermesAgent/1.0)"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        log.warning("arXiv fetch failed: %s", exc)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: list[NewsItem] = []
    for e in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", ns) or "").strip()
        raw_url = (e.findtext("a:id", "", ns) or "").strip()
        if not title or not raw_url:
            continue
        updated = (e.findtext("a:updated", "", ns) or "").strip()
        ts = 0
        if updated:
            try:
                ts = int(datetime.fromisoformat(
                    updated.replace("Z", "+00:00")).timestamp())
            except ValueError:
                pass
        if ts and ts < floor:
            continue
        url = re.sub(r"v\d+$", "", raw_url)  # canonical unversioned abs page
        abstract = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns) or "").strip()
        it = NewsItem(
            title=title, url=url, source="arXiv",
            published=safe_iso(ts) if ts else _today_iso(),
            snippet=abstract[:300],
        )
        it.ts = ts or None
        out.append(it)
    return out


def fetch_reddit() -> list[NewsItem]:
    """Reddit via its Atom RSS feeds (the .json API is 403-blocked on
    datacenter IPs; the RSS endpoint is far more permissive)."""
    import xml.etree.ElementTree as ET

    out: list[NewsItem] = []
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    atom = "http://www.w3.org/2005/Atom"
    for sub in REDDIT_SUBREDDITS:
        # /hot = recency+upvote ranked (the only popularity signal Reddit
        # exposes over RSS; the .json API is 403-blocked on datacenter IPs).
        url = f"https://www.reddit.com/r/{sub}/hot/.rss"
        resp_content: Optional[bytes] = None
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, params={"limit": 100},
                                    timeout=REQUEST_TIMEOUT)
                if resp.status_code == 429 or resp.status_code >= 500:
                    log.warning("Reddit rate-limited (%d) on r/%s; backing off.",
                                resp.status_code, sub)
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    log.warning("HTTP %d fetching %s", resp.status_code, url)
                    break
                resp_content = resp.content
                break
            except requests.RequestException as exc:
                log.warning("Reddit RSS request failed for r/%s: %s", sub, exc)
                time.sleep(RETRY_DELAY)
        if resp_content is None:
            continue
        try:
            root = ET.fromstring(resp_content)
        except ET.ParseError as exc:
            log.warning("Reddit RSS parse failed for r/%s: %s", sub, exc)
            continue
        for entry in root.iter(f"{{{atom}}}entry"):
            title = _text(entry, f"{{{atom}}}title")
            link = _text(entry, f"{{{atom}}}link", attr="href")
            updated = _text(entry, f"{{{atom}}}updated")
            content = _text(entry, f"{{{atom}}}content")
            if not title:
                continue
            out.append(NewsItem(
                title=title, url=link or "", source=f"r/{sub}",
                published=_parse_rss_date(updated or ""),
                snippet=_strip_html(content or "")[:300],
            ))
        time.sleep(3.0)  # Reddit throttles rapid requests from datacenter IPs
    return out


def _text(entry: Any, tag: str, attr: Optional[str] = None) -> str:
    el = entry.find(tag)
    if el is None:
        return ""
    return el.get(attr) if attr else (el.text or "")


def _parse_rss_date(value: str) -> str:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except (ValueError, TypeError):
            continue
    return _today_iso()


def fetch_github(lookback_hours: int) -> list[NewsItem]:
    """GitHub repo search over configured topics, limited to recently-pushed
    repos (star-ranked). The project-signal lane for topics where GitHub IS
    the news (e.g. window managers); disabled unless config sets topics."""
    since = (_utcnow() - timedelta(hours=lookback_hours)).date().isoformat()
    out: list[NewsItem] = []
    headers = {"Accept": "application/vnd.github+json", "User-Agent": _NAME}
    for topic in (GITHUB_TOPICS or []):
        data = _get_json("https://api.github.com/search/repositories", headers=headers,
                         params={"q": f"topic:{topic} pushed:>{since}",
                                 "sort": "stars", "order": "desc", "per_page": 8})
        if not data:
            continue
        for r in data.get("items", []) or []:
            out.append(NewsItem(
                title=r.get("full_name") or "",
                url=r.get("html_url") or "",
                source="GitHub",
                published=(r.get("pushed_at") or _today_iso())[:10],
                snippet=(r.get("description") or "")[:300],
                score=r.get("stargazers_count"),
            ))
    return out


def _lane_of(item: NewsItem) -> str:
    if item.source.startswith("r/"):
        return "reddit"
    if item.source in ("exa", "tavily"):
        return "premium"
    return item.source


def _corroborate(items: list[NewsItem]) -> None:
    """Stamp each item with how many distinct communities carry the same
    (normalized) title. Cross-community coverage is the strongest free
    'this story matters' signal, and it feeds both the velocity lane
    (``communities``) and the follow-up gate (``corroboration``)."""
    communities: dict[str, set[str]] = {}
    for it in items:
        communities.setdefault(_norm_title(it.title), set()).add(_lane_of(it))
    for it in items:
        n = len(communities.get(_norm_title(it.title), ()))
        it.communities = n
        if n > 1:
            it.corroboration = max(it.corroboration or 0, n)


def ingest(lookback_hours: int) -> list[NewsItem]:
    """Multi-lane discovery; each enabled lane is capped and sorted by its
    own signal.

    Lane order is signal-strength order: HN (real upvote counts) -> Google
    News (multi-publisher corroboration) -> arXiv (research) -> GitHub
    (star-ranked repos, the project signal for topics where GitHub is the
    news) -> Reddit hot -> premium search (when keyed). This replaces the
    old global (date, score) sort, under which score-less Reddit 'new' posts
    (and star-counted repos) displaced strong HN stories from earlier in the
    window. Lane availability is config-driven per topic."""
    since_ts = int(_utcnow().timestamp() - lookback_hours * 3600)
    days = max(1, int(lookback_hours / 24))

    def _gated(items: list[NewsItem]) -> list[NewsItem]:
        return [i for i in items if _relevant(i.title) and not _excluded(i.title)]

    fetched: dict[str, list[NewsItem]] = {
        "Hacker News": _gated(fetch_hn(since_ts)),
        "Google News": _gated(fetch_gnews(since_ts)),
        "arXiv": fetch_arxiv(since_ts) if ARXIV_CATS else [],  # pre-categorized: no relevance gate
        "GitHub": ([i for i in fetch_github(lookback_hours) if not _excluded(i.title)]
                   if GITHUB_TOPICS else []),
        "reddit": _gated(fetch_reddit()),
        "premium": _gated(fetch_exa(since_ts) + fetch_tavily(days)),
    }

    kept = _dedupe([i for v in fetched.values() for i in v])
    _corroborate(kept)

    pool: list[NewsItem] = []
    for lane in _LANE_ORDER:
        items = [i for i in kept if _lane_of(i) == lane]
        items.sort(key=_LANE_SORT[lane], reverse=True)
        pool.extend(items[:POOL_CAPS.get(lane, 0)])
    log.info(
        "Ingested %d unique items (window %dh); evaluation pool %d.",
        len(kept), lookback_hours, min(len(pool), MAX_ITEMS_TO_EVALUATE),
    )
    return pool[:MAX_ITEMS_TO_EVALUATE]


# Lane assembly table: order = signal-strength order; each lane sorts by its
# own native signal. Availability is decided inside ingest() from config.
_LANE_ORDER: list[str] = ["Hacker News", "Google News", "arXiv", "GitHub", "reddit", "premium"]
_LANE_SORT: dict[str, Any] = {
    "Hacker News": lambda i: i.score or 0,
    "Google News": lambda i: (i.corroboration or 1, i.published),
    "arXiv": lambda i: i.published,
    "GitHub": lambda i: i.score or 0,
    "reddit": lambda i: i.published,
    "premium": lambda i: (i.score or 0, i.published),
}


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible)
# --------------------------------------------------------------------------- #

def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (stdlib only). Real environment variables take
    precedence over file values, so GitHub Actions secrets still win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def llm_config() -> tuple[Optional[str], str, str]:
    key = (os.environ.get("LLM_API_KEY")
           or os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPENAI_API_KEY"))
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return key, base, model


def call_llm(system: str, user: str) -> Optional[dict[str, Any]]:
    """Return the parsed JSON object from the model, or None on any failure."""
    key, base, model = llm_config()
    if not key:
        log.info("No LLM key configured — skipping narrative reconciliation.")
        return None
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    text: Optional[str] = None
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60.0)
            if resp.status_code == 429 or resp.status_code >= 500:
                log.warning("LLM rate/5xx (%d); retrying.", resp.status_code)
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            if resp.status_code == 400 and "response_format" in resp.text:
                # Some compatible backends reject json_object; retry without it.
                body.pop("response_format", None)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            break
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(RETRY_DELAY)
    if text is None:
        log.error("LLM call exhausted retries; proceeding without reconciliation.")
        return None
    return _parse_json_object(text)


def _parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """Defensively extract a JSON object from model output."""
    text = text.strip()
    # Strip ```json ... ``` fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Evaluation / reconciliation
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_TEMPLATE = """You are a research editor maintaining a running log of developing
<<TOPIC>>. Given the current story dossiers and today's incoming news items, return a
SINGLE JSON object with exactly two keys:

  "matches": [ ... ]   news items that continue an existing story
  "new_stories": [ ... ]  substantial NEW developing narratives (not one-off links)

Stories with "bucket": "breaking" are fresh breakouts the system auto-tracks
for velocity. Treat them like any existing story: match a genuine delta to
one, and never re-create a breaking story as a new story.

Each match entry:
  {"story_id": "<existing id>", "delta": "<1-2 sentences: what specifically changed
   today>", "update": "<1 sentence cumulative state>",
   "source_idx": <integer idx of the news item this is based on>,
   "importance": "<low|medium|high>"}

Each new_story entry:
  {"id": "<kebab-case slug>", "title": "<short title>", "overview": "<2-3 sentences>",
   "key_entities": ["<entity>", ...], "update": "<initial timeline update>",
   "source_idx": <integer idx of the news item this is based on>, "reason": "<why this is an ongoing narrative>"}

Never copy a URL string: "source_idx" is the integer "idx" of the news item that best
supports the update (or justifies a new story). Each news item is listed with its numeric
"idx". If no single item supports it, omit "source_idx".

Signal fields on news items: "score" = community upvotes (Hacker News);
"corroboration" = number of distinct communities/sources independently carrying
the same story (higher = stronger, and it counts toward the follow-up gate).
Prefer higher-signal items when choosing sources and judging significance.

Match discipline:
  * A match must carry a GENUINE new delta — information that advances the
    story beyond what its recent_timeline already says. If the incoming items
    only restate or echo the story's known state, do NOT match it (no match is
    a valid outcome).
  * "source_idx" must point to the item that specifically supports today's
    delta. An item that merely shares keywords or entities with the story is
    not its source; attaching it corrupts the timeline.

Story creation policy — follow-up required before tracking:
  * Open a NEW story only when the incoming items already show a topic covered by
    MULTIPLE independent sources (>=2 distinct URLs/outlets) or by coverage spanning
    more than one day. A single item from a single source is NOT enough to seed.
    The pipeline enforces this in code: a new story whose source item appears in
    fewer than 2 communities is dropped, no matter what you return.
  * Aim for 0-2 new stories per run. Prefer fewer, higher-signal stories over many
    speculative ones.
  * Prefer specific, focused stories over broad catch-all stories. Only match an item
    to an existing story if it genuinely continues that story's narrative; do not
    absorb unrelated items into existing stories.
  * Skip items that cannot develop further (routine tutorials, job postings, personal
    anecdotes, single-source one-off links with no follow-up potential).

Respond with JSON only, no commentary."""


# Reference topic (AI Narrative); load_config() re-substitutes per instance.
# (<<TOPIC>> placeholder + str.replace, not str.format — the prompt body is
# full of literal JSON braces that would break format().)
_PROMPT_TOPIC: str = "AI stories"
SYSTEM_PROMPT: str = SYSTEM_PROMPT_TEMPLATE.replace("<<TOPIC>>", _PROMPT_TOPIC)


def build_eval_prompt(stories: list[dict[str, Any]], items: list[NewsItem]) -> str:
    dossiers = []
    for s in stories:
        d = {
            "id": s["id"], "title": s["title"], "overview": s["overview"],
            "key_entities": s.get("key_entities", []),
            "bucket": s.get("bucket", "narrative"),
            "last_updated": s.get("last_updated"),
            "recent_timeline": (s.get("timeline") or [])[-MAX_TIMELINE_ITEMS_SENT:],
        }
        dossiers.append(d)
    news = []
    for i, it in enumerate(items[:MAX_ITEMS_TO_EVALUATE], start=1):
        rec = it.to_json()
        rec["idx"] = i
        news.append(rec)
    return json.dumps({"stories": dossiers, "news": news}, ensure_ascii=False)


def apply_evaluation(
    stories: list[dict[str, Any]], result: dict[str, Any], items: list[NewsItem],
) -> tuple[int, int, set[str], set[str]]:
    """Mutate ``stories`` in place; return (n_matches, n_new, matched_ids, new_ids)."""
    by_id = {s["id"]: s for s in stories}
    today = _today_iso()
    idx_url: dict[int, str] = {
        i: it.url for i, it in enumerate(items[:MAX_ITEMS_TO_EVALUATE], start=1) if it.url
    }

    def _src(m: dict[str, Any]) -> str:
        idx = m.get("source_idx")
        try:
            return idx_url[int(idx)] or ""
        except (TypeError, ValueError, KeyError):
            return ""

    n_matches, n_new = 0, 0
    matched_ids: set[str] = set()
    new_ids: set[str] = set()

    for m in (result.get("matches") or []):
        sid = (m or {}).get("story_id")
        if not sid or sid not in by_id:
            continue
        story = by_id[sid]
        timeline = story.setdefault("timeline", [])
        if timeline and timeline[-1].get("date") == today:
            continue  # same-day guard: already updated today (re-run)
        src_url = _src(m)
        # Cross-story URL guard: a source already consumed by ANOTHER story
        # is contamination (the LLM attached the wrong article), not a delta.
        if src_url:
            nu = norm_url(src_url)
            others = _seen_urls(stories) - _seen_urls([story])
            if nu in others:
                log.info("Skipping match for %s: source already consumed by another story (%s).",
                         sid, src_url)
                continue
        delta = (m.get("delta") or "No specific delta identified.").strip()
        update = (m.get("update") or delta).strip()
        entry = {
            "date": today,
            "update": update,
            "delta": delta,
            "source_url": src_url,
        }
        if _timeline_entry_duplicate(story, entry):
            # Re-coverage of what the timeline already says (same article, or
            # a rehash of a recent delta). Skip WITHOUT bumping
            # last_updated — the story's quiet clock keeps running, so it
            # cools and ages out naturally instead of staying "heating".
            log.info(
                "Skipping duplicate match for %s (source: %s).",
                sid, entry.get("source_url") or "(none)",
            )
            continue
        timeline.append(entry)
        story["last_updated"] = today
        # Trajectory is NOT asked of the LLM — it is computed from the
        # timeline in compute_trajectory() after the run. `importance`
        # (low|medium|high) is the LLM's severity judgment and stays.
        matched_ids.add(sid)
        n_matches += 1

    taken_slugs = set(by_id.keys())
    for ns in (result.get("new_stories") or []):
        ns = ns or {}
        if not ns.get("title"):
            continue
        title = ns["title"].strip()
        seed_update = (ns.get("update") or "Narrative identified.").strip()
        entry = {
            "date": today,
            "update": seed_update,
            "delta": seed_update,
            "source_url": _src(ns),
        }
        # Cross-story duplicate guard: the LLM opened a card for a narrative
        # we already track (different title/outlet, same story) — including
        # breaking-bucket cards, which the LLM is told to match, not
        # re-create. Fold it into the existing story instead of letting one
        # narrative inflate the tracked set as two cards.
        dup = next(
            (s for s in stories
             if _is_story_duplicate(title, ns.get("key_entities") or [], s)
             or _title_jaccard(title, s.get("title") or "") >= TITLE_JACCARD_MIN),
            None,
        )
        if dup is not None:
            log.info("Merging duplicate new story %r into tracked story %r.",
                     title, dup["id"])
            if _timeline_entry_duplicate(dup, entry):
                log.info("Duplicate-seed entry for %r already covered; dropping entry.",
                         dup["id"])
            else:
                dup.setdefault("timeline", []).append(entry)
                dup["last_updated"] = today
                matched_ids.add(dup["id"])
            continue
        # Gate 2 is enforced in code, not the prompt: a narrative seed must
        # rest on an item carried by >= GATE2_CORROBORATION_MIN distinct
        # communities. Velocity-only stories belong to the breaking bucket,
        # which the code seeds deterministically.
        idx = ns.get("source_idx")
        try:
            i0 = int(idx)
            comm = items[i0 - 1].communities or 0 if 1 <= i0 <= len(items) else 0
        except (TypeError, ValueError):
            comm = 0
        if comm < GATE2_CORROBORATION_MIN:
            log.info("Dropping LLM narrative seed %r (source item in %d communities "
                     "< %d; velocity-only stories belong in the breaking bucket).",
                     title, comm, GATE2_CORROBORATION_MIN)
            continue
        sid = ns.get("id") or slugify(title)
        base, i = sid, 1
        while sid in taken_slugs:
            sid = f"{base}-{i}"
            i += 1
        taken_slugs.add(sid)
        stories.append({
            "id": sid,
            "bucket": "narrative",
            "title": title,
            "overview": (ns.get("overview") or "").strip(),
            "key_entities": ns.get("key_entities") or [],
            "created": today,
            "last_updated": today,
            "timeline": [entry],
        })
        new_ids.add(sid)
        n_new += 1
    return n_matches, n_new, matched_ids, new_ids

# --------------------------------------------------------------------------- #
# Trajectory (computed from the timeline, never LLM-judged)
# --------------------------------------------------------------------------- #

HEATING_MIN_ENTRIES: int = 3        # timeline entries in last 7d -> "heating"
COOLING_MAX_QUIET_DAYS: int = 3     # quiet <= this -> "cooling"; beyond -> "fading"


def compute_trajectory(story: dict[str, Any], today: Optional[str] = None) -> str:
    """Derive the story badge from its own timeline. Deterministic — the same
    story state always yields the same badge, so there is no day-to-day label
    drift the way an LLM free-label would produce.

    Signals:
      quiet_days  days since the last timeline entry (0 = updated today)
      n7          timeline entries dated within the last 7 days (pace)

    States:
      heating   updated today AND n7 >= HEATING_MIN_ENTRIES (sustained coverage)
      steady    updated today, lighter pace
      cooling   quiet 1..COOLING_MAX_QUIET_DAYS (still live, no fresh delta)
      fading    quiet beyond that; heading for the DAYS_TO_RETAIN prune window

    Severity/stakes are NOT this axis — they live in the per-entry
    ``importance`` field (low|medium|high) the LLM still assigns.
    """
    today = today or _today_iso()
    try:
        t = datetime.fromisoformat(today).date()
    except ValueError:
        t = _utcnow().date()

    timeline = story.get("timeline") or []
    last = story.get("last_updated") or (timeline[-1].get("date") if timeline else today)
    try:
        quiet = max(0, (t - datetime.fromisoformat(last).date()).days)
    except ValueError:
        quiet = 0

    cutoff = (t - timedelta(days=6)).isoformat()
    n7 = sum(1 for e in timeline if (e.get("date") or "") >= cutoff)

    if quiet == 0:
        return "heating" if n7 >= HEATING_MIN_ENTRIES else "steady"
    return "cooling" if quiet <= COOLING_MAX_QUIET_DAYS else "fading"


def prune_stale(stories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop narratives quiet longer than the retention window.

    No status machinery: a story simply leaves tracking once it has not been
    updated for more than DAYS_TO_RETAIN days. Breaking stories pass through
    untouched — their lifecycle is expire_breaking()'s shorter TTL, which
    run() applies first."""
    return [
        s for s in stories
        if _bucket(s) == "breaking"
        or days_since(s.get("last_updated") or _today_iso()) <= DAYS_TO_RETAIN
    ]


def cap_stories(stories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enforce the hard ceiling on the narrative bucket (the breaking bucket
    has its own cap and TTL).

    Deterministic — no LLM judgment. Keeps the tracked set from growing past
    MAX_STORIES even when the seeding gate is generous."""
    def _key(s: dict[str, Any]) -> tuple[str, str]:
        return (s.get("last_updated") or "", s.get("created") or "")

    narratives = [s for s in stories if _bucket(s) == "narrative"]
    breaking = [s for s in stories if _bucket(s) != "narrative"]
    if len(narratives) > MAX_STORIES:
        narratives = sorted(narratives, key=_key, reverse=True)[:MAX_STORIES]
    return narratives + breaking


def _seen_urls(stories: list[dict[str, Any]]) -> set[str]:
    """Every source URL already consumed into a tracked story's timeline
    (normalized — see ``norm_url``)."""
    seen: set[str] = set()
    for s in stories:
        for entry in (s.get("timeline") or []):
            u = norm_url(entry.get("source_url") or "")
            if u:
                seen.add(u)
    return seen


def filter_seen(items: list[NewsItem], seen: set[str]) -> list[NewsItem]:
    """Drop incoming items whose (normalized) URL was already consumed by a
    tracked story.

    This is the cross-run dedup: without it, a 48h lookback re-ingests the same
    URLs every day and the LLM re-seeds/re-matches them (duplicate stories)."""
    return [it for it in items if not it.url or norm_url(it.url) not in seen]


# --------------------------------------------------------------------------- #
# Duplicate detection (deterministic guards around LLM output)
# --------------------------------------------------------------------------- #

def norm_url(url: str) -> str:
    """Canonical URL form for dedupe: lowercased scheme/host, no ``www.``,
    fragment stripped, tracking query params removed, trailing slash dropped.
    Two renderings of the same article compare equal."""
    u = (url or "").strip()
    if not u:
        return ""
    parts = urlsplit(u)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    return urlunsplit((parts.scheme.lower(), host, path, urlencode(query_pairs), ""))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).lower().strip()


def _text_similar(a: str, b: str) -> bool:
    """Near-identical check for delta/update text. Exact match or a
    SequenceMatcher ratio at/above DELTA_SIMILARITY_MIN. Calibrated: repeats
    of the same story score 1.00; genuinely new deltas score <= 0.27."""
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= DELTA_SIMILARITY_MIN


def _content_words(title: str) -> set[str]:
    stop = {"a", "an", "the", "of", "for", "in", "on", "to", "and", "or",
            "is", "are", "with", "its", "their", "that", "this", "by"}
    return {w for w in re.findall(r"[a-z0-9']+", (title or "").lower())
            if w not in stop}


def _title_jaccard(a: str, b: str) -> float:
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_story_duplicate(title: str, entities: list, story: dict[str, Any]) -> bool:
    """Deterministic duplicate-story guard. True when a proposed *new* story
    is really the same narrative already tracked as ``story``.

    Requires BOTH: most of the new story's key entities already belong to
    that story (sibling stories of one company share an entity — overlap
    alone is not enough), AND the titles share substantial content words
    (same-product coverage of unrelated things overlaps on the product name
    — overlap alone is not enough)."""
    have = {str(e).lower() for e in (story.get("key_entities") or [])}
    new = {str(e).lower() for e in (entities or [])}
    if not have or not new:
        return False
    if len(new & have) / len(new) < ENTITY_COVERAGE_MIN:
        return False
    return _title_jaccard(title, story.get("title") or "") >= TITLE_JACCARD_MIN


def _timeline_entry_duplicate(story: dict[str, Any], entry: dict[str, Any]) -> bool:
    """True if appending ``entry`` to the story's timeline would repeat
    information it already carries: the same (normalized) source URL, or
    near-identical delta/update text, in ANY existing entry.

    This is what stops a story that keeps getting re-covered (same article
    still in the lookback window, or outlets rehashing the same fact) from
    accumulating repeat entries — each of which would bump ``last_updated``
    and inflate the 7-day entry count that drives the "heating" trajectory.
    """
    timeline = story.get("timeline") or []
    nu = norm_url(entry.get("source_url") or "")
    if nu and any(norm_url(e.get("source_url") or "") == nu for e in timeline):
        return True
    return any(
        _text_similar(entry.get("delta") or "", e.get("delta") or "")
        or _text_similar(entry.get("update") or "", e.get("update") or "")
        for e in timeline
    )


# --------------------------------------------------------------------------- #
# Two-bucket tracking model (breaking vs narrative)
# --------------------------------------------------------------------------- #

def _bucket(s: dict[str, Any]) -> str:
    return (s.get("bucket") or "narrative").lower()


def _item_age_hours(it: NewsItem) -> Optional[float]:
    """Item age in hours at run time. Exact when the lane provided a
    timestamp; day-precision fallback assumes midday, so a same-day item
    reads as fresh and a previous-day item as ~24-48h old."""
    if it.ts:
        return max(0.0, (_utcnow().timestamp() - it.ts) / 3600.0)
    try:
        d = datetime.fromisoformat(it.published).date()
        ts = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
        return max(0.0, (_utcnow() - ts).total_seconds() / 3600.0 - 12.0)
    except ValueError:
        return None


def _eligible_breaking(it: NewsItem) -> bool:
    """Velocity lane: a very fresh item with a hard signal floor — high
    community points OR mainstream multi-outlet corroboration."""
    age = _item_age_hours(it)
    if age is not None and age > BREAKING_MAX_AGE_HOURS:
        return False
    return (it.score or 0) >= HEAT_POINTS_FLOOR \
        or (it.communities or 0) >= BREAKING_CORROBORATION_FLOOR


def select_breaking(stories: list[dict[str, Any]], items: list[NewsItem],
                    today: str) -> list[dict[str, Any]]:
    """Seed today's breaking bucket. Deterministic, code-only, zero LLM:
    the top-velocity eligible items become cards, ranked by points then
    corroboration. An item is skipped when its URL or title already belongs
    to a tracked story in EITHER bucket — one story, one card."""
    seed_urls = _seen_urls(stories)
    out: list[dict[str, Any]] = []
    ranked = sorted(
        (i for i in items if _eligible_breaking(i)),
        key=lambda i: (i.score or 0, i.communities or 0, i.ts or 0),
        reverse=True,
    )
    for it in ranked:
        if len(out) >= MAX_BREAKING:
            break
        if norm_url(it.url) in seed_urls:
            continue
        # Vs tracked stories: title overlap means "same narrative, different
        # outlet" — skip (the merge guard handles it the other direction).
        if any(_title_jaccard(it.title, s.get("title") or "") >= TITLE_JACCARD_MIN
               for s in stories):
            log.info("Breaking seed skipped (title overlaps tracked story): %s", it.title)
            continue
        # Vs this batch's own seeds: identical normalized title = same story
        # carried by two lanes (strict on purpose — sibling breakouts share
        # generic words and must not be collapsed).
        if any(_norm_title(it.title) == _norm_title(x["title"]) for x in out):
            continue
        if it.score and it.source == "GitHub":
            overview = f"Breakout: {it.score:,} stars — recently active repo."
            heat = {"points": it.score, "communities": it.communities, "kind": "stars"}
        elif it.score:
            overview = f"Breakout: {it.score} Hacker News points in under a day."
            heat = {"points": it.score, "communities": it.communities, "kind": "points"}
        else:
            overview = f"Breakout: {it.communities} independent communities, no HN presence."
            heat = {"points": None, "communities": it.communities, "kind": None}
        story = {
            "id": slugify(it.title),
            "bucket": "breaking",
            "title": it.title,
            "overview": overview,
            "key_entities": [],  # filled by the LLM on a later match, if any
            "created": today,
            "last_updated": today,
            "heat": heat,
            "timeline": [{
                "date": today, "update": it.title, "delta": it.title,
                "source_url": it.url,
            }],
        }
        all_ids = {s["id"] for s in stories} | {x["id"] for x in out}
        base, i = story["id"], 1
        while story["id"] in all_ids:
            story["id"] = f"{base}-{i}"
            i += 1
        out.append(story)
    return out


def select_narrative_autoseed(stories: list[dict[str, Any]], items: list[NewsItem],
                              today: str) -> list[dict[str, Any]]:
    """Deterministic narrative seeding for stories that are big AND
    corroborated — the class that used to fall through both lanes (no longer
    fresh enough for breaking, missed by the LLM's narrative discretion).
    Code-only, no LLM: such a story always reaches the card."""
    seed_urls = _seen_urls(stories)
    out: list[dict[str, Any]] = []
    ranked = sorted(
        (i for i in items
         if (i.score or 0) >= AUTO_SEED_POINTS_FLOOR
         and (i.communities or 0) >= AUTO_SEED_COMMUNITIES_MIN),
        key=lambda i: (i.score or 0, i.communities or 0),
        reverse=True,
    )
    for it in ranked:
        if norm_url(it.url) in seed_urls:
            continue
        if any(_title_jaccard(it.title, s.get("title") or "") >= TITLE_JACCARD_MIN
               for s in list(stories) + out):
            log.info("Auto-seed skipped (title overlaps tracked story): %s", it.title)
            continue
        story = {
            "id": slugify(it.title),
            "bucket": "narrative",
            "title": it.title,
            "overview": (f"Auto-tracked: {it.score} Hacker News points across "
                         f"{it.communities} independent communities."),
            "key_entities": [],  # the LLM fills these in on a later match
            "created": today,
            "last_updated": today,
            "timeline": [{
                "date": today, "update": it.title, "delta": it.title,
                "source_url": it.url,
            }],
        }
        all_ids = {s["id"] for s in stories} | {x["id"] for x in out}
        base, i = story["id"], 1
        while story["id"] in all_ids:
            story["id"] = f"{base}-{i}"
            i += 1
        out.append(story)
    return out


def _registrable_host(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def promote_breaking(stories: list[dict[str, Any]]) -> list[str]:
    """Graduate a breakout into the narrative bucket once it earned
    follow-up coverage: 2+ timeline entries from distinct publishers.
    Without this, a continuing story would die at the breaking TTL and
    respawn as a second card — the cross-bucket version of the
    duplicate-story bug."""
    promoted: list[str] = []
    for s in stories:
        if _bucket(s) != "breaking":
            continue
        hosts = {_registrable_host(e.get("source_url") or "")
                 for e in s.get("timeline", []) if e.get("source_url")}
        hosts.discard("")
        if len(hosts) >= 2:
            s["bucket"] = "narrative"
            promoted.append(s["id"])
            log.info("Promoted breakout %r to narrative bucket (follow-up coverage).",
                     s["id"])
    return promoted


def expire_breaking(stories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Breaking stories age out after BREAKING_TTL_DAYS without promotion.
    (Call promote_breaking() first — a promoted story is already
    'narrative' and follows the 7-day retention rule instead.)"""
    today_d = _utcnow().date()
    kept: list[dict[str, Any]] = []
    expired: list[str] = []
    for s in stories:
        if _bucket(s) == "breaking":
            try:
                created = datetime.fromisoformat(s.get("created") or _today_iso()).date()
            except ValueError:
                created = today_d
            if (today_d - created).days >= BREAKING_TTL_DAYS:
                expired.append(s["id"])
                log.info("Expired breakout %r (no follow-up within %dd).",
                         s["id"], BREAKING_TTL_DAYS)
                continue
        kept.append(s)
    return kept, expired


def _drop_url_shadows(stories: list[dict[str, Any]]) -> None:
    """One card per source URL, across buckets. If an LLM-seeded narrative
    story reuses a URL already carried by an earlier card (typically a
    breaking seed of the same item), drop the narrative and keep the first.
    List order wins: existing + breaking stories precede LLM-created ones."""
    seen: set[str] = set()
    keep: list[dict[str, Any]] = []
    for s in stories:
        urls = {norm_url(e.get("source_url") or "") for e in s.get("timeline", [])} - {""}
        if urls & seen:
            log.info("Dropping shadow card %r (URL already carried by an earlier card).",
                     s["id"])
            continue
        seen |= urls
        keep.append(s)
    stories[:] = keep


# --------------------------------------------------------------------------- #
# Markdown generation
# --------------------------------------------------------------------------- #

def _md_link(title: str, url: str) -> str:
    return f"[{title}]({url})" if (title and url) else (title or url or "")


def render_briefing(items: list[NewsItem], stories: list[dict[str, Any]],
                    matched_ids: Optional[set[str]] = None,
                    new_ids: Optional[set[str]] = None) -> str:
    matched_ids = matched_ids or set()
    new_ids = new_ids or set()
    today = _today_iso()
    lines: list[str] = []
    lines.append(f"# {_NAME.title()} — Daily Narrative Briefing — {today}")
    lines.append("")

    # --- Breaking (last 24h, velocity-ranked, code-seeded) ---
    breaking = [s for s in stories if _bucket(s) == "breaking"]
    lines.append("## Breaking — Top Velocity (last 24h)")
    lines.append("")
    if not breaking:
        lines.append("_No qualifying breakouts in the window._")
        lines.append("")
    for s in breaking:
        heat = s.get("heat") or {}
        sig = (f"{heat['points']} pts" if heat.get("points")
               else f"{heat.get('communities')} communities")
        latest = s.get("timeline", [])[-1]
        lines.append(f"### {s['title']} _[{sig}]_ ({latest.get('source_url', '')})")
        lines.append("")
        if s.get("overview"):
            lines.append(f"**Overview:** {s['overview']}")
            lines.append("")

    # --- Developing stories ---
    lines.append("## Developing Stories (Deltas & Trajectory)")
    lines.append("")
    developing = [s for s in stories if s["id"] in matched_ids]
    if not developing:
        lines.append("_No developing stories matched incoming news today._")
        lines.append("")
    for s in developing:
        lines.append(f"### {s['title']}")
        lines.append("")
        lines.append(f"**Overview:** {s.get('overview') or ''}")
        lines.append("")
        latest = s.get("timeline", [])[-1]
        lines.append(f"**Today's delta:** {latest.get('delta', '')}")
        lines.append("")
        traj = s.get("trajectory")
        if traj:
            lines.append(f"**Trajectory:** {traj}")
            lines.append("")
        ents = s.get("key_entities") or []
        if ents:
            lines.append(f"**Entities:** {', '.join(ents)}")
            lines.append("")
        src = latest.get("source_url")
        if src:
            lines.append(f"**Source:** {src}")
            lines.append("")

    # --- New narratives ---
    lines.append("## New Narratives Emerging")
    lines.append("")
    # Fresh cards from either seeder (code auto-seed or LLM), unless the LLM
    # already matched new coverage to them today (they are "developing" then).
    emerging = [s for s in stories
                if s.get("created") == today and s["id"] not in matched_ids]
    if not emerging:
        lines.append("_No new narratives identified today._")
        lines.append("")
    for s in emerging:
        lines.append(f"### {s['title']}")
        lines.append("")
        lines.append(f"**Overview:** {s.get('overview') or ''}")
        lines.append("")
        if s.get("key_entities"):
            lines.append(f"**Entities:** {', '.join(s['key_entities'])}")
            lines.append("")
        latest = s.get("timeline", [])[-1]
        if latest.get("source_url"):
            lines.append(f"**Source:** {latest['source_url']}")
            lines.append("")

    # --- Status table ---
    lines.append("## Tracked Stories")
    lines.append("")
    lines.append("| Story | Bucket | Last Updated | Days Quiet |")
    lines.append("| --- | --- | --- | --- |")
    for s in sorted(stories, key=lambda x: x.get("last_updated") or "", reverse=True):
        quiet = days_since(s.get("last_updated") or today)
        lines.append(f"| {s['title']} | {_bucket(s)} | {s.get('last_updated') or '—'} | {quiet} |")
    lines.append("")

    # --- Raw items (only when no LLM reconciliation ran) ---
    if not matched_ids and not emerging:
        lines.append("## Incoming Items (unreconciled digest)")
        lines.append("")
        lines.append("_Narrative reconciliation was skipped (no LLM configured or an error "
                     "occurred). Raw items for the window below._")
        lines.append("")
        for it in items[:MAX_ITEMS_TO_EVALUATE]:
            lines.append(f"- {_md_link(it.title, it.url)} _(via {it.source}, {it.published})_")
        lines.append("")

    lines.append("---")
    lines.append(f"_Generated {_utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC · "
                 f"{len(items)} items ingested · {len(stories)} stories tracked._")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def dispatch(text: str, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log.info("Wrote briefing to %s", output_path)
    _dispatch_webhooks(text)


def _dispatch_webhooks(text: str) -> None:
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url:
        try:
            # Discord embeds cap content at 2000 chars; send a truncated summary.
            requests.post(url, json={"content": text[:1900]}, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            log.warning("Discord webhook failed: %s", exc)
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text[:4000],
                      "disable_web_page_preview": True},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            log.warning("Telegram send failed: %s", exc)


# --------------------------------------------------------------------------- #
# Per-topic configuration
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict[str, Any]:
    """Apply a topic's config.json onto the module globals.

    The constants declared at the top of this module ARE the reference
    (AI Narrative) topic; a config file overrides whatever part of them it
    wants, so a new topic can supply just lanes/queries/keywords. The
    ``thresholds`` block is a generic override map onto the named constants
    (e.g. {"HEAT_POINTS_FLOOR": 500}) — every tunable knob is reachable
    without touching code."""
    global HN_QUERIES, MIN_HN_POINTS, GN_QUERIES, ARXIV_CATS, REDDIT_SUBREDDITS
    global GITHUB_TOPICS, POOL_CAPS, RELEVANCE_TERMS, EXCLUDE_RE, _NAME, _PROMPT_TOPIC
    global SYSTEM_PROMPT, DEFAULT_LOOKBACK_HOURS
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    _NAME = cfg.get("name", _NAME)
    lanes = cfg.get("lanes") or {}
    if "hn" in lanes:
        HN_QUERIES = lanes["hn"].get("queries", HN_QUERIES)
        MIN_HN_POINTS = int(lanes["hn"].get("min_points", MIN_HN_POINTS))
    if "gnews" in lanes:
        GN_QUERIES = lanes["gnews"].get("queries", GN_QUERIES)
    if "arxiv" in lanes:
        ARXIV_CATS = lanes["arxiv"].get("cats") or None  # explicit null disables the lane
    if "reddit" in lanes:
        REDDIT_SUBREDDITS = lanes["reddit"].get("subreddits") or None
    if "github" in lanes:
        GITHUB_TOPICS = lanes["github"].get("topics") or None
    if cfg.get("relevance_terms"):
        RELEVANCE_TERMS = tuple(cfg["relevance_terms"])
    if cfg.get("exclude_regex"):
        ex = cfg["exclude_regex"]
        if isinstance(ex, dict):  # {"pattern": "...", "flags": "ignorecase"}
            flags = re.IGNORECASE if ex.get("flags") == "ignorecase" else 0
            EXCLUDE_RE = re.compile(ex["pattern"], flags)
        else:
            EXCLUDE_RE = re.compile(ex)
    if cfg.get("pool_caps"):
        POOL_CAPS = dict(cfg["pool_caps"])
    if cfg.get("lookback_hours"):
        DEFAULT_LOOKBACK_HOURS = int(cfg["lookback_hours"])
    if cfg.get("topic_description"):
        _PROMPT_TOPIC = cfg["topic_description"]
        SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("<<TOPIC>>", _PROMPT_TOPIC)
    for k, v in (cfg.get("thresholds") or {}).items():
        if k in globals():
            globals()[k] = v
        else:
            log.warning("config: unknown knob %r ignored", k)
    log.info("Loaded topic config %s (name=%s).", path, _NAME)
    return cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(state_path: str, output_path: str, lookback_hours: int) -> dict[str, int]:
    stories = load_stories(state_path)
    for s in stories:
        s.setdefault("bucket", "narrative")
    items = ingest(lookback_hours)
    items = filter_seen(items, _seen_urls(stories))

    # Bucket maintenance before evaluation: breakouts that earned follow-up
    # coverage graduate to the narrative bucket, ones that didn't age out,
    # and today's velocity leaders seed the breaking bucket deterministically.
    promote_breaking(stories)
    stories, expired = expire_breaking(stories)
    breaking = select_breaking(stories, items, _today_iso())
    stories.extend(breaking)
    if breaking:
        log.info("Breaking bucket: seeded %d — %s",
                 len(breaking), "; ".join(s["title"] for s in breaking))
    # Big AND corroborated stories get a narrative card in code — no LLM
    # discretion left on the biggest stories of the window.
    auto = select_narrative_autoseed(stories, items, _today_iso())
    stories.extend(auto)
    if auto:
        log.info("Auto-seeded %d high-signal narrative(s) — %s",
                 len(auto), "; ".join(s["title"] for s in auto))

    result = call_llm(SYSTEM_PROMPT, build_eval_prompt(stories, items)) if items else None
    matched_ids: set[str] = set()
    new_ids: set[str] = set()

    if result and isinstance(result, dict):
        try:
            n_matches, n_new, matched_ids, new_ids = apply_evaluation(stories, result, items)
            log.info("Reconciliation: %d matches, %d new stories.", n_matches, n_new)
        except Exception as exc:  # noqa: BLE001 — never let eval crash the run
            log.exception("Evaluation application failed: %s", exc)
    else:
        log.warning("No LLM result — emitting an unreconciled digest.")

    # One card per source URL, across buckets (breaking seed vs LLM seed of
    # the same item). Runs after evaluation, before prune/cap.
    _drop_url_shadows(stories)

    stories = prune_stale(stories)
    stories = cap_stories(stories)
    # Trajectory is computed, not LLM-judged: recompute after every mutation so
    # the stored badge always reflects the current timeline.
    for s in stories:
        s["trajectory"] = compute_trajectory(s)
    save_stories(state_path, stories)

    briefing = render_briefing(items, stories, matched_ids, new_ids)
    dispatch(briefing, output_path)
    print(briefing, end="")
    return {"items": len(items), "stories": len(stories),
            "breaking": sum(1 for s in stories if _bucket(s) == "breaking")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json",
                        help="topic config (default ./config.json; reference AI-topic defaults if absent)")
    parser.add_argument("--state", default="stories.json", help="path to stories.json")
    parser.add_argument("--output", default="daily_briefing.md", help="output markdown path")
    parser.add_argument("--lookback-hours", type=int, default=None,
                        help="override the config's lookback window")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if os.path.exists(args.config):
        load_config(args.config)
    elif os.environ.get("STORY_TRACKER_CONFIG"):
        load_config(os.environ["STORY_TRACKER_CONFIG"])

    # CLI wins; otherwise the config's value (already loaded into the default).
    lookback = args.lookback_hours if args.lookback_hours is not None else DEFAULT_LOOKBACK_HOURS

    try:
        counts = run(args.state, args.output, lookback)
    except Exception as exc:  # noqa: BLE001
        log.exception("Fatal error: %s", exc)
        return 1
    log.info("Done: %d items, %d stories (%d breaking).",
             counts["items"], counts["stories"], counts.get("breaking", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())