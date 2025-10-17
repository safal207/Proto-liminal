"""RSS collector for LIMINAL ProtoConsciousness."""
from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
from datetime import datetime, timezone
from typing import Iterable, List

import feedparser
import requests

from utils_io import ensure_parent_dir, iso_utc, iter_existing_hashes, safe_write_jsonl

LOGGER = logging.getLogger(__name__)
USER_AGENT = "LiminalProtoCollector/0.1"


def setup_logging(log_path: str = "logs/collector.log") -> None:
    if logging.getLogger().handlers:
        return
    ensure_parent_dir(log_path)
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def load_feeds(path: str) -> List[str]:
    feeds: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            feeds.append(url)
    return feeds


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def merge_text(entry) -> str:
    parts: List[str] = []
    summary = entry.get("summary")
    if isinstance(summary, str):
        parts.append(summary)
    contents = entry.get("content")
    if isinstance(contents, list):
        for item in contents:
            value = item.get("value") if isinstance(item, dict) else None
            if isinstance(value, str):
                parts.append(value)
    combined = "\n".join(filter(None, parts))
    return strip_html(combined)


def detect_language(text: str) -> str:
    if not text:
        return "und"
    latin_letters = len(re.findall(r"[A-Za-z]", text))
    cyrillic_letters = len(re.findall(r"[\u0400-\u04FF]", text))
    total_letters = latin_letters + cyrillic_letters
    if total_letters == 0:
        return "und"
    if latin_letters / total_letters >= 0.7:
        return "en"
    if cyrillic_letters / total_letters >= 0.7:
        return "ru"
    return "und"


def compute_hash(link: str | None, title: str | None) -> str:
    base = (link or "").strip()
    title = (title or "").strip()
    if base:
        payload = f"{base}|{title}"
    else:
        payload = title
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_entry(entry, feed_url: str, ts_collected: str, min_length: int) -> dict | None:
    title = entry.get("title", "")
    link = entry.get("link", "")
    text = merge_text(entry)

    if len(text) < min_length:
        return None

    published = iso_utc(entry.get("published_parsed"))

    authors: List[str] = []
    if isinstance(entry.get("authors"), list):
        for author in entry["authors"]:
            name = author.get("name") if isinstance(author, dict) else None
            if isinstance(name, str) and name.strip():
                authors.append(name.strip())
    else:
        author = entry.get("author")
        if isinstance(author, str) and author.strip():
            authors.append(author.strip())

    tags: List[str] = []
    if isinstance(entry.get("tags"), list):
        for tag in entry["tags"]:
            term = tag.get("term") if isinstance(tag, dict) else None
            if isinstance(term, str) and term.strip():
                tags.append(term.strip())

    record = {
        "ts_collected": ts_collected,
        "source": "RSS",
        "feed_url": feed_url,
        "link": link or None,
        "title": title or None,
        "text": text,
        "lang": detect_language(text),
        "authors": authors,
        "tags": tags,
        "published_at": published,
    }
    record["hash"] = compute_hash(link, title)
    return record


def collect(
    feeds_path: str,
    out_path: str,
    min_length: int = 40,
    max_items: int = 1000,
    timeout: int = 10,
) -> dict:
    feeds = load_feeds(feeds_path)
    if not feeds:
        LOGGER.warning("No feeds provided")
        return {
            "feeds": 0,
            "fetched": 0,
            "written": 0,
            "skipped_dup": 0,
            "skipped_short": 0,
        }

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    ensure_parent_dir(out_path)
    existing_hashes = iter_existing_hashes(out_path)
    seen_hashes = set(existing_hashes)

    ts_collected = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    fetched = 0
    written = 0
    skipped_dup = 0
    skipped_short = 0

    for feed_url in feeds:
        if written >= max_items:
            break
        try:
            response = session.get(feed_url, timeout=timeout)
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except (requests.RequestException, Exception) as exc:  # broad to capture parser errors
            LOGGER.warning("Failed to fetch feed %s: %s", feed_url, exc)
            continue

        entries = parsed.get("entries", [])
        for entry in entries:
            if written >= max_items:
                break
            record = normalize_entry(entry, feed_url, ts_collected, min_length)
            if record is None:
                skipped_short += 1
                continue
            fetched += 1
            item_hash = record["hash"]
            if item_hash in seen_hashes:
                skipped_dup += 1
                continue
            safe_write_jsonl(out_path, record)
            seen_hashes.add(item_hash)
            written += 1

    summary = {
        "feeds": len(feeds),
        "fetched": fetched,
        "written": written,
        "skipped_dup": skipped_dup,
        "skipped_short": skipped_short,
    }
    LOGGER.info(
        "Collection summary: feeds=%(feeds)d fetched=%(fetched)d written=%(written)d "
        "skipped_dup=%(skipped_dup)d skipped_short=%(skipped_short)d",
        summary,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect RSS feeds into JSONL")
    parser.add_argument("--feeds", required=True, help="Path to file with feed URLs")
    parser.add_argument("--out", required=True, help="Path to output JSONL file")
    parser.add_argument("--min-length", type=int, default=40, dest="min_length")
    parser.add_argument("--max-items", type=int, default=1000, dest="max_items")
    parser.add_argument("--timeout", type=int, default=10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging()
    collect(
        feeds_path=args.feeds,
        out_path=args.out,
        min_length=args.min_length,
        max_items=args.max_items,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
