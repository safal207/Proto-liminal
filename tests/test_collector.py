"""Tests for the RSS collector."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import collector


FEED_URL = "http://example.com/feed"


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
<channel>
<title>Example</title>
<item>
<title>First story</title>
<link>http://example.com/1</link>
<description><![CDATA[Summary one with <b>bold</b> text for testing.]]></description>
<content:encoded><![CDATA[Longer content block for the first entry with enough characters.]]></content:encoded>
<pubDate>Tue, 20 May 2025 10:00:00 GMT</pubDate>
<author>Author One</author>
<category>markets</category>
</item>
<item>
<title>Second story</title>
<link>http://example.com/2</link>
<description><![CDATA[Summary two contains different text but still quite lengthy.]]></description>
<content:encoded><![CDATA[Additional content for the second entry so that it passes the threshold.]]></content:encoded>
<pubDate>Tue, 20 May 2025 12:00:00 GMT</pubDate>
<category>economy</category>
</item>
</channel>
</rss>
"""


SHORT_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Example</title>
<item>
<title>Short story</title>
<link>http://example.com/short</link>
<description>tiny</description>
</item>
</channel>
</rss>
"""


class DummyResponse:
    def __init__(self, content: str, status_code: int = 200):
        self.content = content.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error {self.status_code}")


def patch_session(monkeypatch: pytest.MonkeyPatch, feed_map: dict[str, str]) -> None:
    class DummySession:
        def __init__(self):
            self.headers = {}

        def get(self, url: str, timeout: int):
            if url not in feed_map:
                raise AssertionError(f"Unexpected URL requested: {url}")
            return DummyResponse(feed_map[url])

    monkeypatch.setattr(collector.requests, "Session", DummySession)


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feed_content: str, **kwargs):
    feeds_path = tmp_path / "feeds.txt"
    feeds_path.write_text(f"{FEED_URL}\n", encoding="utf-8")
    output_path = tmp_path / "out.jsonl"
    patch_session(monkeypatch, {FEED_URL: feed_content})
    collector.collect(
        feeds_path=str(feeds_path),
        out_path=str(output_path),
        min_length=kwargs.get("min_length", 10),
        max_items=kwargs.get("max_items", 10),
        timeout=kwargs.get("timeout", 5),
    )
    return output_path


def test_collects_entries_into_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_path = run_collector(tmp_path, monkeypatch, SAMPLE_FEED, min_length=20)
    records = read_records(out_path)
    assert len(records) == 2
    first = records[0]
    assert first["title"] == "First story"
    assert "Summary one" in first["text"]
    assert first["lang"] == "en"
    expected_hash = collector.compute_hash("http://example.com/1", "First story")
    assert first["hash"] == expected_hash


def test_deduplication_on_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_path = run_collector(tmp_path, monkeypatch, SAMPLE_FEED, min_length=20)
    run_collector(tmp_path, monkeypatch, SAMPLE_FEED, min_length=20)
    records = read_records(out_path)
    assert len(records) == 2  # No duplicates added on second run


def test_min_length_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_path = run_collector(tmp_path, monkeypatch, SHORT_FEED, min_length=20)
    records = read_records(out_path)
    assert records == []
