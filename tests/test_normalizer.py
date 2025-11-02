"""
Tests for normalizer.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from utils_text import (
    strip_html,
    remove_urls,
    remove_emoji,
    squash_spaces,
    normalize_unicode,
    detect_lang,
    clean_text,
)
from normalizer import (
    compute_hash,
    compute_hash2,
    normalize_timestamp,
    normalize_record,
)


def test_strip_html():
    """Test HTML stripping"""
    assert strip_html("<p>Hello</p>") == "Hello"
    assert strip_html("<div>Test<br/>Content</div>") == "TestContent"
    assert strip_html("No HTML here") == "No HTML here"
    assert strip_html("&nbsp;&amp;&lt;") == " &<"


def test_remove_urls():
    """Test URL removal"""
    assert remove_urls("Check https://example.com") == "Check "
    assert remove_urls("Visit www.example.com") == "Visit "
    assert remove_urls("No URL here") == "No URL here"
    assert remove_urls("Multiple https://a.com and https://b.com links") == "Multiple  and  links"


def test_remove_emoji():
    """Test emoji removal"""
    assert remove_emoji("Hello 😄") == "Hello "
    assert remove_emoji("Test 🚀 rocket") == "Test  rocket"
    assert remove_emoji("No emoji") == "No emoji"


def test_squash_spaces():
    """Test space normalization"""
    assert squash_spaces("Hello    world") == "Hello world"
    assert squash_spaces("  Leading and trailing  ") == "Leading and trailing"
    assert squash_spaces("Line\n\nbreaks") == "Line breaks"


def test_normalize_unicode():
    """Test unicode normalization"""
    # NFC normalization
    text = "café"  # May have different representations
    normalized = normalize_unicode(text)
    assert normalized == "café"


def test_detect_lang():
    """Test language detection"""
    # English
    assert detect_lang("This is an English sentence") == "en"

    # Russian (if langdetect supports it)
    russian_text = "Это русский текст для тестирования"
    lang = detect_lang(russian_text)
    assert lang == "ru" or lang == "und"  # May fail if langdetect not installed

    # Too short
    assert detect_lang("Hi") == "und"

    # Empty
    assert detect_lang("") == "und"


def test_clean_text():
    """Test complete text cleaning"""
    dirty = "<p>Check https://example.com 😄</p>  Multiple   spaces  "
    clean = clean_text(dirty)

    # Should not contain HTML
    assert "<p>" not in clean
    assert "</p>" not in clean

    # Should not contain URL
    assert "https://" not in clean

    # Should have normalized spaces
    assert "   " not in clean


def test_compute_hash():
    """Test hash computation"""
    h1 = compute_hash("https://example.com", "Title")
    h2 = compute_hash("https://example.com", "Title")
    h3 = compute_hash("https://example.com", "Different")

    assert len(h1) == 40  # SHA1 hex length
    assert h1 == h2  # Same input = same hash
    assert h1 != h3  # Different input = different hash


def test_compute_hash2():
    """Test enhanced hash computation"""
    h1 = compute_hash2("https://example.com", "Title", "2025-10-31T10:00:00Z", "en")
    h2 = compute_hash2("https://example.com", "Title", "2025-10-31T10:00:00Z", "en")
    h3 = compute_hash2("https://example.com", "Title", "2025-10-31T11:00:00Z", "en")

    assert len(h1) == 40
    assert h1 == h2
    assert h1 != h3  # Different timestamp


def test_normalize_timestamp():
    """Test timestamp normalization"""
    # ISO format with Z
    assert normalize_timestamp("2025-10-31T10:00:00Z") == "2025-10-31T10:00:00Z"

    # ISO format without timezone
    result = normalize_timestamp("2025-10-31T10:00:00")
    assert result is not None
    assert "2025-10-31" in result

    # None/empty
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("") is None


def test_normalize_record_basic():
    """Test basic record normalization"""
    record = {
        'link': 'https://example.com/article',
        'title': 'Test Article',
        'text': '<p>This is a test article with some content.</p> Visit https://example.com',
        'published_at': '2025-10-31T10:00:00Z',
        'ts_collected': '2025-10-31T10:05:00Z',
        'source': 'RSS',
        'feed_url': 'https://example.com/feed',
    }

    normalized = normalize_record(record)

    assert normalized is not None
    assert normalized['link'] == 'https://example.com/article'
    assert normalized['title'] == 'Test Article'
    assert 'text_clean' in normalized
    assert 'lang' in normalized
    assert 'hash' in normalized
    assert 'hash2' in normalized
    assert 'published_at_utc' in normalized

    # Text should be cleaned
    assert '<p>' not in normalized['text_clean']
    assert 'https://' not in normalized['text_clean']


def test_normalize_record_short_text():
    """Test filtering of short texts"""
    record = {
        'link': 'https://example.com/article',
        'title': 'Test',
        'text': 'Too short',
        'published_at': '2025-10-31T10:00:00Z',
    }

    # Should be filtered out (min_length=80 by default)
    normalized = normalize_record(record, min_length=80)
    assert normalized is None


def test_normalize_record_missing_fields():
    """Test handling of missing fields"""
    record = {
        'text': 'Some text content',
    }

    # Should be filtered out (no link or title)
    normalized = normalize_record(record)
    assert normalized is None


def test_normalize_record_language_filter():
    """Test language filtering"""
    record = {
        'link': 'https://example.com/article',
        'title': 'Test Article',
        'text': 'This is an English article with enough content to pass the minimum length requirement for normalization testing purposes.',
        'published_at': '2025-10-31T10:00:00Z',
    }

    # Allow only Russian
    normalized = normalize_record(record, allow_languages={'ru'})

    # English article should be filtered out
    assert normalized is None


def test_normalize_record_max_chars():
    """Test text truncation"""
    long_text = "A" * 7000
    record = {
        'link': 'https://example.com/article',
        'title': 'Test Article',
        'text': long_text,
        'published_at': '2025-10-31T10:00:00Z',
    }

    normalized = normalize_record(record, max_chars=6000, min_length=50)

    assert normalized is not None
    assert len(normalized['text_clean']) <= 6000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
