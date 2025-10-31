"""
Module: utils_text.py
Purpose: Text cleaning and normalization utilities
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
import re
import unicodedata
from typing import Optional

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None
    LangDetectException = Exception


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text

    Args:
        text: Input text with potential HTML

    Returns:
        Text without HTML tags
    """
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
    }

    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from text

    Args:
        text: Input text with potential URLs

    Returns:
        Text without URLs
    """
    if not text:
        return ""

    # Remove http(s):// URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove www. URLs
    text = re.sub(r'www\.\S+', '', text)

    return text


def remove_emoji(text: str) -> str:
    """
    Remove emoji characters from text

    Args:
        text: Input text with potential emoji

    Returns:
        Text without emoji
    """
    if not text:
        return ""

    # Emoji pattern covering common emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE
    )

    return emoji_pattern.sub('', text)


def squash_spaces(text: str) -> str:
    """
    Reduce multiple spaces to single space and trim

    Args:
        text: Input text with potential multiple spaces

    Returns:
        Text with normalized spacing
    """
    if not text:
        return ""

    # Replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to NFC form

    Args:
        text: Input text with potential unicode variations

    Returns:
        Normalized unicode text
    """
    if not text:
        return ""

    # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
    text = unicodedata.normalize('NFC', text)

    return text


def detect_lang(text: str) -> str:
    """
    Detect language of text using langdetect

    Args:
        text: Input text

    Returns:
        ISO 639-1 language code (e.g., 'en', 'ru') or 'und' if detection fails
    """
    if not text or len(text.strip()) < 10:
        return "und"

    if detect is None:
        # langdetect not installed
        return "und"

    try:
        lang = detect(text)
        return lang
    except (LangDetectException, Exception):
        return "und"


def clean_text(text: str) -> str:
    """
    Apply all cleaning operations to text

    Args:
        text: Raw input text

    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""

    # Apply all cleaning steps in order
    text = strip_html(text)
    text = remove_urls(text)
    text = remove_emoji(text)
    text = normalize_unicode(text)
    text = squash_spaces(text)

    return text
