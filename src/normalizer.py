"""
Module: normalizer.py
Purpose: Normalize raw JSONL data - clean text, detect language, deduplicate
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
import argparse
import glob
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from utils_text import clean_text, detect_lang

LOGGER = logging.getLogger(__name__)


def compute_hash(link: str, title: str) -> str:
    """
    Compute SHA1 hash from link and title

    Args:
        link: Article link
        title: Article title

    Returns:
        Hexadecimal SHA1 hash
    """
    content = f"{link}|{title}"
    return hashlib.sha1(content.encode('utf-8')).hexdigest()


def compute_hash2(link: str, title: str, published_at_utc: str, lang: str) -> str:
    """
    Compute enhanced SHA1 hash including timestamp and language

    Args:
        link: Article link
        title: Article title
        published_at_utc: Publication timestamp in UTC
        lang: Language code

    Returns:
        Hexadecimal SHA1 hash
    """
    content = f"{link}|{title}|{published_at_utc}|{lang}"
    return hashlib.sha1(content.encode('utf-8')).hexdigest()


def normalize_timestamp(ts: Optional[str]) -> Optional[str]:
    """
    Normalize timestamp to UTC ISO format

    Args:
        ts: Input timestamp (various formats)

    Returns:
        ISO 8601 UTC timestamp or None if invalid
    """
    if not ts:
        return None

    try:
        # Try parsing ISO format first
        if 'T' in ts:
            if ts.endswith('Z'):
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            elif '+' in ts or ts.count(':') > 2:
                dt = datetime.fromisoformat(ts)
            else:
                dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        else:
            # Try common date formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
            ]:
                try:
                    dt = datetime.strptime(ts, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
            else:
                LOGGER.warning(f"Could not parse timestamp: {ts}")
                return None

        # Convert to UTC
        dt_utc = dt.astimezone(timezone.utc)

        # Return ISO format
        return dt_utc.isoformat().replace('+00:00', 'Z')

    except Exception as exc:
        LOGGER.warning(f"Timestamp normalization error: {ts} - {exc}")
        return None


def normalize_record(
    record: Dict,
    allow_languages: Optional[Set[str]] = None,
    min_length: int = 80,
    max_chars: int = 6000
) -> Optional[Dict]:
    """
    Normalize a single record

    Args:
        record: Raw record dictionary
        allow_languages: Set of allowed language codes (None = no filter)
        min_length: Minimum text length
        max_chars: Maximum text length (truncate)

    Returns:
        Normalized record or None if filtered out
    """
    # Extract fields
    link = record.get('link', '')
    title = record.get('title', '')
    text = record.get('text', '')

    if not link or not title:
        LOGGER.debug("Skipping record without link or title")
        return None

    # Clean text
    text_clean = clean_text(text)

    # Truncate if too long
    if len(text_clean) > max_chars:
        text_clean = text_clean[:max_chars]

    # Filter by length
    if len(text_clean) < min_length:
        LOGGER.debug(f"Skipping short text: {len(text_clean)} < {min_length}")
        return None

    # Detect language
    lang = detect_lang(text_clean)

    # Filter by language
    if allow_languages and lang not in allow_languages:
        LOGGER.debug(f"Skipping language: {lang} not in {allow_languages}")
        return None

    # Normalize timestamp
    published_at = record.get('published_at')
    published_at_utc = normalize_timestamp(published_at)

    if not published_at_utc:
        # Use collection timestamp as fallback
        ts_collected = record.get('ts_collected')
        published_at_utc = normalize_timestamp(ts_collected)

    if not published_at_utc:
        # Last resort: current time
        published_at_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    # Compute hashes
    hash_val = compute_hash(link, title)
    hash2_val = compute_hash2(link, title, published_at_utc, lang)

    # Build normalized record
    normalized = {
        'ts_collected': record.get('ts_collected', ''),
        'source': record.get('source', 'RSS'),
        'feed_url': record.get('feed_url', ''),
        'link': link,
        'title': title,
        'text': text,
        'text_clean': text_clean,
        'lang': lang,
        'published_at': published_at or '',
        'published_at_utc': published_at_utc,
        'hash': hash_val,
        'hash2': hash2_val,
    }

    return normalized


def normalize_jsonl(
    input_paths: List[str],
    output_path: str,
    allow_languages: Optional[List[str]] = None,
    min_length: int = 80,
    max_chars: int = 6000,
    dedup: bool = False
) -> Dict:
    """
    Normalize JSONL files

    Args:
        input_paths: List of input JSONL file paths (can include globs)
        output_path: Output JSONL file path
        allow_languages: List of allowed language codes
        min_length: Minimum text length
        max_chars: Maximum text length
        dedup: Enable deduplication by hash2

    Returns:
        Statistics dictionary
    """
    # Expand globs
    all_files = []
    for pattern in input_paths:
        matched = glob.glob(pattern)
        if matched:
            all_files.extend(matched)
        else:
            # Not a glob, use as-is
            all_files.append(pattern)

    if not all_files:
        LOGGER.warning(f"No files matched: {input_paths}")
        return {
            'files_processed': 0,
            'records_read': 0,
            'records_written': 0,
            'duplicates_removed': 0,
        }

    # Convert languages to set
    allow_lang_set = set(allow_languages) if allow_languages else None

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deduplication set
    seen_hashes: Set[str] = set()

    # Statistics
    stats = {
        'files_processed': 0,
        'records_read': 0,
        'records_written': 0,
        'duplicates_removed': 0,
    }

    # Process files
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for file_path in all_files:
            LOGGER.info(f"Processing: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            record = json.loads(line)
                            stats['records_read'] += 1

                            # Normalize
                            normalized = normalize_record(
                                record,
                                allow_languages=allow_lang_set,
                                min_length=min_length,
                                max_chars=max_chars
                            )

                            if normalized is None:
                                continue

                            # Deduplication
                            if dedup:
                                hash2 = normalized['hash2']
                                if hash2 in seen_hashes:
                                    stats['duplicates_removed'] += 1
                                    continue
                                seen_hashes.add(hash2)

                            # Write
                            out_f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                            stats['records_written'] += 1

                        except json.JSONDecodeError:
                            LOGGER.warning(f"Invalid JSON at {file_path}:{line_num}")
                        except Exception as exc:
                            LOGGER.error(f"Error processing {file_path}:{line_num} - {exc}")

                stats['files_processed'] += 1

            except FileNotFoundError:
                LOGGER.warning(f"File not found: {file_path}")
            except Exception as exc:
                LOGGER.error(f"Error reading {file_path}: {exc}")

    LOGGER.info(f"Normalization complete: {stats['records_written']} records written")

    return stats


def main():
    """CLI interface for normalizer"""
    parser = argparse.ArgumentParser(
        description="Normalize raw JSONL data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/normalizer.py \\
    --inp data/raw/news_*.jsonl \\
    --out data/clean/news_norm_20251031.jsonl \\
    --allow-lang en,ru \\
    --min-length 80 \\
    --max-chars 6000 \\
    --dedup
        """
    )

    parser.add_argument(
        '--inp',
        required=True,
        nargs='+',
        help='Input JSONL file(s) or glob pattern(s)'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output JSONL file'
    )
    parser.add_argument(
        '--allow-lang',
        dest='allow_lang',
        help='Comma-separated list of allowed languages (e.g., en,ru)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=80,
        dest='min_length',
        help='Minimum text length (default: 80)'
    )
    parser.add_argument(
        '--max-chars',
        type=int,
        default=6000,
        dest='max_chars',
        help='Maximum text length, truncate if longer (default: 6000)'
    )
    parser.add_argument(
        '--dedup',
        action='store_true',
        help='Enable deduplication by hash2'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Parse languages
    allow_languages = None
    if args.allow_lang:
        allow_languages = [lang.strip() for lang in args.allow_lang.split(',')]

    # Normalize
    stats = normalize_jsonl(
        input_paths=args.inp,
        output_path=args.out,
        allow_languages=allow_languages,
        min_length=args.min_length,
        max_chars=args.max_chars,
        dedup=args.dedup
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Normalization Summary")
    print("=" * 60)
    print(f"Files processed:      {stats['files_processed']}")
    print(f"Records read:         {stats['records_read']}")
    print(f"Records written:      {stats['records_written']}")
    print(f"Duplicates removed:   {stats['duplicates_removed']}")

    if stats['records_read'] > 0:
        retention_rate = (stats['records_written'] / stats['records_read']) * 100
        print(f"Retention rate:       {retention_rate:.1f}%")


if __name__ == "__main__":
    main()
