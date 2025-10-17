"""Utility IO helpers for collector module."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Set


def ensure_parent_dir(path: str) -> None:
    """Ensure that the parent directory for ``path`` exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_write_jsonl(path: str, record: dict) -> None:
    """Append a JSON record to a JSONL file, ensuring newline termination."""
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def iter_existing_hashes(path: str, limit_back: int = 50_000) -> Set[str]:
    """Return a set of hashes already present in the JSONL file.

    Only the last ``limit_back`` records are considered to limit memory usage.
    """
    file_path = Path(path)
    if not file_path.exists():
        return set()

    hashes: Set[str] = set()
    buffer: deque[str] = deque(maxlen=limit_back)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer.append(line.rstrip("\n"))

    for line in buffer:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        hash_value = data.get("hash")
        if isinstance(hash_value, str):
            hashes.add(hash_value)
    return hashes


def iso_utc(dt) -> str | None:
    """Convert a ``datetime`` or ``time.struct_time`` to ISO-8601 UTC string."""
    if dt is None:
        return None

    from datetime import datetime, timezone
    import time

    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    if isinstance(dt, time.struct_time):
        converted = datetime(*dt[:6], tzinfo=timezone.utc)
        return converted.isoformat().replace("+00:00", "Z")

    return None
