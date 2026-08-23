"""Low-level readers for the public Hyperliquid December 2025 archive."""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ORDER_DTYPE = np.dtype([
    ("ts", "<u8"),
    ("userId", "<u4"),
    ("isBuilder", "?"),
    ("statusId", "<u1"),
    ("isAsk", "?"),
    ("limitPx", "<u4"),
    ("sz", "<u4"),
    ("oid", "<u8"),
    ("timestampDiff", "<u4"),
    ("triggerCondition", "<i4"),
    ("triggered", "?"),
    ("isTrigger", "?"),
    ("hasChildren", "?"),
    ("isPositionTpsl", "?"),
    ("reduceOnly", "?"),
    ("orderTypeId", "<u1"),
    ("tifId", "<u1"),
    ("triggerPx", "<u4"),
    ("origSz", "<u4"),
])

POWERS = np.array([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=np.float32)
DECEMBER_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
DECEMBER_END = datetime(2026, 1, 1, tzinfo=timezone.utc)


def decodePrice(encoded: np.ndarray) -> np.ndarray:
    """Decode the archive's packed price/size representation."""
    decimals = encoded >> 29
    value = encoded & 0x1FFFFFFF
    return (value / POWERS[decimals]).astype(np.float32)


def taskHour(taskId: int) -> tuple[str, int]:
    """Map a zero-based 744-task index to a December UTC day and hour."""
    hourTime = DECEMBER_START + timedelta(hours=taskId)
    return hourTime.strftime("%Y%m%d"), hourTime.hour


def adjacentHours(day: str, hour: int) -> list[tuple[str, int]]:
    """Return the target hour and its available immediate neighbors."""
    center = datetime.strptime(f"{day}{hour:02d}", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    times = [center + timedelta(hours=offset) for offset in (-1, 0, 1)]
    return [(time.strftime("%Y%m%d"), time.hour) for time in times
            if DECEMBER_START <= time < DECEMBER_END]


def readOrderFile(path: Path) -> np.ndarray:
    """Read one binary order-status hour and reject a partial record."""
    raw = gzip.open(path, "rb").read()
    if len(raw) % ORDER_DTYPE.itemsize:
        raise ValueError(f"Partial 54-byte record in {path}: {len(raw):,} bytes")
    return np.frombuffer(raw, dtype=ORDER_DTYPE)
