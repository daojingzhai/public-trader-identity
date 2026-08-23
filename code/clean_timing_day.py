"""Produce raw-order-monotone timed diffs with explicit uncertainty gate closures."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path("/home/dz328/scratch_pi_as3993/dz328/toxicity_dec2025_rerun_20260803/stage0")
LIBRARY = Path(__file__).with_name("lnds.so")
COINS = ("BTC", "ETH", "SOL")
MAX_UNCERTAINTY_NS = 100_000_000
MISSING_LOW = np.iinfo(np.int64).min
MISSING_HIGH = np.iinfo(np.int64).max


def staleRows(frame: pl.DataFrame) -> tuple[np.ndarray, int]:
    """Drop whole replay chunks identified by backward exact open timestamps."""
    isNew = frame["diffType"].to_numpy() == "new"
    exact = np.asarray(frame["anchorExact"].fill_null(False).to_numpy(), dtype=bool)
    indexes = np.flatnonzero(isNew & exact)
    times = frame["anchorTs"].to_numpy()[indexes].astype(np.int64)
    difference = np.zeros(len(frame) + 1, dtype=np.int32)
    highWater = MISSING_LOW
    start = None
    target = None
    segmentCount = 0
    for index, timestamp in zip(indexes, times):
        if start is not None:
            if timestamp <= target:
                continue
            difference[start] += 1
            difference[index] -= 1
            segmentCount += 1
            start = None
            target = None
        if timestamp < highWater:
            start = int(index)
            target = int(highWater)
            continue
        highWater = int(timestamp)
    if start is not None:
        difference[start] += 1
        difference[len(frame)] -= 1
        segmentCount += 1
    return np.cumsum(difference[:-1]) > 0, segmentCount


def longestNondecreasing(times: np.ndarray) -> np.ndarray:
    """Return the exact maximum nondecreasing-subsequence mask from the C++ helper."""
    library = ctypes.CDLL(str(LIBRARY))
    function = library.longest_nondecreasing
    function.argtypes = [
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        ctypes.c_uint64,
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
    ]
    function.restype = ctypes.c_uint64
    contiguousTimes = np.ascontiguousarray(times, dtype=np.int64)
    keep = np.zeros(len(times), dtype=np.uint8)
    keptCount = function(contiguousTimes, len(times), keep)
    if int(keep.sum()) != keptCount:
        raise ValueError("LNDS helper returned an inconsistent keep mask")
    return keep.astype(bool)


def mergedClosures(lower: np.ndarray, upper: np.ndarray) -> pl.DataFrame:
    """Merge overlapping wide-uncertainty intervals into gate closures."""
    if not len(lower):
        return pl.DataFrame({"validTo": [], "validFrom": []}, schema={
            "validTo": pl.Int64, "validFrom": pl.Int64,
        })
    order = np.argsort(lower, kind="stable")
    lower, upper = lower[order], upper[order]
    starts = [int(lower[0])]
    ends = [int(upper[0])]
    for intervalLower, intervalUpper in zip(lower[1:], upper[1:]):
        if intervalLower <= ends[-1]:
            ends[-1] = max(ends[-1], int(intervalUpper))
            continue
        starts.append(int(intervalLower))
        ends.append(int(intervalUpper))
    return pl.DataFrame({"validTo": starts, "validFrom": ends})


def run(taskId: int) -> None:
    """Clean and time one coin-day from all 24 raw-order hourly partitions."""
    dayIndex, coinIndex = divmod(taskId, len(COINS))
    day = f"202512{dayIndex + 1:02d}"
    coin = COINS[coinIndex]
    paths = [ROOT / "timed_diffs_v3" / f"{coin}_{day}_{hour:02d}.parquet"
             for hour in range(24)]
    frame = pl.concat([pl.read_parquet(path) for path in paths], rechunk=True)
    originalRows = len(frame)
    stale, staleSegments = staleRows(frame)
    frame = frame.filter(pl.Series("keep", ~stale))

    anchorExact = np.asarray(frame["anchorExact"].fill_null(False).to_numpy(), dtype=bool)
    exactIndexes = np.flatnonzero(anchorExact)
    exactTimes = frame["anchorTs"].to_numpy()[exactIndexes].astype(np.int64)
    keepExact = longestNondecreasing(exactTimes)
    trusted = np.zeros(len(frame), dtype=bool)
    trusted[exactIndexes[keepExact]] = True
    demoted = anchorExact & ~trusted

    anchorValues = np.full(len(frame), MISSING_LOW, dtype=np.int64)
    anchorValues[trusted] = frame["anchorTs"].to_numpy()[trusted].astype(np.int64)
    lower = np.maximum.accumulate(anchorValues)
    futureValues = np.full(len(frame), MISSING_HIGH, dtype=np.int64)
    futureValues[trusted] = anchorValues[trusted]
    upper = np.minimum.accumulate(futureValues[::-1])[::-1]

    dayStart = int(np.datetime64(
        f"{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00", "ns").astype(np.int64))
    dayEnd = dayStart + 86_400_000_000_000
    lower = np.where(lower == MISSING_LOW, dayStart, lower)
    upper = np.where(upper == MISSING_HIGH, dayEnd, upper)
    candidateMin = frame["candidateMin"].to_numpy()
    candidateMax = frame["candidateMax"].to_numpy()
    candidateKnown = ~np.isnan(candidateMin) & ~np.isnan(candidateMax)
    canRefine = (~trusted & candidateKnown & (candidateMin >= lower) &
                 (candidateMax <= upper) & (candidateMax >= candidateMin))
    lower[canRefine] = candidateMin[canRefine].astype(np.int64)
    upper[canRefine] = candidateMax[canRefine].astype(np.int64)
    lower[trusted] = anchorValues[trusted]
    upper[trusted] = anchorValues[trusted]

    # Conservative availability is monotone in authoritative raw order.
    available = np.maximum.accumulate(upper)
    if np.any(np.diff(available) < 0):
        raise ValueError("availableTime is not monotone")
    uncertainty = available - lower
    wide = ~trusted & (uncertainty > MAX_UNCERTAINTY_NS)
    closures = mergedClosures(lower[wide], available[wide])
    closureEnds = closures["validFrom"].to_numpy()
    gateId = dayIndex * 1_000_000 + np.searchsorted(closureEnds, available, side="right")

    sourceAnchorTs = frame["anchorTs"]
    sourceAnchorExact = frame["anchorExact"].fill_null(False)
    trustedAnchorTs = np.full(len(frame), MISSING_LOW, dtype=np.int64)
    trustedAnchorTs[trusted] = anchorValues[trusted]
    timed = frame.with_columns(
        sourceAnchorTs.alias("sourceAnchorTs"),
        sourceAnchorExact.alias("sourceAnchorExact"),
        pl.Series("anchorExact", trusted),
        pl.Series("_trustedAnchorTs", trustedAnchorTs),
        pl.Series("trustedAnchor", trusted),
        pl.Series("demotedAnchor", demoted),
        pl.Series("lowerTs", lower),
        pl.Series("upperTs", available),
        pl.Series("timingUncertaintyMs", uncertainty / 1_000_000),
        pl.Series("boundedBeforeGateReopen", wide),
        pl.Series("replayGateId", gateId.astype(np.int64)),
        pl.Series("availableNs", available),
    ).with_columns(
        pl.from_epoch("availableNs", time_unit="ns").alias("availableTime"),
        pl.when(pl.col("_trustedAnchorTs") == MISSING_LOW).then(None)
        .otherwise(pl.col("_trustedAnchorTs")).alias("anchorTs"),
    ).drop("availableNs", "_trustedAnchorTs")

    outputDir = ROOT / "clean_diffs"
    closureDir = ROOT / "gate_closures_day"
    metricDir = ROOT / "clean_timing_metrics"
    for directory in (outputDir, closureDir, metricDir):
        directory.mkdir(parents=True, exist_ok=True)
    timed.write_parquet(outputDir / f"{coin}_{day}.parquet", compression="zstd")
    closures.with_columns(
        coin=pl.lit(coin), day=pl.lit(day),
        validTo=pl.from_epoch("validTo", time_unit="ns"),
        validFrom=pl.from_epoch("validFrom", time_unit="ns"),
    ).write_parquet(closureDir / f"{coin}_{day}.parquet")

    totalSpan = max(1, int(available[-1] - available[0]))
    closedNs = int((closureEnds - closures["validTo"].to_numpy()).sum())
    result = {
        "coin": coin,
        "day": day,
        "inputRows": originalRows,
        "outputRows": len(timed),
        "staleSegments": staleSegments,
        "staleRows": int(stale.sum()),
        "exactAnchors": len(exactIndexes),
        "trustedAnchors": int(trusted.sum()),
        "demotedAnchors": int(demoted.sum()),
        "unresolvedRows": int((~trusted).sum()),
        "wideUncertaintyRows": int(wide.sum()),
        "closureIntervals": len(closures),
        "closedDurationSec": closedNs / 1e9,
        "wallClockRetainedRate": 1 - closedNs / totalSpan,
        "uncertaintyMsP50": float(np.median(uncertainty[~trusted]) / 1e6),
        "uncertaintyMsP95": float(np.quantile(uncertainty[~trusted], 0.95) / 1e6),
        "uncertaintyMsMax": float(uncertainty[~trusted].max() / 1e6),
        "availableTimeViolations": int((np.diff(available) < 0).sum()),
    }
    (metricDir / f"{coin}_{day}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    run(int(sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]))
