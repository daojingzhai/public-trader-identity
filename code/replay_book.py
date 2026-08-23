"""Continuously replay one coin's cleaned December book and emit block-close BBO/depth."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path("/home/dz328/scratch_pi_as3993/dz328/toxicity_dec2025_rerun_20260803/stage0")
LIBRARY = Path(__file__).with_name("replay_book.so")
COINS = ("BTC", "ETH", "SOL")
PRICE_SCALE = {"BTC": 1, "ETH": 10, "SOL": 100}
MAX_PRICE_KEY = {"BTC": 250_000, "ETH": 200_000, "SOL": 200_000}
COMPOSITE_GATE_MULTIPLIER = 1_000_000
METRIC_NAMES = (
    "liveOrders", "duplicateNew", "conflictingNew", "missingRemove", "missingUpdate",
    "crossedGroups", "outOfRange", "timeViolations", "blocks",
    "crossingNewObserved", "resetCount", "resetInputRows", "warmupGroups",
    "warmupInputRows", "emptyGroups", "discardedLiveOrders", "bookGateId",
)


def libraryFunctions() -> tuple:
    """Load and type the C++ stateful replay entry points."""
    library = ctypes.CDLL(str(LIBRARY))
    create = library.book_create
    create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int64, ctypes.c_int64]
    create.restype = ctypes.c_void_p
    process = library.book_process
    process.restype = ctypes.c_uint64
    pointerTypes = [
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
    ] + [np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")] * 9 + [
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    ]
    process.argtypes = [ctypes.c_void_p, ctypes.c_uint64, *pointerTypes]
    metrics = library.book_metrics
    metrics.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS")]
    return library, create, process, metrics


def loadDay(coin: str, day: str) -> pl.DataFrame:
    """Load one cleaned source day in authoritative row order."""
    return pl.read_parquet(ROOT / "clean_diffs" / f"{coin}_{day}.parquet", columns=[
        "availableTime", "oid", "side", "px", "diffType", "newSz", "replayGateId",
    ])


def exactKey(frame: pl.DataFrame, index: int) -> tuple[int, int]:
    """Return an exact Int64-ns timing key without Python datetime downcasting."""
    return (int(frame["availableTime"].cast(pl.Int64)[index]),
            int(frame["replayGateId"][index]))


def splitDeferredSuffix(frame: pl.DataFrame, coin: str, nextDay: str | None
                        ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Defer a final key split across source files so replay applies it atomically."""
    if nextDay is None:
        return frame, frame.clear()
    nextKey = (pl.scan_parquet(ROOT / "clean_diffs" / f"{coin}_{nextDay}.parquet")
               .select("availableTime", "replayGateId").head(1).collect())
    lastTime, lastGate = exactKey(frame, -1)
    nextTime, nextGate = exactKey(nextKey, 0)
    if (nextTime, nextGate) != (lastTime, lastGate):
        return frame, frame.clear()
    splitIndex = int(frame.with_row_index("_row").filter(
        (pl.col("availableTime").cast(pl.Int64) == lastTime)
        & (pl.col("replayGateId") == lastGate)
    )["_row"][0])
    return frame.head(splitIndex), frame.slice(splitIndex)


def processDay(handle: int, process: ctypes._CFuncPtr, coin: str, day: str,
               frame: pl.DataFrame) -> pl.DataFrame:
    """Apply one logical day and return every unique atomic block close."""
    times = np.ascontiguousarray(frame["availableTime"].to_numpy().astype("datetime64[ns]")
                                 .astype(np.int64))
    oids = np.ascontiguousarray(frame["oid"].to_numpy(), dtype=np.uint64)
    asks = np.ascontiguousarray((frame["side"].to_numpy() == "A").astype(np.uint8))
    keys = np.ascontiguousarray(np.rint(frame["px"].to_numpy() * PRICE_SCALE[coin]),
                                dtype=np.int32)
    diffValues = frame["diffType"].replace_strict(
        {"new": 0, "remove": 1, "update": 2}, return_dtype=pl.UInt8).to_numpy()
    diffs = np.ascontiguousarray(diffValues, dtype=np.uint8)
    sizes = np.ascontiguousarray(frame["newSz"].to_numpy(), dtype=np.float64)
    gateIds = np.ascontiguousarray(frame["replayGateId"].to_numpy(), dtype=np.int64)
    if np.any(np.diff(gateIds) < 0):
        raise ValueError(f"replayGateId is not globally nondecreasing: {coin} {day}")
    sameTime = np.diff(times) == 0
    if np.any(np.diff(gateIds)[sameTime] < 0):
        raise ValueError(f"replayGateId decreases within an equal-time group: {coin} {day}")
    groupCount = int(np.count_nonzero(
        np.r_[True, (np.diff(times) != 0) | (np.diff(gateIds) != 0)]
    ))

    outputTimes = np.empty(groupCount, dtype=np.int64)
    outputBlocks = np.empty(groupCount, dtype=np.uint64)
    outputTimingGates = np.empty(groupCount, dtype=np.int64)
    outputBookGates = np.empty(groupCount, dtype=np.uint64)
    outputGroupRows = np.empty(groupCount, dtype=np.uint64)
    floatOutputs = [np.empty(groupCount, dtype=np.float64) for _ in range(9)]
    outputValid = np.empty(groupCount, dtype=np.uint8)
    outputReset = np.empty(groupCount, dtype=np.uint8)
    outputCount = process(
        handle, len(frame), times, oids, asks, keys, diffs, sizes, gateIds,
        outputTimes, outputBlocks, outputTimingGates, outputBookGates, outputGroupRows,
        *floatOutputs, outputValid, outputReset,
    )
    arrays = [values[:outputCount] for values in floatOutputs]
    bid, ask, bidSize, askSize, bidDepth, askDepth, bidNear, askNear, warmupAge = arrays
    denominator = bidNear + askNear
    return pl.DataFrame({
        "coin": coin,
        "blockNumber": outputBlocks[:outputCount],
        "blockTimeNs": outputTimes[:outputCount],
        "timingGateId": outputTimingGates[:outputCount],
        "bookGateId": outputBookGates[:outputCount],
        "groupRows": outputGroupRows[:outputCount],
        "bidPx": bid,
        "askPx": ask,
        "bidSz": bidSize,
        "askSz": askSize,
        "mid": (bid + ask) / 2,
        "spread": ask - bid,
        "bidDepthUsd": bidDepth,
        "askDepthUsd": askDepth,
        "bidNearUsd": bidNear,
        "askNearUsd": askNear,
        "nearImbalance": np.divide(bidNear - askNear, denominator,
                                   out=np.zeros_like(denominator), where=denominator > 0),
        "warmupAgeMs": warmupAge,
        "bookValid": outputValid[:outputCount].astype(bool),
        "resetOccurred": outputReset[:outputCount].astype(bool),
    }).with_columns(pl.from_epoch("blockTimeNs", time_unit="ns").alias("blockTime")) \
      .with_columns(
          (pl.col("timingGateId") * COMPOSITE_GATE_MULTIPLIER
           + pl.col("bookGateId").cast(pl.Int64)).alias("gateId")
      ).drop("blockTimeNs")


def run(coin: str) -> None:
    """Replay all 31 days continuously for one market."""
    library, create, process, metricsFunction = libraryFunctions()
    initialWarmupNs = int(
        float(os.environ.get("DEC25_INITIAL_WARMUP_MS", "3600000")) * 1_000_000
    )
    resetWarmupNs = int(
        float(os.environ.get("DEC25_BOOK_WARMUP_MS", "60000")) * 1_000_000
    )
    handle = create(
        PRICE_SCALE[coin], MAX_PRICE_KEY[coin], initialWarmupNs, resetWarmupNs
    )
    outputDir = ROOT / "bbo_depth"
    metricDir = ROOT / "replay_metrics"
    resetDir = ROOT / "book_resets"
    outputDir.mkdir(parents=True, exist_ok=True)
    metricDir.mkdir(parents=True, exist_ok=True)
    resetDir.mkdir(parents=True, exist_ok=True)
    daily = []
    deferred = None
    deferredBoundaries = 0
    prependedBoundaries = 0
    endDay = int(os.environ.get("DEC25_REPLAY_END_DAY", "31"))
    for dayNumber in range(1, endDay + 1):
        day = f"202512{dayNumber:02d}"
        source = loadDay(coin, day)
        prependedRows = len(deferred) if deferred is not None else 0
        if prependedRows:
            prependedBoundaries += 1
            deferredKey = exactKey(deferred, -1)
            sourceKey = exactKey(source, 0)
            if deferredKey != sourceKey:
                raise ValueError(f"Deferred boundary key mismatch: {coin} {day}")
            source = pl.concat([deferred, source])
        nextDay = f"202512{dayNumber + 1:02d}" if dayNumber < endDay else None
        replayInput, deferred = splitDeferredSuffix(source, coin, nextDay)
        deferredBoundaries += int(not deferred.is_empty())
        if replayInput.is_empty():
            raise ValueError(f"No complete group available for {coin} {day}")
        output = processDay(handle, process, coin, day, replayInput)
        output.write_parquet(outputDir / f"{coin}_{day}.parquet", compression="zstd")
        reopen = (output.filter(pl.col("bookValid"))
                  .select(pl.col("blockTime").alias("reopenTime")).sort("reopenTime"))
        (output.filter(pl.col("resetOccurred"))
         .select("coin", pl.col("blockTime").alias("resetTime"), "timingGateId",
                 "bookGateId", "gateId", "groupRows").sort("resetTime")
         .join_asof(reopen, left_on="resetTime", right_on="reopenTime", strategy="forward")
         .write_parquet(resetDir / f"{coin}_{day}.parquet", compression="zstd"))
        metricValues = np.empty(len(METRIC_NAMES), dtype=np.uint64)
        metricsFunction(handle, metricValues)
        dayMetrics = {name: int(value) for name, value in zip(METRIC_NAMES, metricValues)}
        dayMetrics.update({
            "coin": coin,
            "day": day,
            "initialWarmupMs": initialWarmupNs / 1e6,
            "resetWarmupMs": resetWarmupNs / 1e6,
            "sourceRows": len(source) - prependedRows,
            "prependedBoundaryRows": prependedRows,
            "deferredBoundaryRows": len(deferred),
            "checkpointTimeNs": int(replayInput["availableTime"].cast(pl.Int64)[-1]),
            "checkpointTimingGateId": int(replayInput["replayGateId"][-1]),
            "outputRows": len(output),
            "validRows": int(output["bookValid"].sum()),
        })
        daily.append(dayMetrics)
        print(json.dumps(dayMetrics), flush=True)
    if endDay == 31 and (
        deferredBoundaries != 2 or prependedBoundaries != 2 or not deferred.is_empty()
    ):
        raise ValueError(
            f"Expected two fully consumed atomic day boundaries: {coin}; "
            f"deferred={deferredBoundaries}, prepended={prependedBoundaries}"
        )
    (metricDir / f"{coin}.json").write_text(json.dumps(daily, indent=2) + "\n")
    library.book_destroy(ctypes.c_void_p(handle))


if __name__ == "__main__":
    coinIndex = int(sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"])
    run(COINS[coinIndex])
