"""Decode one UTC hour and attach only defensible order-book timing anchors."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import orjson
import polars as pl

from zenodo_io import adjacentHours, decodePrice, readOrderFile, taskHour


INPUT_ROOT = Path("/home/dz328/scratch_pi_as3993/dz328/toxicity_dec2025/input")
OUTPUT_ROOT = Path(
    "/home/dz328/scratch_pi_as3993/dz328/toxicity_dec2025_rerun_20260803"
)
COINS = ("BTC", "ETH", "SOL")
TERMINAL_STATUS_IDS = (2, 4, 5, 7, 10, 11, 12, 13, 14, 16)
HOUR_NS = 3_600_000_000_000
ANCHOR_GRACE_NS = 1_800_000_000_000
MISSING_LOW = np.iinfo(np.int64).min
MISSING_HIGH = np.iinfo(np.int64).max


def orderMaps(coin: str, day: str, hour: int) -> tuple[pl.DataFrame, ...]:
    """Return open, post-open state, and terminal timestamp maps for one coin-hour."""
    frames = []
    for neighborDay, neighborHour in adjacentHours(day, hour):
        path = (INPUT_ROOT / f"orders_{coin.lower()}" / neighborDay /
                f"{coin.lower()}_{neighborHour:02d}.data.gz")
        if not path.exists():
            continue
        records = readOrderFile(path)
        frames.append(pl.DataFrame({
            "oid": records["oid"],
            "statusTs": records["ts"].astype(np.int64),
            "statusId": records["statusId"],
            "statusSizeBits": decodePrice(records["sz"]).view(np.uint32),
        }))

    statuses = pl.concat(frames)
    aggregate = [
        pl.col("statusTs").min().alias("candidateMin"),
        pl.col("statusTs").max().alias("candidateMax"),
        pl.col("statusTs").n_unique().alias("candidateCount"),
    ]
    opens = (statuses.filter(pl.col("statusId") == 1).group_by("oid").agg(aggregate)
             .rename({name: f"open{name[9:]}" for name in
                      ("candidateMin", "candidateMax", "candidateCount")}))
    states = (statuses.filter(pl.col("statusId") != 1)
              .group_by("oid", "statusSizeBits").agg(aggregate)
              .rename({name: f"state{name[9:]}" for name in
                       ("candidateMin", "candidateMax", "candidateCount")}))
    terminals = (statuses.filter(pl.col("statusId").is_in(TERMINAL_STATUS_IDS))
                 .group_by("oid").agg(aggregate)
                 .rename({name: f"terminal{name[9:]}" for name in
                          ("candidateMin", "candidateMax", "candidateCount")}))
    return opens, states, terminals


def tradeMaps(day: str, hour: int) -> dict[str, tuple[pl.DataFrame, ...]]:
    """Return per-coin oid timestamp maps from both legs of neighboring-hour trades."""
    rows = {coin: {"oid": [], "tradeTs": [], "fillSizeKey": []} for coin in COINS}
    for neighborDay, neighborHour in adjacentHours(day, hour):
        path = INPUT_ROOT / "trades" / neighborDay / f"{neighborHour}.gz"
        if not path.exists():
            continue
        for line in gzip.open(path, "rb"):
            record = orjson.loads(line)
            coin = record["coin"].upper()
            if coin not in rows:
                continue
            timestamp = int(np.datetime64(record["time"], "ns").astype(np.int64))
            fillSizeKey = round(float(record["sz"]) * 100_000_000)
            for sideInfo in record["side_info"]:
                rows[coin]["oid"].append(int(sideInfo["oid"]))
                rows[coin]["tradeTs"].append(timestamp)
                rows[coin]["fillSizeKey"].append(fillSizeKey)

    maps = {}
    for coin in COINS:
        frame = pl.DataFrame(rows[coin], schema={
            "oid": pl.UInt64, "tradeTs": pl.Int64, "fillSizeKey": pl.Int64,
        })
        general = frame.group_by("oid").agg(
            pl.col("tradeTs").min().alias("tradeMin"),
            pl.col("tradeTs").max().alias("tradeMax"),
            pl.col("tradeTs").n_unique().alias("tradeCount"),
        )
        bySize = frame.group_by("oid", "fillSizeKey").agg(
            pl.col("tradeTs").min().alias("fillTradeMin"),
            pl.col("tradeTs").max().alias("fillTradeMax"),
            pl.col("tradeTs").n_unique().alias("fillTradeCount"),
        )
        maps[coin] = general, bySize, frame
    return maps


def bookFrames(day: str, hour: int) -> dict[str, pl.DataFrame]:
    """Stream the interleaved book file into typed, coin-specific frames."""
    columns = {coin: {name: [] for name in
                      ("eventIndex", "user", "oid", "side", "px", "diffType", "origSz",
                       "newSz", "fillSz")}
               for coin in COINS}
    counters = {coin: 0 for coin in COINS}
    path = INPUT_ROOT / "book_diffs" / day / f"ex{hour}.gz"
    for line in gzip.open(path, "rb"):
        record = orjson.loads(line)
        coin = record["coin"].upper()
        if coin not in columns:
            continue
        rawDiff = record["raw_book_diff"]
        if rawDiff == "remove":
            diffType, originalSize, newSize = "remove", None, 0.0
        elif "new" in rawDiff:
            diffType = "new"
            originalSize = newSize = float(rawDiff["new"]["sz"])
        else:
            diffType = "update"
            originalSize = float(rawDiff["update"]["origSz"])
            newSize = float(rawDiff["update"]["newSz"])
        output = columns[coin]
        output["eventIndex"].append(counters[coin])
        output["user"].append(record["user"])
        output["oid"].append(int(record["oid"]))
        output["side"].append(record["side"])
        output["px"].append(float(record["px"]))
        output["diffType"].append(diffType)
        output["origSz"].append(originalSize)
        output["newSz"].append(newSize)
        output["fillSz"].append(originalSize - newSize if originalSize is not None else None)
        counters[coin] += 1

    schema = {
        "eventIndex": pl.UInt32,
        "user": pl.Utf8,
        "oid": pl.UInt64,
        "side": pl.Utf8,
        "px": pl.Float64,
        "diffType": pl.Utf8,
        "origSz": pl.Float64,
        "newSz": pl.Float64,
        "fillSz": pl.Float64,
    }
    frames = {}
    for coin, values in columns.items():
        frame = pl.DataFrame(values, schema=schema)
        frames[coin] = frame.with_columns(
            pl.Series("newSizeBits",
                      np.asarray(values["newSz"], dtype=np.float32).view(np.uint32)),
            (pl.col("fillSz") * 100_000_000).round().cast(pl.Int64).alias("fillSizeKey"),
        ).with_columns(
            (pl.col("eventIndex").rank("ordinal").over("oid", "fillSizeKey") - 1)
            .cast(pl.UInt32).alias("bookFillOrdinal"),
            pl.len().over("oid", "fillSizeKey").cast(pl.UInt32).alias("bookFillCount"),
        )
    return frames


def attachCandidates(frame: pl.DataFrame, maps: tuple[pl.DataFrame, ...],
                     trades: tuple[pl.DataFrame, pl.DataFrame], day: str,
                     hour: int) -> pl.DataFrame:
    """Attach type-specific candidate times and retain exact anchors only."""
    opens, states, terminals = maps
    generalTrades, sizeTrades, tradeRows = trades
    hourStart = int(np.datetime64(
        f"{day[:4]}-{day[4:6]}-{day[6:]}T{hour:02d}:00:00", "ns").astype(np.int64))
    lowerLimit = hourStart - ANCHOR_GRACE_NS
    upperLimit = hourStart + HOUR_NS + ANCHOR_GRACE_NS
    tradeSequence = (tradeRows.filter(pl.col("tradeTs").is_between(
        lowerLimit, upperLimit, closed="both"))
        .sort("oid", "fillSizeKey", "tradeTs")
        .with_columns(
            (pl.col("tradeTs").rank("ordinal").over("oid", "fillSizeKey") - 1)
            .cast(pl.UInt32).alias("bookFillOrdinal"),
            pl.len().over("oid", "fillSizeKey").cast(pl.UInt32).alias("tradeFillCount"),
        ).select("oid", "fillSizeKey", "bookFillOrdinal", "tradeTs", "tradeFillCount"))
    joined = (frame.join(opens, on="oid", how="left")
              .join(states, left_on=["oid", "newSizeBits"],
                    right_on=["oid", "statusSizeBits"], how="left")
              .join(terminals, on="oid", how="left")
              .join(generalTrades, on="oid", how="left")
              .join(sizeTrades, on=["oid", "fillSizeKey"], how="left")
              .join(tradeSequence, on=["oid", "fillSizeKey", "bookFillOrdinal"], how="left"))

    isNew = pl.col("diffType") == "new"
    isUpdate = pl.col("diffType") == "update"
    hasState = pl.col("stateMin").is_not_null()
    hasFillTrade = pl.col("fillTradeMin").is_not_null()
    hasTradeSequence = (pl.col("tradeTs").is_not_null() &
                        (pl.col("bookFillCount") == pl.col("tradeFillCount")))
    hasTerminal = pl.col("terminalMin").is_not_null()
    candidateMin = (pl.when(isNew).then(pl.col("openMin"))
                    .when(isUpdate & hasTradeSequence).then(pl.col("tradeTs"))
                    .when(isUpdate & hasFillTrade).then(pl.col("fillTradeMin"))
                    .when(isUpdate).then(pl.col("tradeMin"))
                    .when(hasState).then(pl.col("stateMin"))
                    .when(hasTerminal).then(pl.col("terminalMin"))
                    .otherwise(pl.col("tradeMin")))
    candidateMax = (pl.when(isNew).then(pl.col("openMax"))
                    .when(isUpdate & hasTradeSequence).then(pl.col("tradeTs"))
                    .when(isUpdate & hasFillTrade).then(pl.col("fillTradeMax"))
                    .when(isUpdate).then(pl.col("tradeMax"))
                    .when(hasState).then(pl.col("stateMax"))
                    .when(hasTerminal).then(pl.col("terminalMax"))
                    .otherwise(pl.col("tradeMax")))
    anchorSource = (pl.when(isNew).then(pl.lit("open_status"))
                    .when(isUpdate & hasTradeSequence).then(pl.lit("fill_trade_sequence"))
                    .when(isUpdate & hasFillTrade).then(pl.lit("fill_trade"))
                    .when(isUpdate).then(pl.lit("trade"))
                    .when(hasState).then(pl.lit("zero_status"))
                    .when(hasTerminal).then(pl.lit("terminal_status"))
                    .otherwise(pl.lit("trade")))
    joined = joined.with_columns(
        candidateMin.alias("candidateMin"),
        candidateMax.alias("candidateMax"),
        anchorSource.alias("anchorSource"),
    ).with_columns(
        ((pl.col("candidateMin") == pl.col("candidateMax")) &
         pl.col("candidateMin").is_between(lowerLimit, upperLimit, closed="both"))
        .alias("anchorExact"),
        (pl.col("candidateMin").is_not_null() &
         ~pl.col("candidateMin").is_between(lowerLimit, upperLimit, closed="both"))
        .alias("anchorOutsideHour"),
    ).with_columns(
        pl.when(pl.col("anchorExact")).then(pl.col("candidateMin"))
        .otherwise(None).cast(pl.Int64).alias("anchorTs")
    )
    return joined


def attachBounds(frame: pl.DataFrame) -> pl.DataFrame:
    """Use candidate ranges as bounds; raw file order is diagnostic, not a clock."""
    rawAnchors = frame["anchorTs"].to_numpy()
    exact = ~np.isnan(rawAnchors)
    anchorValues = np.where(exact, rawAnchors, MISSING_LOW).astype(np.int64)
    orderViolation = np.zeros(len(frame), dtype=bool)
    exactIndexes = np.flatnonzero(exact)
    if len(exactIndexes) > 1:
        backward = np.diff(anchorValues[exactIndexes]) < 0
        orderViolation[exactIndexes[1:][backward]] = True

    candidateMin = frame["candidateMin"].to_numpy()
    candidateMax = frame["candidateMax"].to_numpy()
    bounded = ~np.isnan(candidateMin) & ~np.isnan(candidateMax) & (candidateMax >= candidateMin)
    lower = np.full(len(frame), MISSING_LOW, dtype=np.int64)
    upper = np.full(len(frame), MISSING_HIGH, dtype=np.int64)
    lower[bounded] = candidateMin[bounded].astype(np.int64)
    upper[bounded] = candidateMax[bounded].astype(np.int64)
    uncertainty = np.where(bounded, (candidateMax - candidateMin) / 1_000_000, np.nan)
    usable = bounded & (uncertainty <= 100)
    return frame.with_columns(
        pl.Series("anchorOrderViolation", orderViolation),
        pl.Series("anchorUsable", usable),
        pl.Series("_lowerRaw", lower),
        pl.Series("_upperRaw", upper),
        pl.Series("uncertaintyMs", uncertainty),
    ).with_columns(
        pl.when(pl.col("_lowerRaw") == MISSING_LOW).then(None)
        .otherwise(pl.col("_lowerRaw")).alias("lowerTs"),
        pl.when(pl.col("_upperRaw") == MISSING_HIGH).then(None)
        .otherwise(pl.col("_upperRaw")).alias("upperTs"),
    ).drop("_lowerRaw", "_upperRaw")


def metrics(frame: pl.DataFrame, coin: str, day: str, hour: int) -> dict:
    """Summarize anchor coverage and timing uncertainty for one coin-hour."""
    bounded = frame.filter(pl.col("uncertaintyMs").is_not_nan())
    unanchored = frame.filter(~pl.col("anchorExact").fill_null(False))
    unanchoredBounded = unanchored.filter(pl.col("uncertaintyMs").is_not_nan())
    exactTimes = frame.filter(pl.col("anchorExact"))["anchorTs"].to_numpy()
    timeDifferences = np.diff(exactTimes)
    backwardJumps = -timeDifferences[timeDifferences < 0] / 1e9
    runStarts = np.r_[0, np.flatnonzero(timeDifferences < 0) + 1, len(exactTimes)]
    runLengths = np.diff(runStarts)
    return {
        "coin": coin,
        "day": day,
        "hour": hour,
        "rows": len(frame),
        "diffCounts": dict(zip(*frame["diffType"].value_counts().select(
            "diffType", "count").to_dict(as_series=False).values())),
        "candidateRate": frame["candidateMin"].is_not_null().mean(),
        "exactRate": frame["anchorExact"].fill_null(False).mean(),
        "within100msRate": frame["anchorUsable"].fill_null(False).mean(),
        "outsideHourRate": frame["anchorOutsideHour"].fill_null(False).mean(),
        "rawOrderViolations": int(frame["anchorOrderViolation"].sum()),
        "rawBackwardJumpSecP50": float(np.median(backwardJumps)) if len(backwardJumps) else None,
        "rawBackwardJumpSecP95": (float(np.quantile(backwardJumps, 0.95))
                                  if len(backwardJumps) else None),
        "rawMonotoneRuns": len(runLengths),
        "rawRunLengthP50": float(np.median(runLengths)) if len(runLengths) else None,
        "rawRunLengthP95": float(np.quantile(runLengths, 0.95)) if len(runLengths) else None,
        "rawRunLengthMax": int(runLengths.max()) if len(runLengths) else None,
        "boundedRate": bounded.height / len(frame) if len(frame) else 0.0,
        "uncertaintyMsP50": bounded["uncertaintyMs"].median() if len(bounded) else None,
        "uncertaintyMsP95": bounded["uncertaintyMs"].quantile(0.95) if len(bounded) else None,
        "uncertaintyMsMax": bounded["uncertaintyMs"].max() if len(bounded) else None,
        "unanchoredRows": len(unanchored),
        "unanchoredBoundedRate": (unanchoredBounded.height / len(unanchored)
                                  if len(unanchored) else 1.0),
        "unanchoredWithin100msRate": (float((unanchoredBounded["uncertaintyMs"] <= 100).mean())
                                      if len(unanchoredBounded) else 0.0),
        "unanchoredUncertaintyMsP50": (unanchoredBounded["uncertaintyMs"].median()
                                        if len(unanchoredBounded) else None),
        "unanchoredUncertaintyMsP95": (unanchoredBounded["uncertaintyMs"].quantile(0.95)
                                        if len(unanchoredBounded) else None),
    }


def run(taskId: int) -> None:
    """Decode and audit one December hour across all three study coins."""
    day, hour = taskHour(taskId)
    outputDir = OUTPUT_ROOT / "stage0" / "timed_diffs_v3"
    metricDir = OUTPUT_ROOT / "stage0" / "timing_metrics_hour_v3"
    outputDir.mkdir(parents=True, exist_ok=True)
    metricDir.mkdir(parents=True, exist_ok=True)

    frames = bookFrames(day, hour)
    trades = tradeMaps(day, hour)
    results = []
    for coin in COINS:
        timed = attachBounds(attachCandidates(
            frames[coin], orderMaps(coin, day, hour), trades[coin], day, hour))
        timed = timed.with_columns(
            coin=pl.lit(coin), day=pl.lit(day), hour=pl.lit(hour, dtype=pl.UInt8),
        ).select(
            "coin", "day", "hour", "eventIndex", "user", "oid", "side", "px",
            "diffType", "origSz", "newSz", "fillSz", "anchorTs", "anchorSource", "anchorExact",
            "anchorOutsideHour", "anchorOrderViolation", "anchorUsable",
            "candidateMin", "candidateMax", "lowerTs", "upperTs", "uncertaintyMs",
        )
        timed.write_parquet(outputDir / f"{coin}_{day}_{hour:02d}.parquet",
                            compression="zstd", statistics=True)
        results.append(metrics(timed, coin, day, hour))

    metricPath = metricDir / f"{day}_{hour:02d}.json"
    metricPath.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"taskId": taskId, "day": day, "hour": hour, "metrics": results}))


if __name__ == "__main__":
    run(int(sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]))
