# December 2025 book reconstruction

Code and diagnostics for Appendix C of

> Daojing Zhai (2026), *Public Trader Identity: Adverse Selection and Return Predictability*.

Appendix C replicates the paper's design on the public Hyperliquid December 2025 release of
Albers, Cucuringu, Howison and Shestopaloff (Zenodo, DOI
[10.5281/zenodo.18184441](https://doi.org/10.5281/zenodo.18184441); BTC, ETH, SOL).
The paper's July 2026 sample comes from a Hyperliquid node, whose stream carries every event
inside a consensus-block envelope (block number, block time, receipt time) and ten-second
oracle snapshots. The public December release does not retain that envelope: the book diffs
are flat, with no timestamp or block number, and there is no oracle stream. This repository
holds the code that recovers timing and ordering from the public files alone and replays the
order book into the gate-valid BBO tape used in Appendix C, together with the audit tables.

## Contents

- `reconstruction_appendix_dec2025.pdf` — audit tables: timing anchors and closure
  intervals; replay volume and integrity; trade-location and status-book cross-checks;
  admitted data volume.
- `code/` — the six core Stage-0 files.

## Input layout (extracted Zenodo archives)

```
orders_{coin}/{YYYYMMDD}/{coin}_{HH}.data.gz   accepted order statuses, 54-byte binary records
trades/{YYYYMMDD}/{H}.gz                        trades, NDJSON, all coins interleaved
book_diffs/{YYYYMMDD}/ex{H}.gz                  book diffs, NDJSON (coin, user, oid, side, px, raw_book_diff)
```

`zenodo_io.py` is the only file that knows these formats.

## Pipeline

| step | script | unit | what it does |
|---|---|---|---|
| 1 | `decode_timing_hour.py` | coin-hour (744 tasks) | joins each book diff on `oid` to statuses and trades from the hour and its neighbours; `new` → open-status time, `update` → matching trade, `remove` → zero-size/terminal status; emits a candidate [min, max] time per diff and an exact anchor when min == max |
| 2 | `clean_timing_day.py` + `lnds.cpp` | coin-day (93) | keeps raw file order as the sequence; drops stale chunks; a longest-nondecreasing-subsequence pass demotes anchors inconsistent with global order; `availableTime` = running max of the upper bound; intervals with uncertainty > 100 ms become closures; writes `clean_diffs/{coin}_{day}.parquet` with `availableTime` and `replayGateId` |
| 3 | `replay_book.py` + `replay_book.cpp` | coin (continuous over 31 days) | applies every `(availableTime, replayGateId)` group atomically; one BBO/depth row per group (`blockNumber` is a running counter, not a chain block); crossed book → reset and new `bookGateId`; `bookValid` after a 3600 s initial / 60 s post-reset warm-up; writes `bbo_depth/{coin}_{day}.parquet` |

Build the shared libraries before steps 2 and 3:

```
g++ -O3 -shared -fPIC -o lnds.so lnds.cpp
g++ -O3 -shared -fPIC -o replay_book.so replay_book.cpp
```

Python dependencies: `polars`, `numpy`, `orjson`.

## Caveats

- This is research code as run on Yale's Bouchet cluster (SLURM arrays; 1.7B / 1.1B / 0.6B
  diffs per coin). Input and output roots are hard-coded near the top of each script
  (`INPUT_ROOT`, `OUTPUT_ROOT`, `ROOT`) and need to be changed.
- Steps 1–2 are day-parallel; step 3 is sequential within a coin and needs tens of GB of RAM.
- The scripts that relabel gate IDs month-globally, build the gate interval table, and run the
  fill-ledger and status-only-book cross-checks reported in the PDF are not included here; the
  downstream scoring and forecasting code will be released with the full replication package.

## Citation

```
@unpublished{zhai2026identity,
  author = {Daojing Zhai},
  title  = {Public Trader Identity: Adverse Selection and Return Predictability},
  year   = {2026},
  note   = {Working paper}
}
```

MIT license.
