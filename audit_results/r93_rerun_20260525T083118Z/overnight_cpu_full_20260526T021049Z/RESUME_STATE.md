# Sequential Round 3 — partial run record

**Launched**: 2026-05-26 12:10:49 UTC (sequential, single stream)
**Killed**:  2026-05-26 12:54 UTC (clean kill, no orphans)
**Reason**:  switched to 3-stream parallel for ~6 hr time saving

## What this BASE actually contains

This BASE dir holds **only the partial cora_2024 results** that landed before the
kill. The 3-stream parallel restart uses NEW base dirs (see below) and picks
up from where this run left off.

### cora_2024 (50/170 persisted)

- **Persisted iids**: 10..59 (50 instances)
- **Verdicts**: 14 CERTIFIED + 2 UNKNOWN + 34 UNKNOWN_TIMEOUT
- **Remaining for follow-up**: iids 60..179 (120 instances) → assigned to Stream 1

### Benchmarks not started in this sequential run

| Benchmark | iids needed | Stream |
|---|---|---|
| soundnessbench | 10..49 | 1 |
| traffic_signs_recognition_2023 | 5..44 | 1 |
| nn4sys (lindex_200+) | 107..193 | 1 |
| metaroom_2023 | 5..99 | 2 |
| vggnet16_2022 | 5..17 | 2 |
| yolo_2023 | 10..71 | 2 |
| cifar100_2024 | 5..199 | 3 |
| tinyimagenet_2024 | 5..199 | 3 |

## Parallel restart base dirs

- `overnight_cpu_full_stream1_<ts>/`  — light/medium (8 GiB peak)
- `overnight_cpu_full_stream2_<ts>/`  — medium (24 GiB peak)
- `overnight_cpu_full_stream3_<ts>/`  — heavy CNN (24 GiB peak)

Each stream's BASE has its own `README.txt`, `<bench>/driver.log`,
`<bench>/per_instance_*.json`, `<bench>/watchdog_summary.json`. Ingest into
`CONSOLIDATED_RESULTS/` after all streams complete by adding
`_source_<label>` symlinks and running `python3 build_csvs.py`.
