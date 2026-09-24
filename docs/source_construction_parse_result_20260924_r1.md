# Construction parsing reuse: controlled synthetic result

## Outcome and decision

The frozen 18-call synthetic comparison completed, with **18/18 complete
constructions and unchanged independent checks**, zero errors/timeouts, and
identical source, construction and check-result identities within each fixture
across all three modes and repetitions. This is a construction-efficiency
result, **not 18 output certificates**. There were no real requests, solver
calls, proposed lower bounds or new SAFE/UNSAFE conclusions in this experiment.

Keep the adapter opt-in. The signal supports a separately controlled integration
into the full proof supervisor; it does not support enabling it by default,
claiming that it fixes the sealed real-request timeout, or rerunning that request
without a new scope. Do not enlarge this synthetic experiment or tune its cache.

## Identity and controls

- Clean execution commit: `85a8e4374462e7ed083658a441917d61a5a44048`.
- Protocol: [frozen protocol](source_construction_parse_protocol_20260924_r1.md).
- Configuration: `configs/backend_controls/source_construction_parse_r1.json`.
- New controls: [17 passed](source_construction_parse_controls_20260924_r1.json),
  including mutable-input/returned-container contamination, scope/shape binding,
  collisions, bounded eviction, deadlines, exceptions and incomplete evidence.
- Before execution, 53 unchanged regression tests also passed:
  `scoped_proof.tests`, `scoped_source.tests`, `source_enclosure.tests`,
  `source_enclosure.portable_tests`, `full_source.tests` (17.053 seconds).
- The 494 original implementation files remained unchanged. Only the separate
  adapter was used; the checker was always the original uncached implementation
  in a fresh `python -S` process, with no new producer/cache, model or solver
  import. This is independent program checking, not independent human review.
- Machine had approximately 91 GiB available at admission. All modes used the
  same CPU-only environment with numerical thread limits of two; they were not
  CPU-affinity isolated from other users. There was no GPU workload in this study.

All source/guard/property obligations are unchanged. Cache entries contain only
exact parses of content-bound CSR matrices, never bounds or validation verdicts.
Original producer algebra and original miss parsing both remain unchanged.

## Complete measured costs

Each row reports medians of three executions. Each complete execution charges
source generation, worker loading/imports, construction, serialization, fresh independent
checking, owned-process cleanup and terminal publication. The return clock also
includes the final cost-ledger write. One-time batch-supervisor startup,
configuration/source-hash admission, controls and later administrative review
are outside these per-call clocks; they are not included in the reported sum.

| Synthetic width | Mode | Build process (s) | Check process (s) | Complete call (s) | Max sampled parent+worker RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 | Original reference | 0.3439 | 0.4910 | 0.8475 | 55.65 |
| 16 | Adapter, cache off | 0.3833 | 0.5010 | 0.8948 | 56.50 |
| 16 | Adapter, cache on | 0.2754 | 0.4965 | 0.7873 | 60.48 |
| 32 | Original reference | 0.9011 | 1.5526 | 2.4782 | 93.32 |
| 32 | Adapter, cache off | 1.0023 | 1.4892 | 2.5110 | 93.45 |
| 32 | Adapter, cache on | 0.6853 | 1.5280 | 2.2351 | 95.15 |

The cached build-process median is **19.9% / 23.9% lower than the original**;
the complete-call median is **7.1% / 9.8% lower**. Relative to the disabled
adapter, complete-call medians decrease 12.0% / 11.0%. These are ratios of
medians, not median paired speedups or population estimates. The disabled
adapter is slower than the original, so using only that baseline would overstate
the practical improvement. Three repetitions on two small deterministic graphs
do not establish robust real-model latency gains.

All 18 returned-call clocks sum to 29.3243 seconds. The unchanged checking
process remains most of the cached complete-call cost; construction optimization
does not remove that work. Sampled RSS rises, especially on width 16. The cache
limits retained entries/payload/cells, not total interpreter RSS, and the sampled
maximum is not a guaranteed instantaneous peak.

## Mechanism evidence, not a real-timeout attribution

Both synthetic sizes perform 402 CSR lookups per adapter call. Width 16 reduces
actual exact parses from 402 to 135 (267 hits); width 32 reduces them to 145
(257 hits). The frozen 64-entry bound causes 71 / 81 evictions. Final live entry,
payload and cell counts are zero in every call. No capacity was increased after
seeing these results.

Within the equally instrumented adapters, median inclusive CSR parsing time
decreases from 0.2129 to 0.1075 seconds at width 16, and 0.6701 to 0.3352 at
width 32. The improvements are concentrated in shared joins, property projection
and LP construction; affine/ReLU and pair-guard timings change little. These
times are nested and must not be added to complete-call time. Matrix parsing is
therefore a demonstrated repeated cost **in these synthetic graphs**, not the
established unique cause of the sealed CIFAR4088 construction timeout.

## Audit and retained artifacts

[Machine-readable saved-only review](source_construction_parse_results_20260924_r1.json)
records the fixed roster, all costs, raw-file hashes, matrix/check identities,
fine-grained timers and cache counters. The review independently reran the
unchanged exact construction checker for both distinct source/bundle pairs,
then reproduced in `--check` mode. It made no new solver calls.

Raw root: `data/moe/results/source_construction_parse_synthetic_20260924_r1`.
All 236 files, 184,793,257 bytes, are retained locally, not committed as raw
data. The compact review is committed. Reproduction of the saved review:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m source_construction_lab.review --check
```

Do not rerun the exclusive frozen execution in the same directory. Historical
results, input98, the 23 source-gap gains, numerical gates, 25% scheduling and
external baseline comparisons remain unchanged.

## Next bounded gate

If this adapter is taken further, separately version the full proof execution
integration. Retain intake, original source checking, lower-bound proposals and
complete obligation aggregation under one unchanged 300-second request budget.
First exercise cached and original synthetic end-to-end controls, including
expiry during construction and partial evidence, and verify byte-identical new
matrices. Only then freeze any real comparison. The current result authorizes
neither borrowing old positive bounds nor calling the source-proof goal solved.
