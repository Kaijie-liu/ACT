# HybridZ endpoint execution under a bounded CPU budget

The existing multi-pair endpoint algorithm now runs through the bounded CPU
lifecycle engine. Twelve fixed controls pass, all fourteen scheduled calls are
retained, and an independent saved-evidence audit passes. This establishes
execution, reception and accounting on tiny supplied HZ fixtures, not a real
model certificate, GPU result or complete network-to-output proof.

## Protocol and retained attempts

The [protocol](hz_endpoint_supervision_design_20261001.md) and
[configuration](../configs/hz_endpoint_supervision_20261001.json) were committed
and pushed at `3671b4e1a` before execution. The mathematical modules, 128 proposal
iterations, source coefficients, gate ranges and acceptance threshold did not
change. Four normal controls and ten fault calls were fixed in advance.

Both archives remain under `baseline_runs/hz_endpoint_supervision_20261001_r*`.
[R1](hz_endpoint_supervision_20261001_r1.json) passed the initial controls, but
review found that the batch harness lacked an explicit stop after unconfirmed
cleanup. R2 adds immediate observation/pending records, a batch fail-stop control,
a maximum prefix count, comparison of every reference endpoint bound, and nested
cost diagnostics. [R2 is the accepted audit](hz_endpoint_supervision_20261001_r2.json).
Its summary SHA-256 is
`b0ccf9f1a6a0a04d6aa0115d4f561406e9c4d6cd7c8edecc8ff8494e1c9fc727`.
R1 is not silently replaced or presented as having the later batch guarantee.

## Outcomes and accounting

| Fixed request | Checked endpoints | Required duties covered | Mathematical conclusion | Observed API seconds |
|---|---:|---:|---|---:|
| Three-expert separation | 6 | 3/3 | Positive given HZ and gate premises | 2.9848 |
| Shared input relation | 1 | 1/1 | Positive given HZ and gate premises | 2.9024 |
| Independent input product | 1 | 1/1 | UNKNOWN nonpositive | 2.9569 |
| Deliberately partial separation | 4 | 2/3 | UNKNOWN missing evidence | 3.0000 |

The complete normal endpoints match the preceding archive individually, not
only in their minimum. Separation's minimum remains approximately 0.0677579;
the shared/independent minima remain 1/4 and −3/4. Equal gate endpoints share
one identical query, with both semantic labels checked. No mathematical gain is
claimed from adding supervision.

All ten injected faults reached their intended location: five ERROR terminals
and five TIMEOUT terminals. They cover creation, proposal, first-pair prefix,
serialization, source and endpoint corruption, checker failure/delay/missing
output, and reception. None receives an online positive conclusion. Offline
checking of saved prefixes is separate; for example, the partial-delay call
retains two checked endpoints and four missing endpoints. Valid late files never
repair its TIMEOUT.

The four normally terminated checks contain twelve endpoint records; eight
belong to complete requests and four to the explicitly incomplete case. Across
all saved final prefixes, fifty-six endpoint records can be checked offline,
including repeats and failed calls. This count is not a number of certificates
or independent network properties.

Every call includes import/creation, merging and query construction, proposal,
serialization, checking, reception, owned-process cleanup and publication.
The nested operation timings are not added again to stage or API cost. On the
separation control, recorded creation including imports is about 0.872 seconds,
preparation 0.0123 seconds, and the three proposals together 0.0276 seconds.
The independent-check operation takes about 0.845 seconds and includes its
first ACT imports. These tiny-fixture observations are not kernel benchmarks,
scaling estimates or speedups. Normal calls have a thirty-second control budget;
fault budgets are fixed at eight, ten or thirty seconds, all inside the interface's
300-second maximum.

## Acceptance and remaining trust

The [policy](../scripts/hz_endpoint_supervised.py) and
[worker](../scripts/hz_endpoint_worker.py) reuse the existing bounded lifecycle
engine without changing it. The parent anchors fixed launch plans and actual
phase output bytes. Original HZ sources, gates and properties must match the
frozen basis; the checker independently reconstructs each newly allocated joint
frame and its endpoint objectives. Per-pair proof prefixes retain the complete
roster, using null candidates for unfinished pairs. The batch stops if cleanup
cannot be confirmed, retaining all pending slots.

`CHECKED_ENDPOINT_CPU_EXECUTION` means the checking workflow finished, including
a possible UNKNOWN. `obligations_complete` and `positive` are separate fields.
Only a timely externally observed API return supports budget acceptance. The
process executor confirms leader reaping and no live same-group members; it
does not prove cleanup of escaped descendants or impose a machine-checked bound
on arbitrary operating-system or filesystem stalls.

Network/guard lowering, source factor provenance and gate coverage remain
premises. The inner check retains `source_complete=false` and
`deployed_float_SAFE=false`. This entry is CPU-only and is not a relocated
`python -S` proof checker. No native solver, CUDA, full-size source, real model,
checkpoint or dataset was used. No production default or old result was changed.

## Verification and next boundary

The twelve new controls and eighty-five endpoint, support, device, propagation,
handoff, navigation and historical-closure regressions pass. Two read-only AI
reviews checked acceptance and lifecycle details; they are not independent human
technical review. The complete R2 audit rechecks saved evidence without rerunning
the candidate optimizer. The two small execution archives occupy about 2.17 MB
together; no evidence or cache was deleted in this stage.

All six research gates remain open. The next bounded step is a source-connection
review: determine how the already checked propagation outputs, factor identities,
guards and gate evidence can supply this endpoint interface on one declared
object, without splicing unrelated matrices or trusting a new float conversion.
Freeze that contract before implementation. Do not increase these fixtures or
admit real/full-size requests from the timing numbers above. The previously
refused physical GPU batch remains unstarted; any fresh admission is a separate
record, not an automatic retry loop.

Recheck the archived execution using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_endpoint_supervision audit \
  /data1/Kane/MOE/baseline_runs/hz_endpoint_supervision_20261001_r2
```
