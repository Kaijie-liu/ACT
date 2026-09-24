# Full-budget shared-residual comparison: route gain, no complete output proof

Both frozen requests completed their terminal lifecycle and are now sealed.
**ZERO new complete output certificates.** The shared representation made
checked route exclusions available within budget, but full-output evidence did
not close. No failed call was retried, extended or replaced.

## Registered same-object comparison

Implementation `cc55c2ed8ba3425f6e4b0a508dadc16ae5efee52`; clean launch
`2c7f9deed7ad894d780ad68ac82dc56ad268a1a9`. Config SHA256
`e06061d1eacd0164dca1a0473c7945a01832614c35dc2de8caf8f2b9cf2113c8`.
Existing seed0 checkpoint, next sequential manifest rank3/CIFAR4099, label4,
exact 2/255, unchanged property and center; observed engineering case, not a
fresh holdout. Pairwise then shared, once each, 300 s / two CPU threads / sampled
8 GiB. All 252 original pair/property duties remain in each denominator.

Both captures have source SHA256:
`312a60342d30f86576a4baa32038ae355b4597fa950a947d1d35edc51a1e9093`.

## Whole-request results and all-phase cost

| Item | Pairwise route evidence | Shared residual evidence |
|---|---:|---:|
| Effective terminal | TIMEOUT | RESOURCE_LIMIT |
| Whole accounted time | 298.0496 s | 287.5545 s |
| Intake | 8.3016 s | 8.2608 s |
| Route preparation | 103.8555 s | 18.2279 s |
| Route check | 185.8853 s, unfinished | 23.8906 s, completed |
| Retained construction/publication phase | not reached | 237.1674 s, resource stop |
| Peak sampled parent + owned-group RSS | 0.8805 GiB | 8.0567 GiB |
| Online route exclusions | none accepted | 25/28 pairs |
| Original duties discharged by checked route exclusion | 0 | 225/252 |
| Published retained construction | absent | absent |
| Native output LP calls / candidates / checked output bounds | 0 / 0 / 0 | 0 / 0 / 0 |
| Complete positive output requests | 0 | 0 |

The publication reserve is unchanged: useful work stops at 298 s, with 2 s
reserved for terminal records. The shared run hit the sampled 8 GiB admission
limit earlier; this was **not an OS OOM diagnosis or a GPU memory failure**.
Whole-time accounting includes all reached phases and cleanup. The shorter
shared terminal time is a resource stop, not a full-request speedup.

Both route-preparation phases completed, so their 103.8555→18.2279 s costs are
a valid descriptive same-case observation. The old route check is censored:
do not turn 185.8853/23.8906 into an exact checking speedup. Both whole requests
remain incomplete, so there is no complete-certificate runtime comparison.

## What the saved trace actually localizes

The shared run finished candidate preparation at elapsed 26.452 s and published
its independently checked route receipt at 50.333 s. The remaining pairs were
`{4,7}`, `{5,7}`, `{6,7}`; experts4–7 were needed. These are **retained potential
routes**, not three independently witnessed reachable routes.

More precise than the phase label `construct`:

1. `checked_lazy_source_construction` entered at 51.0801 s and returned at
   282.9478 s (231.8677 s). Its untrusted in-memory construction returned.
2. `serialize_construction` entered at 282.9479 s.
3. The resource watchdog stopped the owned process about 4.604 s later.
   Neither `construction.json` nor its partial file was published.
4. Source checking, output proposal and final aggregation were never started.

`residual_proof/worker.py` publishes the whole returned bundle through
`scoped_proof/io.py::save`, which currently performs
`json.dumps(...).encode()` before opening the temporary output. This makes a
full serialized representation coexist with the live construction object.
The event trace and absent partial file support locating the observed limit
in whole-buffer serialization before file opening. They do NOT distinguish
every allocator/JSON-string/byte-buffer contribution. There is no heap profile
of the stopped process.

The builder also holds all retained expert traces and per-pair joint, guarded,
projected and output-LP objects until publication. That code pattern motivates
bounded-memory evidence publication, but it is not a measured allocation-level
causal decomposition. An in-memory return is **not** a serialized or independently
checked source/output proof, and its matrices cannot be compared from this
archive because they were not published in either arm.

Even eliminating the serialization resource peak would leave only about 15 s
before the work deadline at that point, with source checks and all output
queries/checks still outstanding. Thus a serialization fix alone is not evidence
that a complete request would close at 300 s.

## Independent audit and exact real-case differential

Saved-only batch/source/route audit: PASS, zero issues, 209.996 s separately
timed; complete archive operation 210.100 s. A fresh-process archive replay and
captured checker-output differential also passed, 208.498 s separately timed.
Neither invokes a model, candidate provider or native solver. These audit
times are not added to the request budget or counted as online success.

The offline pairwise check recovered the same **56 ordered lower bounds,
residual corrections and nonzero-residual counts**, and the same **28 pair
decisions**, as the shared checker. Normalized comparison SHA256:
`334c9bb97278fd6c7f6cc393e8157cf39259b8cfbc2c013f35a1850920abc861`.
Consequently this real case supports unchanged router evidence strength, not
just equal aggregate exclusion counts. It does not retroactively accept the
pairwise route receipt that was missing at its deadline.

All 565 frozen bindings are intact. All 99 raw files, 290,397,400 bytes, are
preserved outside Git. The archive additionally rejected 12 corruptions of the
actual cost records and four malformed operation traces; open/partial event
controls passed. The integration gate had 121 passing tests. These controls
and a successful audit do not constitute a complete output proof or human
technical review.

Evidence:
[protocol](residual_proof_protocol_20260925_r1.md),
[freeze](residual_proof_freeze_20260925_r1.md),
[controls](residual_proof_controls_20260925_r1.json),
[batch audit](residual_proof_execution_audit_20260925_r1.json),
[hash-bound archive and traces](residual_proof_execution_archive_20260925_r1.json),
[fresh replay / exact differential](residual_proof_execution_replay_20260925_r1.json).

Read-only reproduction, with archived files present:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/replay_residual_proof_comparison.py
```

## Decision and next bounded work

Seal both calls. Shared residuals achieved their local objective: identical
checked route evidence became available online and execution progressed beyond
the old stopping phase. **They did not achieve the full-output endpoint.**
There is still no output LP evidence from which to diagnose insufficient
relaxation or claim the model unsafe. No output acceptance gate is relaxed.

Next engineering control should investigate bounded-memory canonical evidence
publication, checking exact serialized identity, interruption/partial-file
handling, atomic receipt acceptance and full cost, without dropping sources or
checks. Separately measure construction/source-check costs before claiming the
remaining budget can close outputs. No new real request, sample, increased
memory/time cap or numerical change is automatically authorized by this result.
In particular, do not rerun this case merely to recover its unsaved matrices.

Inputs98/4088/4096/4098 and now4099 remain sealed. Historical23 source-gap claims,
production tables, external comparisons and high-accuracy certification claims
remain unchanged. A complete declared-real-graph proof, deployed floating-point
correctness, actual route-changing witnesses and external competitive advantage
are still distinct requirements.
