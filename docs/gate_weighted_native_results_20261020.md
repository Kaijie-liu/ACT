# M1b: native calls completed; batch reception FAILED

Execution: `8b6a8d51aeee2bb41869b7dc5fccc7a0978a96a8`.
Freeze SHA-256:
`bed01d909897a436f587502d9de2558f076a172a32ad683fc6c7c699e856c22f`.
One admission and one invocation; no retry, model/checkpoint/data load or GPU.
Here “no model load” means no **trained** model: three literal synthetic
`OutputLevelMoE` objects were constructed and native CPU propagation/HiGHS executed.

**Overall result remains FAIL.** The worker recorded 12/12 complete invocations
and 8/8 interface-fault groups PASS, but the owned-session observer rejected an
unreadable RSS observation. The controller never reached online saved-record
reception. Offline review cannot recover online acceptance or pass the M2 gate.

## Observed native behavior, not a performance result

Each table row is one fixed synthetic object and requested path, comparing M1
off/on. Counts are actual joint-F0 function entries; all recorded entries returned
status0. They are not all propagation/support/HiGHS calls.

| Object / requested path | off → on calls | off / on terminal | through-audit seconds off → on |
|---|---:|---|---:|
| A / staged | 1 → 0 | SAFE / SAFE | .367392 → .090019 |
| A / monolithic | 1 → 0 | SAFE / SAFE | .104360 → .080584 |
| B / staged | 2 → 0 | SAFE / SAFE | .130609 → .163497 |
| B / monolithic | 1 → 0 | SAFE / SAFE | .150443 → .103064 |
| C / staged | 1 → 1 | UNSAFE / UNSAFE | .096963 → .087935 |
| C / monolithic | 1 → 1 | UNSAFE / UNSAFE | .073449 → .158779 |

The compact records are authoritative for full-precision times. Each arm has
four SAFE and two UNSAFE invocations; six off/on pairs show zero observed
SAFE/UNSAFE/solved gain or loss. Total joint-F0 calls7→2, with six generated
gate-weighted pair/property facts replacing five joint calls: monolithic
aggregation makes those units different.
All six new positive facts use native `constant_exact` gates with gap0 and
weight1/2. This batch therefore does not demonstrate a positive shortcut with
a variable gate; C exercises the variable-gate fallback instead.

A/C requested staged uses the existing single-pair direct monolithic kernel.
B/staged uses multi-pair staged; B contains all three tie-legal unordered pairs.
Both C-on calls retain unavailable gate sources and one fallback property;
the full-model violation is replayed in all four C invocations. The original
gap policy was not changed. The saved monolithic evidence does not retain the
raw rejected support/gap metadata, so do NOT identify missing-gap as its uniquely
observed cause. Source inspection makes it a possible explanation, not a logged fact.

Two pairs are slower despite fewer/equal joint calls. These tiny fixed objects,
one shared batch process, startup/warm effects and failed outer reception do not
establish request speedup, new real SAFE, source-complete proof or M2 efficacy.
All positive conclusions retain the numerical-policy grade conditional on the
given expert intervals, gate ranges and upstream lowering.

## Failure and complete recorded cost boundary

The session record reports:

* `execution_error`: ValueError, `live process RSS unavailable`,
  `phase=confirm_stat`, PID487131.
* First and last identity both state R, SID487131, start_ticks1132932238;
  status was read (1150 bytes) but RSS was absent.
* Worker stdout had already published PASS and the bound result hash; recorded
  process returncode0. The combination is NOT a successful observation contract.
* Cleanup status `LEADER_REAPED_NO_LIVE_SESSION`, cleanup_error null.
  The remaining identity-only roster lists a zombie and no live member; no
  instantaneous RSS, escaped-session or global-zombie guarantee is inferred.
* Controller `ValueError('worker/session failure')`, exit1, original status FAIL.
  No `saved_record_audit` or final-FD measurement was reached.

Do not claim an algorithm, memory-capacity or deadline failure: the recorded
rejection is the RSS observation contract. A transient exit/teardown race is
plausible but the trace does not establish its unique cause. The existing observer
and hard limits were reused unchanged; do not launch a repair/retry chain under
this consumed attempt. A later read-only ps check found no PID487131; that does
not retroactively change the original observation.

Observed controller duration through terminal file export: **10.346614s**.
Worker entry to before export:8.664561s; worker import/construct/propagate/query/
serialize/audit/replay all occurred inside the common D. Cleanup .111103s.
Observed session-plus-caller RSS881,319,936B; worker sample742,797,312B.
These are samples, not guaranteed instantaneous peaks. AS remained8GiB/process.
Worker signed-fact Work29,874 = arithmetic813 + coefficients28,985 + rows76;
no new allowance per invocation. Upstream native operations are wall-time charged,
not included in that Work metric. Runtime retained74 files,349,129B, including
12 request tensors and4 witness tensors. No cleanup/deletion of failure artifacts.

Python/OS pre-entry launch, stdout flush/final exit and final controller FD count
remain unmeasured, not zero. Frozen cap300s/8M Work/128MiB/8GiB AS/2GiB sampled RSS
was not increased. Parent online recomputation was **not executed**.

## Saved-record audit and archive

[Offline audit](gate_weighted_native_audit_20261020.json) rechecks all twelve
saved package/manifest identities, sixteen tensor-file hashes, same-input/
literal/property identities, exact route rosters, actual counters, Work
continuity and six saved M1 arithmetic facts. It reports
`PASS_SAVED_RECORDS_ONLY`, `online_acceptance_recovered=false`.
Its separate offline Work1,251 is not added retroactively to an accepted online
run. Full tensor/model replay is recorded from the charged worker; this offline
checker does not execute it again or independently prove network lowering.

Reproduce the offline, standard-library record audit (no solver):

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B \
  scripts/audit_gate_weighted_native_archive.py \
  /data1/Kane/MOE/results/gate_weighted_native_20261020_r1
```

[Lossless JSON archive](gate_weighted_native_records_20261020.json) stores the
exact58 JSON texts in a base64/gzip mapping (mtime0, uncompressed hash recorded).
Tensor binaries remain at the runtime path; their hashes are in manifests and
the audit. No raw training data, checkpoint or external repository is committed.
An additional read-only AI review checked all644 frozen source hashes, package
hashes, terminal counts and the failure boundary. This is not human review.

## Research decision

M1b has a useful **native functional observation**, but the frozen complete-stage
acceptance failed. Keep the default-off code and original failure; do not promote
the phase, rerun the twelve calls, expand samples or launch M2.

Continue the independent G1 research candidate in
[the within-slice gate–margin design](gate_margin_relation_design_20261020.md).
Existing N1 already implements closed slicing; the prospective difference is
joint gate–margin constraints on the same factor vector. Its analytic separation
is a design control, not a new empirical result. Keep this work independent of
the observer policy. Any renewed execution/observer decision must explicitly
preserve this failure and specify a finite scope, not hide it by changing a
version name or tolerating absent RSS. G1–G6 remain OPEN.
