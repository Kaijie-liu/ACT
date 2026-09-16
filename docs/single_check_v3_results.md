# V3 sole-check evidence tail: controls and inherited80s saved-proof result

Completed2026-09-16. Implementation, protocol and72 passing controls were
committed/pushed at `6a7c3269584e45dd65206f0f2f5a7a337af0badc` before the one
fixed offline execution. Read [contract](single_check_portable_v3.md),
[controls](single_check_v3_controls_attempt001.json) and
[result receipt](single_check_v3_saved114.json).

## What changed and what did not

The V3 packer produces **unverified evidence**, with no result argument or
expected-result oracle. It makes no mathematical checker call. The sole
authoritative result comes from the original complete rational checker inside
the isolated process. Source, theorem, code, cache policy and manifest hashes
remain bound. All support, range, McCormick, dual, reuse and legal-route/output
obligations remain; malformed proofs reject there. This removes redundant
full-check execution, not one of the necessary proof obligations. The contract
explains the equivalence on fixed bytes and the loss of cross-execution result
consistency as a diagnostic. Upstream trusted lowering remains unchanged.

Cache remains optional, same limits and exact mathematics. Legacy V1/V2 and
all original execution/method/source identities are preserved. No actual
upstream generator, reserve, per-solver cap, sample or threshold was modified.

## Controls

72/72 tests PASS, zero errors/failures/skips,29.886s. Eleven new V3 controls
plus all61 earlier controls cover original/V2/V3 exact result differential,
cache on/off, dimensions, ties, partial reuse, incomplete/nonpositive evidence,
invalid proofs that pack but fail the sole check, no result/precheck API,
one checker-adapter invocation, relocation after removing sources, rejected
expected-result injection, deadline after math/before response, inherited
clock and whole-driver admission, and result-coverage/sign/count mutations.
No V3 or earlier frozen source changed during or after controls/execution.

## Fixed archived input114, same80s reserve

This is NOT a new full-model request. It reads exactly the prior archived
rank0/input114 proof sources, performs no proposals/network/checkpoint/data
loading, and uses a new directory without retry. Original start is set to
tail_start-220: **220s simulated prior cost**, not220s of computation performed
by this run. The same total300s deadline and298s work cutoff leave80s/78s for
the actual tail; they are enforced, not subtracted after the run.

| Charged portion | Observed seconds |
|---|---:|
| Packaging subprocess | 15.642 |
| Sole isolated checker subprocess | 38.437 |
| Other startup, supervision, inventories and publication/review | 0.801 |
| **Observed tail through outer publication** | **54.880** |
| **Request clock including simulated prior220s** | **274.880** |

No process was killed; both stages completed. Final total-clock slack25.120s.
The old80s reserve was NOT increased. Packaging's own internal timer15.597s
and isolated checker timer38.375s are included in their process costs, not
additional costs. Driver exit274.110s and in-budget outer review/submission
274.880s include child cleanup and inner terminal serialization. A subsequent
read-only archival review0.009s is separate and cannot rescue failed runs.

Historical V2 on this same stored proof had precheck35.032s, packaging15.695s,
isolated checking38.882s and total90.497s. V3 has no precheck stage/file and
total54.880s. This is a descriptive historical comparison, not an interleaved
benchmark or population speedup. The dominant removed work is the duplicated
full precheck; differences in the other phase times also contain server noise.
Unlike V2's observed90.5s tail, this one V3 tail fits the inherited80s window.

## Proof result and portable artifact

The full exact result is identical to the archived original and prior V2:
`UNKNOWN_NONPOSITIVE`,9 obligations checked,3 positive/6 nonpositive.
Canonical full result SHA256:
`987cff13fc4b2df9e3fb8469823c2d74d6f950cce44efaa18bae8f2dcd580ec0`.
Comparison to the archived result occurred only AFTER independent runtime
acceptance. No old result was used as a verdict oracle; malformed transport
or proofs are still rejected independently.

The package is10,824,516bytes (10.323MiB), retaining the old content-addressed
transport and compression. Isolated cache267/290 hits,23 parses,zero live
entries after check. The original request remains TIMEOUT. The remaining six
nonpositive bounds did not become positive, so this is neither new SAFE nor
proof that this evidence configuration now gives new certificates.

Raw package/log root (not committed):
`data/moe/results/single_check_saved114_20260916_v3`.
Portable subtree: `tail/portable`.
Bundle SHA256: `b98a723fcb0136d8f6245053759ece9c7598d5e5768e8020aaa3067a0bc6581e`.
Statement SHA256: `cae8e865a3d79ea9b1a3e939d5b1711c0fda17c4081321d023849f5309693b13`.
Controls SHA256: `f23dc5a891507482a2adfa6333dab0c97a4d769d753bc39d3bcfa66c651f1a21`.
Result receipt SHA256: `3c9f0d192672188a55d6d7c796236ae62b515ffa03eda896803a68f9340e57db`.

Both runtime admission and read-only archival review pass. A separate shell/jq
review rechecked every saved raw artifact hash, all source-control hashes,
phase costs/complete states, simulated-clock arithmetic, no-precheck markers
and unchanged original request/terminal hashes. Structural auditing is not
another independent implementation of every mathematical bound proof.

## Next bounded step

The duplicate-check tail bottleneck has a demonstrated single-case remedy
without increasing the reserve. It is now reasonable to PREPARE a separately
frozen, small development comparison of a full upstream+tail flow, with actual
capture/proposal costs, unified300s, unchanged80s reserve and explicit terminal
audit. First connect and control-test that full-flow execution; do not silently
replace the sealed cohort's driver or reuse its old results as new outcomes.

Primary engineering endpoints should be complete independent checks and cost,
with missing/nonpositive/timeout split retained. Do not promise extra SAFE:
checked nonpositive bounds remain a separate proof-strength limitation. No
new real-request experiment, holdout or model training was launched here.
