# Checked declared source to actual HybridZ endpoints

Completed finite CPU controls on 2026-10-01. The
[contract](hz_source_connection_design_20261001.md) and
[protocol](../configs/hz_source_connection_20261001.json) were committed and pushed
as `96c26c184988e2d17d055e0aa6a1c0bb68fb0ba0` before implementation/execution.
No real model, checkpoint, dataset, native optimizer or CUDA call was used.
All six research-goal gates remain OPEN.

## Result

The new opt-in path connects the four unchanged H2 declarations to actual
SparseHZ affine/ReLU and shared/private pair construction, then generates fresh
128-step CPU batch dual proposals. The independent checker checks source
lowering, route entry, gate premises, every endpoint objective and exact lower
bounds. No old direct-node LP or given-HZ certificate is reused.

| Frozen source | Positive / required pair–property duties | Checked endpoints | Result |
|---|---:|---:|---|
| weighted_sign | 3 / 3 | 6 | Checked positive for declared real source |
| tied_partial_reuse | 18 / 18 | 18 | Checked positive for declared real source |
| unsafe_tied | 0 / 6 | 6 | UNKNOWN_NONPOSITIVE |
| unresolved_sign | 6 / 6 | 12 | Checked positive for declared real source |
| Deleted final-pair evidence, derived control | 15 / 18 | 15 | UNKNOWN_MISSING_EVIDENCE |

The four normal cases contain 42 freshly checked endpoints. The partial control
rechecks a subset of existing evidence, not 15 additional certificates. The
historical fixture name `tied_partial_reuse` does not imply reuse is enabled:
this protocol proves all eighteen obligations without free facts. Equal gate
endpoints share one identical objective with both endpoint labels covered.

For weighted_sign the smallest checked lower bound is exactly
`194996818405069/18014398509481984`, about 0.0108245, above the unchanged 1e-7
acceptance threshold. Its declaration hash remains
`b50e6ac79b92ac1d25e6a2ad8d83d1958353e90eeba869663e3d9d7c9eff0de2`.
Exact evaluation at x=-1 and x=1 produces distinct legal pairs {1,2} and {0,1};
x=-1/2 admits both tie-legal pairs. All three unordered pairs are nevertheless
retained, including the empty {0,2} branch. This is a declared-source,
route-changing **synthetic** positive, not a new trained-model result.

The other normal minimum bounds are 2023/200, -1977/200 and 999/100 respectively.
A checked negative lower bound is not an UNSAFE witness. No new MC comparison
was executed, and the old MC negative points are not evidence on these matrices.

## What was changed inside the analysis path

[Producer](../scoped_source/hz_source_build.py) calls the existing
`sparse_hz_linear` and `sparse_hz_apply_relu_exact` kernels. It adds a checked
source trace around them, not a production-default replacement. The
[independent checker](../scoped_source/hz_source_check.py) verifies:

- Input containment for the exact declared, clipped domain.
- Exact affine residual compensation around the actual nominal floating result.
- Actual ReLU rows, range classification and fresh factor slots; only the new
  blocked inequality suffix is permuted into the theorem checker's row order.
- Exact conversion of every state coefficient to live binary64, followed by
  whole-snapshot equality. Unsupported nonrepresentable coefficients are
  rejected rather than silently rounded.
- Conditional non-strict route guards in the full checked router factor space,
  with the original input expression restored for both expert propagations.
- Distinct expert-private factors, exact terminal-to-endpoint snapshots,
  sign-derived gate ranges with the first expert's orientation, all legal pairs
  and all required classification margins.

All normal fixtures happen to have zero affine-compensation factors. A separate
control uses the nonzero exact residual of a stored binary64 0.1×0.1 affine
operation, checks its compensation and rejects both factor deletion and an
incorrect zero radius. This control does not establish large-model coverage of
rounding cases. Actual ReLU arithmetic that fails exact reconstruction is also
rejected; the entry is not a generic exact binary64 propagation claim.

## Evidence and controls

[Compact independent audit](hz_source_connection_20261001_r2.json) reports PASS,
zero issues, eighteen controls and five checked packages. The local archive is
`/data1/Kane/MOE/baseline_runs/hz_source_connection_20261001_r2`.
Its summary SHA-256 is
`89ee79c80a157161301fe002af5a1c7b919e6cee7b720681a3c0899f1b6dc3af`;
saved implementation bytes and complete case/test rosters are bound by the audit.
R1 and its [pre-hardening report](hz_source_connection_20261001_r1.json) are
preserved. Read-only review found that the first archive audit did not require
normal evidence completeness, verify scope flags or reject expected test failures.
R2 adds those receipt checks and mutation controls. All five package identities
are unchanged between R1 and R2; no source, candidate algorithm or bound changed.

Controls reject inward inputs, nonzero compensation deletion, corrupted ReLU
coefficients and blocked row order, inward ranges, missing layers, reversed
guards, aliased factors, wrong gate orientation, endpoint/source substitution,
source/property changes, missing pairs, stale proofs and nonrepresentable
coefficients. Expired checking/build deadlines fail closed. A partial proof stays
UNKNOWN. The acceptance path does not call a proposal or propagation routine.
ACT's package initialization still imports numerical modules; this is **not**
a relocated standard-library-only checker.

Eighty-five existing endpoint/support/device/checked-propagation/navigation/closure/
handoff regressions pass. Two read-only AI code reviews found no blocking mathematical
acceptance defect; they are not independent human technical review. The audit
rechecks saved proofs without a new optimizer invocation.

## Cost and guarantee boundary

The measured in-process construction/proposal/serialization/check totals for
the four normal sources were 0.0880, 0.2217, 0.0827 and 0.1032 seconds. Detailed
parts are in the audit. These exclude import time, fixed-fixture creation and
test/archive orchestration. There is no timing comparison or capacity inference.
One cooperative deadline covers each measured call, but the previous endpoint
hard supervisor does not automatically cover this new upstream path.

The resulting positive concerns the **declared real graph with its stored
binary64 parameters**. It relies on declaration/program correspondence and the
checker implementation. It does not prove native floating-point execution,
retrospectively validate the old analyzer, repair historical input-source gaps,
or close G3's real-request and portable-proof requirements.

The two new archives total 1,443,620 bytes (R1: 719,741; R2: 723,879).
The MOE tree remains about 223.80 GB in logical bytes; no cache, result or failed
attempt was deleted. New processes used `-B` and disabled bytecode generation.

## Next boundary

Before any size increase or real intake, separately freeze integration of the
**whole new source path** into bounded CPU supervision: source creation,
propagation, fresh proposals, serialization, checking, reception and cleanup.
Test deadline, error, partial-source and partial-output evidence with complete
costs. Portable source checking remains another explicit delivery. Do not infer
admission from these subsecond controls or retune their 128 iterations.

GPU execution remains refused under the separately archived resource decision;
no automatic polling or retry occurred. Real/full-size inputs and the sealed
98/4088/4096/4098/4099 objects remain closed.
