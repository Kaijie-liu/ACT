# C7 V1: complete exact affine-factor HZ, offline storage passed

This is an additive trial-log entry. The preceding goal turn was progress:
C6 completed its actual 14-term support certificate and identified the dense
composition blocker. This turn implemented and tested a different exact HZ
representation for that SAME residual-CNN suffix. It does not update a score.

Result `results/c7_factored_hz_20260905_v1/result.json` SHA-256:
`f63d45dd21d7bfa6d0c415cd24b10ac99345d053fa72ca82ad7a2bc9087928ef`.
231 tests passed (193 inherited plus 38 new); tests and worker exit zero,
source/provenance drift false. Supervisor wall 98.04544124472886 seconds.
The candidate remains isolated/default-off; no production file was changed.

## What actually changed

Common ORIGINAL operator suffixes are factored across sums, retaining source
identity and shared continuous/binary coordinates. The actual 14 terms and
9 sources become a 36-node shared definition graph. Every retained value
coordinate is encoded with a fresh continuous factor and a power-of-two
normalized equality. All original predicates remain in the same shared frame,
including the three entirely zero later sources and the other zero path.
There is no convex replacement, deleted binary phase, LP/dual decision,
instance menu, PGD, split, BaB or backward bound refinement.

The constructed SparseHZono has nc=254870, nb=1150, neq=244312, nineq=2300;
243162 continuous factors/defining equalities are new. Its value map has only
200 continuous entries, but nonconvexity remains in the coupled binary
predicates. Auxiliary count is a COST, not the simplification metric.
Complete HZ entries are 11,407,286, within 64M.

Every one of the 243162 new equations was independently checked against the
ORIGINAL unfused row oracle/source coefficients. The audit streamed 75,888,888
original entries, verified exact reverse power-of-two scaling, sufficient
auxiliary boxes, unchanged old predicates and original term multiplicities.
Triangular definitions prove unique latent extension and old-prefix witness
projection. Focused Fraction elimination independently recovers the complete
original real affine program, including non-dyadic stored float coefficients.
This is explicitly NOT byte identity to a differently associated rounded
materialized matrix. Live numerical semantics remain separately gated.

## Work, complete offline storage and construction

Support work 84,358,648 plus the preregistered encoding upper bound gives
**168,143,936**, below 256M; largest branch upper bound 89,688,528 below 200M.
This is an operation upper bound, not a measured runtime speedup or a direct
ratio to C6's differently scheduled composition bound.

Candidate construction with tracing took 44.150994 seconds; traced peak
452,019,578 bytes plus 164,704 tracer metadata; conservative resident growth
433,385,472 bytes. All are within the existing 1 GiB construction gate.
The independent coefficient audit took 18.874093 seconds. In the event journal,
the complete_identity_audit elapsed_s field is this STAGE duration because
the payload overrides the generic event clock; use construction/exit records
for whole-run timing. The raw event record is not rewritten.

The COMPLETE 460 registered offline numeric roots include the loaded native
state, source/predicate/operator payloads, new HZ, graph supports, coordinate
slots, exponents, checkpoint graph and all retained numeric reconstruction
state. They occupy **317,194,096 bytes / 30,950,482 entries**. Both are strictly
below the preregistered same-expanded-comparator two-leaf LOWER BOUND:
**629,346,312 bytes / 52,428,800 entries**. The two distinct-content leaves
each have 26,214,400 native expanded entries, every row coefficient verified;
their aggregate stays below 64M and their joint numeric payload below 1 GiB.
No full-reference total is claimed. Arbitrary Python heap and process RSS are
not silently included in the exact numeric metric; diagnostic peak including
reference construction was 1,972,816 KiB.

Checkpoint `results/c7_factored_hz_20260905_v1/lifted_hz.pickle` has
251,574,085 bytes, SHA-256
`6cbe92faf0d6eb5e4113f3e6e0b324a82a00d2d1a6019a73f2c2fd0f522e82df`.
It includes the HZ, definition graph, old prefix cache and reconstruction
metadata, and was subsequently reloaded/re-audited in a separate worker.

## Advancement is NOT approved

The subsequent unchanged-backend ingestion audit fails: see
`C7_CHECKPOINT_INGESTION_AUDIT_20260905.md`. This makes C7 V1 unqualified for
live/terminal advancement despite the valid offline set/work/storage evidence.
There was no ReLU78 application, terminal property/solver run, new CERT or ADV,
quarter-work promotion, family retention or full replay. Formal **1870/2413**
and independent E0 **61/400** remain unchanged. The research goal remains active.
