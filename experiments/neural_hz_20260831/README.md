# Neural-HZ structural trials (isolated)

This directory is the only result write target for the new Neural-HZ shadow trials.
Historical results under `/data1/Kane/HyZor` are read-only inputs and must not
be modified, copied over, or mixed into these records.

Formal baseline: 1,870 / 2,413 solved across 13 families (1,063 CERT and 807
validated ADV). Nothing in this directory changes that headline.

Current checkpoint (2026-09-05):
`S0_C3_CLOSURE_AND_LOADER_AUDIT_20260905.md` closes the fixed C3 target as a
structural zero-hit, records the default-off production BN graph repair and
recovers the exact 1,870 baseline table from read-only Git history. The live
Overleaf table lost revision coloring, so its original live-path hash test
rejects; the untouched 543-row manifest nevertheless rebuilds byte-identically
using the recovered original table. All score vectors remain unchanged.
The existing TLL 29/32 candidate has been requalified on the current source
under `TLL_CURRENT_SOURCE_REPLAY_PREREG_20260905.md` and its V2 loader-compatibility
amendment. The current result is 27/32, retaining the formal 17/17 but losing
two prior candidate solves, so qualification fails. All 16 reported ADV pass
independent native-ONNX/original-VNNLIB replay. The first all-ERROR load attempt
is preserved separately. See `TLL_CURRENT_SOURCE_REPLAY_AUDIT_20260905.md`.

The active objective, hard representation/data limits, score semantics, and
one-structure-at-a-time campaign are frozen in `GOAL_CHARTER.md`. The original
hashed charter is preserved; `GOAL_CHARTER_E0_AMENDMENT_20260831.md` adds the
subsequently frozen external E0 retention semantics without changing the formal
score. The user-authorized continuation and sole current S0-C3 sequence are
recorded in `GOAL_S0_C3_RESUME_AMENDMENT_20260831.md`. Baseline sources and the
exact 13-family vector remain authoritative in `BASELINE_LOCK.md`.

The first composed-stencil rule was pre-registered in
`S0_C1_COMPOSED_STENCIL_PREREG.md`. Exact graph lineage subsequently proved
that its fixed Tiny iid143 ReLU36 target contains an ADD strictly inside every
possible two-Conv core, so S0-C1 is closed as a structural zero-hit without a
production attempt. The proof is in `S0_C1_LINEAGE_AUDIT_20260831.md`; the
production prerequisites and non-claims remain in
`S0_C1_PRODUCTION_PATCH_MAP_V1.md`. Its formal-score, E0, representation and
performance comparators remain frozen in
`S0_C1_COMPARATOR_AMENDMENT_20260831.md`.

The separately pre-registered exact residual-distributive S0-C2 rule is in
`S0_C2_RESIDUAL_DISTRIBUTIVE_PREREG_V1.md`. Its stricter graph-occurrence,
stable-frame, shared-suffix, actual-emission and whole-state boundaries are
recorded transparently in `S0_C2_IMPLEMENTATION_CLARIFICATION_V1.md`.
Authoritative real-graph evidence then proved C2-v1 is also a structural
zero-hit: the Scale nodes required by its frozen branch shape are dangling
siblings, while the variable producer chain shows they are nonidentity
BatchNorm maps. The closure and the independent whole-state alias risk are in
`S0_C2_REAL_GRAPH_ZERO_HIT_AUDIT_20260831.md`.

The current BatchNorm authority is V2, not V1.  The strict real Tiny143 audit
finds 19/19 decomposed BatchNorm pairs in the same systematic sibling shape:
the source graph has exactly 19 producer mismatches plus 19 missing Scale/Bias
events.  One generic 19-edge sibling-to-chain plan is rederived from the
current graph and its private clone passes with zero issue, with every Scale
and Bias numerical payload bound into the digest.  This is clone-only
capability evidence; `BN_GRAPH_FAITHFULNESS_AUDIT_V2_20260831.md` explicitly
supersedes V1 as current evidence and records gain zero.

Treating the missing Scale as identity or silently changing `D+` to `D*` is
forbidden. The successor C3 rule retained its preregistered
`Conv -> D* -> ADD -> D* -> Conv` grammar. Its pure planner passed 37/37 C3
tests and 121/121 across the C1/C2/C3 suites; its final isolated runtime adapter
now passes 264 tests. The 2026-09-05 complete-path negative screen proves three
of the four ReLU36 branches necessarily contain nested ADDs, even with the
most generous permitted ReLU source boundaries. C3-v1 is therefore closed at
this occurrence with gain zero. Trial9 exited naturally with an exclusive
validated UNKNOWN record and sidecar; its nine frozen source files still
verify. The new BN loader option is off by default and has no score credit.

The historical adapter contract and originally planned census are frozen in
`S0_C3_RUNTIME_LINEAGE_ADAPTER_PREREG_V1.md`.  Runtime evidence must travel
inside each term and may not be reconstructed from an external object-id
cache.  The first repaired run is lineage-only because ADD32 has four terms:
three may expose the earlier ADD24, while ADD32 also retains layer40 as a
second consumer. The subsequent negative graph proof closes this occurrence
without runtime migration. It does not hide the earlier ADD, invent an affine
source boundary or relabel inherited terms as identity branches.

The corrected group-intersection contraction arithmetic is executable in the
isolated `test_group_intersection_contraction_oracle_v1.py`; it is not imported
by production. Its source hash is frozen in
`GROUP_INTERSECTION_CONTRACTION_ORACLE_V1.sha256`.

The production-shaped descriptor V2, C1/C2/C3 pure planners, transactional
prototypes and Trial 9 exit sealer remain default-off isolated artifacts. The
descriptor's independent/main-agent defect audit, exactness evidence and
still-open materialization boundaries are recorded in
`S0_C1_ISOLATED_V2_ADVERSARIAL_AUDIT_20260831.md`. None is a production or
score promotion. The real-CSR artifact and whole-state ledger audit, including
the repaired NumPy owner-allocation, CSR-view, multi-span, graph-root and
compound-role/cycle defects, is in
`S0_C2_MATERIALIZATION_ACCOUNTING_AUDIT_20260831.md`.
Its ledger suites pass 59/59 and the combined descriptor/planner/artifact/
ledger checkpoint passes 183 tests. `S0_C1_ISOLATED_V2_SHA256SUMS` is a preserved
historical checkpoint; its entries predate later edits and it does not seal the
current worktree. `CHECKPOINT_20260905_SHA256SUMS` seals the current sources,
audits and new evidence without rewriting that historical manifest.

The 543 formal UNKNOWN/TIMEOUT rows are partitioned into six mutually exclusive
ONNX-structure cohorts by the content-addressed manifest under `manifests/`.
The ordered target cards and promotion rules are frozen in
`STRUCTURE_CAMPAIGN_543_V1.md`.

## Trial 1: inactive equality-factor projection

Pre-registered structure: a continuous HZ factor no longer occurs in the
network output basis, has exactly one defining equality, and can be eliminated
with a strict decrease in predicate nonzeros. The transform is exact, never
pivots a binary factor, reconstructs eliminated input coordinates, is disabled
by default, and rejects fill-increasing substitutions.

Promotion order:

1. mathematical and witness tests;
2. paired baseline/Neural-HZ shadow runs on instances sharing the structure;
3. per-family shadow replay;
4. frozen 2,413-instance replay with every one of the existing 1,870 results
   retained and zero invalid adversarial results.

Files in `results/` are created with exclusive-create semantics. A rerun cannot
silently overwrite an earlier record.

Trial 1 is now closed as a standalone promotion candidate. It exactly removes
the target structure and preserves every sampled verdict, but the decisive
infeasible/CERT cohort became slower and produced no formal gain. The pass
remains opt-in as a measured compiler component; it is not enabled in the
1,870-result baseline.

## Trial 2: predicate-implied binary phase fixing

Pre-registered structure: a ReLU binary factor remains in the HZ output MILP,
but one of its two values cannot intersect at least one existing predicate row
even when every other latent is independently relaxed to its HZ box. Only then
is the binary fixed and substituted. The interval calculation is outward
enclosed; if both values survive, the factor is retained unchanged.

This is an exact simplification of the existing nonconvex HZ predicate set. It
does not branch, split, attack, invoke a backward/dual rescue, or convexify an
unfixed binary factor. Fixed source IDs are retained for concrete witness
reconstruction. The first structural census targets binary-heavy ACAS Xu,
ReluSplitter, and TLL examples. Trial 2 is opt-in and cannot affect the formal
baseline before the same four promotion gates above pass.

Trial 2 is now closed as a structural miss: 14 materialized ACAS Xu/TLL final
HZs contained 2,895 binaries and yielded zero implied fixes. ReluSplitter lost
the HZ before this final-state pass. The tested rule remains default-off and is
recorded as a negative result.

## Trial 3: fill-aware exact ReLU graph quotient

The existing exact ReLU graph introduces two continuous factors, one binary,
and three predicate rows per unstable neuron. Its exact quotient needs one
continuous factor, the same binary, and three inequalities. For preactivation
support `p`, the respective new predicate nnz counts are `p+7` and `2p+5`.

The current selector uses the quotient when fill strictly decreases, or at
equal fill (`p=2` on average) only when the post-transform latent width is at
most 512. This preserves large equality blocks that MILP presolve handles well.
Both dense and sparse HZ implementations are opt-in; the sparse version
reserves only one continuous frame slot. Formal baseline remains 1,870 / 2,413.

Trial 3 v3 is retained as a default-off component, not promoted. It preserves
the TLL N16 retained certificate by selecting no quotient there and improves
the smaller TLL iid 2 certificate, but it has not produced a new formal solve.

## Trial 4: exact shared ReLU graph

Pre-registered structure: two unstable ReLU inputs have byte-identical center,
continuous-generator, and binary-generator rows in the same sparse HZ frame.
They are therefore the same latent affine expression, so their ReLU outputs may
share one extended exact-ReLU graph. Bounds are conservatively aggregated with
the minimum lower and maximum upper endpoint. No tolerance or hash-only match
is accepted, no predicate is relaxed, and near-equal rows remain separate.

The structure is common in TLL. On iid 2 it reduces the final HZ from
`442c/220b/660rows/2596nnz` to `314c/156b/468rows/1908nnz` and preserves CERT.
On retained iid 4 it reduces `1766c/882b/2646rows/10466nnz` to
`970c/484b/1452rows/6132nnz` and preserves CERT at 60 seconds, but is slower
than baseline. On formal UNKNOWN iid 7 it reduces the model to
`918c/458b/1374rows/5866nnz` without changing the 60-second verdict.

All 15 formal TLL UNKNOWN instances contain exact duplicate groups. Positive-
proportional census added no group beyond byte-identical duplicates, so that
extension was closed. Signed proportional census, however, showed that every
additional relation was the strict `+1/-1` case. For byte-exact negations the
candidate uses `ReLU(-x) = ReLU(x) - x`: one shared phase graph represents both
orientations while the original affine row remains explicitly in the output
basis. A compact form additionally replaces each group's two continuous ReLU
factors and equality by one continuous factor and three exact inequalities.
Compact form is selected for a layer only when that layer contains at least
one proven negative member. All-singleton layers retain the original extended
graph, preventing unconditional quotienting from leaking into zero-hit
families.

The compact signed-share arm preserves all five old TLL certificates and adds
five reproducible certificates on formal UNKNOWN iid 7, 12, 14, 18, and 22.
It also reconstructs concrete witnesses for iid 19, 21, 26, 28, and 31. Each witness
passes the converted PyTorch network and an independent original-ONNX Runtime
check against the VNNLIB inequality; the violation slacks are about 1.2786,
0.0412, 0.3721, 0.0708, and 0.7648. TLL therefore moves as a candidate from
`5 CERT + 12 ADV = 17/32` to `10 CERT + 17 ADV = 27/32`.

This is a candidate gain of ten (`+5 CERT`, `+5 validated ADV`), making the
cross-family candidate total 1,880 / 2,413. The formal headline remains frozen
at 1,870 until cross-family retained-result shadows and the complete 2,413-case
gate pass. Both signed-share arms remain explicit opt-in.

Representative CERT plus UNKNOWN/TIMEOUT probes in the non-TLL families found
no signed member, or lost the HZ before the candidate could run. Guarded paired
checks on ACAS Xu iid 0 and Cersyve iid 11 have identical baseline/candidate
continuous count, binary count, row count, and predicate nnz, with zero compact
layers. ViT iid 101 also records zero signed hits; its isolated worker retains
all intermediate states and reaches the known 16 GB lowering limit, while the
production cache-release path remains the retained CERT evidence. These are
structural shadows, not yet the complete 2,413-case promotion replay.

## Trial 5: width-adaptive exact dead signed-ReLU graphs

When a compact signed class has one Dense successor and the exact-real sum of
its successor weights is zero in every output row, its shared nonlinear graph
is dead after that Dense. Trial 5 proves zero with integer binary64 ratios and
removes only the graph's private factors and three local inequalities. It
fails closed on any retained coupling.

Solver behavior required a structural lift budget. All-graph elimination,
pair-only elimination, and unconditional two-pair elimination each produced a
different retained-ADV regression. The first zero-regression rule always
eliminates cardinality-two classes and eliminates cardinality-four classes
only at a ReLU source width of at least 8,192; larger aggregation classes stay
as redundant lifts. This rule retains the complete 28-case TLL solved set
under one candidate hash, with every ADV concretely valid and no errors.

It adds reproduced certificates on iid 24 and iid 8. TLL therefore reaches a
candidate `12 CERT +17 validated ADV = 29/32`, net `+12` over the frozen
family result. The cross-family candidate becomes `1,882 / 2,413`; the formal
headline remains `1,870 / 2,413` until the remaining promotion gates pass.
Full details and negative boundaries are in `TLL_FAMILY_GATE_V2.md`.
