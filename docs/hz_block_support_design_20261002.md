# Block native support for shared HybridZ expert templates

This single representation/execution factor follows
`8a9e9a824759b838b8f78611ad29f2c9a6534054`. Keep the checked expert-template
source algorithm, exact source leaves, gate rules, complete pair/property roster,
128 projected-dual updates and positive threshold unchanged. Replace materialized
joint HZ/LP matrices with direct local-block operations and independent residual
checking. This is not new bound tightening: the old backend already shares its
one pair matrix across multiple objectives.

## Source and block contract

The pair support input consists of actual SparseHZono common, guarded entry,
expert A template and expert B template. Common contains the full router
assignment and restored input expression. Entry must preserve those factors,
outputs and equalities, retain its inequality prefix, and add only shared guards.
Both expert templates must retain the exact common factor/constraint prefix.
Each private block remains disjoint; private rows may use shared columns.

The three constraint blocks are entry in full, A after the **common** constraint
prefix, and B after the common prefix. Never cut templates at the longer guarded
entry prefix. The global factor order is shared/A-private/B-private continuous,
then shared/A-private/B-private binary, exactly as the existing joint builder.
Binary offsets use the final global continuous count. Keep all original global
factors and all rows; binary factors are explicitly relaxed to [-1,1].

For a given objective, project the two expert outputs and the single property
constant exactly. Each local block uses a selection map S_j from the same global
factor vector. With y_j<=0, free t_j and finite bounds l,u, check

    r = c - sum_j S_j^T (A_j^T y_j + E_j^T t_j)
    L = d + sum_j (b_j^T y_j + h_j^T t_j)
          + sum_i min(l_i*r_i, u_i*r_i).

Shared residuals are summed **before** the box minimum, and the proposal forward
step gathers the same global argmin into each block. Separate local argmins or
overwriting shared residual contributions would change the algorithm/relation.
Keep zero initialization, 129 evaluations / 128 updates, .125/sqrt(k+1) steps,
strict per-target best retention and exact zero-candidate fallback.

The independent checker derives row spans, column maps and objectives from the
anchored original snapshots, not a candidate-provided sparse matrix. It checks
every coefficient including zero-dual rows. Every candidate binds the batch and
complete query roster. Nonpositive or incomplete evidence is UNKNOWN, not UNSAFE.

## Direct source connection and finite execution

Use the same four declarations/hashes in the existing source representation
configuration. Both arms freshly propagate the same templates. The block arm
must neither build a joint HZ nor use the old complete-source checker as a hidden
joint prerequisite: independently reuse source-checking leaves, bind the original
template/entry snapshots and aggregate the block endpoint proofs. The old arm is
the current materialized expert-template path, not an artificially weakened
per-target backend. Default production dispatch remains unchanged.

Keep all old finite limits: 128 global factors/outputs, 256 constraints, <=8
support targets, <=4 experts/properties. CPU float64, one thread, no native solve,
physical CUDA, model/data load, sealed inputs or full-size rerun. Each normal arm
uses one cooperative <=300-second clock for source creation, propagation,
preparation, proposals, serialization and checking. Alternate arm order per
declaration. Preserve failures, source prefixes and auxiliary fault-call cost.
Imports and test orchestration are separately visible, not production timing.

## Frozen acceptance controls

Eighteen groups: complete four-source output coverage; independently reconstructed
full numeric base/objective differential; both arms' candidates cross-checked
with exact residual/bound equality; shared continuous/binary offset control;
shared residual cancellation versus independent-input control; common/entry
prefix and factor binding; private template span and alias mutations; zero-dual
row validation; property offset and objective mutation; missing pair/target and
partial evidence; candidate sign/nonfinite/excessive-claim rejection; source and
gate binding; no joint/remap/export/old-checker calls in the block path; fixed
iterations and CPU-only admission; producer deadline/pollution; checker expiry;
checker independence; archive inventory, ordered clocks and complete cost.

Additional given-HZ controls are the already established shared/private binary
fixture and +/- shared-input cancellation fixture in `test_hz_endpoints.py`.
They add no complete source claim. The shared case has a 1/4 zero-dual lower
bound; splitting the input gives -3/4. Full normal sources get fresh proposals.

For all 15 normal pair bases, independently expand the block definition only in
the diagnostic comparator and compare all coefficients, bounds and objectives
against the old arm. Transfer each arm's fresh dual to the other checker; require
the exact same residual and bound for that candidate. Different floating sparse
reduction orders need not generate identical trajectories. Report bound and
complete-outcome differences without tuning the algorithm to remove them.

Save concrete negative queries, independently reconstruct them from the anchored
normal objects and recheck actual refusal. Unexpected timeout is not ordinary
mutation success. Count matrix construction boundaries and local storage; absence
of joint allocation is not a measured peak-memory or speed guarantee.

## Stop and next gate

On semantic differential failure, stop and fix implementation, retaining the
attempt. Do not weaken equivalence, enlarge caps or change gates to pass. Even a
successful finite result does not admit real models, close G1/G2/G3 or fix the
known source/operator/package limits. Hard supervision and portable checking must
explicitly bind the new schema before any realistic capacity decision. No GPU
retry is authorized by this protocol.
