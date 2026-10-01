# Guarded HybridZ views of shared expert templates

This separate representation factor follows `ed9af689deebdae9dd7bacdecdd6cc95351afb8d`.
It addresses repeated **source propagation**, not the already implemented sharing
of support matrices across objectives. The current source producer propagates two
experts for every unordered pair, even though its affine error rule and ReLU box
ranges do not use pair constraints for tightening. All current finite limits,
the 128-step CPU proposal, gate ranges and positive threshold stay unchanged.
No real/full-size request, native solve, CUDA retry or sealed object is admitted.

## Single change and mathematical contract

Construct one common entry with the checked input output expression and the
checked router factors/constraints, before any pair guards. Propagate each expert
once from that entry, with distinct expert-owned private factors. Retain and check
the exact affine/ReLU reference and binary64 enclosure at every layer as before.
These are new proofs for this request, not imported checkpoint-specific facts.

For each pair, independently construct and check its original conditional guard
entry from the same input and router. This version requires that the guard lift
introduce no new factors, preserve all common coefficients/constraints, and append
only inequalities over the shared factors. Refuse incompatible entries rather
than silently widening a template. Attach these checked inequalities to each
expert template by inserting them after the common constraint prefix. Keep all
original columns and expert-private identities; do not merge two experts' factors.

Soundness does not require a point to satisfy every pair simultaneously. For any
input and one legal pair, a common router assignment exists by the source chain.
Each expert template has an extension of that same common assignment; their
private factors are disjoint. The legal guard inequalities hold for the common
assignment, so both extensions survive their pair views. Their joint HZ covers
the actual paired outputs. All tie-legal unordered pairs and all properties are
still obligations; no feasibility exclusion is introduced.

This is source-checked template specialization, not cross-request caching or a
new abstract domain. On this registered box-only propagation path, adding guards
commutes with propagation up to factor names/constraint placement. The finite
comparison must independently check equality of every numeric pair source,
joint support base and endpoint objective, not just equality of positive counts.
No such equality is promised for support-tightened/guard-dependent propagation.
The new checker must reject inward guard-specific ranges reused as global facts.

## Fixed execution and controls

Use the four unchanged declarations and hashes in
`configs/hz_source_representation_20261001.json`. Run both the current row-lifted
per-pair source path and the template path afresh, in fixed alternating arm order.
Each arm has one cooperative at-most-300-second deadline including declaration
creation, propagation, proposals, serialization and checking. Keep all attempts;
do not tune fixtures, ranges, order, support iterations or positivity. Record
generation/check costs separately from orchestration/imports. These controls do
not measure production speed or provide hard-budget supervision.

The eighteen groups are: complete four-source rosters; exact numeric pair/base/
objective differential; expert propagation call counts; shared/private ownership;
common-entry reconstruction; template source/layer inventory; wrong template
selection; guard/RHS omission and inward change; private-column alias/pollution;
property/gate binding; missing pair; partial output evidence; stale endpoint
proof; independently checked view composition on the existing nonrepresentable
affine/ReLU/guard operator controls; guard-dependent range rejection; deadline
and in-place pollution; checker independence; complete cost/archive inventory.

All normal endpoint candidates are freshly generated. The unsafe declaration may
not be accepted positive. Other counts are observations, not acceptance gates.
Expected expert propagation calls are E rather than E(E-1); router runs once in
each arm. Count actual dispatches, not only inferred trace lengths. The checker
reconstructs all views from checked common/template/guard objects without calling
the producer. Fixed mutation queries and hashes are independently replayed from
saved normal objects; unexpected timeouts do not count as mutation rejection.

## Scope and next decision

Do not call a reduction in propagation calls a demonstrated speedup. Pair views,
joint matrix construction, all endpoint queries and serialization remain and may
dominate cost. The finite local row/global/source limits stay intact. Cross-pair
constraint registry, matrix-free support, native comparison, full-budget/portable
integration and realistic capacity require their own decisions. If representation
equality fails, preserve and diagnose; do not weaken this control into mere outcome
agreement. All G1–G6 remain OPEN.
