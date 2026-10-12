# Frozen captured source coefficient construction

2026-10-13; parent `b2057e75b4851025a93e4c853f7b6990a451c275`.
Connect the captured tensor-row reader to coefficient-reference affine and ReLU
construction. This is a source-lowering component, not a solver, a complete
request certificate, a GPU attempt or a change to production acceptance.

## Representation and contract

Use a source forest, not a conjunction of mutually exclusive route guards.
One immutable factor registry and coefficient-node bank serve ordered views.
Each view retains its parent, scope, new constraints and complete ordered output
roster. Scalars are separate from coefficient references. Intermediate views
remain in the inventory even when an inactive ReLU makes them unreachable from
the final outputs. All allocated factors, including unused zero-error columns,
remain in their original order. Complete sources include every unordered top-2
pair; this component does not exclude any pair by feasibility.

The source checker independently traverses the captured declaration and all
tensor bytes. It checks input containment, every supported operator, factor
ownership and allocation, inherited constraints and all pair-guard rows. It does
not invoke the producer or native affine operator. A prefix is explicitly not a
complete lowering, even though both readers must exhaust all source tensors.

For affine rows, actual `sparse_hz_linear` supplies the binary64 nominal output.
The exact rational reference uses the original captured weights and compensated
parent. Add the L1 coefficient and center error as a fresh continuous factor.
Allocate **all layer error factors before all output lift factors**, retaining
the existing nearest-error-then-outward-lift construction. Only a single exact
reference row is temporary. A checked diagonal input basis permits direct
weight-times-radius checking; no general dense matrix bank is required.

For ReLU, active output aliases the complete parent vector; inactive output is
zero. An unstable row with bounds `lo,hi` uses new negative/positive/sign factors
and an equality with coefficient reference `-parent`, local terms
`lo/2, -hi/2, lo/2`, and independent rhs `center-hi/2`. All auxiliaries precede
output lifts, which precede equality lifts; all negative inequalities precede
positive inequalities. Reference inclusion covers all earlier compensation.
Only this disjoint fresh-factor structure may reuse a checked parent L1 radius.
Arbitrary overlapping references cannot use a sum of separate radii as equality.

The native nominal workspace contains parent output matrices, not duplicated
inherited constraints: `sparse_hz_linear` only transforms outputs and passes
constraints through. The forest retains the original constraints unchanged.
The independent checker verifies the enclosure, not the native library's
implementation or deployed floating-point semantics.

## Bounds and accounting

Retain 32768 factors, 16384 active factors, 128 views, depth 64, 65536 nodes/rows,
1048576 local entries, 65536 references, 4096/8192 stored/intermediate bits and
128 MiB serialized forest. No old Store schema or limit is changed. A single
Work object and 300-second deadline cover generation, serialization, parsing
and independent source checking for each control. No reset per layer or reader;
no substitution of the endpoint 96M policy. Each exact source coefficient use,
stored coefficient parse, exact reference and accumulated compensation/radius
is counted; bounded native preparation and serialization time are also charged.
Construction refuses if the remaining limits cannot cover it.

## Fixed controls

Use the existing `hz_tensor_rows_20261013_r1` captured fixtures unchanged:

- Small source: complete router, both experts, ReLU, all final outputs and pair.
  Independently expand the small forest and compare to the old compensated
  source implementation, including every factor and constraint; no solve.
- Dense source: input/router plus expert0 through Linear1 and ReLU2 only,
  256 by 3072 first layer. Drain and bind the remaining tensor bytes but do not
  claim they were lowered. Run this large control once; an honest resource
  refusal is retained, not repaired by raising a cap or changing the fixture.
- Small controls: stable active/inactive/unstable and tie boundaries, three
  experts and all pairs, private factor separation, nonzero compensation,
  missing/extra operators or roots, altered rhs/coefficient/ref, wrong source,
  scope/frame/order pollution, incomplete source traversal, deadline and work
  refusal. Independent checks must reject semantic mutations even with a new
  forest hash. Preserve all failures and test repairs.
- Relocate the small captured source and forest, then check under `python -I -S`
  without native numerical imports, checkpoint or historical source directory.

Record source bytes, local/reference/expanded occurrence counts, factor and view
inventories, cumulative work, generation/native/serialization/check/outer time.
These are component costs, not acceleration or certificate gains. The archive
is new, capped at 64 MiB; dependencies and source identities are bound. No real
model, optimizer, GPU, sealed input or acceptance threshold is changed.

Stop after review and archive. General later-layer capacity, convolution,
support/endpoints consuming this new source forest, and physical CPU/GPU
comparison need separate admission. All G1–G6 remain OPEN.
