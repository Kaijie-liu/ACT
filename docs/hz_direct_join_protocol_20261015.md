# Direct source and portable consumer controls

Implementation protocol, frozen from `3ec0caded13114a33f2ae501b46121fb981093d7`.
The user approved the default-off source/checker/portable connection with fixed
candidates and standard-library stubs. No real checkpoint, dataset, native
optimizer, Torch or GPU execution is authorized. No resource limit changes.

## Implementation boundary

Construct a new binary source format directly from the existing V3 builder's
source-derived coefficient graph, without generating or replaying the old
forest JSON. Reuse the existing V3 construction mathematics. Store unique
`(center,node)` roots and separate ordered view memberships. Store explicit
binary64 node coefficients and scales losslessly; this first join does not
implement the proposed first-affine tensor-alias compression or restricted
live-checkpoint parser. It therefore cannot inherit the 96.8 MB format estimate.

The independent checker uses the existing source/affine/ReLU/guard equations,
but a new closed direct-format intake. It does not invoke the producer. Ordered
memberships remain a checked immutable sequence over a unique root bank, not a
reconstructed old JSON forest. All memberships still count against the original
65,536-row ceiling; unique storage does not authorize a new cap mapping.

The new checked type is explicitly admitted to the consumer's closed type
list. Retain same-Work binding, source reseal, complete pair/property inventory,
all sign/residual/intermediate checks and both proposed/zero candidate roles.
Candidate width is eight and pair-scoped check width two. Old defaults remain.

The producer-facing connection accepts only a bounded fixed-candidate callback
in these controls. No native proposer executes. A versioned portable manifest
binds source, forest, contract, ordered typed frames and checker code inventory.
Portable replay directly invokes Driver.receive, not old candidate JSON.
Package-local checking uses trusted code under `python -I -S`; self-hashing
packaged code is not an independent trust root.

## Budget and evidence contract

Use the existing POLICY, candidate byte limits, archive limits, exact bit limits,
positive threshold and at-most-300-second deadline without changes. One online
Work object covers source construction/checking, fixed candidates, encoding,
publication, exact acceptance, fresh portable recheck, terminal reception and
cleanup. Record phase wall time, cumulative Work, archive/failed-prefix bytes
and owned-buffer peaks. Wall time includes non-Work operations such as hashing,
copying and serialization. A failure never refunds or resets accounting.

The complete production and fresh independent replay use one absolute online
deadline. A copied-package subprocess control is a separate offline audit;
report its copying/startup/check/reception cost separately and never use it to
rescue an online failure. No complete RSS or native-runtime bound is claimed.
Existing process-group deadline controls remain applicable; this stage is not
real-model execution supervision admission.

## Frozen controls

Use only the existing synthetic `mixed`, `deep`, `half`, and `guarded` sources.
No source is selected on its proof result. Two fixed candidate scalar profiles
(exact binary64 and admitted general rationals) exercise the same mathematics.
Compare lower bounds and complete outcomes with old Session, dense Driver,
pair-scoped Driver and the independent expanded scalar oracle. New source
identities imply new query/slot bindings; compare semantic obligations, never
accept a foreign old frame as new evidence.

Additional required controls:

- mutated root, membership order, factor/scope, coefficient, scale, source and
  property, including recomputed outer hashes; truncation and excess payload;
- immutable checked source and root memberships, foreign Work/forest/frame,
  stale or reused ownership tokens;
- missing candidate UNKNOWN versus missing/duplicate/reordered manifest
  obligations rejected; second-block failure cannot produce full acceptance;
- source, publication, recheck, reception and cleanup deadlines; exceptions,
  partial writes and late success fail closed, with prefix costs retained;
- relocated checking without original source, repository, producer or numerical
  libraries; extra/changed/symlink package members rejected;
- cumulative Work shared across online phases and unchanged policy ceilings.

Run relevant existing consumer/wire/archive/navigation regressions. Preserve
any failing first control result and fixes in the stage record; no real trials.
Success means this finite interface contract passes, not capacity admission,
algorithm speedup, a new model certificate or completion of G1–G6.

## Stop and next gate

No isolated control result authorizes higher quotas or full-shape execution.
After controls, reconcile the actual new representation, source/consumer
coexistence and portable replay with the complete Work/bytes/lifetime ledger.
Live capture, native dispatch and memory, general full-size evidence, GPU and
real-model experiments remain separately gated.
