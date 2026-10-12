# Request owned HybridZ store and direct view consumption

Freeze baseline: `012f31d73304e43f78a91904e8402a6879ac76ef`.
This implements the [combined capacity decision](hz_combined_capacity_20261012.md),
without changing old schemas, production defaults, mathematical relaxation,
candidate iterations or acceptance gates. It is not real-model or GPU admission.

## Implementation sequence and first acceptance boundary

Implement one compatible representation in this order:

1. A given-HZ immutable request store, inherited row/factor views, and direct
   pair residual consumption. This first component is the execution authorized
   below. A validated store is not checked network provenance.
2. Row-consuming exact input/affine/ReLU/guard and binary64 enclosure checks,
   including actual SparseHZono coefficient matching; no expansion through the
   old global 128/256 parsers. Inherited rows may be shared only through checked
   unchanged-row edges. Allocate compensation factors at whole-state scope,
   outputs before equalities, not independently at each row tile.
3. Complete endpoint inventories and at-most-eight-column tiles using the same
   block candidate algorithm and CPU exact acceptance. Preserve all 252 duties
   and 252–504 endpoints. Query tiling is not factor/domain splitting.
4. Bind source tensors, full supervision, relocation and finite resources for
   this combined schema, then decide whether real development may be frozen.
   The GPU block kernel remains a separate required factor.

Do not declare the whole sequence complete when its first component passes.
Do not substitute the direct-node triangle LP from old H2 for HybridZ.

## Given HZ storage and scope contract

The externally anchored manifest binds a request identity, fixed resource policy,
one continuous/binary factor registry with original names and owners, all state
views and every row member. Each member is bounded canonical JSON with immutable
rows. Each row contains a constant/RHS and sorted nonzero coefficients in the
registry coordinates; factor kind remains explicit. A parent supplies inherited
factors and constraint rows by reference. A child supplies new factors, output
rows and appended constraints. No view duplicates its parent's row values.

Every view has a common or expert-private scope and an optional pair guard.
An expert template may inherit unconditional common rows, never a different
expert's factors or a pair-only fact. A guard entry preserves common factors
and only strengthens its domain. Shared columns have one identity across all
views. All stored rows and coefficients are checked even when a received dual
is zero. Missing/extra files, unused blocks, duplicate keys, noncanonical rows,
factor aliases, forward/cyclic parents, wrong scope or changed bytes refuse.

The checker reads and anchors bytes independently of the untrusted writer.
Caller mutation cannot alter the immutable captured view. This is protection
against ordinary alias pollution, not malicious mutation of the Python runtime.
The declaration-to-store connection remains a required next checker, not an
assumption that a self-consistent hash proves source lowering.

For a pair, consume common constraints once, its guard suffix, and both expert
private suffixes. Shared residual terms must be summed before taking one box
support. Independently check inequality multipliers are nonpositive, all dual
dimensions and the objective. Binary factors are explicitly relaxed to [-1,1].
The exact result is a given-HZ lower bound, not full MoE SAFE or native FP safety.

## Finite control resource policy

The policy is fixed by the caller and checked against the manifest, not widened
by the package. Limits are refusal limits, not a prediction of memory or speed.

| Resource | Limit for this given-store component |
|---|---:|
| Registry factors per request | 32,768 |
| Active factors per view / pair | 16,384 |
| Views / inheritance depth | 128 / 64 |
| Rows per view / unique stored rows | 16,384 / 65,536 |
| Unique coefficient entries | 1,048,576 |
| Row members / rows per member | 4,096 / 64 |
| Manifest / each row member / entire store bytes | 8 MiB / 1 MiB / 128 MiB |
| Stored / intermediate rational bits | 4,096 / 8,192 |
| Exact row/coefficient visits per control | 8,000,000 |
| Budget per control including construction, archive and checking | 300 seconds |
| Whole suite / attempts | 1,800 seconds / at most 3 repairs |

One CPU thread, no solver, CUDA, model/data loading or dependencies installed.
The first component has cooperative deadlines and byte/work refusal, not a
hard-process memory claim. Process hard supervision remains step 4. Store data
and full-scope failures are retained under new `baseline_runs/hz_request_store_*`
directories; limit the entire stage to 64 MiB of control archives. If those
limits are inadequate, stop and report; do not enlarge them after observing a
failure. Offline rechecking costs are reported separately from generation.

## Frozen first component controls

1. Fixed common/template/entry construction: three common continuous factors,
   one common binary, three experts with one continuous and one binary private
   factor each; all three pairs. Output/constraint coefficients are small exact
   dyadic rationals. Direct-view residuals/bounds must match a separately
   expanded coefficient oracle, including shared-term cancellation and zero
   dual rows. Expansion is allowed only in this small differential oracle.
2. Fixed capacity layout: 3,856 continuous +128 binary common factors; each of
   eight experts adds 2,324 continuous +384 binary factors. Every factor occurs
   in a nonzero output coefficient. Store the common 128 equalities/256
   inequalities once, 384 equalities/768 inequalities per expert, and twelve
   inequalities per pair. All 28 pair views actually use 8,504 continuous +896
   binary factors and 2,700 constraint rows. The whole registry has 25,648
   factors and unique constraints total 9,936. These are deterministic given-HZ
   layout coefficients, not a claimed conversion of a trained or synthetic NN.
3. On every large pair, directly traverse all required rows and evaluate two
   fixed complete multiplier records (all zero, and all inequality multipliers
   -1/32 with equality multipliers 1/64) against a fixed exact objective. Check
   independently derived analytic residuals/bounds. This is 56 given-HZ checks,
   not the 504 endpoint or 252 property experiment, which remains pending.
4. Mutation controls: missing/extra member, changed member bytes, wrong external
   anchor, cross-request manifest, wrong resource policy, factor name/kind alias,
   wrong/forward parent, common-to-private contamination, guard-to-common
   promotion, missing/duplicated row coverage, invalid/noncanonical coefficient,
   inwardly altered supplied result, missing/wrong-sign dual, and caller alias
   pollution. Refusal reasons and concrete mutated identities must be retained.
5. Resource/partial controls: registry/active/row/nnz/member/byte/arithmetic/work
   excess and expired deadline; no partial result is a completed lower bound.
   Completed prefixes may be saved for diagnosis but not promoted to acceptance.
6. Relocate the stored object and independently check it under `python -B -I -S`
   with only the bound standard-library checker code. No repository, checkpoint,
   training data or optimizer is read. Check original/relocated coefficient
   identities and exact results agree; don't confuse this with portable full
   network/source proof.

Use the same fixed objects at the default 64 rows/member and at 1 row/member
only for the small differential. Do not run a chunk-size timing search. Normal
objects, negative constructions, all attempts, actual storage/object counts and
all executed costs are audited. Success requires genuine reference sharing and
direct row consumption, not merely successful serialization.

### Coefficient recipe fixed before first execution

For both layouts, assign contiguous registry blocks: common continuous then
common binary, followed by each expert's continuous then binary factors. Common
outputs number 3 (small) or 3,072 (large); distribute common factors by their
index modulo the output count, with coefficient 1 and constant 0. Each expert
has 2 (small) or 10 (large) outputs. Distribute its ordered active factors by
position modulo that count, coefficient 1, with constant `(expert+1)/16` on
each output. Every factor is consequently consumed, not just listed.

Common equality/inequality counts are 1/2 (small), 128/256 (large); expert
private counts 1/2 or 384/768. Row `j` uses its block's factor `j modulo block
length` with coefficient 1; equality RHS is 0 and inequality RHS is 1. Pair
guards have 2 or 12 inequalities, RHS 0, with alternating +1/-1 coefficients
on common factor `j modulo common length`. These are supplied abstract rows,
not a network/router guard derivation.

Every query takes all left output weights +1, all right weights -1 and offset
1/8. Its exact objective constant is
`output_count*(left_expert-right_expert)/16 + 1/8`; common output residuals
cancel. The two complete dual records are zero and the fixed values above.
An independent control oracle updates residuals using these integer index
formulae, never by invoking the store's row iterator or producer. It checks
every residual coordinate as well as the final bound. The small expanded oracle
also compares flattened supplied coefficients. Positive end-to-end certificates
are not the aim of these capacity objects.

## Continue or stop

If these given-HZ controls pass, stop their timing/width study and implement
the source and endpoint consumers in steps 2–3. A format-only result cannot
close G1/G3 or admit a real request. If a capacity or exact-check defect remains,
repair only within the frozen envelope and retain failures. Do not drop rows,
properties, checks or enlarge budgets. All six long-term goal gates stay OPEN.
