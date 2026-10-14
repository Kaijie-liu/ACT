# Pair scoped workspace and checking microblock design

**A scoped, in-place consumer with two-endpoint checking blocks has a proposed
packed-payload upper of 117,009,527 bytes, below the existing 128 MiB arena.**
This removes the previously identified dense-vector obstruction in the design,
not in running code. Pair scoping alone does not provide that capacity guarantee.
Full-source admission, cumulative candidate size, complete Work and process
memory remain unresolved; no real-model or GPU execution is admitted.

The design is based on clean, synchronized `feat/moe-route-verification` at
`8e4d9165732ee10bdee82facef9ab78c04687ed7`. Its
[static calculation](../scripts/assess_scoped_workspace_design.py) and
[report](hz_scoped_workspace_design_20261014.json) bind source text and the
registered topology. No source construction, model-sized allocation or solver
is involved. The [implemented dense-path refusal](hz_owned_capacity_20261014.md)
and historical reports retain their original identities and results.

## Scope follows checked ancestry rather than labels alone

For pair `{a,b}`, retain the input, the **entire router**, both expert chains and
that pair's guard view. Common/Flatten aliases keep their original node IDs.
Private factors from other experts are omitted from workspace storage only;
their original global identity and logical zero coordinates remain defined.

This rule is justified for a complete `CheckedScaledForest` or
`CheckedWireForest`, bound to the same source, request and cumulative Work:

- The source checker reconstructs the first affine layer from input factors
  and fresh compensation factors. Later affine references point only to checked
  parent outputs, with fresh local compensation.
- ReLU rows use fresh factors and the explicitly checked parent reference.
  Aliases preserve parent outputs rather than inventing coefficients.
- Each expert starts at common, whose ancestry includes the router; guard rows
  refer only to router outputs. Complete roles, allocation order and inventories
  are checked against the source-derived construction.

Consequently the selected node set is closed under predecessor references and
its local factor set contains every possible nonzero coefficient. Induction in
descending global node order shows that pullback from the selected objective
or constraint roots is zero outside this set. Shared input/router coordinates
are **not duplicated**; expert-private coordinates are **not identified**.

This reuses an already established source invariant. It does not require
rechecking every source coefficient once per pair. Deriving bounded maps from
checked ownership, view ancestry and the complete ordered row roster is still
charged. A nonzero write outside the map must fail immediately. Unchecked
owner labels, incomplete sources, foreign receipts, arbitrary graph families
or future direct-source formats do not inherit this contract. Existing wire
read/identity/reseal checks remain mandatory.

Do not select all `scope == -1` nodes: other pairs' guards also use that scope.
Select the checked ancestry and this pair's guard view. Ordered equality and
inequality memberships retain duplicates and their `(view, kind, position)`
identity. For wire metadata, use the checked node descriptor rather than
indexing a full node and accidentally decoding all its coefficients.

## A tighter structural factor bound

The old ledger allowed seven new factors per ReLU. The checker establishes a
tighter bound: an unstable ReLU has three main factors and at most one lift for
each of its output and equality rows. Its two inequality rows have zero centers
and only `+1/-1` coefficients, so binary64 rounding is exact and their lift
error is zero. Stable aliases and constant-zero outputs add no such factors.
Thus five factors per ReLU suffice for this specific checked construction.

For the registered router `3072–128–8` and two experts `3072–256–128–10`:

| Scoped quantity | Upper |
| --- | ---: |
| Affine outputs A | 924 |
| ReLU units U | 896 |
| Guard rows | 12 |
| Nodes `3072 + A + 4U + 12` | 7,592 |
| Factors `3072 + 2A + 5U` | 9,400 |

The corresponding full-source factor upper is 25,648, versus the earlier loose
32,048. Old reports are not rewritten; global index storage below deliberately
retains their larger counts. These are conditional structural uppers, not
observed sizes or a claim of simultaneous attainability. Eight objective factor
vectors alone have a 155,288,000-byte upper with the current 2,065-byte slots.
Therefore merely restricting dense vectors to a pair does not establish a
width-eight arena guarantee.

## Separate candidate width from checking width

The candidate tile remains eight endpoints, with unchanged query identities,
candidate values and immutable direct-frame format. The proposed consumer uses
explicit checking microblocks of two endpoints. This is a **new execution
schedule**, not an invisible optimization of the old width-eight trace.

1. Prepare immutable sparse objective descriptors with at most four root
   entries and one offset per endpoint. Compute entries in the same expert/class
   order, including intermediate exact checks and duplicate-root updates.
   Present them to native code in global node order. Native `Operator.propose`
   reads these roots and offsets, not the exact objective factor vectors.
2. Generate, encode and publish the whole candidate tile once. Both proposed
   and explicit-zero roles remain present. Do not rerun the native proposal or
   reset bytes, Work or the deadline for each checking microblock.
3. For a microblock, obtain exact objective factor vectors using one private
   node accumulator at a time. Transfer ownership for in-place pullback rather
   than copying the accumulator. Retain the resulting objective vectors.
4. Check every dual row and role, construct private dual node accumulators and
   consume them in place. Keep the dual/residual factor vectors distinct from
   objective vectors until the original final factorwise addition.
5. Check proposed and zero bounds, release microblock vector storage and proceed.
   A later fault or cutoff poisons the invocation. Earlier receipts remain only
   partial evidence; no positive terminal can precede all endpoints and duties.

Every column retains descending global node order, original local/reference
order and `exact(multiply)` followed by `exact(add)`. Full global factor scans,
implicit out-of-scope zeros, scale-group first-encounter order, dual signs,
explicit zeros, residual corrections and the strict `>1/10000000` gate remain.
Do **not** combine objective and dual seeds before pullback: early cancellation
could bypass an intermediate 8,192-bit failure that the old path rejects.

Moving objective pullback after candidate publication delays some failures that
previously occurred before native execution. All spent candidate work remains
charged. Same-candidate per-column arithmetic can be preserved; the global
interleaving, first observed fault and budget outcome need not be identical.

## Proposed packed payload and added scans

Use the existing general working scalar capacity, not a silent dyadic-only
restriction. For `q` live checking columns, node/factor banks require at most
`2065*q*(7592 + 2*9400)` bytes. The proposed packed envelope additionally reserves:

- the existing global root index and all ordered memberships: 856,864 bytes;
- global/local ID maps, view flags and complete row triples: 306,626 bytes;
- eight sparse objective descriptors: 82,752 bytes;
- the unchanged native eight-endpoint frame: 362,944 bytes;
- one scale-group bank, at most 3,072 input scales plus unit scale: 6,358,037 bytes;
- `q` dual RHS values, two result bounds per tile endpoint and three reusable
  scalar totals: `(q+19)*2064` bytes.

The group bank is conservatively included at vector peak even though a stricter
lifetime schedule can release the node accumulators before norm evaluation.
Sealing overlaps two candidate frames, but occurs before the large vector
banks exist and is smaller than the reported envelope. This accounting assumes
the specified packed objects; current Python dictionaries do not inherit it.

| Checking columns q, candidate width still 8 | Packed payload upper, bytes | Fits arena component |
| --- | ---: | --- |
| 1 | 62,507,983 | Yes |
| 2 | 117,009,527 | Yes |
| 3 | 171,511,071 | No |
| 4 | 226,012,615 | No |
| 8 | 444,018,791 | No |

Two is the largest width fitting this declared layout, chosen from static byte
arithmetic rather than model outcomes. It leaves 17,208,201 bytes in the arena,
not that much proven process-RSS headroom. Arithmetic temporaries, Python
metadata, source/native coexistence, external aliases and archive lifetime are
outside this component envelope.

At the maximum 504 endpoints, the unchanged 84 candidate tiles become 252
checking microblocks. Retaining full-node and dual-row traversals adds uppers of
3,275,328 node-row visits and 453,600 constraint-row visits, compared with one
dual pullback per candidate tile. The 504 single-objective pullbacks remain.
Scope construction also has up to 1,521,128 metadata/row entries to inspect
over 28 pairs. These are scan subledgers, **not complete Work or time bounds**;
byte accesses, map validation and primitive accounting must be bound to an
implementation. Memory savings do not establish acceleration.

## Decision and bounded next step

Eight new static tests check the source bindings, tighter shape arithmetic,
payload formulas, complete microblock partition and extra scans; thirteen
handoff/navigation checks also pass. Read-only AI reviews checked ancestry and
arithmetic-order reasoning; these are not runtime differential tests or human
technical review. Recompute the design report with:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_scoped_workspace_design.py --check-report docs/hz_scoped_workspace_design_20261014.json
```

Proceed only to a separately scoped optional interface implementation/control
decision. The proposed controls must cover foreign/corrupt scope maps, shared
and private factor separation, out-of-scope nonzero writes, duplicate/cancelling
roots, intermediate-bit rejection, both candidate roles, irregular final
microblocks, second-block failure, deadline before/within/after checking,
immutable publication, full-duty aggregation and byte/Work non-refund. Compare
both bounds with the old consumer for identical candidates, not merely the
selected maximum. Bind candidate and checking widths separately. This stage
does not execute those controls or implement that new consumer.

The old full-source membership and 8M Work refusals still precede this path.
Native candidate cumulative bytes and the general-profile frame envelope also
remain unresolved. This design does not change their format or caps. Full cost
is still **NOT CLOSED**, G1–G6 remain open, and neither quota adjustment nor
real-model/GPU execution is authorized. Scoped storage reuses established
HybridZ shared/private structure; by itself it is not a new algorithmic result.
