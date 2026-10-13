# Lossless evidence encoding and bounded intake proposal

There is a concrete **single-package storage design** for the registered
eight-expert V3 source, but not yet an executable capacity contract. A proposed
fixed-record envelope is 110,611,136 bytes, below the retained 128 MiB archive
limit. It relies on source-row references, exact binary64 value encoding, an
explicit compressed zero-candidate role, and one immutable copy of each tile.
None of those interfaces is implemented or approved by this analysis. Full
resident memory, live capture, and the extra decoding/checking cost remain open.

This follows the [full resource ledger](hz_full_resource_ledger_20261014.md),
not a new solve or an attempt to overturn its 140,203,072-byte JSON counterexample.
Starting state was clean `feat/moe-route-verification` at
`74c2b609caf14e4ba6c098b0ce95b3a616c731f5`. The
[static arithmetic report](hz_lossless_evidence_design_20261014.json) binds
22 code/metadata anchors through its parent analysis. No checkpoint, captured
source, numerical library, solver or device was loaded.

## Why exact binary encoding is possible in this source family

The source contract is sequential `Flatten/Linear/ReLU`, not an arbitrary HZ
or graph. In every output view its node IDs are distinct:

1. Input and Linear allocate a new global node per output, including zero rows.
2. Flatten and common views copy the ordered output list.
3. An active ReLU retains its distinct parent node. Every inactive or unstable
   unit allocates a new node, distinct from all parents and other new nodes.

This induction prevents a later Linear from combining two weights into the same
parent reference. Its references therefore equal individual source binary64
weights; ReLU and guard references are ±1. The independently checked node order,
operator reconstruction and complete inventories are essential premises. See
[source construction](../scoped_source/source_dag.py),
[affine references](../scoped_source/affine_references.py) and the
[independent checker](../scoped_source/check_source_dag.py).

The V3 checker already requires local coefficients, root centers/RHS and scales
to be exact finite binary64 values. Encoding their eight bytes preserves those
real numbers; it is not rounding to a weaker representation. Generic DAG sums
do not have this property: `1 + 2^-53` is an exact dyadic that is not binary64.
The proposed codec must reject unsupported graph contracts or use a separately
bounded rational format. It must never silently apply `float` to such a value.

| Data | Proposed exact representation |
|---|---|
| Captured weights, bias and input-center bytes | Original bytes, including source signed-zero identity |
| Checked local/root/scale and registered sequential references | Little-endian finite binary64, exact rational interpretation |
| Native proposed duals | Full dense binary64 vectors; no sparsity assumption |
| Explicit zero duals | Length-, role- and row-bound zero-vector encoding; all logical values still checked |
| Radius, margin, clip and property offset | Original exact source literals; not converted to binary64 |
| Independently computed bounds and intermediate results | Canonical signed numerator/denominator integers; not binary64 |

The incoming candidate's existing 4096-bit limit and the 8192-bit intermediate
limit remain different. Stored diagnostic results may need the latter; they
never authorize acceptance without recomputation.

## Source references preserve the logical forest

The first affine bank repeats 6,684,672 source weights already present in the
portable package. A source-row descriptor can bind the source manifest, tensor,
graph/layer, row, width and ordered input factor map, then store the at-most-two
compensation terms separately. The decoder must recover every exact nonzero
weight and still check zeros, centers, compensation, lifts, scale and ownership.
It cannot substitute the fact that a pointer has a hash for checking a row.

[TensorRows](../scoped_source/tensor_rows.py) supplies the metadata and bounded
row traversal, but not a checked reusable raw-row handle. Its reader is
single-pass; a receipt requires complete consumption and the final source
identity check. A new reference decoder plus the old checker would therefore
perform an extra source pass. That pass, conversion and hashing must be counted.

Keep separate identities for wire bytes and logical V3 fields. The latter
currently includes source-receipt statistics; changing intake statistics or
policy cannot silently preserve its meaning. A codec must bind both identities
and prove exact field/order correspondence under the same stated policy.
Physical references do not reduce logical root occurrences, factor identities,
candidate counts, or the Work cost of mathematical checks. Full-size logical
documents still exceed existing structural limits; a new wire format is not a
way to bypass those gates.

## Proposed single canonical archive envelope

The following widths and reserves are **new design choices**, not measured
sizes, existing format capabilities or approved runtime limits. Let
F=32048, N=19496, L=6732720, W1=6684672, R=277280, H=136232, V=82,
E=504 endpoints, M=2700 dual rows per pair. The complete 252 duties remain.

| Component | Proposed envelope | Bytes |
|---|---|---:|
| Captured source and original manifest | Raw payload + 4 MiB | 59,909,824 |
| Factor identities | 256F | 8,204,288 |
| Node records | 64N | 1,247,744 |
| First-affine source-row descriptors | 64 × 2176 | 139,264 |
| Remaining local entries | 16(L−W1) | 768,768 |
| Reference entries | 16R | 4,436,480 |
| All logical root occurrences | 16H | 2,179,712 |
| Scales and view metadata | 8F + 512V | 298,368 |
| Forest header, roles and source receipt | Reserved 1 MiB | 1,048,576 |
| **Forest subtotal** | | **18,323,200** |
| Native raw proof with both roles | E(8M+512) + 256 KiB | 11,406,592 |
| Exact receipts and aggregation | Reserved 8 MiB | 8,388,608 |
| Checker code and execution metadata | Reserved 8 MiB | 8,388,608 |
| Bounded logs, failure, progress and terminal | Reserved 4 MiB | 4,194,304 |
| **One complete package** | | **110,611,136** |
| Remaining within 128 MiB | | **23,606,592** |

The factor width assumes the registered generated ASCII names, at most 200
bytes; no truncation or replacement of arbitrary Unicode is allowed. Sixteen
bytes cover an ID, a binary64 value and fixed tags/padding. Node and source-row
records must have fixed checked offsets/counts. The 512-byte endpoint header
can hold six 32-byte identities, two row bindings and two candidate hashes,
leaving 192 bytes for version, role/status, ordinals, pair/property, counts,
offsets, zero descriptors and the native producer's two `lower=0` markers.
The final field layout still requires its own independent controls.

The 256 KiB proof manifest reserve must include the width-one worst case of
504 tile descriptors, not only 84 width-eight tiles. Partial publication and
final evidence must reference the **same immutable tile**. Retaining a second
full proof would require a different ledger. All 1,008 candidate roles and
2,721,600 logical dual values are independently processed. A missing zero role
or wrong length/binding is a rejection, not an instruction to synthesize one.
Zero encoding is not permission to discard checks or decrement their Work.

Three exact endpoint bounds plus each duty bound and each gate record can be
stored within 4,568,768 bytes using 2064-byte rational records, fixed metadata
and a 256 KiB manifest. Thus the proposed 8 MiB receipt reservation has a
concrete bound. Incoming arbitrary-rational candidates are not covered by the
native F64 proof envelope; no candidate is rounded merely to fit.

The log reservation is for one bounded attempt: at most 504 tile events and
fixed phase/terminal events, with prechecked per-record and combined stdout/
stderr limits. It does not cover unlimited retries. Exceeding any reservation
must produce a nonaccepting resource terminal, not omit evidence. Code and
metadata inventories still need to be frozen and checked against their reserves.

## Relocation and storage layout are separate decisions

Two copies of the proposed complete envelope total 221,222,272 bytes. This
design therefore does **not** supply a two-copy 128 MiB capacity contract; an
upper bound above the cap does not prove that every actual request overflows.
The archived dense runner copies online data and code to a second location and
retains both. That behavior cannot be silently replaced by a one-copy total.

A later protocol must specify persistent archive, relocation temporary peak,
and retention separately. Hardlinks do not demonstrate an independent portable
copy. Moving a new owned package or reclaiming a verified reproduction copy
would require an explicit lifecycle protocol; neither is authorized here.
Original source, failures, frozen records and checkpoints are not removed.
Offline checking remains separately charged and cannot finish an online proof.

Sharing partial and final tile storage is also a new archive-layout factor,
not merely a byte codec. Publication-before-check, immutable bytes, acceptance
state, missing-duty accounting and interrupted prefixes must all survive.

## Compact storage alone does not solve memory

The current checker fully decodes JSON and creates Fraction tuples;
`CheckedScaledForest` retains those tuples. `Session` retains all raw tiles and
then decodes and serializes a whole proof again. A binary-to-JSON adapter would
retain these peaks and add decoding cost. It is a reference for exactness, not
the proposed production memory solution.

A separately controlled bounded-intake interface would need:

- Prevalidated descriptor lengths, integer arithmetic on sizes, byte caps,
  source identities and deadlines **before** allocation or following a reference.
- Checker-owned immutable row sources, bounded decoded windows and independently
  repeated source validation; no mutable producer cache trusted as evidence.
- Exact row/factor/owner semantics and the same endpoint arithmetic, with wire
  identity and logical identity explicit. A generated logical hash is not a
  replacement for source reconstruction or independent bound checks.
- Incremental candidate publication and final coverage aggregation over stored
  tiles, without a full proof DOM; all decode, read, check and cleanup costs charged.

This is a consumption-interface change separate from lossless transport.
Changing native kernels, skipping coalesce, batching support, or changing HZ
enclosure is not included. The old Work upper cannot be copied to new code
without charging its additional decode/source passes and recomputing the ledger.

## Native memory uncertainty is now more specific

The installed `torch/version.py` reports 2.9.1+cu128 and git
`5811a8d7da873dd699ff6687092c225caffcf1bb`; the file was read without importing
Torch. Conditional on that upstream CPU dispatch, the named COO arrays can be
counted. A local float64 COO uses 24L bytes. Its transpose copies indices and
values, so retaining both uses 48L = 323,170,560 bytes; it is not a free view.
See the pinned [transpose implementation](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/TensorShape.cpp).

Coalesce allocates input/output COO, flattened indices, sorted keys and a
permutation: 72L bytes. Index initialization adds an 8L temporary; the radix
branch instead adds two 8L vectors. The named-array component is bounded by
88L = 592,479,360 bytes. This excludes sorting/runtime allocator overhead and
all Python state. Sources:
[COO coalesce](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/sparse/SparseTensor.cpp),
[index initialization](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/Sorting.cpp),
[sorting branches](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/SortingKernel.cpp).

These are derived component bounds, not observed peak RSS or binary execution
proof. Source-version metadata does not prove the installed binary's exact
dispatch or build flags. The saved report records the four inspected upstream
source hashes and the installed version-metadata hash; offline arithmetic
recomputation does not refetch or execute those sources. Python objects,
Fraction temporaries, sort/BLAS runtime,
allocator high-water behavior, and previous-loop `points`/`residual` lifetimes
still require accounting. The 8 GiB RSS/16 GiB address-space stop rules are not
completion guarantees. No memory optimization or native call was executed here.

## Remaining capture and archive lifetime costs

A follow-up code inspection at `e3c27d4f9e43dd6506d6231e1630abf2faab9205`
separates the existing captured-source consumer from the earlier live-object
path. It does not connect or execute those paths. In particular, the current
dense runner copies an already captured source; its Work bound does not cover
the following capture costs.

The [live capture function](../scoped_source/capture.py) first stores all
parameter tensors as base64 strings. Its inventory calls
[`tensor(t)[0]`](../router_source/checker.py), which still decodes the entire
tensor, builds a list of Python floats and converts every entry to `Fraction`,
even though capture uses only the returned identity. This is not row streaming.

The existing [capture-bound wrapper](../scoped_source/sparse_intake.py) then
calls [graph validation](../scoped_source/graph.py). For every Linear layer,
validation parses weight and bias once for inventory correspondence, then
`operator()` parses both again and constructs the whole layer's nonzero-weight
dictionaries. The caller's `dims,_,_=operator(...)` does not prevent those
objects from being constructed. These checks are not removed by making the
eventual evidence file compact.

For a successful invocation of **that existing capture-plus-validation path**
on the registered shape, let P=6,961,368 parameter values and n=3,072 input
values. Calls to `Fraction.from_float` inside `tensor()` are:

| Operation | Number of value conversions |
|---|---:|
| Capture parameter and center identities | P+n |
| Validate the center and parameter inventory | P+n |
| Validate Linear dimensions through `operator()` | P |
| **Total** | **3P+2n = 20,890,248** |

This is a code-derived operation count, **not** the custom Work counter, an
elapsed-time prediction, or a simultaneous object count. It excludes model
loading, JSON hashing, base64 decoding, packing, propagation and endpoint work.
One largest weight tensor already contains 786,432 values, compared with the
later `TensorRows` reader's 3,072-value input row. The full capture path needs
its own lifetime account or a separately approved bounded implementation; a
streaming reader downstream cannot establish a streaming capture claim.

The old archive's scope is also broader than one portable bundle:

| Lifetime | Existing files and accounting scope |
|---|---|
| Before execution | Original captured `SOURCE_ROOT`, outside the run archive; retained |
| Online request | Copied source, forest and full proof under `query/bundle`; separate raw candidate tiles under `query/partial`; receipts, progress, code and logs |
| Relocation | A second independent file copy of code and the complete bundle under `relocated`; both copies retained |
| Final accounting | All files recursively under the run root, including relocation and terminal, count toward 128 MiB |

See [`reserve`, `worker` and `execute`](../scripts/run_scaled_dense_capacity.py)
and the [archive auditor](../scripts/audit_scaled_dense_capacity.py). The source
can therefore have its original plus two run-local copies in project storage.
The external original is not charged to this particular run's archive bytes;
the two run-local copies are. There is no existing deletion, move or hardlink
transition that makes the proposed one-copy envelope an unchanged-run total.
Candidate bodies may appear three times inside the archive: as partial tiles,
inside the online proof, and inside the relocated proof. The byte accounting
uses logical file lengths, not filesystem metadata or allocated blocks. The
supervisor control tests also use temporary directories outside the archive;
normal context-manager cleanup is not a hard-kill orphan-retention contract.

Furthermore, `reserve()` checks before selected large writes and copies, with
a 2 MiB allowance for small files. It is **not a continuous filesystem byte
limit**: stdout/stderr, progress, receipts and other metadata use ordinary file
writes. The supervisor polls time and RSS, not disk growth. Final
`terminal_bytes()` changes an oversized archive's status to
`ARCHIVE_LIMIT_REFUSED`; the auditor rejects oversize success but permits the
preserved refusal archive. This prevents an oversized accepting result, not
all temporary or failed-run byte excess. The proposed bounded log and terminal
writers are thus implementation requirements, not controls already supplied by
the old runner. No historical result is relabelled by this observation.

The outer timer is recorded before terminal serialization and writing. A new
end-to-end protocol must also charge and deadline-check that final work; the
existing pre-publication timestamp is not a complete terminal-publication cost.

The next implementation decision must therefore cover checked row intake,
candidate publication, bounded log/terminal writers and a stated relocation
lifecycle. Live-model capture is a separate unclosed adapter: it is neither
silently included in the codec nor free preprocessing. Complete resident memory
and all-in online cost remain unproved; no new source or execution is admitted.

## Bounded next scope and stop rule

There is now a testable design hypothesis: preserve the exact V3 logical
representation while removing repeated source-weight bytes and full-proof
serialization copies. Request separate approval for a **versioned lossless
codec plus independently checked bounded intake**, with archive layout an
explicit separate factor. Do not simultaneously change HZ relaxation, native
candidate algorithm, batching width, GPU, resource quotas or real inputs.

Before implementation, freeze exact field layouts, source/intake identities,
logical-byte versus physical-byte rules, relocation/retention, new counted work
and peak objects. Controls must cover exact logical/identity and same-candidate
bound differentials; wrong source/expert/row/map; alias pollution and mid-read
mutation; malformed offsets, nonfinite/endian/signed-zero cases; generic rational
refusal without rounding; missing or polluted zero roles; all ReLU cases and
nonzero compensation; complete tie/property coverage; deadlines/exceptions and
partial evidence; and isolated portable rechecking with no checkpoint/solver.

If a complete memory/storage contract remains missing, stop before execution.
Do not use another source prefix, a small benchmark, a new Work number, or a
silent retention change to claim that the full registered request fits. There
is no new quota, time prediction, full-source run, real experiment or GPU retry.
G1–G6 remain OPEN; encoding is enabling infrastructure, not a new MoE algorithm
or a substitute for real proof and external comparison.

## Static checks

Nine new static arithmetic/identity checks and twenty parent checks pass with
the existing `act-py312`, `-B -I -S` and an explicit repository import path.
Symbolic node-ID cases and a few binary64 scalar round trips are not model
conversion or codec tests. The saved report is exactly regenerated by
[`analyze_lossless_evidence_design.py`](../scripts/analyze_lossless_evidence_design.py);
test coverage is in [the static tests](../scripts/test_lossless_evidence_design.py).
No producer, parser, supervisor, cap, dependency or frozen result was changed.

Final selected regression: **60 tests pass**, including the earlier capacity,
manuscript, review-handoff and layout controls. Three read-only agent reviews
pass within their separate mathematics/identity, encoding/storage and native
component scopes; these do not replace human review. Workspace checking retains
619 frozen source identities and 4,632 history lines, verifies 435 navigation
links and eight local report links, and records no experiment or deletion.

The capture/archive follow-up adds no implementation or experiment. Its
conversion formula was recomputed from the bound topology, and two read-only
agent reviews checked the capture and archive code paths separately. Forty-eight
existing static, manuscript, handoff and layout regressions pass after the
documentation update. Those tests do not measure capture cost or establish a
complete memory bound. No new result directory, source, candidate or file
deletion was created by the follow-up.
