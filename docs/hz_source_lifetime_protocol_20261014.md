# Restricted source and worker-lifetime contract — design only

2026-10-14; based on `f395e899de4bde927ddda775e387b43b2f0681f6`.
This follows the [unique-root proposal](hz_unique_root_protocol_20261014.md)
and [runtime sub-ledger](hz_runtime_lifetime_20261014.md). The user authorized
static closure of a new protocol, **not a quota increase, production change or
real/GPU execution**. All mathematical checks and the 300-second budget remain.

The deliverable here is a restricted source-format and lifetime specification,
with [reproducible integer accounting](hz_source_lifetime_protocol_20261014.json).
It does not implement a deserializer or supervisor. **Complete cost remains
NOT CLOSED.** In particular a parameterized arena is not a Python RSS bound,
and a deadline is not a guarantee that successful computation finishes by it.

## 1. Source identity before allocation

The existing writer saves `format`, dataset, factory configuration, state dict,
metrics and training metadata, without optimizer state. The loader calls a
general pickle reader and builds a model; the factory may probe CUDA even with
CPU map location. The new path must not call that loader or instantiate Torch
modules. Historical checkpoint/state hashes exist; **no live checkpoint or
input has been selected or read by this analysis**.

The registered factory directory is fixed at 52 tensors and 6,961,368 values:

| Network | Affine dimensions | State names |
|---|---|---|
| Router | 3072→128→8 | `router.{1,3}.{weight,bias}` |
| Each of eight experts | 3072→256→128→10 | `experts.i.{1,3,5}.{weight,bias}` |

There are 63 expected module-metadata paths. The canonical binary64 parameter
payload is 55,690,944 bytes; including the 3,072-value center, 55,715,520 bytes.
This does **not** establish the original checkpoint dtype. Bind separately:

- frozen checkpoint full-file hash and length;
- exact factory configuration, names, shapes, storage dtype/offset/strides;
- canonical promoted state and center identities, with the historical hashing
  domains explicitly distinguished from the new direct-format domain;
- frozen program/factory source and declared eval, normalized selected-softmax
  top-2 semantics. No arbitrary runtime hooks or replacement class is covered.

Finite binary32 coefficients can be promoted exactly to binary64; binary64
coefficients retain their bits, including signed zero. Nonfinite coefficients
are rejected. Source-row validation, compensation, factor ownership, guard,
ReLU, endpoint and complete-property checks are **not removed** by this adapter.

## 2. Restricted, non-executing checkpoint parser

Proposed profile: single-disk STORED ZIP, protocol-2 data-only pickle and only
the expected state schema. The pinned Torch writer defaults to protocol 2;
actual historical ZIP/pickle compatibility has not been inspected. Unknown
formats return `UNSUPPORTED_SOURCE_FORMAT`, never a generic-unpickle fallback.

The ZIP index has at most 52 storage members plus six named metadata members:
`data.pkl`, `byteorder`, `version`, `.data/serialization_id`, `.format_version`,
`.storage_alignment`. ZIP64 fields/data descriptors, when present, require
explicit bounded offset interpretation and local/central consistency, not
integer truncation. Reject compression, encryption, duplicate names, path
escapes, invalid offsets, conflicting storage types and overlapping physical
regions. No extraction into arbitrary paths is performed.

The byte interpreter handles primitive scalars/containers, tuple, memo and
specific symbolic tags. `GLOBAL` yields an enum, not import/getattr; `REDUCE`
recognizes only the registered empty OrderedDict and tensor-descriptor rules;
`BINPERSID` records a storage reference. `BUILD` handles only the expected
state-dict metadata. No pickle, rebuild, constructor or device function executes.
Tensor rebuild fields also matter: hooks must be empty, requires-grad must
match the registered state schema, and optional tensor metadata is absent/empty
or has only false neg/conj flags. A true negative/conjugate view or unknown
metadata cannot be ignored while copying raw storage. Reject it. Saved device
location is checked as provenance, never used to restore a device object.

Memo keys are sparse records sorted by ID/producer position, not an allocation
indexed by a possibly huge memo ID. A GET must reference an earlier PUT;
duplicate PUTs are unsupported. Aliases push IDs, never expanded subtrees.
Container mutations remain in the event/edge graph until final validation;
early validation cannot hide a later write. Reject duplicate dictionary keys
and cycles using an iterative graph walk. Strings remain checked byte spans,
without unbounded interning or recursive copying.

For `P` metadata bytes the proposed typed arenas are:

| Arena | Bytes |
|---|---:|
| Nodes, opcodes | `128P + 32P` |
| Stack/marks, memo | `16P + 16P` |
| At most `4P` edges | `64P` |
| Colors, iterative DFS, sorting | `P + 24P + 32P` |
| Total, excluding raw metadata/runtime | **`313P`** |

Each opcode consumes at least one metadata byte; bulk append edges charge the
preceding pushes, and ID references do not recursively expand. The `4P` edge
invariant and exact record layout must be proved against the implemented
grammar before this becomes an implementation bound. All arena bytes as well
as IDs need representable uint64 offsets (`313P ≤ 2^64−1` in this layout);
check before allocation. This is a proposed format
restriction, not a changed resource allowance. If raw metadata is retained,
add `P`. Worst-case sorting, memo lookup, UTF-8 checking and name comparisons
are separately charged; this is not a linear-time or average-hash-table claim.

Do not assume contiguous or nonaliased parameters. Rank is at most two here.
Check nonnegative offset/strides and
`offset + sum((shape[j]-1)*stride[j]) < storage_numel` before access. Same storage
ID must denote the same dtype and bytes. Coordinate gathering preserves logical
parameter values for supported noncontiguous/overlapping read-only views. Two
independent construction/binding passes require at most `2N` scalar reads,
`4N` stride contributions and `16N` requested input bytes; worst case is `2N`
random read calls. Requested bytes do not bound filesystem block amplification.

## 3. Source snapshots and the previously omitted disk component

Hashing an original file and then reopening it by offset does not prevent
concurrent modification. Neither `stat`, `O_RDONLY` nor a hardlink proves
immutability. This design charges a bounded copy to a private request-owned
snapshot, validates the frozen digest, closes writable handles and subsequently
parses that same object. Protection against a writer to the private object is
an explicit ownership assumption/implementation obligation, not a hash theorem.

Let `F` be checkpoint bytes, `R` dataset bytes, `A_w` the prior final-store
envelope. Conservative logical live storage is **`A_w + F + R`**, not `A_w`.
The design does not rely on early deletion for this bound. Snapshot writes are
`F+R`; copy/hash and verification scans read `2(F+R)` and hash `2(F+R)`.
Parser and gather accesses, CRC contexts, promoted-tensor hash contexts, source
output, independent source comparison, fsync and cleanup are additional costs.
Two metadata decode passes account only for `2P`/`2K` sequential input; borrowed
string-span comparisons may reread bytes and must be separately counted, or use
an explicitly charged raw-metadata bank. Parameter hash payload counts exclude
center hash payload (49,152 bytes across two historical identity passes), name/
shape framing and the separate new direct-format hash contexts.

For a conservative malformed-inclusive ZIP schedule, each independent index
pass can request at most `(2S+5)F + 65557` bytes, `S≤58`; two checks are charged.
This bound requires bounded per-entry calls and early member-count rejection;
it deliberately does not assume that a malformed directory has disjoint local
sections. A smaller bound for well-formed files must not replace the rejecting
path. Both raw-file SHA and CRC scans are counted, even when sharing file reads.

No `F`, `P` or `R` is frozen for a live request in this design. Therefore even
the earlier ~96.8 MB **final store** envelope is not a complete attempt-disk
admission result. It cannot silently absorb checkpoint/data snapshots or a
second archive. Partial prefixes remain charged; only a later explicitly scoped
implementation may define recoverable cleanup of its own temporary snapshots.

## 4. Input reader preserves the numerical object

Proposed input profile: bound CIFAR-10 test-batch file/hash/length, original
index and label, `(10000,3072)` uint8 C-order data. A non-executing descriptor
parser validates the NumPy dtype/reconstruct/BUILD tags without calling them.
It checks the complete file, 10,000 labels and the 30,720,000-byte data span,
retaining only the selected 3,072-byte row. It must not load the training batches
or class-name pickle merely to answer this request. That is a new adapter,
not the historical loader's unchanged execution or timing.

For arena accounting `K = R - 30,720,000` removes the single **validated leaf
payload** once, not all references to it; opcode/length headers still count.
The data scan/hash remains in the budget. Use at most `313K` typed metadata
payload under the same proved grammar invariant. Checkpoint and input parser
arenas may be combined by `max(313P,313K)` only if the first is released;
otherwise they add. Raw metadata, buffers and runtime are additional.
Before that release, extract the independent bounded tensor descriptors and
remove all borrowed arena references. Merely deleting one name or making calls
sequential does not establish allocator/RSS release.

The old path sets default float64 before `ToTensor().div(255)`. Do not replace
it by float32 division followed by a cast, or by exact rational `p/255`.
A proposed integer-constructed RN64 table needs 256 entries (2,048 bytes),
at most 68-bit intermediate integers and a 24,576-byte center output. Source
inspection does **not** prove equality to every dispatched native division.
A later authorized finite 256-value differential or an implementation proof is
required; no such native control ran here. No old input identity is relabeled.
For the proposed table, 0 and 255 are exact. For `1≤p<255`, choose
`e=floor(log2(p/255))∈[-8,-1]` and round `p*2^(52-e)/255` to the nearest even
integer by quotient/remainder; the input numerator is at most 68 bits. This
explains the bound, not native implementation equivalence.

Likewise `Fraction(float(2/255))` and `Fraction("2/255")` are different radius
objects. Preserve original type/literal and exact normalized value. The current
request string gate is 4,096 characters, not the candidate's 4,096-bit gate:
string numerator/denominator magnitudes can require 13,607 bits. Four literal
records need at most 13,672 bytes plus provenance text; intermediate center±r
can need 13,669/13,668 bits, and clipping comparison products 27,276 bits.
The latter is only literal-clip versus center±radius; comparison of two
constructed endpoints may require 27,337 bits and is counted separately.
These are ingress bounds, not permission to weaken subsequent 8,192-bit checks.

## 5. Whole-worker lifetime and one budget

Proposed phases use a stdlib parent and at most one fresh-exec worker at a time:

`SOURCE → quiesce/receive → PRODUCTION → quiesce/receive → PORTABLE → quiesce/receive → cleanup/terminal`.

SOURCE creates/checks the direct source; PRODUCTION retains native candidate
and exact-consumer state; PORTABLE independently checks the full store with
`-I -S`. All 252 properties, up to 504 endpoints and both candidate roles remain.
Workers exchange the same immutable store, not three full copies. The parent
holds finite metadata/FDs, never complete source/Fraction/tensor objects.

One absolute 300-second deadline and cumulative resource ledger begin before
loading. No phase or tile resets them. Raw candidates and partial-prefix bytes
are durable records, not completion. Reception checks exact roster and binding,
not an output string saying COMPLETE. Late or partial evidence cannot establish
success, and offline recheck cannot repair an online timeout.

Conditional process RSS is
`parent + max(source, production_native + exact + overlap, portable, reception)`.
Current production interleaves `prepare/propose/accept`, so **native and exact
must add**, not take a maximum. Fresh exec removes inherited Python/Fraction
cache state, but its release claim requires every thread/allowed descendant to
exit and worker-owned mappings/FDs to be released before the next worker. The
parent's surviving finite handles remain in its own cost term. Host file
page cache and kernel objects are not included in a process-RSS formula.

The capacity entrypoint currently observes only its leader. Other existing
entries are stronger: `wire_archive.run_child` always attempts group kill and
bounds logs; `scoped_proof.supervisor.execute` samples group plus parent RSS.
Neither establishes complete containment/release against escaped native
descendants. Python audit hooks and `set_num_threads(1)` are not native
no-fork or whole-runtime thread-count proofs. Do not invent available cgroup or
seccomp authority. A limited native/no-descendant assumption or separately
authorized containment implementation must be explicit.

On any error/deadline: stop owned work, quiesce, retain charged prefixes and
publish a nonaccepting terminal. Do not transmit traceback objects to parent;
bounded error codes/spans avoid constructing arbitrary `str(exc)` first.
Cleanup and final publication are counted. SIGKILL/wait has no general finite
OS/I/O completion guarantee: record overrun and reject, never backdate success.

## 6. Decision and next static boundary

This stage specifies source provenance, restricted format, value/byte counts,
snapshot costs and lifetime obligations. It does not reintroduce old JSON
logical bytes or duplicate root values. It neither changes mathematical checks
nor claims a complete implementation cost bound.

Still open: concrete source-format compatibility and conversion equivalence;
exact-runtime scratch/allocator correspondence; native temporaries, thread and
dispatch envelope; whole-worker ownership/cleanup; full comparison/IO/runtime
cost aggregation. `full_runtime_upper`, new quota and execution admission stay
null/false. Next work is limited to those static terms; repeated source-format
design or tiny timing runs cannot substitute for closing them. If no defensible
bound is available for the selected Python/native path, report that stop and
the specific implementation assumption needed rather than silently raising a
stop limit. G1–G6 scientific status and sealed results are unchanged.

Validation: nine new stdlib static tests check roster, formulas, ingress domain,
identity drift and nonadmission. They do **not** test an implemented parser,
deadline supervisor or real compatibility. Reproduce with the existing Python:

```text
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys,unittest; sys.path.insert(0,"/data1/Kane/MOE/ACT"); unittest.main(module=None,argv=["unittest","scripts.test_source_lifetime_protocol"])'
```

The nine tests plus 52 prior static/navigation/manuscript regressions pass.
Three read-only design reviews checked checkpoint, input and worker contracts;
they are AI code reviews, not the independent human technical review required
by G6. Review corrections include tensor neg/conj metadata, all-arena byte
addressability, borrowed-span reread costs and the wider endpoint comparison.
