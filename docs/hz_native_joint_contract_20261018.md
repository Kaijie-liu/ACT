# Native joint ownership and binding decision — static, 2026-10-18

Base: 8caba399a5494108c5bc606c7966244a936a3cee, clean
feat/moe-route-verification at stage start. This follows the
[generation contract](hz_native_generation_contract_20261018.md), not another
native experiment. The [record](hz_native_joint_contract_20261018.json) and
[stdlib analyzer](../scripts/assess_native_joint_contract.py) bind the inspected
implementation/environment and recompute small schema/count facts only.

## Decision first

The remaining boundary is now specific: **a joint, versioned self-owned resource
interface must address first metadata construction and partial observations,
not just native tensor storage or successful output files.** Do not launch
another small control or full-shape retry under the old caps. Do not extend this
analysis into an unbounded investigation of every CPython/native scratch buffer.

This record closes several previously omitted *components and lifetime joins*,
not the complete heap. It supplies a minimal proposed interface below; that
interface is **not implemented or authorized by this record**. Full self-owned
heap and failure-archive totals remain null, not zero. New numerical policy,
implementation, controls and real/GPU execution require their own decisions.

The approved 8-GiB finite native controls are already complete. Their approval
does not admit the registered eight-expert shape. Original Work, membership,
logical-nnz and candidate-stream restrictions, all checks and 300 seconds remain.
No G1–G6 claim, historical source limitation or experiment result changes.

## 1. Returned integer capacity is provenance-dependent

Bind CPython 3.12.12, 30-bit digits and the inspected closed source path. Let p
bound **all Fraction numerators/denominators in the listed project fragment**,
including its ungated intermediate values and decimal-token provenance; it is
not only the final bit length or a bound on library workspace. A reviewed
conservative returned-object envelope is:

~~~text
D(b) = ceil(b/30)
C(p) = 2*D(p+2)+4                  requested digit capacity
J(c) = max(32, 24+4*max(1,c))       one returned PyLong, bytes
F(p) = 48+2*J(C(p))                Fraction plus two returned integers
~~~

This distinguishes allocated capacity from normalized logical size. New
arithmetic allocates from normalized digit counts, so old excess capacity is not
multiplied again at every subsequent operation. The 32-byte minimum includes
the medium-integer allocation path. Division/reconstruction is bounded only for
the inspected input spans. These are fixed-implementation audit envelopes, **not
a theorem for arbitrary Python integers or a bound on recursive scratch**.
See the [fixed integer implementation](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/longobject.c).
The local division-reconstruction helper, _pylong.py, is also hash-bound; the
record must fail if that implementation changes, not only if fractions.py does.

With the unchanged B=8192 acceptance gate:

| Inspected fragment | Provenance bits p | Returned digits |
| --- | ---: | ---: |
| Two checked operands | 16,385 | 1,098 |
| Norm update n + abs(s)*norm[parent] | 24,577 | 1,644 |
| Input midpoint (lo+hi)/2, before gate | 34,922 | 2,334 |
| ReLU enclosure, up to four error terms | 37,075 | 2,476 |
| Result rational parsing, 4 × 4,936 characters | 19,744 | 1,322 |

The largest listed Fraction component is 19,904 bytes. This is not its typical
size, a measured peak, a new bit allowance, or a universal per-object multiplier.
The prior 34,921 figure covers two-operand endpoint arithmetic; including the
midpoint division needs 34,922. Neither changes the 8,192-bit acceptance check.
Arbitrary preallocated integers borrowed through F(int)/F(Fraction) are outside
this closed provenance contract. Logical index/getsizeof subtotals in the older
constructor ledger remain what they were; they are not silently upgraded to
requested allocator capacity.

The active shared builder/checker override the old expanded affine and guard
paths. Old dense expansion is not part of this sum. Conversely, checking only
Work.exact calls would miss compound arithmetic executed before those calls.

## 2. Source/JSON objects and retained cache references

For a raw manifest of H bytes, retain its actual arrays, objects, strings and
numeric-token counts, including optional trust metadata. A conservative named
decoded-tree component can be expressed as:

~~~text
sum LP(array lengths) + sum DP(object member counts)
 + sum (80+4*string_codepoints)
 + sum (32+4*ceil((4*integer_token_length+31)/30))
 + 24*float_count
~~~

LP/DP are the previous monotone-container envelopes. Counts and total lexical
lengths are bounded by H, but this formula excludes decoder memo/pair lists,
recursion, encoder workspace and allocator storage. In particular the previous
17H text-overlap subtotal is not the entire parse heap. The duplicate-key hook
can create pairs before rejection. The fixed [JSON decoder](https://raw.githubusercontent.com/python/cpython/v3.12.12/Modules/_json.c)
and [Unicode representation](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/unicodeobject.c)
must remain distinct from project-owned retained objects.

A TensorRows instance has 13 attributes and retains 53 Tensor, K Chunk and
52 Layer instances; tensor shape tuples contain rank+78 entries, chunk tuples K.
The full raw manifest/trust tree is not retained after initialization. Use the
bound actual rank/layout rather than assuming four dimensions or 95 chunks.
Managed-instance dictionaries and class-shared keys must be counted; a slots
formula must not be applied to these dataclasses. These inventories are finite
but have not been combined into a claimed absolute heap total.

Snapshot.inputs reconstructs all endpoint Fractions and their pair tuples.
Using the prior generator-to-tuple envelope TP:

~~~text
Inputs(n) = TP(n)+n*TP(2)+2*n*F(17460)
Inputs(3072) = 59,289,936 bytes
two overlapping endpoint collections = 118,579,872 bytes
~~~

These deliberately loose requested-component envelopes apply during input
construction/checking, not throughout every phase. Snapshot shares its reader's
stored inputs; do not count a third collection for that assignment. During
iterator advance the caller's old row and the newly constructed row can coexist:
at most 6146 parameter/bias Fractions for two registered 3072-wide rows, plus
new list/tuple references. PackedRow retains binary bytes, not a complete
source-weight Fraction matrix. This overlap also applies during Snapshot intake.

One previously omitted persistent reference set is CPython's Fraction hash LRU:
the bound fractions.py uses maxsize=16384 and keys containing numerator and
denominator. Scale grouping and gate sets trigger it. Normal generated sources
give at most min(16384,n+4)=3076 entries with binary64/unit/half scales; their
referenced integer payload alone is at most 1,058,144 bytes using 37 digits per
integer. Keys, cache nodes, dictionary and result hashes are additional.
The cache can survive forest/session cleanup. It retains integers, not the
Fraction objects. A malformed or prewarmed interpreter cannot inherit the
normal-source 3076/37-digit bounds; the global entry cap alone does not bound
arbitrary imported integer payload. No cache was cleared or changed here.

## 3. Joint phase map — sum overlaps, not unrelated maxima

| Phase | Required overlapping own components |
| --- | --- |
| Second source reader during intake | First reader; second reader initialization; pinned raw manifest; refs/inodes/archive/census; two decoded requests at equality |
| Build | One full Snapshot bank; growing forest/norms/scales; up to two affine rows during advance; pending layer/ReLU lists; duplicate endpoints during input stage |
| Independent check | Published packet; one new Snapshot bank; parsed forest; local/global norms and visible-factor sets; up to two source rows during advance; duplicate endpoints during initial input check |
| Property plan | Checked forest with retained first-layer bytes; already allocated RootTable; tasks; generic pullback temporaries |
| Native construction | Checked forest/plan/roots; arm-specific constructor containers and tensor payload; separately scoped library scratch |
| Exact candidate consumption | Native operator and best dual still alive; candidate/frame; exact arena/temporaries; checked endpoint values and receipts |
| Production finish/seal | Checked forest/plan/receipts; new TensorRows and complete source recheck; archive/result metadata; hash cache |
| Replay finish/seal | Same non-native components plus last loop raw frame; no native operator |
| Reception | Package/member records, result aggregation and census/suffix; no fourth lowering |

Production closes the native lifecycle before the final source seal. Snapshot
banks from sequential source passes must not be added as if all three survive.
On the other hand caller and controller coexist with the active actor:

~~~text
joint(t) = caller(t) + controller(t) + active_actor(t)
joint peak = max_t joint(t), not max(one actor)
~~~

The prior 110,559,088-byte exact arena and 62,712,192/330,081,664-byte named
native tensor subtotals are ingredients, not complete phase sums. Retained
Fraction counts, metadata skeletons and phase-specific temporaries still need
the same identity-aware join. The map does not assert these additions already
fit 2 GiB or 8 GiB. No RSS or duration was measured this stage.

## 4. Existing rejection gates versus missing construction bounds

Already present, and not to be reimplemented as a claimed novelty:

- Raw JSON reads check lengths before decoding: 4 MiB source manifest and
  1 MiB native records. Successfully retained 13 metadata/log files have caps
  summing to 7,733,248 bytes. The four-log subtotal 1,245,184 bytes is **inside**
  that number, not an extra allocation or a guarantee of native log demand.
  These 13 files are the request-root roster; the outside caller receipt is
  additional. Its saving time was already included in the original 300 seconds.
- Archive checks at most 4096 attempts and 200-character relative names before
  directory creation. A loose own-directory bound is 4096×100 per Archive,
  including failure prefixes; source intake permits only two levels. Therefore
  normal owned directories are not intrinsically unbounded. Unowned/native
  additions and external directory mutation remain explicit rejection/trust
  boundaries, not entries silently covered by the normal roster.
- Existing AS/FSIZE limits apply to managed controller/actors. The outside
  caller's profile validation and invocation encoding do not inherit actor AS.

Still missing from a complete pre-construction contract:

- validate_profile computes identity(value) before validating the closed shape;
  write_json and some save paths encode before testing the output cap.
- str(exc)[:400] first constructs the complete string. The *retained* encoded
  message is at most 4802 bytes (6002 for 500 characters), but this is not a
  bound on initial formatting, exception class name, args or traceback.
  The owned-bounded helper's fixed error codes already avoid that particular
  problem; they do not cover every native actor/caller error site.
- Cleanup collects live as well as zombie PIDs and reads proc entries. A
  success-only empty live list cannot bound a failure terminal. For separately
  bound count z and PID width p, each roster has length at most
  max(2,1+z*(p+1)); neither z nor p is newly selected here. Sampled 2-GiB stopping
  does not bound the first overshooting reading or post-cleanup RSS field.
- A late caller failure can retain both a populated result and an error.
  Successful archive formulas cannot simply be reused for that branch.

This does not invalidate the completed finite controls or turn an incomplete
run into UNSAFE. It prevents claiming an all-path capacity guarantee from them.

## 5. One proposed next interface, not four new micro-experiments

Keep a non-executable immutable binding sheet with exact bytes/hashes for
profile, root, run ID, version, property and source manifest/layout. Missing
fields are null with a reason; observed OS values are not frozen contracts.

| Proposed component | Required behavior |
| --- | --- |
| Frozen envelope | Bound canonical/raw bytes, schema and source layout; do not precompute request propagation or proof for free |
| Metadata construction | Reserve schema/field representation before construction/encoding against unchanged caps; over-cap refuses |
| Observation/diagnostic intake | Bound incremental collection and avoid unbounded exception formatting; incomplete observation explicitly refuses acceptance |
| Partial terminal | Distinguish complete, partial and absent output; preserve failure prefixes, outside caller cost and late-result refusal |

This proposal may reuse existing cap values, but applying them earlier or to
new observation fields changes interface/rejection semantics and still needs
implementation authorization. It must not truncate an observed live set into
an empty set, remove a mathematical/identity/residual/aggregation check, reset
Work per tile, or hide preprocessing outside 300 seconds. It does not resolve
the separate full-topology Work/member/nnz/candidate-cap refusals.

**Continue** only with a single shared, versioned self-owned resource/interface
proposal and explicit counts/lifetime joins for both arms. Numerical resource
policy is a separate decision. **Stop** repeated synthetic controls, isolated
metadata patches, old-limit full-shape attempts and recursive library-scratch
investigations without a new decision. Library/native overhead may use the
adopted hard-fail boundary; project-owned JSON/forest/census must not be renamed
as native overhead. Do not report full capacity until the remaining joins and
bindings are actually checked.

## Review and verification

Read-only AI reviews covered source/JSON lifetime, integer provenance/capacity,
and metadata/cleanup. A second source review cross-checked the integer envelope
and required its closed-provenance qualification. This is not an independent
human technical review or empirical replication.
The final source review also corrected an initially omitted old/new row overlap
during iterator advance; metadata review separated retained caps from early
construction limits and kept the outside caller receipt out of the 13-file sum.
Integer review added the previously omitted _pylong.py environment binding and
distinguished 3072 coefficients from a 3073-Fraction row including bias.

~~~bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_joint_contract.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_generation_contract.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_storage_lifetimes.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_counted_envelope.py --check
git diff --check
~~~

All four static checks and whitespace checking passed after these corrections.
No production imports, model/source construction, native calls, GPU, dependency
changes, new quota or runtime edits belong to this static stage. Its output is
a better delimited resource decision, **not progress evidence for algorithmic
superiority or a newly closed real MoE certificate**.
