# C129 preflight — complete owned evidence and unchanged C62 ledger payment

Disposition: static/saved-scalar preflight only, not an executed qualification.
The proposed source200M/setup2M/comparison10M/proof6M/evidence20M/custody9M/
ledger7M split totals254M without increasing whole256M or branch200M. For the
current fixed four-fixture program, complete C62 ledger work is conservatively
bounded by6682176<7000000, including2162688 reporting reservations fixed BEFORE
any C129 freeze or run. Complete ledger JSON is bounded by342016 bytes and the
complete terminal JSON plus overhead by787434 bytes; each receives a1MiB
allowance. The earlier131072-ledger/524288-success proposal is superseded before
execution, not retroactively financed after a failure.
The proposed complete custody fee is8223344<9000000. This bound does NOT reuse
C128v2's failed partial ledger as if it were a complete certificate.

The complete C129 worker still has to execute the unchanged strict owner walk,
known-metadata census,64M-entry gate and BOTH1GiB measurements successfully.
Both C128 versions remain failed. No new source/network/F4/solver admission,
wall-speed claim, score gain, lowered fee or skipped proof follows here.

## Authenticated saved inputs and inspected source

Paths below are relative to `experiments/neural_hz_20260831`.

| Input | SHA256 |
| --- | --- |
| `results/c128_channel_support_20260927_v2/exit.json` | `53d2b6bd379a9b33313956fa762feaa92aee5681a537535cd03f90035f8b6c36` |
| `results/c128_channel_support_20260927_v2/preregistered.json` | `968399807e07fd9d408b1667274496701f09da5a33542dd29560036ea2a91420` |
| `CHECKPOINT_C128_CHANNEL_SUPPORT_20260927_SHA256SUMS` | `0bfdf6ebd26a89cd427a6f9cf06f6b6d42967548c6d939ee6f2df838c415d2b5` |
| `c62_physical_measure_v1.py` | `3bb3e74748d707830415b6d29c76789bd33e9042ee3e0d647b70b87562179bc6` |
| `c5_partial_csr_owner_ledger_v3.py` | `043017915c743f0ff843c56e8266d19c20e279a7923af4d402785872de67df5c` |
| `c97_birth_emission_v1.py` | `88daef3b69cb2ac64cf3585ca546fb415832e1dd25be11aeefc8e79a2e45d1a1` |
| `c97_direct_csr_v1.py` | `e8b518e93feac19e5a025517429daa847fa8b312b5b3cf7e34166de17564425e` |

The pinned v2 exit authenticates all24 complete stage/case/point JSONs used for
the scalar inventory. Their hashes, exact bytes and complete manifests were
independently checked in the C128 final audit. This calculation reads those
saved JSON trees and current source text only; no numerical NPZ payload, model,
source restore, experimental import, test or proof run is performed. The new
three C129 implementation files are reviewed as source and must be frozen with
this preflight before any numerical execution.

## Complete custody and retained-root shape

The current `c129_complete_source_v1` still compares ACTUAL live old/new arrays
and every original semantic/owner/inverse field. Its local dictionaries of CSR
views expire only after that comparison. Original expression, both complete
generated states, every graph/source/HZ owner, metadata and all proofs remain
reachable from `held[mode]`. Removing a temporary duplicate view dictionary is
not deletion of its original physical owner or proof.

`c129_source_worker_v1` retains a detached snapshot for each source/old/new
stage, and the original already-owned dense owner-oracle arrays for each owner
stage. Thus `stage_arrays` contains16 maps:12 complete snapshots and4 complete
owner maps. There are no retained snapshot reports, recipe caches, scalar-word
arrays or bytes-backed views beyond these arrays. `snapshot` makes every array
owned, C-contiguous and read-only with its exact dtype and shape, including
zero-length fields. It changes no HZ, graph, powers, UID, source or proof rule.

| Population | Maps | Arrays | Logical elements |
| --- | ---: | ---: | ---: |
| Original source/operator snapshots | 4 | 116 | 45608 |
| Complete old/new state snapshots | 8 | 384 | 980774 |
| Complete original owner oracles, not recopied | 4 | 8 | 15208 |
| Total staged evidence | 16 | 508 | 1041590 |

All12 snapshot calls prepay1024 header units and8 units per copied element
before any copy allocation. The fixed header covers at most128 named arrays;
actual source/old/new populations are29/48/48. The element tariff covers the
complete read, copied allocation, write and retention, not a byte/hash surrogate:

The fixed1024 term is a NEW composite header tariff for the whole bounded
mapping, not1024 element operations. Its at-most128 records cover all eight
fixed groups: name/exact-ndarray custody; primitive dtype domain; size/byte
totals; aggregate caps; copied dtype/shape; owned-base/contiguity; read-only
flags; and destination-map retention. Dispatch and the empty destination map
belong to that fixed scope. Header checks on zero-length arrays are included
even when their element fee is zero. No header, post-copy ownership check or
flag write is delegated to an uncharged later pass; the actual fixture uses
at most48 records per call. This is a tariff for the new complete snapshot
program, not a discount to an old constructor or evidence-encoding tariff.

```
copied elements = 45608 + 980774 = 1026382
custody work = 12*1024 + 8*1026382 = 8223344
custody headroom = 9000000 - 8223344 = 776656
```

All original full comparison, owner/inverse and archive encoding tariffs remain
payable separately. No copied element is refunded out of those old payments.
The500 new owned arrays add at most8211056 numeric payload bytes at eight bytes
per element, with every array/header/temporary and all live originals still
inside the resource measurements. This payload bound is not a complete RSS,
traced-memory or retained-entry certificate.

## What the unchanged C62 routines charge

For a supplied complete `held` root, `numeric_layout` charges16 for each newly
visited object header, then `128*R+1024` for its R registered numeric/typed roots.
It executes the unchanged full C5 owner/dtype/span/CSR-role checks. `metadata`
charges8 per newly visited header, including live HZ/CSR dictionaries and array
base headers; numeric backing bytes are not counted again as Python metadata.
Neither routine iterates scalar values inside a numeric ndarray as Python
metadata. A fingerprint is not called by this fixed worker, so no fingerprint
pass is silently assumed free or substituted for an original binding check.

We use an upper H for EACH header walk, even though their actual traversals
differ. Counting every tree occurrence, including repeated keys/values and
repeated serialized copies, bounds identity-deduplicated header visits without
depending on integer/string interning or empirical v2 counts:

```
T(scalar) = 1
T(list/tuple) = 1 + sum(T(item))
T(dict) = 1 + sum(T(key) + T(value))
```

Arrays are counted as headers, not expanded into their elements. Their live
base chains are explicitly reserved below. The owner-layout tariff and complete
serialized reporting reserves are added separately; no partial-v2 work credit
appears anywhere in this derivation.

## A. Every saved scalar, proof and evidence tree

Counting all24 authenticated v2 JSON files independently, without sharing any
object between files, gives the following exact occurrence totals:

| Complete tree population | Header occurrences |
| --- | ---: |
| Four original source stage JSONs | 1676 |
| Four old stage JSONs | 3684 |
| Four new stage JSONs | 3684 |
| Four owner stage JSONs | 292 |
| Four complete case JSONs | 9412 |
| Four full original/recovered point JSONs | 118988 |
| Total | 137736 |

The point populations alone have39632 rational pairs; all their pair containers,
numerators, denominators, population lists and wrapper fields are included.
This count is intentionally much larger than the actual distinct-object count
when small integers or report/metadata objects are shared. Full manifests,
shape lists, hashes, exact point receipts, both source reports, original
bindings, nodes, every alias-work field and all proof flags are included.

C129 adds two boolean fields to each complete case report:four extra key/value
occurrences per case,16 overall. Consequently A=137752. New terminal data flags
are covered by the extra-wrapper allowance below; they are not new numeric
roots. Value changes, path strings or source hash changes do not increase this
header count. Any future schema/population growth must be re-counted before
freeze, not dismissed because the fields are “only metadata”.

## B. Complete immutable freeze, including growth

The full saved v2 freeze has5353 occurrences:5075 in its2537-entry source-hash
map,158 in its157-file test list, and120 elsewhere. Growing source hashes to
3000 and test files to160 would give:

```
120 + (1 + 2*3000) + (1 + 160) = 6282
```

Reserve B=8192 occurrences for the ENTIRE new freeze, not only its hash map.
This includes new C129 conditions, this document's own frozen hash, all prior
run/dependency/document paths, every digest, test filename, provenance and
original input bindings. The supervisor must reject BEFORE numerical work if
source hashes exceed3000, test files exceed160, or the complete plain-JSON
freeze occurrence count exceeds8192. That pre-run structural check does not
read numeric artifacts or alter the worker's unchanged complete ledger.

## C. All live source/HZ/graph headers not represented by saved JSON

The live scalar report objects are already represented in A:the metadata
packets directly reference each generated `fields['report']`, and complete
case/stage evidence retains these objects. Still reserve an additional8192
headers PER fixture,32768 overall, for all live non-JSON structure and base
headers. A conservative source-level sub-envelope per fixture is:

| Live structure | Bound per fixture |
| --- | ---: |
| Three HZ objects, each including its six CSR objects, vars dictionary, three vectors, flags and array bases | 3*512=1536 |
| Two standalone diagonal CSR operators, including complete headers/bases | 2*64=128 |
| Two returned-state/fields/construction/graph container populations and their non-HZ array headers/bases | 2*1024=2048 |
| Original expression/Conv/keep/case wrappers and other fixed live scalar headers | 512 |
| Subtotal | 4224 |
| Reserved ceiling | 8192 |

There are four graph nodes and27 non-HZ primitive arrays per generated state
(nine fields, sixteen node arrays, two UID arrays). Each state has only its
fixed five top-level fields, sixteen field entries and five construction
entries; these are validated by `packet`. Graph node dictionaries retain their
complete fixed schemas, including original source/operator pointers. Original
and generated HZs have precisely the11 validated HZ fields. The64-per-CSR
envelope includes data/index/indptr array and base headers, shape tuple/scalars,
ordinary SciPy canonical/sorted flags and ACT's `_act_hz_zero_free` flag.

The ordinary producers use owned allocation/concatenation and finite view/base
chains, not foreign buffers or arbitrary object-array trees. The surplus3968
headers per fixture above the explicit sub-envelope covers remaining ordinary
view/base and container duplication without assuming deduplication savings.
This is a bound for these fixed fresh constructors and their current supported
schemas, not a universal claim about arbitrary mutated SciPy objects. Unknown
types, incompatible aliases and actual pool exhaustion still reject normally.
No live source/HZ/graph is filtered out of either C62 walk.

## D. Detached arrays, holders, receipts and authentication

Reserve D=4096 further headers for the complete top-level held structure,
sixteen stage-array maps, all508 snapshot/oracle array headers and their map
keys, stage-record/report/point-holder dictionaries, all24 exact JSON receipts,
the four point-receipt map entries, authentication counters and fixed C129
reporting flags. The16 array maps plus every key and array use at most
16+2*508=1032 occurrences;24 seven-field receipts use at most360 more.
The remaining2704 cover the fixed holder structures, duplicate references,
authentication scalars and new report fields. All500 snapshots own their
storage (`base is None`), so they add no unbounded backing-container trees.
Owner arrays and original live base headers are already covered above.

Reports, artifacts, decoded freeze, full rational point populations and actual
owner oracles are retained. This allowance is not permission to omit evidence
or to replace a full point population with a digest. The C62 metadata census
still explicitly reports inherited opaque IDs and does not claim complete
Python allocator occupancy.

## Complete numeric-root bound and final ledger arithmetic

The numeric-header walk stops at numeric arrays, HZs, CSR matrices and its
registered opaque expression/operator types; the unchanged owner visitor then
checks their full contents. A conservative explicit root population is:

```
500 detached source/old/new arrays
+ 8 complete owner-oracle arrays
+ 8*27 live non-HZ state/graph/UID arrays
+ 8 generated HZ objects + 4 original HZ objects
+ 4 expression objects + 4 Conv operators + 8 diagonal CSR operators
+ 4 original keep arrays
= 756 roots <= 1024 reserved roots
```

Shared references may reduce this count but are not needed for the inequality.
All CSR internal payloads remain in the actual owner visitor; treating one CSR
as one registered root does not omit its data/index/indptr storage. Set R=1024.

```
H = 137752 + 8192 + 32768 + 4096 = 182808
numeric header walk <= 16*H = 2924928
known metadata walk <= 8*H = 1462464
full numeric owner layout <= 128*R+1024 = 132096
prepaid success/failure/ledger JSON = 1048576+65536+1048576 = 2162688
COMPLETE ledger-category upper = 6682176
ledger-category headroom = 7000000-6682176 = 317824
```

All three reporting allowances are nonrefundable, including an unused failure
allowance. Their exact-byte limits must still be checked at publication. The
ledger receipt and terminal wrapper are created after the held-layout census;
they add no numeric arrays and are separately prepaid reporting metadata, not
a claim that the earlier held census includes future terminal objects.

## Complete ledger JSON bound, including every storage role

The C62 work bound alone did NOT justify the initial131072-byte ledger
allowance. It is superseded before freezing by1048576, without changing the7M
category or254M total. The following is a complete structural serialization
bound, not an empirical bytes-per-record ratio from a previous experiment.

The unchanged C5 `StorageProvenance` serializes exactly six fields:token,
storage_kind,resident_bytes,resident_entries,entry_semantics,roles. It assigns
one record per final ndarray backing owner and adds every strong alias role;
it does not stop traversing a compound object just because it was counted once.
Consequently storage records and root counts cannot be conflated.

There are at most1012 distinct visited ndarray leaves/backing records:

```
500 snapshots + 8 owner oracles + 216 live field/node/UID arrays
+ 4 original keep arrays                                  = 728
+ 12 HZs * (3 vectors + 6 CSR matrices * 3 buffers)         = 252
+ 4 expression bias arrays + 4 Conv kernels
+ 8 separate CSR operators * 3 buffers                    =  32
TOTAL distinct ndarray leaves/backing records             = 1012
```

This deliberately ignores storage sharing that could decrease the record
count. A view contributes its final owner, not an additional record per base
link. The four expression biases must be included even though they are reached
through opaque expression roots rather than separately registered flat roots.

All alias roles must also fit, including repeated visits through expressions:

```
728 direct dense roots
+ 12 explicit HZ roots * 21 array visits                  = 252
+ 4 expression roots * (bias1 + source21 + Conv1 + CSR6)   = 116
+ 4 explicit Conv roots + 8 explicit CSR roots * 3        =  28
TOTAL possible storage-role registration occurrences      = 1124
```

Use rounded bounds1024 records and2048 roles. The flat numeric root ID has at
most four digits. With the fixed one-term/three-operator geometry, the longest
role is52 ASCII characters:
`active['1023'].terms[0].source.predicate.Auc.indices`.
Reserve64 characters per role. These generated labels contain no double quote,
backslash or non-ASCII character needing extra JSON escaping; two quotes and
one separating comma therefore cost at most67 bytes per role.

For each record, a12-character `storage-1024` token, `numpy` storage kind,
the longest23-character entry convention, empty role-list brackets and even
20 decimal digits for EACH numeric counter serialize to181 compact bytes.
Use192 bytes per record including record separators. This bound does not depend
on first passing the1GiB gate; the16GiB address-space ceiling is already fewer
than20 digits. All six object-count kinds, fixed C5 global fields/no-claims,
known-metadata dictionary and all eight opaque64-bit IDs are included in a
separate8192-byte fixed allowance, also covering outer framing/newline and the
1024 reporting overhead. No numeric storage provenance or role is shortened.

```
complete ledger encoded bytes + fixed reporting overhead
<= 192*1024 + 67*2048 + 8192
= 342016 < 1048576
```

The actual exact byte string still must pass `JsonAllowance` before publication.
No role suppression, pointer substitution, shorter proof population or existing
ledger-schema modification is used to reach this bound.

## Complete terminal JSON bound, including duplicated evidence records

The terminal includes the full ledger AGAIN, all four complete case reports,
all sixteen stage records, all25 serialization receipts, category/work data,
measurement, authentication and status fields. Those copies are all counted.
The full point populations remain strongly held, proved and separately saved,
but the unchanged return schema contains their complete receipts/counts rather
than repeating their rational lists in `result.json`.

For a conservative source-schema byte bound, retain every exact string/key in
the saved JSON schemas, count brackets/commas/colons, and replace EVERY numeric,
boolean or null scalar by a32-byte encoding budget. Case/stage numbers are
fixed-geometry dimensions/work/counts and process IDs, not arbitrary512-bit
coefficient payloads;32 bytes exceeds all signed64-bit ID/counter encodings and
the bounded scalar encodings here. Hashes retain their full64 characters.
Stage artifact filenames are unchanged. Add both new C129 case flags in full.
This gives:

| Complete serialized subtree | Conservative bytes |
| --- | ---: |
| Four-report dictionary, all nested source metadata/manifests/point receipts | 178353 |
| All sixteen stage-record dictionary entries | 174905 |
| Complete ledger, using the bound above | 342016 |
| All25 receipt objects,1024 bytes each | 25600 |
| All other fixed result/data/work/measurement/authentication fields and framing | 65536 |
| Additional terminal reporting overhead | 1024 |
| Complete terminal upper | 787434 |

The per-case report bounds are44565/44571/44613/44559 bytes. Source/old/new/owner
stage schemas are bounded directly by their complete fixed key/value trees;
they are not estimated by scaling their historical file sizes. A receipt has
seven fields, a filename of at most34 characters, a64-character hash, three
bounded integer quantities, a boolean and the fixed encoding-name string:
1024 bytes each is conservative. The remaining result/data schemas have fewer
than128 fixed scalar fields in total and at most32 work-part entries; allowing
256 bytes per fixed field and512 per work entry fits49152, leaving16384 for
their dictionary/list delimiters and keys in the65536 allowance. No full freeze
map, numerical arrays or full rational point lists appear in those remaining
terminal fields. They remain retained and counted at their proper live or
separate evidence boundary.

Thus `787434 < 1048576` pays the complete terminal without relying on a shortened
success payload. Ledger1048576/success1048576/failure65536 are established
BEFORE the sole run. The16 stage, four case and four point allowances stay
unchanged. Their24 full JSON publications and the25th ledger receipt are still
mandatory. A serializer failure remains failure; no post-run reserve transfer
or fallback success is permitted.

## Whole fixed-program preflight and execution boundary

| Category | Conservative current bound | New fixed category cap |
| --- | ---: | ---: |
| Both complete source builds on all four fixtures | 200000000 | 200000000 |
| Fixture creation and full original bindings | 1083872 | 2000000 |
| Complete headers and actual live-array comparisons | 7911728 | 10000000 |
| All eight complete owner/inverse proofs | 4802560 | 6000000 |
| All sixteen numeric archives and24 full JSON records | 19303264 | 20000000 |
| All twelve complete detached snapshots | 8223344 | 9000000 |
| Complete C62 header/owner/metadata ledger and reporting | 6682176 | 7000000 |
| Total | 248006944 | 254000000 |

The fixed ordinary geometry, all proof/evidence populations and all original
fee rules are unchanged. Source reservations remain paid in full; this total
does not refund measured source work. Any implementation or freeze growth beyond
the explicitly counted schema needs a revised pre-run bound within the same
global caps. V2's234693760 incomplete counter is NOT carried as a complete
qualification and its partial ledger is NOT subtracted from this new bill.

This establishes pre-run payment feasibility for the current owned-evidence
representation, not success of the alias, memory, time, source or final evidence
gates. The complete inherited tests, all four unchanged C16/K32/h6 fixtures,
all eight builds, unchanged strict ledger,64M retained entries,BOTH1GiB limits,
CPU1/GPU0/AS16GiB,60s tests and240s worker remain mandatory. A failure retains
its staged evidence and closes the version rather than borrowing an allowance.

Formal1870/2413 and separate E0 CIFAR25/Tiny36=61/400, all13 families/every old
solve, true nonconvex HZ and original source/owner/inverse semantics remain
protected. This ordinary custody plan does not remove the independent real
source/kernel/output branch obstruction or authorize a real-network retry.
