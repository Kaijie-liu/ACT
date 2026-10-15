# Shared-source archive demand and object lifetime contract

The conditional **complete successful archive demand** is now accounted for,
including source, forest pages, both candidate roles, checker code, inventories,
progress, both worker result copies and the parent receipt. This is a static
follow-up to the [counted-loop envelope](hz_shared_execution_envelope_20261015.md),
not a runtime change or full-model admission. The existing complete-topology
Work, membership and dense-arena refusals remain. All checks, acceptance gates,
resource limits and the 300-second request budget are unchanged.

Base: `a3681b44f3bad58e52cb64e4f6b881c4e93afecc`, branch
`feat/moe-route-verification`. The [calculator](../scripts/assess_shared_archive_contract.py),
[tests](../scripts/test_shared_archive_contract.py) and
[bound report](hz_shared_archive_contract_20261015.json) read only implementation
syntax, hashes and registered dimensions. They do not import production modules,
generate source/candidates, launch workers or execute models, solvers or GPU work.

## What source layout this bound actually covers

The registered dimensions are input 3072, router `3072–128–8`, and eight experts
`3072–256–128–10`. The metadata envelope additionally assumes center shape
`[1,3,32,32]`, the fixed ASCII trust text emitted by `capture.py`, and the existing
`factored_source.pack()` default of full 1 MiB chunks **per tensor** followed by
its remainder. Four domain numeric strings are each bounded by 4096 characters.
There are no additional provenance fields.

These assumptions are not new parser restrictions or granted execution scope.
The general parser also accepts smaller aligned chunks, other center ranks and
optional trust content. Its generation demand does not follow from this bound.
More importantly, the shared supervisor still accepts only `half/guarded` fixed
sources; it has **not** been connected to the full captured-source packer. The
calculation describes the current format's conditional demand, not an executable
full-model protocol. A new source adapter/specification must account for its own
additional metadata and capture costs.

| Component | Conditional byte upper | Files |
| --- | ---: | ---: |
| 52 parameter tensors plus center | 55,715,520 | 95 chunks |
| Source manifest, including maximum domain strings | 50,656 | 1 |
| Forest header | 1,652,770 | embedded in pages |
| Forest binary sections | 5,329,156 | embedded in pages |
| Forest page index | 11,835 | 1 |
| **Source plus forest** | **62,759,937** | **204** |

The 6,981,926-byte forest uses 107 pages. Header demand includes factor names,
all views/roles, first-affine aliases, sequence spans and the full source receipt.
The source manifest is obtained from small schema skeletons and record-length
algebra: state inventory 7,801 bytes, tensor descriptors/chunks 18,782, networks
6,990. Its fixed-domain upper is 34,280; expanding the four numeric strings adds
16,376. No tensor or forest is materialized to compute these numbers.

Memberships remain logically expanded. Source ownership transfer avoids storing
a second source, but does not remove independent source traversals or checks.

## Complete successful metadata, not a sum of file caps

All generated fields are accounted for recursively. Fixed ASCII relative paths
are bounded by 200 characters, finite float text by 24, and nonnegative owner
counters by 20 digits. The latter is a representation bound justified by loose
allocation/copy counts, not a new numeric allowance. Twelve Work-counter digits
cover the prior conditional full-loop calculation; the report also retains the
seven-digit calculation for the **unchanged** 8M policy.

| Generated member | Byte upper, conditional full-loop counters |
| --- | ---: |
| Contract | 4,550 |
| Manifest | 36,074 |
| Bundle inventory | 100,250 |
| Standalone 252-property result | 1,257,460 |
| Inner terminal | 1,236 |
| Worker candidate report, including two complete results | 2,666,907 |
| Supervisor invocation | 12,648 |
| Parent receipt, including one complete result | 1,294,980 |
| 368 progress lines | 64,400 |
| Successful silent stub worker log | 0 |
| **Generated metadata total** | **5,438,505** |

The 26 currently packaged checker files add 244,614 bytes. The full inventory is
324 files: 96 source files, 107 forest pages, one page index, 84 candidate tiles,
26 code files, five inner metadata/results and five supervisor files.

| Candidate scalar representation | Whole-request storage upper |
| --- | ---: |
| Binary64 typed candidates | **91,311,216 bytes** |
| General-rational typed candidates | **2,900,002,416 bytes** |

The binary64 total fits the aggregate 128 MiB component in this calculation, but
**does not admit the request**: Work/memberships/dense ownership already refuse,
the supervisor has no full-source entry, and individual result/report uppers
exceed existing 1 MiB caps. An upper exceeding a cap proves a missing worst-case
capacity guarantee, not that an actual result necessarily overflows. No large
report or full-shape execution was generated or observed here.

The returned call/census is not automatically written into this archive. If a
caller saves it, that extra result/inventory and escaped root name must also be
charged. Offline relocation and later audit outputs are separate. Failed prefixes
continue to count under existing archive/census rules; the successful schema
above is **not** a bound on arbitrary exceptions, native reports or provenance.
In particular, an unbounded OS error string cannot inherit a success constant.

## Actual object lifetime, including what the owned arena misses

The complete packed parameter payload is 55,690,944 bytes; after checking, aliases
retain only the first-affine packed rows, whose payload is 53,494,784 bytes.
Both require Python row/key/container overhead. Builder and independent checker
graphs do not coexist across the build return, and production graph/arena cleanup
precedes replay. Those releases do not imply that the allocator returns RSS.

| Phase | Important live objects beyond the current owned-buffer ledger |
| --- | --- |
| Capture/build/encode | Full packed bank, builder graph, encoded pages, header and writer fill |
| Independent source check | Packet, new full bank, graph, parsed metadata, two norm lists and visibility sets |
| Production consumption | Packet still retained, checked graph, first-affine rows, plan, arena, candidate lists and receipts |
| Portable source check | Saved production result, inventory, packet, new full bank and checking temporaries |
| Portable consumption | Saved production result, graph/first-affine rows, plan, arena, caller frame and receipts; packet released |
| Worker report generation | Both full results, snapshots/events and JSON encoding temporaries |
| Parent reception | Original `pending`, re-read report, source/manifest/bundle, independent result and encoding temporaries |

Concrete non-arena bounds include `V·F = 2,627,936` logical visibility memberships, two
`N=19,496` norm lists, and up to `N+2F=83,592` ordinary-dict/Fraction entries
during planning. The previous residual can survive while the next pullback is
evaluated. These are logical element counts, not allocated hash/list capacities,
bytes or measured peaks; fixed-build capacity functions are still needed.

The source row loop can retain the previous row while evaluating the next one.
A conservative temporary includes `2(W+1)` source Fractions, `3W` logical references,
two chunk buffers and two row-fill/publication payloads, where `W=3072`. The
one-row wording in the existing Snapshot docstring is not a bound for this
assignment overlap. No runtime docstring or behavior is changed in this stage.
Replay sealing can hold caller bytes plus owned mutable and immutable buffers:
three frame payloads in the full process, although only two belong to its arena.

An auditable live-object formula is:

```text
Mfixed + Marchive + max(
  Ball + Gbuild + encoding temporaries + Packet,
  Packet + Ball + Gchecked + checking temporaries,
  Packet + Bfirst + Gchecked + Plan + Arena + Receipts + CandidatePython + norm temporaries,
  Rproduction + PackageMeta + Packet + Ball + Gchecked + checking temporaries,
  Rproduction + PackageMeta + Bfirst + Gchecked + Plan + Arena + Receipts + CallerFrame + norm temporaries,
  Rproduction + Rreplay + report objects + JSON temporaries
)
```

Parent reception needs its own additional envelope. Instantiating this formula
requires fixed Python/allocator container sizes, numeric/JSON temporaries and
allocator retention bounds. `Work.exact()` checks bit length after evaluating an
expression; unchecked arithmetic intermediates are not bounded merely by the
accepted 8192-bit result size. Thus neither full live-heap nor RSS upper is claimed.

The current 2 GiB RSS rule samples parent plus owned group at roughly 10 ms
intervals; it is not an OS hard memory cap and can miss peaks. After the child
exits, parent `receive()` is outside that sampling loop. `RLIMIT_FSIZE` limits
files, not heap. These are distinct from the already implemented hard time and
late-result rejection controls.

## Validation and next decision

Eight new static tests plus sixteen prior ledger tests pass under `python -B -S`.
Three read-only reviews independently covered source/layout, metadata/supervisor,
and object lifetimes. The first new suite had **three failures**: small schema
examples omitted conservative shape-string width, nullable-parent width and the
bounded sequence-count width. Only the calculator skeletons were corrected; the
frozen source dimensions, production code, tests' expected totals and all limits
were unchanged. The prior reports remain intact.

The first saved-report equality check also rejected one serialization error:
passing a 64-bit counter through a JavaScript Number changed its final digits.
The archive now retains the exact Python integer `7420863689825944064`; the
independent saved-report equality check passes. No rounded counter is accepted
as equivalent, and no runtime result or scientific evidence was affected.

The current shared-source implementation/control stage remains complete. This
follow-up closes conditional archive generation demand and identifies actual
lifetime terms; it does not close G1–G6, demonstrate speedup or create certificates.
Do not repeat the source-only ledger, expand controls to chase a passing size, or
retry the full topology under unchanged refusals.

The next **decision**, before more runtime work, must separately specify:

1. A bounded source/capture profile and report/ownership representation, including
   early size checks and duplicate result handling while preserving full checking.
2. A comparable dense/scoped ownership contract. Sparse root descriptors alone
   cannot make the dense width-eight complete-consume buffer fit.
3. Whether memory means a proved logical live-object envelope or an OS-enforced
   process/group limit covering reception too. Neither implies guaranteed success.
4. Any independently justified resource-policy revision. Existing mathematical
   checks already exceed 8M without representation overhead, so packing alone
   cannot fix that refusal. No new quota is inferred from these loose uppers.

Changes to runtime representation/enforcement or resource policy require explicit
scope. Real/native/GPU work remains prohibited. Capture/native proposal costs and
successful wall-time/RSS guarantees remain open, rather than being replaced by
another partial JSON audit.
