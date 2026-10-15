# Shared execution envelope: complete counted loops, remaining admission gaps

The shared implementation remains default-off and control-only. This follow-up
closes the **post-capture source/consumer Work counting envelopes** for both arms,
but does not admit the registered full topology. It also identifies a dense-arm
owned-buffer refusal and a report-demand gap under the old cap. No runtime
implementation, mathematical check, acceptance gate, resource ceiling or 300-second
budget is changed. No model, candidate, worker, solver or GPU is executed here.

The base is `82411e0b4045a7015b732bd549321ccffab3544d` on
`feat/moe-route-verification`. The [calculator](../scripts/assess_shared_execution_envelope.py),
[static tests](../scripts/test_shared_execution_envelope.py) and
[machine-readable envelope](hz_shared_execution_envelope_20261015.json) bind the
[prior refusal](hz_shared_source_decision_20261015.md) and its implementation
hashes. Prior implementation controls and frozen results are not rerun or edited.

## What is now bounded

These are conditional complete-traversal upper bounds for the bound dimensions,
assuming all required candidates are present and arithmetic/encoding checks pass.
The unchanged caps refuse earlier; these are **not observed executions, suggested
quotas, completion-time guarantees or measured performance comparisons**.

The source inventory uses input dimension `n=3072`, affine outputs `A=3288`,
ReLU units `U=3200`, first/later affine weight positions `W1=6684672`,
`W2=273408`, nodes `N<=19496`, factors `F<=32048`, references `R<=277280`,
views `V=82`, guards `G=336`, and expanded memberships `H<=136232`.
Explicit locals have upper `l=n+2A+12U=48048`; logical locals have upper
`L=W1+l=6732720`. This last identity is between conservative bounds:
actual logical nnz equals explicit nnz plus **nonzero** aliased weights, not
the number of dense weight positions.

### Source construction and two independent checks

Let `P=W1+W2+A`, and `A1=2176` first-affine output rows. The following table
uses `h<=H` stored sequence entries; the report substitutes `h=H` conservatively.

| Source stage | Row visits | Coefficient visits | Recorded exact operations |
| --- | ---: | ---: | ---: |
| One Snapshot construction | `A` | `2P` | `0` |
| One complete Snapshot replay | `A` | `W1+W2` | `0` |
| Builder equations | `F+N+U` | `2W1+W2` | `n+W1+2W2+2A+9U+G+3L+R` |
| Encode | `H+2N+h` | `l+R+F` | `0` |
| One intake | `3N+A1+h+H` | `l+R+F` | `0` |
| Checker initialization/norms | `V+N` | `L+R+2W1` | `3L` |
| Checker equations, excluding replay | `F+V+U` | `W1+W2` | `n+W1+2W2+A+9U+G+R` |

There are three Snapshots/replays: builder, production checker, fresh portable
checker. Encode runs once; intake runs twice. Intake's `3N` includes descriptors,
root decoding and RootBank validation. SourceLocal scans **all** first-affine
positions, including zeros. Norm grouping does not assume equal factor scales.

| Source subpath | Work upper |
| --- | ---: |
| Build, including Snapshot and replay | 62,330,600 |
| Encode | 668,832 |
| Each fresh independent checker | 76,712,652 |
| Build + encode + two checkers | **216,424,736** |

Source file re-sealing is counted in each consumer's finish below, not twice.
Source transfer, page I/O, hashes and publication still cost wall time even where
the current Work instrumentation only calls `tick()`.

### Complete consumers, fixed proposal and both frames

Set `J=L+R=7010000`, `B=28` pairs, `D=252` properties, `E=504` endpoints,
`T=84` pair-local tiles, `M<=2700` dual rows per pair, classes `C=10`.
For check blocks, `K=T` in dense and `K=28*(4+4+1)=252` in scoped, from
candidate widths 8/8/2 and checking two at a time. This is the registered upper
roster, not a formula for arbitrary actual endpoint counts. `Q` counts source
chunks including center; `Q<=4096` is a conservative member-policy bound, not
a measured chunk count or a new chunking protocol.

| Consumer stage | Dense | Pair-scoped |
| --- | --- | --- |
| RootTable | `4H+3V+1` | same |
| Plan | `12+B(N+W1+3J+4F+4)+D+E(1+2C)` | same |
| Scope construction | `0` | `T[4(N+F)+4]` |
| Objectives | `E(3N+2C+46+W1+6J)` | `E(5N+2C+58+W1+11J)` |
| Consume excluding dual pullback | `E(36+8M+21F)+3MK` | `E(36+13M+32F)+3MK` |
| Dual pullback | `(6E+K)N+2E+KW1+(K+5E)J` | `(4E+K)N+E+KW1+(K+10E)J` |
| Finish, including seal and receipt identities | `E+D+Q+9` | same |

The upper includes buffer get/set checks, SourceLocal zero visits, each tile's
scope rebuild, every present endpoint's full merge and both norms. It deliberately
uses global bounds rather than assuming tight active pair closures. The large
numbers must not be interpreted as the scoped path being empirically slower.

Fixed proposal adds `E(33+2M)` only in production. Production frame encoding,
sealing and validation adds `3T+5E+4ME`; replay intake/sealing/validation adds
`3T+E`. Both candidate roles are retained. Binary64 and general-rational profiles
share these Work bounds, **not** their byte demand or bit/time costs.

| Bound component | Dense | Pair-scoped |
| --- | ---: | ---: |
| Production consumer + fixed proposer + frame | 44,612,645,184 | 82,450,188,984 |
| Independent replay consumer + frame | 44,604,461,736 | 82,442,005,536 |
| Above plus source build/encode/two checks | **89,433,531,656** | **165,108,619,256** |

These close the instrumented loops for this **fixed-proposer, captured-source**
subpath. They do not cover a future native capture/proposer. Many Fraction,
string, hashing, I/O, container and cleanup operations do not increment Work;
thus a finite counted-work upper is not a full computational-cost upper.

## Dense ownership is still a separate capacity problem

The shared path now uses the same at-most-four-root objective descriptor in both
arms. This removes the old dense objective-root allocation but not dense dual
accumulators and their pullback copies. Working scalar slots remain 2065 bytes;
an eight-objective all-present consume retains `2k(N+F)` such slots (`k=8`).
Even `N>=n+A+G=6696`, `F>=n=3072` gives

    2*8*2065*(6696+3072) = 322,734,720 bytes > 128 MiB.

This is a conditional **owned-buffer lower bound**, independent of the Work
refusal; it is not RSS. Including upper roots/descriptors and the binary64 tile,
the conservative dense arena upper is 1,704,299,744 bytes. Root internment alone
does not make this reference arm fit. The report gives preparation, descriptor
sealing, frame sealing and consumption separately. It excludes upstream objects,
caller buffers, Fraction temporaries and allocator-retained memory.

The previously bounded pair-scoped arena must not be confused with a complete
two-arm or whole-process memory contract. This stage neither increases its cap
nor silently replaces dense's registered consumer strategy.

## Results and supervisor reports require their own envelope

At the existing 8192-bit work bound, one signed reduced Fraction string needs at
most `2*2467+2=4936` characters. The closed 252-row result schema has upper
`233+252*(52+4936)+251 = 1,257,460` bytes. This exceeds 1 MiB but is **not an
observation of overflow**: actual results can be much shorter.

More importantly, the child `candidate.json` embeds both the production result
and `replay.result`. It is not just a reference to `result.json`. Any *actual*
result exceeding half the 1 MiB report cap would make that report exceed the cap
before other metadata. The parent receipt and saved result also each hold a full
copy. Endpoint receipt strings are in-memory and must not be invented as extra
on-disk members. No report has been silently truncated or changed here.

The phase/progress skeleton is bounded separately: successful `T=84` execution
has 186 phase rows and 368 progress lines. Under the existing seven-digit Work
cap the progress upper is 58,880 bytes. For the conditional full-loop upper,
twelve counter digits are used instead; the report preserves **both** calculations
and does not quietly substitute a larger quota. Whole report demand remains
open until archive events, all snapshots, hash maps, source layout and manifests
are included. Fixed file caps are rejection limits, not evidence that arbitrary
valid results fit.

Successful core file-read accounting remains online `17Tsrc+2X+9(S+G)+9Cbytes`,
parent `4Tsrc+4(S+G)+4Cbytes`: source tensor payload includes center `X`, forest
payload is `S`, page index `G`, candidate stream `Cbytes`. Metadata/code/imports,
in-memory hashing, future checkpoint/live-source capture input I/O and later
offline audits are additional. These are logical file-byte reads, not physical
device I/O. Single stored source does not imply a single read.

## Stop/continue contract

No current full-shape attempt is justified. Proven refusals remain Work and
expanded memberships, now also complete-width-eight dense ownership. Oversized
candidate/report **uppers** separately show missing capacity guarantees, not
inevitable overflow. None of these values is a proposed new resource limit.

The next design must bind, together:

1. The captured-source identity and bounded chunk/header layout, separately from
   a future charged live checkpoint capture and native proposer.
2. Candidate profile, both roles, same complete obligation roster, production and
   replay strategy. Restricting profiles or changing replay/counting is a new
   contract, not an equivalent packing fix.
3. The complete vector of logical visits/memberships, owned arena, transient
   objects, metadata, per-file/member/archive demand and actual RSS enforcement.
4. The common 300-second deadline through reception/cleanup, with every failure
   and partial artifact retained. A cap can enforce refusal without guaranteeing
   successful completion.

Remaining open items are whole generated archive/metadata demand, live-object
and RSS bounds, native capture/proposal, and any completion-time guarantee.
Do not copy these deliberately loose Work uppers into production quotas. Runtime
or quota changes and real/native/GPU execution still need separate scope approval.
G1–G6 and the historical scientific source gaps remain open.

## Checks and review

Three read-only agent reviews independently inspected source accounting, consumer
ownership/counting, and report/supervisor boundaries. They are code reviews,
not a machine proof of the upper bounds or independent human review.

First static suite: 16 tests attempted, 9 passed and 7 errored at one new-script
field mismatch (`guard_rows` versus frozen inventory `guards`). No runtime
construction was reached. The calculator was corrected, not the frozen input.
Second suite: **16 passed** (7 new arithmetic/identity tests, 9 existing ledger
tests). This is not another implementation performance run or 16 new certificates.
Tests cover phase arithmetic, seal non-duplication, actual descriptor/dual
ownership, both report copies, source binding mutation and unchanged authority.

Reproduce without site packages:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_shared_execution_envelope scripts.test_shared_source_decision scripts.test_shared_source_cost
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m scripts.assess_shared_execution_envelope --check-report docs/hz_shared_execution_envelope_20261015.json
```
