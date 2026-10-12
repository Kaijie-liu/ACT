# Later affine layers retain checked coefficient references

The [protocol](hz_affine_references_design_20261013.md) was frozen and pushed at
`1163cf8ef` before implementation. Final execution code `39868aa4c` was committed
and pushed before the once-only two-source archive. [Compact results](hz_affine_references_20261013.json)
bind the code, captured sources, representations, terminal and external audit.

## Result and boundary

The new opt-in source schema eliminates expansion at **later** affine layers,
including in the independent checker. It preserves exact source-weight
references, independently rounded centers with checked error compensation,
separate guard rhs, and shared versus expert-private factors. A separate CPU
operator now handles multiple reference levels. V1/default execution is unchanged.

| Captured synthetic source | Local entries | References | Same-V2 expanded oracle entries | Parent seconds | Cumulative work / 8M |
|---|---:|---:|---:|---:|---:|
| mixed | 73 | 48 | 109 | 1.809612 | 4,401 |
| deep | 116 | 75 | 257 | 1.809151 | 7,941 |

The expanded counts describe the **same new representation**, not V1. Local
entries alone omit reference storage; mixed even has more local-plus-reference
entries than its expanded oracle. These counts do not establish a memory or
speed advantage. The finite native forward/adjoint differential covers 176 and
260 scalar results respectively; exact recursive and eager results agree.
CPU floating products agree within the frozen 1e-12 control tolerance, which is
not a safety acceptance threshold.

Both complete source representations pass relocated `python -I -S` checking
without torch, NumPy, SciPy, solver, checkpoint or historical run access. Mixed
consumes all 92 stored values (90 parameters plus two inputs); deep all 107
(105 plus two). Each retains three unordered pair guards. The deep graph has
two reference-processing levels and nonzero R², requiring accumulated adjoints.

**No V2 output-bound optimization or full output proof was run.** The existing
V1 complete endpoint consumer intentionally still rejects this new schema/handle.
No dense source, real model, solver or GPU was executed; all G1–G6 stay OPEN.

## What changes mathematically

For parent outputs `(c_j,n_j)`, each later affine row separately computes
`b + sum_j w_j c_j` and combines the coefficients referencing each `n_j`.
References can cancel while centers remain different, so centers are never
merged by node identity. Exact source weights remove intermediate coefficient
rounding from this path; binary64 center error still gets fresh checked factors,
with every row error allocated before any row lift. Input-diagonal affine layers
continue using the previous independently compensated native operation.

An overlapping-reference node uses the triangle radius
`B_n = ||local_n||_1 + sum_p |s_np| B_p`. It is a sound bound, not necessarily
the expanded exact L1 norm. ReLU decisions may therefore differ from V1 and
may be looser; factor inventories and output bounds need not match V1.
The independent checker rebuilds all source-driven references and compensation;
it does not call the producer or accept its reported norm.

For strictly lower triangular references R, the native operator computes
`C=(I-R)^(-1)L`, using ascending levels for forward products and descending
levels for transpose products. The latter propagates the **current accumulated**
adjoint, not just the original alpha. The exact control oracle independently
expands the same DAG. Native L and its transpose can hold separate value banks;
there is no single-physical-allocation claim.

## Controls, repairs and cost accounting

The archived suite passes 56 tests: 16 new source/operator controls, 17 V1
source regressions, 20 V1 endpoint regressions and three old audit controls.
Three additional stdlib archive tests check six evidence mutations and simulated
launch error/timeout terminal accounting over copies; no numerical requests
were rerun. Controls include shared-ancestor cancellation, independent centers,
zero weights, nonzero center lifts/error ordering, factor and expert binding,
guard/pair omissions, source suffix corruption, deadline/work refusal, and
producer/checker no-expansion enforcement.

The first development suite had 49/51 passing tests. Two test expectations were
wrong: the fixed deep graph has two reference levels, not more than two; its
nonzero center compensation appears in other later layers, not expert0/2.
Only those assertions were corrected. The fixed sources, tolerance and algorithm
were not changed to repair them. This review is retained in the archive.

Total parent-observed archive workflow time before terminal publication is
6.881851 seconds, including all five calls, imports, fixture creation, source
lowering/checking, exact and native differential, serialization, copying and
relocation. The two worker times before receipt publication are 1.397302 and
1.402317 seconds; the parent values above additionally cover startup and exit.
The source lowering native-only sub-times are nested in total build times and
must not be added again. Final terminal I/O follows the explicitly labelled
`outer_before_terminal_seconds`; no complete-request speedup is claimed.

The separate stdlib offline audit takes 0.042248 seconds. It recomputes sources
and exact oracle results, checks source anchors, native values, portable
identities, immutable archive hashes, workload counts, phases and parent costs.
Its result is outside the archive to preserve the original byte ledger:
`/data1/Kane/MOE/runs/hz_affine_references_20261013_r1_audit.json`.
The archive has 928,918 logical bytes (about 1.7 MiB allocated); the 16 MiB cap
includes its terminal. Storage remains 87 GiB free / 98% used; no scientific
record or checkpoint was removed.

## Capacity decision and next question

The existing dense prefix remains closed and unchanged. It consumes 7,202,670
counts. Keeping first-layer checking and the current complete endpoint consumer,
one native preparation already needs at least 800,622 more counts, beyond the
797,330 remaining. An independently sufficient refusal is the required expert-0
zero candidate: 157×3073 + 99×2 = 482,659 local terms, each with one visit and
two exact operations, giving **8,650,647** with the old prefix alone. This omits
other layers, expert 1, planning, proposals and final acceptance. It is a static
lower bound for the fixed one-pair/one-property/two-endpoint source, not a new
execution or a general scaling theorem.
The protocol's coarser 157×3072×3 bound excludes 355 additional local entries
(active-row compensation and unstable-ReLU output terms); the refined bound
includes them. The offline audit checks these recorded constants, not a fresh
large-graph derivation. No dense computation was hidden in the audit.

V2 solves the later affine expansion problem but not this full-query capacity
problem. Stop these small operator controls; do not run a large source-only
stage and present it as complete-query progress. Before another execution,
analyse a bounded source-row/factor-scale representation that can remove actual
per-weight work, plus identity-bound multi-target exact residual consumption.
In particular, retaining input diagonal scales until after shared residual
aggregation is a candidate algebraic direction, **not an admitted protocol or
implemented benefit**. Simply caching a native buffer cannot fix the exact
endpoint lower bound above. All original source and candidate checks must remain;
do not trust a producer's norm, change counting definitions or raise 8M.

For that prospective scale separation the full formula would be
`C=(I-R)^(-1)L D`, with each input factor's checked radius on D and scale one
for new error/ReLU factors. Equal scales must **not** merge distinct factors
before taking absolute values. ReLU also needs the weighted local radius
`sum_j |L_nj|D_j`, not the unscaled norm. Nonzero input centers still require
checked per-weight center arithmetic; clipped inputs can have different radii.
The fixed dense source's zero center and common radius are not evidence of
generic or trained-source capacity. These costs and every proposed/zero/final
endpoint evaluation must appear in any next admission calculation.

Convolution, correspondence with trained checkpoints, compatible V2 complete
proof consumption and full CPU/GPU costs remain open. Sealed requests, old
holdouts and the refused physical GPU experiment remain closed.
