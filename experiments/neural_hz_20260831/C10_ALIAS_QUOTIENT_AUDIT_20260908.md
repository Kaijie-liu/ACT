# C10 exact quotient V1: actual component passes; live path still unproved

2026-09-08, redu-hz. Exclusive completed transaction:
results/c10_alias_quotient_20260908_v1/. Sources and provenance were frozen
before the first real target execution; no source or provenance drift.
37 new proof tests plus 405 inherited tests: **442 passed in 4.67 s**.
Supervisor/worker exits 0; supervisor 26.922976199537516 s, worker
20.26559586916119 s. The development fixture correction is separately recorded
in C10_ALIAS_DEVELOPMENT_NOTE_20260908.md; it was made before the target freeze.

## The mathematical change now exists, not just a census

The uniform independent frontier selects 100,603 of the 100,727 locally exact,
window-safe aliases. It blocks 124 candidates due to selected-child/parent
dependencies; no ratio chains or benchmark-identity choices are introduced.
The rewrite substitutes 227,185 occurrences in 37,500 equality rows and erases
exactly 100,603 homogeneous defining rows. On this actual target no parent-column
collision occurs; exact/inexact sibling collisions and cancellations are
covered by the independent toy tests, not falsely claimed as real-target hits.

Independent Fraction audit checks every erased definition and all 37,500
changed rows, plus 109,137 unchanged rows. All 143,937 surviving equalities and
2,700 inequalities are checked. Every eliminated variable has the exact unique
extension x_j=r*x_parent with |r|<=1, and no selected parent is eliminated.
The original input prefix and all 1,350 binary factors remain intact. No phase
split, binary pivot, convex replacement, solver status or rescue path is used.

| Metric | Original component | Quotient plus reconstruction |
| --- | ---: | ---: |
| Raw global continuous frame width | 255,298 | 255,298 |
| Ordinary lowered continuous factors | 254,965 | 154,362 |
| Binary factors | 1,350 | 1,350 |
| Equalities | 244,540 | 143,937 |
| Total HZ coefficient nonzeros | 11,201,930 | 11,000,724 |
| Unique resident numeric bytes | 138,382,224 | 137,577,400 |
| Retained numeric entries | 11,449,370 | 11,549,973 |

The 100,603 selected global slots are UNUSED, not renumbered. Ordinary lowering
prunes them through its existing source-column maps. The certificate owns four
explicit arrays totaling 3,219,296 bytes; it contains no reference to the full
original HZ. Net numeric byte decrease is only **804,824 bytes (~0.58%)**.
Numeric entries INCREASE by **100,603**, because four reconstruction entries
replace two coefficients plus a RHS per removed definition. Thus a roughly
39.5% reduction in lowered continuous variables is NOT a comparable memory or
nonzero reduction; the coefficient reduction is only 201,206 (~1.80%).

## Resource and ingestion evidence

Frozen logical component work upper 236,441,920 <256,000,000. This includes
118,064,528 census work, eight additional payload-pass charges, MAIN/frontier
work, 227,185 exact products and sorting upper 1,610,232 with the registered
multiplier. Measured construction 6.124202028848231 s; conservative resident
growth upper 290,430,976 bytes; traced peak 292,129,030 plus 67,808 tracer metadata.
Both unchanged 1 GiB construction checks pass. Fraction audit 3.210275238379836 s.

Native ingestion after the proof retains **10,960,724 / 10,960,724** predicate
matrix coefficients, zero differences, unchanged row/column bounds and
integrality. Value-map Gc's 40,000 nonzeros are separate from that matrix count.
Native model has 154,362 continuous and 1,350 binary variables. No presolve or
solve was invoked. Construction-report native_ingestion_executed=false describes
the constructor itself; the enclosing result's native_ingestion records the
subsequent read-only ingestion check. Neither is a solver result.

The diagnostic process peak was 2,888,432 KiB with BOTH source checkpoints,
independent oracle and native model present. This is not candidate-only peak
memory. The numeric ledger exposes five roots; Python shallow measurement is
3,224,568 bytes (includes array object sizes and is not additive to numeric
owner bytes). Python allocator overhead remains outside the numeric gate.

## What this does NOT qualify

This is an offline exact component, not a fresh-network live publication or
capability run. Original checkpoint HZ contents were verified unchanged.
Original/native prefix caches were not deleted or replaced. Production defaults
and existing source candidate digest are unchanged. No CIFAR target, same-
structure cohort, 13-family or full 2413 replay is claimed.

Naively appending this component's registered 236,441,920 work to the prior
C9 234,443,780 bound gives 470,885,700, beyond the unchanged 256M whole-path
ceiling. This is not a runtime measurement but it rules out certifying that
composition using those bounds. Also, retaining the old full C9 HZ behind live
proof objects would negate the intended storage saving. The next substantive
work must be a fused definition-emission/alias representation with a complete
work proof and compact reconstructible lineage; it must not simply add a
postpass, drop proof storage, discard native caches or loosen thresholds.

A fused candidate still needs the same all-row independent proof, original
input recovery, native ReLU/global phase slots, complete reachable numeric
roots against the frozen comparator, and a fresh ordinary terminal run under
unchanged budgets. The existing C9 terminal UNKNOWN remains closed, not rerun
with extra time. Formal **1870/2413**, independent historical-origin E0
**61/400**, new formal/capability gain **0**.

## Sealed artifacts

- result.json SHA: 2061284868f51ef33ebbae51d3f5c6497b6eb6b404092675793d974235fdbea5
- quotient.pickle: 137,601,322 bytes, SHA
  bce0ba6ce9d1a75b9a97e5de585d4857ad8e827f8c171f26f3361b99b7d73d15
- quotient HZ content SHA:
  7d32e48d3c9b24360e83ae35c72e27fc29ace03b04f1f5763f006899585ba8f6
- original HZ content SHA (unchanged):
  b2024dfbe2af20f7c9e729bf79c835d5cf7b2050ab57113cc33bebe78f9801c8

All processes in this transaction have exited. The checkpoint contains only
the quotient, certificate, scalar proof, source hashes and provenance. Logs,
events, preregistration, test results and exit hashes are preserved.
