# C119 — positive complete ordinary HZ, failed aggregate qualification

Status: v1 CLOSED. Full research goal ACTIVE. Formal gain 0. No target, solver,
default, production edit, commit, push or benchmark promotion occurred.

## Results actually obtained

All 146 files / 3384 tests passed. Combined collection and execution took
51.812539003789425 s under the unchanged60s gate; pytest reports40.68s and13
inherited warnings. The 21 new tests include independent actual polynomial
elimination, inverse extension, shared IDs, masks, scaling, both binary branches
and corrupted-packet rejection. No old test was modified or skipped.

The two fixed complete C16/K32/h6 sources were freshly generated. Every512
original output equation was bound exactly in original source coordinates.
Every actual F4 definition was independently proved, including72 bilinear basis
coefficients, full output coverage, odd denominators, native binary64 literals,
redundant auxiliary boxes and original output pivots. Complete C91 row, owner,
map, source-binary/EQ/INEQ preservation and inverse checks passed for both F4
states. These are actual SparseHZ states, not packet-size estimates.

| Complete source | Direct numeric bytes / entries | F4 numeric bytes / entries | Actual Ac nnz, direct → F4 |
| --- | --- | --- | --- |
| Dense | 1095680 / 94353 | 820800 / 75993 | 76417 → 40961 |
| Historical masked | 863560 / 74626 | 786632 / 72762 | 57697 → 38737 |

F4 saves274880B/18360entries in dense and76928B/1864entries in masked. Both use
1728 new continuous factors and retain the original binary factor, EQ/INEQ,
shared source/frame and all512 old scalar inverse equations. Complete predicate
nnz including binary/inequality coefficients is76996→41540 and58132→39172.
Known metadata grows by2209B per F4 state; inherited opaque expression identity
is the same before/after. Numeric byte reductions are not total RSS reductions.
The exact inverse test point is an ordinary box point, NOT a feasible network
witness; universal feasible-set preservation follows from source/row/box proofs.

The unchanged C96 four-F2 comparator was also constructed and fully proved.
Dense F2 has2560 new factors and complete950528B/88753entries, so actual dense
F4 is smaller by129728B/12760entries. Masked F2 has per-packet entry increment
+2312 (four packets, +9248 total); with the unchanged old131072 reserve it is
rejected BEFORE an actual complete F2 state is installed. No physical masked-F2
state is invented. C113's once-prepared variant removes8 V factors per packet
and reported+2160entries; it is a DIFFERENT comparator, not a contradiction.
Both F4 complete-packet entry bounds are negative; shared aux/emission and
positive-entry gates pass. Declared whole emissions are799744 and768768.

## Why this is not a qualified advancement

After BOTH complete case reports and packet archives had been saved, the final
combined held-root walk exhausted the single256M diagnostic budget:

`used=255999988, requested=16, cap=256000000`

The failing operation is `c62_complete_numeric_header_walk`. The combined
numeric-owner/64M-entry/metadata ledger did NOT complete. Its missing result
cannot be replaced by selected case numbers, an unpaid union, a reset pool,
refund of reserved source work, dropped roots or an unchanged retry.

Each case's own full held-state ledger completed, but that is a narrower result.
The entire measured body retained both cases and passed BOTH transient gates:
RSS growth169906176B; trace peak73262087B plus tracer metadata48637568B. Measured
body25.20052471011877s, worker28.204405788332224s; build returnedFalse. Supervisor
87.34951049275696s, session94206 exit1, all owned jobs terminal. This was a work
cap failure, not an observed memory excess or algebra failure.

The pool includes64M nonrefundable old-source reservations,21233664 new F4
constructor-transform units,12845056 independent F4 kernel-proof units,
44564480 old-F2 polynomial products and both partial/final evidence encoding.
The measured C97 generation UPPER bounds1038508 and857788 do not refund the
preregistered32M-per-source reservations. Fees are not wall-clock CPU counts.

Source/production drift=False; both frozen checks passed. Authentication traffic
was4300 source-file hash calls/19352870090B and12 evidence hash calls/6889472B,
separate from generation tokens. Historical artifacts are unchanged.

## Retained evidence and next boundary

Exclusive directory: `results/c119_denominator_f4_20260922_v1`. Contains complete
collection/JUnit inventory, tests, fatal log, phase events, all10 pre-proof
constructor packet archives, two complete packet archives, both complete source
reports, failed worker result and supervisor exit. Pre-proof JSONs remain
explicitly UNPROVED; later proofs are in their complete source reports.

- preregistered SHA256:08cf8bb383c99664616b16930f1c7036f29f4459756cb566a52b2cc1d7a253df
- dense report SHA256:930a705a183b15459f9bf73c8f71c5f458fc19b72984fcbd40ef5022d3814c85
- masked report SHA256:43c9382208a7e668905fa1793c14436d6c5c49e0bc9c02011adf3a717b1a7bee
- worker result SHA256:185235a4634fbee10ee59e981b6511fcf8286da7d3e07aa91e4700873d1e5147
- failed exit SHA256:adf8b07777e65b1022a1939870e109de66187d6e168d272b680ed9e1fbdbfdcd

Next: materially reduce the NEW exact F4 numerator construction using a proved
separable integer program, while keeping this independent oracle, both ordinary
guards, all proof/owner/inverse roots and complete combined ledger. A complete
new work bound is required before numerical execution. See the focused handoff.
Do not retry the unchanged target, change caps/prices, drop proofs, substitute
an easier mask or promote these fixtures as CIFAR/Tiny/network solves.

Formal baseline remains1870/2413, all13 family counts protected. Separate E0
remains CIFAR10025/200 and TinyImageNet36/200. No full2413 or400 replay was run.
