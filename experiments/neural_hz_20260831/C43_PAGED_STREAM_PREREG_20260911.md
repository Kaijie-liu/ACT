# C43: exact exported stream statistics in bounded numeric pages

C42v1 closed at its1M distinct-value Python dictionary cap; no complete
distribution was obtained. This NEW implementation changes the metadata
representation, not the frozen C42 cap or source artifact. No network/HZ is
decoded or constructed. Formal1870/2413,13-family baseline, E061/400, full
Neural-HZ goal and every progression/prohibition requirement stay unchanged.

## Material bounded representation change

Retain C42's Reader, opcode argument/framing logic, complete source byte hash,
STOP/EOF checks,128-byte LONG domain and64KiB read bound. No reducer/unpickler.
Replace the per-value dictionary with pages indexed by exact floor(value/256),
each an owned256-element uint8 array. Count ALL allocated slots/bytes, including
unused zeros. A slot stores min(frequency,16), because C42 exported only the
exact bins1/2/3/4-7/8-15/16+. Update these bins incrementally at their exact
boundaries. Total occurrences and distinct values remain unsaturated/exact.
The [-5,256] range remains just a labelled range, not an identity/cache proof.

The page table may contain at most250000 Python keys (STRICTLY FEWER than
C42's1M dictionary keys) and64000000 numeric slots/bytes under the unchanged
global64M entry cap. Thus dense domains can cover more values with fewer
Python objects, while very sparse domains fail EARLIER. This is not a larger
work/memory envelope or a relaxation of a failed source's eligibility.
Never drop overflow values, infer from a prefix or claim exact individual
counts above16; every EXPORTED total/category is exact or the census fails.

## Fixed pricing and evidence scope

CPU1/GPU0, AS16GiB, whole256M, worker240s/tests60s, both transient1GiB checks
including tracer metadata. No nested source branch/native/solver execution.
Retain512 header,8/opcode,1/LONG byte and16384 prepaid<=256 evidence events.
New prices:16/integer page lookup/summary update;256+64 per allocated numeric
page;16/first occurrence of a distinct value;256 final small summary. Charge
before lookup/construction/update. The full arrays are numerical metadata;
their allocation/zeroing is NOT free file hashing. Streaming opaque file IO
uses C42's unchanged reusable64KiB buffer and separate authentication timing.

Only this completed nonexecuting statistic is in scope. It is not an HZ score,
decoded heap census, proof of safe scalar sharing, full pickle stack proof,
actual source/new-state admission, checkpoint decrease or native payment.
Opaque inner archives are not inspected. No old archive/default/production
change, no actual decoder/generator/terminal, no family/shadow expansion.
Any cap/domain/time/gate failure closes this version without an unchanged retry.

## Tests and freeze

Counter15 tests0.95s:independent exact Counter oracle over random/signed/wide
values, all saturation boundaries, paid sparse capacity/default guards and
dense page geometry. Initial combined22 passed/1 fixture failure1.44s: its
protocol0 large bytes literal exceeded C42's original64KiB line domain.
The positive equivalence fixture was made within that SAME domain; an explicit
test now requires BOTH old/new parsers to reject the oversized protocol0 line.
No parser/domain rule changed. A1000001-distinct dense synthetic population
uses3907 pages/1000192 real numeric slots and retains exact totals, directly
testing the material change beyond the previous dictionary capacity.

Exclusive results/c43_paged_stream_census_20260911_v1. Freeze all dependencies,
new modules/tests/worker/supervisor/preregistration, C42 audit and C34/C42
anchors before actual stream inspection. Inherited+new tests must pass60s.
Persist complete measured stats/progress/results/exit; all old results remain
immutable. No C43 actual stream has been scanned during development.

Final combined focused check:24 passed1.46s; repeated pre-freeze check24
passed1.44s. No actual C43 stream was scanned during either synthetic check.
