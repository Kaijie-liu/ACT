# Gate margin relation finite native control

2026-10-20. One M3 synthetic control attempt under the standing authorization.
Starting implementation is `6161413cd`; the freeze binds the additional owned
schema check and this case/reception adapter. No trained model, input data,
GPU, support query or native optimizer is executed. M1b remains FAIL and M2
remains gated. This is a representation control, not a speedup experiment.

## Scientific question and fixed calls

Does adding two checked same-factor gate–margin inequalities exclude a negative
point in an existing F0 relaxation while retaining a positive exact LP lower
bound? Both arms use the same literal source, property, slice and supplied
gate/difference ranges. Only the additional rows differ.

Eight paired construction specifications, off/on once each, are fixed in
`scripts/gate_margin_rows_cases.py`. The order alternates by specification.
Four old-method reference constructions are additionally fixed: **20 actual
F0 constructions in total**, not 20 full verification requests.

| Case | Fixed purpose | Expected enhanced action |
| --- | --- | --- |
| P0, P1 | Positive margin on two closed half-intervals | Two rows each |
| N | Negative margin and reversed expert roles | Two rows |
| Z | Zero margin and constant half weight | Two valid zero rows |
| E | Equal experts | Rows added, output bound unchanged |
| Q | Shared/private continuous and binary factors, private rows before slice rows | Two rows with the binary offset checked |
| X | Margin crosses zero, no retained sign proof | Original F0 unchanged |
| R | Native binary64 gate center/radius fails outward embedding | Original F0 unchanged |
| B0–B3 | Four fixed quarter-intervals using old F0 | No new relation layer |

For P0/P1 the same actual stored F0 admits relaxed points with output `-1/16`.
The enhanced stored F0 should admit an exact residual-compensated lower bound
`1/16`, using two fixed dual multipliers. N has an old relaxed point `-3/16`
and the same enhanced bound. These points are not network counterexamples.
Z/E should retain bounds `1/2` and `1/8`. Q tests mapping, not positivity.

The stronger four-slice reference uses upper gate bound
`2303/4096 + i/16`, justified by
`sigma(1/4) <= 1849/3289 < 2303/4096` and the `1/4` derivative bound.
The expected stored LP minimum is `1/4096` on every slice, checked using a
fixed dual and a feasible point. It is not a tight-sigmoid native solve.
The separate tight-range two-slice separation and stronger four-slice
limitation remain in the [static audit](gate_margin_relation_static_audit_20261020.md).
This control cannot establish superiority over N1, H2 or GenBaB.

## Exact checks and fixed faults

The worker captures actual SparseHZ/F0 matrices. The stdlib receiver
rechecks source composition, identities, sign premises, added rows and exact
dual residuals. The analytical cases also compare the actual objective,
McCormick and retained-slice rows with the registered literal expectations.
Q checks its private-factor objective, equalities and retained inequalities.
It still trusts network/guard lowering, declared gate semantics and original
F0 lowering in the general API; this control does not discharge those premises.

The freeze lists exactly 28 faults and their required rejection messages:
schema/extra fields; request/domain/pair/property; gate semantics; active
source changes; router mapping and private metadata; missing slice/sign rows;
inward inequalities; lambda/product/binary columns; changed old output;
missing new row or control record; deadline binding; ledger reset;
expired Work; stored-bit overflow; malformed/duplicate CSR; binding digest
and size; independent-input mode.
Malformed structural bindings are rehashed so those tests reach the structural
check. An unrelated ValueError is a failure, not an expected refusal.
No extra F0 constructions occur in the fault list.

## Budget and reception

One new directory:
`/data1/Kane/MOE/results/gate_margin_rows_20261020_r1`.
One execution/one admission, no retry after unexpected failure.
The existing public `scoped_proof.owned_session.execute` and original cap
installer are reused unchanged. No observer injection, monkeypatch or new
supervisor is introduced.

The common deadline is controller entry +300s; worker cutoff is +240s,
execution +250s and cleanup +270s, leaving reception/publication within +300s.
All source construction, binding, F0 construction, serialization, worker
checking, parent reception and cleanup are charged. Timing starts at the
controller's Python entry; full OS/Python cold-start time is not claimed.
Stdout flush/final-exit duration is explicitly excluded from the measured
through-terminal-file-export number but remains subject to the hard deadline.

The whole batch shares 8M Work and 128 MiB artifacts; each managed process has
8 GiB AS and the existing session/controller 2 GiB sampled RSS policy.
The receiver resumes the worker's bound counts, never a fresh allowance.
All normal and fault prefixes are checked for continuity. Native instructions
not represented by Work still consume wall time. Fixed per-record native
construction timings are descriptive, not full-request off/on speedups:
the fixture binding preparation is separately charged once per paired control.

The outer receipt must be COMPLETED, return code 0, no execution/cleanup error,
and LEADER_REAPED_NO_LIVE_SESSION. A live-RSS observation failure remains FAIL,
even if the worker finishes. No exit delay, policy relaxation, offline promotion
or observer repair is allowed. Every prefix is retained under a distinct name.

## Continue or stop

Continue only if all fixed constructions, algebraic checks, faults, complete
reception and costs pass. That permits considering a separately frozen next
scope; it does not launch M2, trained-model development or GPU.
Any unexpected failure preserves the prefix and ends this attempt. A passed
control supplies a bounded mechanism check, not G1/G3 completion, full-request
benefit, a source-complete strict certificate or external competitiveness.
