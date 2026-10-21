# A2 strong comparison stops the fixed tilt mechanism

The once-only synthetic comparison completed all nine arms and78 native LP
calls. Its frozen scientific gate returned `STOP_CURRENT_FIXED_TILT_MECHANISM`:
finite tilt did not improve the checked bound over either independent zero tilt
or matched zero certificates. This is a negative mechanism result, not a model
counterexample, full-network certificate or general impossibility theorem.

Execution revision `c5544fa6b94bfd33c563c13970b171128164649d` and
[frozen protocol](margin_a2_control_20261021.md) bind the experiment. The
[lossless deduplicated archive](margin_a2_records_20261021.json) preserves every
run file, including relocated code/data, native events and failed fault inputs'
rejection records. Freeze SHA256:
`c48b76d91ff802f816a8d805ff43be2f64ca1ee88ed9452c9f5f9bf9d1fac931`.
Original terminal SHA256:
`85d1b45f84e506580c75576f1cac3ceeb7bb1d295f8a070d2e0af422c6d048ae`.

## Bounds and charged costs

One unchanged3,079-factor synthetic HZ, one full ten-class property and common
continuous P were used. Each value below is independently checked on that
given source, not on a trained network. Higher lower bounds are better.

| Arm | Checked lower | Native calls | Shared-cost allocation through accounting cut |
| --- | ---: | ---: | ---: |
| zero | −10521/8192 = −1.284301758 | 2 | 5.2045s |
| finite | −10521/8192 | 4 | 6.3251s |
| transfer | −10521/8192 | 0 additional,4 originating | 7.2262s |
| endpoint_2 | −10521/8192 | 4 | 6.1280s |
| N1_2 | −10521/8192 | 10 | 11.9082s |
| M3_2 | −10521/8192 | 10 | 11.9483s |
| endpoint_4 | −41/32 = −1.28125 | 8 | 7.8973s |
| N1_4 | −41/32 | 20 | 19.4838s |
| M3_4 | −41/32 | 20 | 19.5113s |

Finite gain is exactly0, below the preregistered1/1024 threshold. Four-cell
references improve it by25/8192≈0.003051758. Two-cell endpoints tie finite
and have slightly lower observed allocated cost. The decisive zero-gain stop
does not depend on this small timing difference or on the pessimistic terminal
tail used in the cost gate. No repeated trials or population inference follows.

Finite's four local cell gains over matched zero are0,0,3/2048,19/4096.
The last two improve, but the bottleneck cell does not. All eight transferred
endpoint residual-cancellation gains are0. Local improvement therefore did not
become complete aggregate improvement. All78 proposals retained native duals;
none selected the checked zero-dual fallback.

All12 product cells used the product path; no N1 sign-stable endpoint query
was skipped. In every product cell, the independently paid endpoint fallback
was stronger than the product bound. Accordingly N1/M3's aggregate equality
with endpoints is not an unqualified success of their product relaxation.
M3 has one improved raw product cell in the two-cell grid but no aggregate
improvement; all raw product certificates remain preserved.

These results do not establish exact LP optimality or that the weighted
property is false. HiGHS status0 only proposes duals; the original-coefficient
residual checks establish the reported lower bounds, not feasible upper bounds.
The earlier weaker-candidate/grid experiments remain separate; they do not
override this same-P strong comparison or establish an inherently tighter domain.

## Execution and evidence audit

Original terminal PASS and outer exit0 were observed. Entry-to-terminal export
was64.762928s under the shared300s hard budget; Python startup before the first
timestamp and the final stdout/exit tail are not claimed as full cold-start cost.
Worker elapsed before its export was41.832695s; controller reception10.809762s;
relocated stdlib reception before export11.079627s; copying0.007181s. These are
nested/component scopes, not independent request timings to sum indiscriminately.

The measured accounting cut was64.757664s. Disjoint per-arm variable times sum
to61.250520s; shared cost3.507144s is charged in full to every standalone arm,
and transfer also pays finite's originating variable cost. The table is this
shared-setup allocation, not nine independently measured cold-start executions.

The frozen conservative cost gate additionally charges the unused235.242336s
as an upper bound on terminal tail, equally to all arms. It is **not measured
runtime**. Both this sensitivity calculation and the through-cut calculation
return the same stop. The raw field `terminal_tail_in_cost_gate=false` is a
misleading label: the actual *measured* export tail is absent, but the recorded
`decision_costs` do include the frozen upper bound. The original field is
preserved; the audit reconstructs the formula rather than trusting that label.

All156 intent/return records bind to the expected original LP/objective/cell
roster; all78 returns are status0. Worker, controller and relocated receivers
agree on every arm. Four new accounting faults were rejected. Cumulative Work
continues44,668,546→60,376,366→76,084,186; it is account-only under the registered
development policy, not an8M admission claim. The two receptions each add
15,707,820 visits. Exact stored/intermediate bit limits are unchanged.
The78 native solver calls total0.650395s; the much larger batch time includes
all source construction, serialization and repeated exact reception. Query
counts alone would not describe this implementation's complete cost.

Both owned leaders were reaped with no recorded live session members;
FDs4→4. Direct-child `wait4.ru_maxrss` is744,808,448B for native and34,643,968B
for portable reception, not a process-group simultaneous RSS bound. Every
managed process retains8GiB AS. No GPU, trained checkpoint or dependency change.

Saved-only `python -I -S -B scripts/audit_margin_a2_20261021.py` passed in about
11.11s, rechecking original coefficients, query identities, all bounds, portable
hashes, continued Work, durations and decision arithmetic. This audit cannot
repair an online failure. Retained run files1,833,495B plus compressed lossless
archive260,546B total2,094,041B, below128MiB. Portable checker source is included
in the deduplicated archive; the producer and frozen protocol are versioned at
the execution commit. Rerun source-hash audits at that revision.

Two separate read-only AI reviews confirmed execution/accounting and independently
rebuilt78 original-coefficient candidate bounds plus8 transferred endpoint
residuals using standard-library rational arithmetic. The latter also rebuilt
the product rows and finite boxes. These are independent implementation checks,
not the required external human technical review. No further solver call occurred.

## Research decision

Stop this registered fixed weighted-median-tilt mechanism. Do not promote it
to A3 real development or trigger A5 external deployment on this evidence.
Do not change this source, grid, tilt choice, query allowance or time and retry
the experiment. The stopped question is this concrete mechanism, not all
possible margin-conditioned support representations.

Any next algorithm proposal must identify a distinct lost relation or value-
function approximation, state how it differs from existing slice/dual-transfer
capabilities, and establish a falsifiable mathematical reason before another
experiment. Do not substitute tuning, another supervisor, more controls or a
larger source for that reasoning. G1–G6 remain OPEN, real-development ledger
0 stages/0 calls/0seconds; historical source gaps and sealed boundaries remain.
