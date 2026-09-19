# Four real LP diagnostics: native fidelity V2

Execution version: `6556b14ace261c2488bb404a30f5730bcfd266ea`, clean and pushed
before launch. All four original obligations were run once in the separately
frozen V2 directory. The original LP bytes, order, one-attempt rule, 300/298/218
second deadlines, native 10-second cap, 4096-bit rational cap and other sparse
limits were unchanged. V1's import failure remains sealed.

## Result

**4/4 LIMIT at exact rational construction; 0 completed independent feasibility
checks, 0 checked feasible upper bounds.** All four passed native import and
before/after readback, and mapped the returned basis to original coordinates.
No job was omitted, retried, given more time or assigned a different property.

| Input / property | Import and readback | Native call (s) | Exact constructor (s) | Recorded pivots | Terminal | Complete supplied-LP clock (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 220 / 0 | kOk, unchanged | 0.304683 | 0.906467 | 3760 / 4331 | rational bit LIMIT | 12.910765 |
| 222 / 1 | kOk, unchanged | 0.071899 | 0.862316 | 3051 / 4616 | rational bit LIMIT | 12.269820 |
| 230 / 2 | kOk, unchanged | 0.126139 | 1.319606 | 4939 / 6416 | rational bit LIMIT | 18.183476 |
| 232 / 0 | kOk, unchanged | 0.154412 | 1.282148 | 4081 / 6029 | rational bit LIMIT | 17.706483 |

Native and constructor timings are **nested** in the complete clock, not extra
cost to add. The constructor returns its own unresolved status normally, so its
process stage can be COMPLETED while its scientific result is LIMIT. Neither
means an exact feasible point was produced. Packaging/checking were not reached;
their missing durations remain null, not measured zero.

The import threshold is fixed at 1e-12 in this version. All originally submitted
coefficients, objective and bounds survive full binary64 readback. The native
logs still warn about small costs/row bounds during optimization; these are
retained verbatim. They are distinct from the old **import** kWarning and do
not establish any independent numerical guarantee. The native basis and result
remain untrusted. The capture stage still refuses non-kOk import or changed
readback rather than ignoring a warning.

## What is now separated

The original import-fidelity blocker is closed for these four submissions.
The current stop is instead the constructor's `rational bit budget`: an exact
intermediate value exceeded the unchanged 4096-bit numerator/denominator cap.
Recorded fill insertions are zero in all four runs. Peak stored basis entries
are 35,429 / 34,594 / 59,773 / 58,056, far below the two-million live-entry cap;
operations are 566,000 / 540,970 / 785,151 / 743,438, below twenty million.
These observations locate **this execution's** stop at arithmetic growth, not
wall-clock exhaustion, matrix fill cap or native optimizer time limit.

They do **not** establish that the exact solution itself requires more than
4096 bits. The frozen constructor does not save the offending value or a full
pivot trace; intermediate swell versus final solution size is not separated.
Nor do they show the returned floating basis has an exactly feasible rational
point: the original equality residuals, inequalities and boxes still require
independent checking after any reconstruction.

The read-only original-constant inventory reports maximum numerator/denominator
bit sizes of **144/142, 138/137, 142/140 and 154/151** respectively. These input
constants are well within the 4096-bit contract: the observed failure occurs
during elimination, not on initially admitting an oversized stored coefficient.
All original equalities are present and their residuals are nonbasic fixed-zero
anchors in these four returned hints; no equality was deleted to obtain a basis.
This narrows the arithmetic diagnosis without repeating elimination or claiming
that a different arithmetic algorithm would necessarily close the proofs.

All four native calls report Optimal and negative objectives (approximately
−1.926, −3.532, −1.365, −9.708). These are diagnostic hints only. No exact feasible
point was checked, so they are **not** certified LP upper bounds, do not prove
that the LP relaxation cannot certify positivity, and are not network UNSAFE.
The earlier checked lower bounds remain context, not a new primal–dual sandwich.

## Full cost and archival scope

Four request publication clocks total **61.070543 s**. Disjoint recorded phases
and the residual (startup, in-request review, bookkeeping and publication)
sum to each request's complete clock. Outside those request clocks, the runner
records 2.808383 s preflight, 0.000376 s resource checks, 17.724875 s post-terminal
audits and 41.156099 s automatic final audit. Independent archival review is
separately timed in the machine archive, not subtracted from execution costs.

These are supplied-LP diagnostics, **not full MoE requests**. Historical network
propagation, guard/range proofs and F0 construction were not rerun; their costs
are not claimed here. No performance comparison or acceleration follows from
these four short unresolved runs.

The first independent archival pass took 49.885372 s; a second fresh-process
reconstruction took 51.606879 s including its nine controls and complete hash
comparison. Both pass with zero issues and no new native solve, basis
reconstruction or bound-check call. The archive binds 121 files. These read-only
audit costs are additional disclosed work, not part of the request clocks.

The separate `fidelity_diagnostic_archive` reader independently reconstructs
the saved roster, terminal/cost arithmetic, complete before/after import
fidelity and raw artifact hashes. It also inventories original stored rational
bit sizes and basic/anchored coordinate counts without optimization or basis
reconstruction. Nine archive controls include hash/options/readback mutation,
missing denominators, missing cost, and rejection of native objectives as exact
upper bounds. A PASS here means archival consistency, not four feasibility
proofs. See the matching machine JSON for measured counts and identities.

## Next research boundary

Do not rerun this freeze, increase the bit cap, extend time, change pivot order
or treat native Optimal as exact evidence. The useful next development question
is whether a separately controlled exact-arithmetic construction (for example,
denominator-cleared/fraction-free or checked modular reconstruction) avoids
intermediate swell under an explicit resource contract. Such a method must keep
all original rows and bounds and still feed the unchanged independent checker.
Its benefit is not guaranteed: it may uncover an inexact/infeasible basis or a
genuinely large exact solution. Start with analytic controls and observable
failure metadata, not another unregistered real retry. No need to expand samples,
train models or reopen sealed CROWN searches.
