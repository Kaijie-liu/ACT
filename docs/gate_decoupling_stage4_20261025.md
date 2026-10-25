# Hard-core gate decoupling — stage4

This is the fifth revision's diagnostic, not a new confirmation cohort or a
claim that changing engines is an algorithmic contribution. The scientific
question is whether replacing gate equalities by narrow independent ranges
allows the strong engine to close histories it previously did not close.
The opposite outcome must remain visible. No real stage4 call has run.

## Same-history graph and scope

For the canonical selected pair `(a_l,b_l)` at layer `l`, write
`m_l=r_a(h_l)-r_b(h_l)`, `g_l=sigmoid(m_l)` and
`h_(l+1)=E_b(h_l)+g_l(E_a(h_l)-E_b(h_l))`; the residual family also adds `h_l`.
Both static compilers retain the same routers, experts, original input, guards,
properties and affine final head. All classes and property constants remain.
The implementation is opt-in in
[gate_decoupling.py](../act/back_end/moe/gate_decoupling.py); it is not wired into
the live1b queue or an automatic production fallback.

Original-prefix CROWN records give the two ordered upper bounds
`U_ab >= r_a-r_b` and `U_ba >= r_b-r_a`; therefore the canonical first expert's
margin range is `[-U_ba,U_ab]`. Source identity must bind the original model,
materialized box, properties and source version, and the record must match the
exact history prefix. Missing, nonfinite or inverted needed directions do not
become a fabricated range. The helper owns immutable tuples, so changes to the
caller record cannot mutate an already constructed gate domain.

V2 replaces each gate equality by an independent interval variable. Every
original history assignment embeds by setting those variables to its original
gates. Later gate ranges need to cover original executions, not every artificial
state introduced by earlier decoupling. Guards are evaluated at each state in
the new graph; an original legal assignment still satisfies them. This proves
mathematical outer containment assuming valid original ranges and lowering.
The implementation uses floating sigmoid plus `nextafter` padding and numerical
CROWN bounds. This **does not prove** a correctly rounded transcendental envelope
or source-complete containment of the original network; accepted results stay
at the registered numerical-policy level.

For fixed original input and all earlier independent gates, the final output
property is affine in the last gate, including the registered residual addition
and affine head. The final-layer guard depends on its input, not that last gate.
Thus two endpoint implications `guard => qF+c>0` imply the same property for the
whole last-gate interval. Each property is compiled as
`max(qF+c, all outsider-minus-selected atoms)>0`. Ties give zero guard atoms and
remain obligations. The two endpoint graphs retain every property and share one
eventual V2 call budget; a missing/nonpositive endpoint cannot establish closure.

Earlier gates cannot in general be endpointized: a later ReLU can implement
`abs(g-1/2)-1/4`, positive at both endpoints but negative at the midpoint.
Consequently multilayer V2 retains its inner products and the nonlinear engine;
only a single-layer V2 endpoint graph is purely affine/ReLU. A nonlinear final
suffix is rejected by this implementation, not silently linearized.

V3 fixes every gate to its interval midpoint. It is an affine/ReLU counterfactual,
not a subgraph or outer envelope of the original function. Its receipt says
`DIAGNOSTIC_CLOSED`, never original-history closed. A relaxed/frozen violation
also does not establish original UNSAFE without full-model replay. Fixing gates
can change downstream guard reachability, including making a history vacuous;
V3 closure alone is not a unique diagnosis of easy experts. Likewise V2 failure
does not by itself distinguish relaxation weakness from search/guard/budget.

## Controls prepared; no numerical execution yet

The [R1 control freeze](gate_decoupling_control_20261025_r1.json) binds15 source
files,10 semantic/receipt test methods and exactly3 external API calls. Static
parsing and source-hash checks pass; none of those10 tests or3 API calls has run.
This is a prepared prototype, not a passed native-integration stage.

The control batch uses the existing owned-session supervisor, existing isolated
GenBaB CPU environment, unchanged engine configuration and one shared300s/128MiB
budget with8GiB per-process AS. No checkpoint, training data, GPU or installation.
There is no new supervisor. The
[fixed controls](../tests/test_gate_decoupling.py) cover direction/ties/range source,
missing/inverted/nonfinite bounds, alias isolation, all properties and both
endpoints, late reception, L1/L2/L3 graphs, residual/affine suffixes,10 classes,
property constants, midpoint-positive/original-negative behavior, and the inner
endpoint counterexample. Three API queries use one analytic-positive two-layer
fixture (V2_LO,V2_HI,V3) under the same control deadline. They test API semantics,
not speed, trained-model success, or the registered stage4 statistical gate.

Unexpected failure stops the control batch and retains its files. Only tool
defects may be repaired under the existing user policy; no changed acceptance
threshold or reset of the cumulative budget. Native controls wait until1b ends,
so this diagnostic preparation does not add a competing numerical process.
Static parsing/review is not recorded as native-control passage.

## Real diagnostic preparation

The [read-only source inventory](gate_decoupling_sources_20261025_r1.json) now
binds the79 already defined1a/B2 objects. It is **not an execution freeze** and
does not include the still-running1b cohort. Its generator is
[gate_decoupling_catalog.py](../scripts/gate_decoupling_catalog.py), using only
the standard library; checkpoint/tensor contents are not loaded. A separate
read-only cross-check passes619 distinct raw-file hashes, the55+24 denominator,
canonical product ordinals and all24 B2 pair/property rosters. Catalog SHA256:
`ae03542cc691ece2da75629431b1af7ffe066b75f472d17968a35f7298e147e1`.

| Cohort | Saved returned-unproved history | Saved first in-flight history | Saved pending/unvisited only | Earlier identity missing |
| --- | ---: | ---: | ---: | ---: |
| 1a,55 objects | 3 | 29 | 13 | 10 |
| B2,24 objects | 21 | 2 started-without-return | 1 | 0 |

The two B2 interrupted starts are retained ERRORs (seed0/4237 pair5,6 and
seed2/4374 pair0,3); seed1/4212 pair2,3 was unvisited after a verified prefix.
These are not24 mathematically unsuccessful single-history solves. Three B2
objects also have previously saved witness records, kept in the inventory;
their selected history need not be the witness's history. No new replay was
performed here and no local positive can become whole-request robustness.
Only3/79 have reusable complete prefix-range records;76 need charged range
reconstruction if a history can be identified. This distinguishes preparation
cost/missing provenance from expert or gate difficulty before any V2/V3 result.

The hard-core catalog remains1a55 plus B2 24 plus the pending1b nonpositive set;
known counterexamples, tool errors and missing sources retain separate labels.
The existing1a inventory has3 complete first-history/range records,42 saved-cut
first-pending ordinals and10 missing earlier-unproved identities. Do not replace
those10 by the last active history or pretend saved-cut counters recover later
unrecorded work. Necessary reconstruction must use the registered diagnostic
budget; an unresolved source stays explicit, not an easier substituted history.

The real call table is frozen after1b completes, before any V2/V3 outcome. It
must expose every object and any unexecuted suffix under250 local calls/21h,
retain the user's Wilson decision rules and single-/multi-layer grouping, and
charge loading, bound reconstruction, graph construction, all endpoint queries,
reception and cleanup to each300s call. Prefix/source checks and local closure
cannot retroactively complete an old request. The implementation and its tests
are not a real execution freeze or authority to exceed the existing call cap.
