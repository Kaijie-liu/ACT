# Checked route frontier before lazy source construction — new opt-in mechanism

## Why this is a different intervention

The sealed 4096 source proof spent its work budget constructing expert/pair
objects; neither arm reached a checked output bound. The source constructor
unconditionally propagates all experts and constructs all E-choose-2 pairs.
This code fact motivates reducing **proved irrelevant work**, not changing
the LP relaxation, native solver, budget, cache size or positive acceptance.
No claim that an actual router margin is positive follows from those logs.

This version is separate from `scoped_source`/`scoped_proof` and does not edit
their frozen implementations. It is not enabled in production or wired to a
real-checkpoint supervisor. Inputs98,4088,4096 and historical23 gains stay sealed.
No checkpoint, dataset, historical numerical bound or old route census is used
in the controls or timing study. The current implementation supports the same
declared Linear/ReLU/Flatten selected-softmax top2 graphs, not convolution/BN.

## Exact coverage argument

For every unordered pair S, its tie-legal guard implies r_i >= r_j for every
i in S and j outside S. Suppose a checked whole-domain router LP lower bound
L on r_j-r_i satisfies L>0 for one such insider/outsider. Every concrete input
extends to the checked router HZ and its binary-relaxed LP, hence r_j-r_i>=L>0;
the guard for S is impossible. This discharges S. A zero/negative/missing bound
never discharges it, even if a solver or floating probe says it is unreachable.

The original E-choose-2 roster is partitioned into checked exclusions and
retained pairs. All retained pairs keep every class-margin obligation and the
unchanged shared-input join, guard, property projection and McCormick LP. Only
experts outside the union of retained pairs avoid propagation. Captured source
identity/parameter inventory is still checked for the entire declared model.

The original output-transaction denominator is unchanged: each original
pair/property is either `DISCHARGED_BY_CHECKED_ROUTE_EXCLUSION` or needs its own
checked positive output bound. Retained does not imply reachable; several
retained pairs do not prove route change. Output violation/negative relaxation
bound does not establish a full-model UNSAFE result.

The router candidate provider is fixed: exact pullback through the FINAL
affine lift only, no native LP, iterative search, basis elimination, or tuning.
Its dual is untrusted. The stdlib checker rebuilds the margin LP from the
checked router state and checks the exact residual-corrected bound. Inherited
constraints, factor identity and source/request binding are all checked; the
producer's claimed sign is not an acceptance input. Arbitrary valid duals are
mathematically acceptable to the interface, but this study never searches them.
The equality candidate preserves common penultimate factors and can cancel a
shared score component that independent output intervals lose. It may also
prove no exclusions, in which case it adds overhead and all pairs remain.

## Controls and boundaries

Controls cover multiple/tie-legal routes, zero/negative/absent route evidence,
E/C/depth/width changes and zero radius, exact retained-matrix differential
against the exhaustive builder, input containment, source/factor/LP/property/
run binding, duplicate/dropped obligations, partial/nonpositive output bounds,
and fresh producer/solver-free `python -S` checking. Analytic positive requests
exercise full aggregation, including an E8/C10 252-row roster with 243 checked
exclusions and 9 checked output bounds. They are synthetic controls, not new
real-model certificates or source repairs for historical numerical results.

A synthetic-only two-phase supervisor controls build and fresh independent
source/routes/output checking under one clock. All source generation, dual
proposal, checking before pruning, expert propagation, LP materialization,
hashing, serialization, imports, parent receipt and owned-process cleanup are
charged. Hard deadline, exception, partial files, missing checker receipt,
late positive file and cost-ledger overrun fail closed. Unchanged general
supervisor regressions also run. No resource or publication reserve is free.

## Frozen synthetic timing, declared before measurement

Exactly12 calls: two analytic E8/C10/width8/depth2 graphs, two methods and three
repetitions with alternating method order. Fixed seed724, radius1/8, no data.
Both have nontrivial hidden expert layers and constant safe final outputs;
router final weights are identical across scores. One has strictly ordered
score offsets, one all ties. These deliberately test useful pruning and its
no-pruning overhead, not a representative trained-model workload.

Exhaustive means the unchanged full source constructor and independent
all-output aggregation. Frontier means the opt-in checked exclusions and lazy
construction followed by independent full-roster aggregation. Both use the
same zero-dual candidates for retained constant-output LPs and no native solver.
Retained matrices must be byte-identical and both must close the same analytic
request. Compare build process, check process, total cost, sampled RSS and
serialized evidence bytes. Report all attempts, both fixtures and any slowdown.

Per call30s,2CPU threads, sampled8GiB;2s publication reserve INSIDE30s. Missing
or late proof never becomes a completed proof. No repeats beyond the12-call
roster, fixture changes, timeout extensions or cache tuning after results.
New result root `data/moe/results/checked_route_frontier_synthetic_20260924_r1`.
Implementation and controls are committed before execution; a saved-only
review independently repeats checking, coverage and differential analysis.

Success here means a checked method/control and measured synthetic tradeoff,
not exceeding MetaMoE/CROWN. Before real evidence, separately integrate and
control checkpoint intake, frontier evidence reception, partial route proofs,
native output proposal and full proof aggregation under the unchanged300s.
Then freeze a bounded observed-development comparison, without reopening a
sealed run or changing production acceptance. A real router proving no useful
exclusions or adding excess overhead is an explicit possible stopping result.

Trusted boundary: intended program equals the declared graph; stored center
corresponds to intended preprocessing; checker implementation/runtime. Checked
source-to-LP containment and exact bounds do not prove native float execution.
