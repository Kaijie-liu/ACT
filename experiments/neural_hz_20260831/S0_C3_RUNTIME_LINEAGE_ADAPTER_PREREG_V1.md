# S0-C3 Runtime Lineage Adapter Preregistration V1

Date: 2026-08-31  
Branch: `redu-hz`  
Repository base commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`

This preregistration narrows the next S0-C3 step to graph repair and faithful
runtime observation.  It authorizes no Tiny/CIFAR verifier run, descriptor
emission, HZ rewrite, score claim or default enablement.  The formal baseline
remains 1,870/2,413 with every one of the 13 family counts frozen; E0 remains
61/400 and arithmetically separate.

## Sole question

Determine whether the complete live Tiny143 ReLU36 lazy expression really
contains the already-preregistered C3 grammar

```text
source -> Conv -> Diagonal* -> ADD
       -> Diagonal* -> Conv -> Diagonal*.
```

The answer must be derived from the repaired current ACT graph, complete SSA
value flow and the actual operator objects stored in every live term.  A
positive answer may only authorize the existing pure planner.  A negative
answer closes this occurrence without changing the grammar.

## Two independent default-off gates

The eventual production implementation must use two independent flags, both
false by default.

1. A loader graph flag repairs only complete marked BatchNorm sibling pairs.
   Starting from `upstream -> SCALE` and `upstream -> BIAS`, it may produce
   `upstream -> SCALE -> BIAS` only when variable production, exact widths,
   finite current payloads, pair markers and every other graph invariant agree.
   The operation is all-or-nothing on private predecessor/successor clones,
   rederives its plan from current state and rebuilds successors from accepted
   predecessors.  Any unrelated issue or CAS change rejects the candidate.
2. A runtime-lineage flag records an event tuple inside each lazy term.  Flag
   off preserves the old construction and must not call a repair, recorder or
   adapter helper.  Flag on still changes no numerical set by itself.

The Tiny143 read-only V2 anchors are 81 layers, 88 edges, exactly 19 repairs,
source digest
`1993b610eb5bac28e5246cba3d994f81be995e94ad1cd39afa05b1bc9195cd60`
and candidate digest
`51c3e6fb0309e9add6063958a6fb891249bf495d7ffacc939346a33bf8952833`.
No digest or target identity may select the generic rule.

## Term-local event custody

Lineage travels with `source` and `operators` in every lazy term; it is not an
external object-id cache and not an expression wrapper.  Each event binds:

- the repaired graph snapshot;
- its layer occurrence and full ordered predecessor tuple;
- its selected input edge;
- complete input and output variable tokens;
- its operator position;
- the current layer-payload digest; and
- for a multiplicative event, the identical live operator occurrence and its
  current payload digest.

Conv and Scale each append exactly one operator and one event referring to the
same object.  Bias appends a transition event and consumes no operator slot.
ADD appends the same graph occurrence to every incoming term with the
appropriate ordered `selected_input`.  Reshape, Dense, nonlinear, unknown and
materialization/checkpoint boundaries are explicit barriers unless a new
exact source boundary with complete ancestry is independently proved.

The candidate Scale representation is `DiagonalLinearOp`, not an unlabelled
CSR diagonal.  Before production use it must be bitwise-equivalent to the
current CSR path for matvec, left composition and materialization, and its
complete storage must be charged.  Flag off retains the current CSR behavior.

Public event/certificate dataclasses are proof vocabulary, never authority.
Immediately before planning, the private adapter rederives graph/value flow
and current payload snapshots, checks operator occurrence identity and performs
a second CAS.  Ordinary inconsistency returns the original expression by
identity.  `KeyboardInterrupt` and `SystemExit` propagate.

## Registered real-graph risk

Trial8 reports that ADD24 owns a three-term lazy expression and has successors
25 and 32.  ADD32 adds the Conv29 branch, producing four terms, and ADD32 itself
has successors 33 and 40.  After faithful BN repair the main term may expose

```text
Conv29 -> Scale30 -> Bias31 -> ADD32
       -> Conv33 -> Scale34 -> Bias35 -> ReLU36.
```

The other three complete terms may still expose ADD24 or a longer prefix.  If
so, the existing C3 rule must reject the whole request as nested-ADD or hidden
prefix.  They may not be relabelled as identity and the caller may not choose a
later structural cut.  A C3 result must be path-local and may not mutate or
release ADD32's cached state while layer40 remains a consumer.

## Frozen execution order

1. Isolated adapter unit and adversarial tests, with no production import.
2. Trial9 exit and source-freeze verification.
3. Default-off production loader repair; flag-off graph identity tests.
4. Exact forward and HZ equivalence for the repair alone.
5. Runtime-lineage-only Tiny143 ReLU36 census of all four terms.
6. Only if every complete term matches the frozen grammar, call the existing
   C3 pure planner; otherwise publish a structural zero-hit and stop C3.
7. If planned, perform actual artifact/whole-state/exact-set shadows before
   ReLU63, same-structure Tiny/CIFAR shadows, family replay and full 2,413.

No UNKNOWN, timeout, stopped census, synthetic pass, graph repair, speedup or
component reduction is a score gain.  Until the complete replay retains all
1,870 solved rows and adds a sound CERT or concretely validated ADV, C3 gain is
zero.
