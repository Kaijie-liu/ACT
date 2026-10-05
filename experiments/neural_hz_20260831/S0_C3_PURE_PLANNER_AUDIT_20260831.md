# S0-C3 Isolated Pure Planner Audit

Recorded on 2026-08-31 on branch `redu-hz` at repository base commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.  This audit covers the
experiment-only S0-C3 P0 planner preregistered before target execution.  It
does not modify ACT production, run Tiny/CIFAR, reopen C2 or change a score.
The formal baseline remains 1,870/2,413 and E0 remains 61/400; S0-C3 gain is
zero.

## Frozen files

```text
efca6160cb3a8e5cc78fbddd9b83a161ffe8dac72d463a672537cfe73c84a9c7  S0_C3_IDENTITY_MIDDLE_PREREG_V1.md
c96ee3f1682cf93e5beae8a5ece8aba810e3b324ba5b1b1002d31a1131da3d21  s0_c3_identity_middle_lineage_planner_prototype.py
2184463bbe8267c9d1e8f065e679fafa73f8422df83f6d45ee3af4d0c2afe9d0  test_s0_c3_identity_middle_lineage_planner_prototype.py
```

The C3-specific suite passes 37/37.  The registered C1, C2 and C3 pure
planner suites pass 121/121 together.  `py_compile` and whitespace checks pass.

## Implemented rule

The module exposes one non-executable rule and never calls C2:

```text
source
 -> ImplicitConv
 -> exact channel-stationary Diagonal*
 -> one latest common ADD
 -> exact shared channel-stationary Diagonal*
 -> one shared ImplicitConv
 -> term-local output Diagonal*.
```

A complete branch may contain zero pre-ADD diagonals.  A branch with no
operator before the common ADD is a genuine identity skip and remains the
identical term object.  Every nonempty pre-ADD segment must begin with exactly
one Conv and contain only diagonals afterward.  Every active complete branch
is selected in one request; zero-support complete terms and all identity skips
remain unchanged.

The source is the already-constructed prefix.  There is deliberately no
caller-supplied `core_start`: the complete operator tuple before ADD is
classified once.  Thus a caller cannot hide a Conv in a prefix or relabel a
complete active branch as identity.

## Graph-event/operator certificate

Every use supplies an ordered structural certificate containing:

- graph occurrence and SSA value tokens;
- variable producer and graph predecessor occurrence tuples;
- the selected incoming ADD edge and complete ordered ADD inputs;
- one exact operator occurrence/index for each Conv or Scale event;
- explicit Bias transitions that consume no operator slot; and
- current Conv kernel/geometry/mask, diagonal and Bias payload snapshots.

Producer/pred mismatch, graph/value self-dependency, reused output value,
missing or extra operator, wrong kind/order/occurrence, Bias without its Scale,
unknown event, hard boundary, nested ADD or a second Conv in either core
segment rejects the entire request.  Shared ADD, outer Conv and post-ADD
suffix require occurrence identity and current snapshot equality.

The certificate dataclasses are an isolated proof vocabulary, not an
authority token.  Production must generate them internally from the same
locked `net.preds`, `in_vars/out_vars` and live HZ lineage, then repeat payload
CAS validation immediately before publication.  A hand-built certificate
cannot authorize a target run.

## Numeric descriptor identity

Empty `D*` and explicitly represented exact-one diagonals have distinct use
lineage and `pre_add_diagonal_count`, but may reuse one numeric descriptor.
The cache key contains:

- current inner Conv snapshot;
- a canonical exact-real product of the stored binary64 diagonal values;
- the deterministic float64 compile product; and
- current outer Conv snapshot.

Binary64 values are dyadic rationals.  The exact key multiplies integer
significands and powers of two channel by channel, canonicalizing trailing
powers and zero.  Consequently `0.1 * 0.1` cannot alias a separately stored
float64 value that merely rounds to `0.01`, while an empty product and any
number of exact ones intentionally share the same linear map.

The planner retains live operators and prospective emission only.  It neither
compiles a descriptor nor creates/reuses a CSR artifact.  C1 arithmetic is
used only for conservative reservation; actual emission, whole-state and RSS
remain separate gates.

## Defects found during independent review

Four material issues were found and repaired before this audit closed:

1. a shared graph snapshot initially included branch-local cursor/selected
   edge state, incorrectly making valid common suffixes appear distinct;
2. a free structural cut could hide a complete branch in the prefix and evade
   all-active selection;
3. a rounded float64 diagonal product could merge exact-real-distinct maps;
   and
4. graph occurrence/value self-cycles and SSA value reuse were not rejected.

The final tests lock all four boundaries, ordinary fail-closed rejection,
`KeyboardInterrupt`/`SystemExit` propagation, resource equality/one-over-limit
behavior, source/frame/predicate field presence, identity preservation, C2's
continued empty-D rejection and absence of iid/family/layer/margin/verdict
selection.

## Production advancement boundary

P0 is complete, but the current target remains NO-GO.  The actual Tiny143
translator graph omits nonidentity Scale events from its live lazy path.  A
production advance requires, in order:

1. the independently audited BatchNorm graph correction, default-off;
2. an internal runtime event/operator adapter with atomic current-payload CAS;
3. real materialization through the CSR artifact transaction;
4. exact HZ value/predicate/frame/bias/witness shadows;
5. complete strong-root accounting at the same consumer-GC boundary,
   including ADD32's second successor, with strict bytes and entries decrease;
6. Tiny iid143 ReLU36, then ReLU63 and registered same-structure shadows;
7. affected-family and four-concurrent no-regression gates; and
8. one complete 2,413 replay retaining every baseline row and family count.

Until all gates pass, there is no Tiny/CIFAR hit, no production integration,
no default enablement and no formal/E0 gain.
