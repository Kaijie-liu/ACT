# Generic declared-source conversion: controls complete, no real proof run

This implements the adapter/control gate of
[the separately scoped protocol](source_output_closure_scope_20260923_r1.md),
not its eventual execution. No real checkpoint, dataset, saved input4088,
input98, historical HZ, old positive bound or candidate exclusion was loaded.
No production verifier, tolerance, budget fraction or frozen result changed.

## Supported object and interfaces

`scoped_source.capture.capture` extracts an eval, CPU/float64
`OutputLevelMoE` with selected-softmax top2 and no shared expert. Router and
experts must be ordinary Sequential **Flatten(1), Linear with bias, ReLU**
graphs ending in Linear. Depth, widths, expert count and shared class count
are read from the supplied object, not from a particular checkpoint. Hooks,
instance-overridden forward, training mode, other operators/dtypes are refused.
Convolution, BN, internal/multilayer MoE and arbitrary Python code are NOT
silently translated by this interface.

Every stored coefficient, parameter name, layer order, shape, model inventory,
center tensor, label, rational radius/clipping and positive-margin offset is
bound into a declared-source digest. The caller must supply the expected
digest from its request manifest; recomputing and accepting an untrusted
replacement manifest is not identity authentication. Extraction invokes no
model forward. Declared-graph/program correspondence remains an explicit trust
assumption; finite probes or parameter hashes do not prove native execution.

`scoped_source.build.build` produces new shared source states and LPs.
`scoped_source.check.check` checks the derivation without importing capture,
build, ACT, torch, NumPy, SciPy or a solver. Shared code is confined to exact
serialization/declared-operator semantics; tests also check semantic concrete
assignments rather than only comparing two files produced by the same builder.

## Checked chain and mathematical contract

1. Form exact rational requested endpoints from the stored binary center.
   Check outward diagonal HZ inclusion, including zero/tiny radii. No radius
   cutoff. The new object need not equal old rounded endpoints.
2. Propagate every router/expert layer. Exact sparse affine equality lifts
   avoid nominal floating products altogether: the new factor is linked by
   an exact equality to the old affine combination and bounded by an L1 range.
   This path therefore needs no nominal affine-error compensation; retained
   compensation-rule regression tests still reject missing/error factors.
3. Independently check active, inactive and unstable ReLU relations and fresh
   continuous/binary factor identities. Every actual source value extends to
   these relations. Later binary relaxation is explicitly an outer relaxation.
4. Join router and both experts on the SAME input factors. Keep each component's
   private factors disjoint and retain all mapped constraints. Unlike the old
   two-expert join, this handles different router/expert output widths.
5. Enumerate all `E choose 2` unordered pairs. Attach exactly
   `r_out - r_selected <= 0` for both selected experts versus all outsiders.
   These are **conditional guards**, not falsely claimed redundant constraints.
   There is no ordering inside a pair, tie tolerance, feasibility exclusion or
   unfinished-route deletion. Every concrete legal top2 execution is covered;
   an enumerated pair need not be reachable.
6. Project the two expert outputs, subtract the required margin from each
   label coordinate, and construct every competing-class obligation from the
   same factors. In a normalized mixture this subtracts the margin once;
   it cancels from the expert difference. Use universal gate range [0,1],
   checked L1 difference range and new exact rational McCormick rows.
7. Check all inherited constraints, objectives, factor maps, pair/property
   inventories and rectangle bounds independently. No old LP certificate or
   partial certificate list is admitted in this construction-only schema.

If positive checked lower bounds for all these LPs are eventually obtained,
outer containment and normalized-mixture algebra would establish the declared
real-graph output property. **This stage does not obtain those bounds.** A
construction check is neither a source-complete positive certificate nor a
proof of LP optimality, network safety or unsafety.

## Control evidence

[Control archive](scoped_source_controls_20260924_r1.json): **35 tests pass**,
17 new generic controls plus 18 existing affine/ReLU/LP/transport regressions.
Individual subcases include:

- E=2/3/4 and multiple depths/widths/classes; E=8,C=10 synthetic graph builds
  and checks all28 pairs/all252 LP constructions (NOT the actual4088 request).
- All tied router scores retain every unordered pair; constant expert
  differences and zero-width input/gate-envelope corner cases remain valid.
- Exact feasible factor extensions on signed/rational inputs traverse every
  layer and legal guard; the LP objective equals the explicitly computed
  weighted expert margin minus the registered offset.
- Reject inward input, missing layer/network/pair/property, wrong source,
  parameter alias, private factor collision, guard sign/RHS, class margin,
  McCormick coefficient/range and historical or partial positive evidence.
- Solver/producer-disabled checks and a fresh **python -S** process consuming
  only synthetic serialized objects; no model/data/solver imports.
- Expired/overlong deadlines, late final checks and old supervisor exception,
  partial-output and deadline controls. The latter are regression tests of the
  old supervisor, NOT evidence that this new pipeline is already wired to it.

Run `python -m unittest scoped_source.tests source_enclosure.tests
source_enclosure.portable_tests full_source.tests -v` in act-py312.
`python -S scripts/validate_scoped_source.py --check` verifies the recorded
source identities/status only; it does not rerun the tests or certify a model.

## What remains before the one real request

This interface shares a finite at-most-300s cooperative deadline. A single
large exact operation may cross it before the next check; therefore it does
NOT replace an outer hard watchdog. The in-memory trace is also not a claim
of scalable streaming/compact artifact construction.

Next integrate it in a **separate version** of the unified budget supervisor,
including loading, source construction/check, new candidate bounds, exact
bound checking, serialization and complete terminal publication. Control
exception, deadline, RSS refusal, late/partial evidence and complete costs;
bind the one frozen source request before any real execution. No budget per
LP masquerading as 300 seconds per request, no free preprocessing. The stored
weights/center/model identity must be tied to the external manifest at intake.

Only after that gate may a separately frozen attempt on4088 generate and
check NEW bounds for ALL252 obligations. No extra samples/time, source-based
row selection, old certificate reuse or input98 reopening. Nonpositive or
incomplete evidence is NOT_CLOSED, not UNSAFE or an identified relaxation
impossibility. Proving route change would additionally require checked legal
route witnesses; all-pair enumeration alone does not establish it.
