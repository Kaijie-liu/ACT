# R1: guarded expert relationship ablation (post-confirmation development)

This protocol is frozen before any new real-model query. It does not amend the
30-input or 100-input schedule confirmations. No effect-based expansion,
replacement, radius change, additional seed, new gate segment, or budget tuning.

## Mathematical contract

For a tie-legal pair S, propagate both experts with the SAME guarded entry
X_S and the SAME marginal propagation/support configuration. Let P_a and P_b
be the resulting factor polytopes with binary factors. Shared composition
identifies the entry factors, but keeps each expert's private factors distinct.
The ablation instead constructs the block-diagonal product P_a × P_b: ALL
equalities, inequalities (including both copies of the guard), factor boxes and
binary integrality constraints remain. It duplicates continuous AND binary
entry factors. It is not coordinate boxing and does not relax private binaries.

Every assignment admitted by shared composition maps into the product by
duplicating its entry assignment. Expert output coefficients/centers are
unchanged. Therefore the common-input output relation is contained in the
Cartesian product of guarded expert marginals. New factor IDs must not be
aligned back to the router. The product owns a fresh execution-local frame;
the original frame is recorded solely as gate-range provenance.

F0 already uses only a scalar interval for the router-derived lambda. That
interval is computed by the SAME routine from the SAME conditioned router in
both arms. It does not couple lambda factors back to the router in either arm.
For any concrete x in X_S, its lambda lies in this interval, its two outputs
are in either domain, and their product satisfies the McCormick inequalities.
Thus each encoding remains a sufficient outer relaxation. Difference support
is recomputed on the chosen domain: widening it is an intended consequence of
discarding correlation, not an independently altered gate rule. Different
finite-budget solver results need not obey ideal optimal-bound monotonicity.

Input reconstruction from the product uses its A-copy only, in the product's
fresh frame. An inconsistent B-copy is NOT a full-model counterexample. The
existing concrete full-MoE replay, box check and classification violation are
mandatory for UNSAFE. A negative relaxation optimum remains UNKNOWN.

Scoped interval facts remain valid on either marginal, and convex combination
of two positive expert facts is positive even with independent inputs. Tier 1,
route enumeration, the 25% rule, stopping rules and acceptance policy are not
changed. Reused obligations are retained in the endpoint accounting.

## Controls before real queries

- Affine tied gate, E_a=x+.2 and E_b=-x+.2 over [-1,1]: shared minimum .2;
  independent minimum -.8. Only shared may certify; the product witness must
  not become UNSAFE. Repeat through both complete-verifier F0 paths.
- Negative-offset unsafe control must not certify in either arm.
- Guard x>=0 must survive in BOTH independent marginals.
- Block-diagonal coefficient checks include private continuous/binary factors,
  and a diagonal feasible assignment reproduces both expert outputs.
- All tie-legal pairs, reuse, malformed provenance and invalid mode tests.

These test arithmetic tolerances are comparison tolerances only. Solver
acceptance remains the frozen HZ/HiGHS numerical policy, not a claim of an
independently checked native floating-point proof.

## Frozen real-model follow-up

Use the first TEN ordered inputs of the already-observed 30-input R2 manifest,
three unchanged checkpoints, epsilon 2/255. The old smoke input (index 3000)
is used for six smoke calls. The new 100-input cohort is NOT used here. This is
10 distinct images, 30 model-input pairs, 60 full method calls, not confirmation.

Both arms are the adaptive scoped-reuse configuration. The ONLY configuration
difference is f0.expert_relation: shared_input vs independent_inputs. Each
independently pays for loading, candidates, guards, facts, propagation,
construction and solving inside the same 300-second wall-clock cap. Graph
size differences are inherent in the abstraction and reported, not hidden.
Order alternates within input/model blocks. One CPU worker, BLAS/OMP=1, nice 10.
No timing run shares a checkout with a writer or another solver experiment.

Smoke gates identity, complete evidence in at least one paired request, fact
conformance, all planned terminals and absence of contradictions; NOT positive
effect size. Failure is retained and stops the pipeline, with no auto-retry or
resume. Full completion is all 60 terminals plus independent structural audit
and concrete UNSAFE replay. Timeouts and incomplete evidence stay in denominator.

Primary report: per-model gained/lost SAFE (shared relative to independent).
Also report gained/lost solved, all four states, pair-count strata, decision
source, F0 row count, factor sizes, costs including censored requests, common
fact equality and available gate-interval equality (missing is not equality).
No post-hoc favorable-radius selection or inferential claim from 60 calls.

If only bounds improve with no complete endpoint gains, report that negative
endpoint result. If the independent encoding is slower, do not ascribe all
coverage differences uniquely to tightness; construction size and solve budgets
are part of this representation comparison. Analytic controls isolate pure
relational cancellation, real-model finite-budget effects need this qualification.

## Separate workstreams

External-tool semantics and complete-request LP evidence remain separate from
this ablation. No new dependencies, external backend, candidate superset,
numerical acceptance change or training modification is part of R1.
