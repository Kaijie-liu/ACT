# Route-Conditioned MoE Verification: Positioning and Risk Register

## Purpose and evidence boundary

This note records the post-Experiment-1D research audit. It separates verified
facts from reviewer-facing risks and prevents the next stage from inheriting
incorrect assumptions. The frozen weighted top-2 evidence remains the
confirmatory ranks 100--199 and the independent Experiment 1D closure; none of
their endpoints is changed here.

The audit was performed at ACT commit
`33e404104699b89ac3a942c835e8bd9512034a5f`. The public baseline repositories
remain external to ACT under `/data1/Kane/MOE/baselines`.

## Corrections to the external review

### The bal010 router is nonlinear

The main checkpoint is not the factory default with an empty hidden layer. Its
frozen `factory_config` contains `router_hidden=[128]`, and the concrete router
is

```text
Flatten -> Linear(3072,128) -> ReLU -> Linear(128,8).
```

Consequently, the exact-router HybridZ result is not the trivial affine image of
an input box. Confirmatory rows require an exact sparse HZ after the router ReLU,
then solve tie-inclusive joint route-feasibility queries in that retained frame.
The 75/86 reduction relative to the ordinary zonotope candidate upper set is
therefore a valid correlation-preserving route-analysis result.

The paper wording should nevertheless remain precise: the result compares an
exact, constraint-aware reachable-router representation with candidate sets
obtained from coordinate-wise bounds. It must not imply that every HybridZ
router remains exact after unsupported or relaxed operations.

### The small-model width result is real but not a scalability result

Across the fixed-radius confirmatory census, guarded expert propagation recorded
10076 pre-support binaries over 903 branches (11.2 per branch on average). This
average alone understates the hard rows. On the 86 route-unstable rows:

| Width statistic | Median | 90th percentile | Maximum |
|---|---:|---:|---:|
| all-expert structural monolithic | 133.5 | 237 | 377 |
| candidate-pruned structural monolithic | 50.5 | 109 | 186 |
| maximum route-conditioned branch | 22 | 41 | 67 |

The corresponding candidate-pruned width ratio has median 0.430 and 90th
percentile 0.530. This supports width separation on the tested model. It does
not by itself establish solver scalability, runtime dominance, or behavior for
large expert networks. A real monolithic allocation and larger models are still
required.

### Repeated solver construction is an implementation bottleneck

The current open backend lowers a sparse HZ into a fresh SciPy/HiGHS model for
each feasibility, support, or minimum query. In the confirmatory F0 stage there
were 111 visited route pairs and 918 visited property rows. Recorded guarded
expert support used 7064 LP and 1849 MILP calls; margin/difference support used
3672 more calls; property minimization adds 918 calls. Thus the documented F0
path issued at least 13503 solver calls, about 225 per invoked sample, before
counting router feasibility and Tier-1 work.

This confirms model construction and repeated optimization as serious costs,
but does not justify an automatic `9x` speedup claim. Combining nine robustness
rows into one disjunctive optimization requires additional selectors and sound
big-M bounds; F0 also builds property-specific disagreement products. The
current implementation already exits after a full-forward validated witness.
Stopping after the first unresolved row would reduce the chance of finding a
later concrete unsafe witness and is not semantics-preserving for the present
three-valued verifier.

The preferred engineering sequence is:

1. add solve-count, row-count, column-count, and model-build-time telemetry;
2. introduce a direct `highspy` incremental backend while retaining SciPy as a
   differential reference;
3. reuse one branch/pair model across support and objective changes where the
   mathematical encoding is unchanged;
4. evaluate property-row disjunction only as a separately tested ablation.

`highspy 1.14.0` is already importable in `act-py312`. Gurobi Python bindings
are installed, but no project license is currently available, so Gurobi cannot
be the required artifact backend.

### Baseline status needs more exact language

The confirmatory `route_invariance_status` is not an unevaluated guess. Exact
tie-inclusive feasible top-k sets are computed; more than one feasible set is a
formal failure of the router-invariance precondition. This is sufficient for
the definition of a route-changing unique certificate.

What remains missing is an explicit end-to-end route-invariance baseline that
also verifies the invariant subset and reports its runtime and total coverage.
Likewise, the existing `monolithic_hz_status` is explicitly a decomposed,
route-unguarded gate-elimination solve. Only its structural width is
monolithic; no monolithic runtime comparison has been run. These limitations
must remain visible in all paper tables.

## Contribution hierarchy

The staged verifier should be presented in this order:

1. **Retained route conditions.** Exact, tie-inclusive routing conditions stay
   in the shared generator frame and are reused by downstream analysis.
2. **Guard-aware support.** Conditional support removes expert ReLU binaries;
   the matched confirmatory table is `n01=17`, `n10=3`, exact McNemar
   `p=0.00258`. This is a coverage result, not an unconditional speedup claim.
3. **Staged output semantics.** Convex gate elimination is the cheap first tier;
   property-directed range-only McCormick is the selected-softmax top-2 second
   tier. F0 resolves 43/60 confirmatory semantic-incompleteness rows without a
   symbolic exponential, division, or segmented sigmoid encoding.
4. **Route decomposition.** When multiple experts are feasible, simultaneous
   expert width is replaced by per-branch width. This is currently a structural
   result until a true monolithic solver baseline is run.
5. **Exact route feasibility.** Candidate and route-set reduction is supporting
   evidence for correlation preservation, with exactness restricted to
   unrelaxed reachable-router HZs.

Convex closure and max-versus-sum branch width are soundness lemmas, not strong
standalone novelty claims. The strongest unifying description is
**path-conditioned abstraction for data-dependent dispatch**, but the current
implementation supports output-level MoE dispatch only. Broader claims require
at least one additional conditional-computation architecture.

## Confirmed limitations

- The frozen checkpoint is a 49.26%-accuracy CIFAR-10 MLP MoE. It is a controlled
  verification benchmark, not a representative modern sparse-MoE model.
- Fixed-radius evidence stops at 2/255; route-boundary search stops at 4/255.
  Boundary-targeted yield must not be described as natural-input prevalence.
- `analyze_topk_sets` enumerates all `binomial(E,k)` sets. This is acceptable for
  `E=8,k=2` but not a scalability solution.
- `condition_topk_membership` uses a sound unconstrained generator upper bound
  for big-M. The current docstring calls it a correlation-preserving support
  bound, but it does not optimize retained constraints. Exact/LP support may
  tighten performance; this is not a current soundness defect.
- F0 intentionally uses an LP relaxation for expert disagreement support. The
  choice is sound and empirically effective, but loses correlation and can
  produce `UNKNOWN_WEIGHTED_RELAXATION`.
- Route A temporarily mutates process-global transfer-function and solver state.
  It restores state on exit, but remains unsafe for concurrent verification.
- The development experiment reports structural monolithic width only. A true
  monolithic router-plus-dispatch-plus-experts formulation is still absent.

## Frozen paper-safe claims

The following wording is supported by the current independent cohort:

> We use a staged route-conditioned verifier. It retains exact feasible routing
> conditions in a shared abstract frame, uses them to tighten downstream expert
> supports, first applies convex gate elimination, and invokes a
> property-directed gate-range McCormick fallback for inconclusive weighted
> top-2 pairs without symbolically encoding the nonlinear gate function.

The paper must report the immutable confirmatory solved rate `56/100` alongside
boundary applicability `76/100`, confirmatory conditional coverage `56/76`, and
the separately labeled follow-up closure `68/76`. It must not replace the failed
endpoint with the closure number.

## Next-stage order

1. Complete the ICML 2025 RT-ER B0 provenance and code/paper discrepancy audit.
2. Resolve the frozen dependency decision before any smoke or training.
3. Implement explicit route-invariance and true monolithic baselines; add solver
   telemetry before claiming runtime improvements.
4. Reproduce the official hard-top-1 ResNet18 model, then test ACT conversion
   and original-scale feasibility without silently shrinking it.
5. Use a clearly named verification-scale derivative only if original-scale
   exact expert solving is intractable.
6. Add lazy route-set enumeration and expert-count scaling before making claims
   in `E`.
7. Treat margin-conditioned F1, arbitrary top-k gate abstraction, intermediate
   MoE, and token-level ViT MoE as later algorithm/scale milestones. F1 remains
   untriggered by the present evidence.

No additional bal010 cohort, F1 run, public training, or baseline execution is
authorized by this positioning note.
