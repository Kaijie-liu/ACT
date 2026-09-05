# AdvMoE two-path verification

## Frozen semantics

The accepted subject is the final epoch-100 checkpoint from the explicitly
labeled numerical-compatibility training run. It is not relabeled as unchanged
official training. The released source remains read/execute-only because it has
no license file, and the checkpoint is identified by hash rather than copied
into ACT.

For each deterministic clean-correct input and radius, the runner constructs
the clipped unit-pixel box and evaluates three compositions with the same plain
CROWN backend:

1. route invariance: a numerical positive lower bound for the selected router
   margin plus a positive property bound for the selected static path;
2. router-independent Route A: positive property bounds for both static paths;
3. a tie-safe eta implication for each path as a bounded guard-representation
   ablation.

AdvMoE has one global hard route shared by all 16 MoE convolutions. Replacing
each routed convolution with the selected contiguous weight slice therefore
produces exactly two static networks, not `2^16` paths. The CROWN adapter also
replaces the final fixed-shape adaptive average pool with `AvgPool2d(4)`. The
literal and lowered paths must agree within the registered absolute tolerance
`1e-6` (with zero relative tolerance); their raw maximum error and prediction
agreement are retained. The nonlinear router uses the already
validated fixed-shape adapter.

The installed CROWN backend is not outward rounded. Consequently, positive
lower bounds are reported only as `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`.
Negative or incomplete bounds are `UNKNOWN`, never `UNSAFE`. Only a concrete
input inside the registered box whose prediction flip replays through the full
dynamic model may be labeled `UNSAFE_FULL_FORWARD_REPLAY`.

## Correctness smoke

The first run is frozen at the first ordered clean-correct test input and
`0.5/255`. It exercises one router bound, both static paths, both eta-compiled
implications, a two-restart 20-step full-model attack, lowering equivalence,
incremental row persistence, and artifact hashing. It is an engineering smoke,
not a prevalence or certification result. Only a zero-error, zero-conflict
independent audit may unlock the 20-sample five-radius execution.

The independent auditor recomputes every CROWN status from stored lower bounds,
recomputes the staged aggregations without calling the runner helper, verifies
the first-clean-correct selection, replays every attack endpoint through the
literal dynamic model, and checks its box membership and perturbation norm. It
also enforces zero formal SAFE counts for this non-outward-rounded backend.

Smoke r1 completes in 14.30 seconds and independently audits `PASS` with zero
issues. The router margin is a positive numerical filter (`0.95591`), while
both unguarded static paths remain unresolved (`-291.25` and `-774.83` minimum
property bounds). The eta implication filters the inapplicable route-0 branch
and remains unresolved on the clean route, which is the expected tie-safe
implication behavior. The full-model attack does not flip the prediction and
the aggregate endpoint remains `UNKNOWN`; no negative relaxation value is
promoted to UNSAFE. The path adapters agree with their literal static paths
within `2.39e-7`, the dynamic selected-path error is `2.39e-7`, and the router
adapter is bit exact. Independent full-test selection replay, endpoint replay,
box containment, status recomputation, and artifact hashes all pass. The audit
is `act/pipeline/moe/results/baseline/advmoe_two_path_seed0_compat_smoke_r1_audit.json`.

The accepted smoke unlocks
`act/pipeline/moe/configs/advmoe_two_path_seed0_compat_full_r1.json`: the same
frozen method on the first 20 clean-correct ordered inputs and
`{0.5,1,2,4,8}/255`. The full run retains all-sample denominators and does not
change the backend, attack, guard ablation, or numerical semantics.

Full attempt r1 stopped before creating a result row because the runner called
the lowering helper with its local default tolerance instead of the frozen
configuration value. Independent replay on the exact 20 selected inputs found
maximum path errors `4.77e-7` and `9.54e-7`, equal predictions, and both paths
inside the registered absolute tolerance `1e-6`. The failure is retained in
`advmoe_two_path_seed0_compat_full_r1_attempt001_failure.json`. Full r2 changes
neither the model nor the scientific configuration: it only connects the
already registered tolerance to the execution gate and records zero relative
tolerance explicitly.

Full attempt r2 then exposed a second execution-only mismatch before writing
any result row: the literal dynamic model used the registered 20-input batch,
whereas the selected static paths were evaluated one input at a time. The two
valid floating-point schedules differ by at most `2.86e-6`, with identical
predictions. Evaluating both static paths at the same registered batch shape
and then selecting the routed rows is bit exact against the dynamic forward.
R3 freezes that like-for-like execution schedule without changing any
scientific method or denominator; the r2 failure remains separately recorded.

## Full r3 result

Full r3 completes all 100 registered rows in 1,437.56 seconds. The enhanced
independent audit reports `PASS` with zero issues after rebuilding every
per-radius table, replaying the first-clean-correct selection over the full
test archive, and replaying all 100 attack endpoints through the literal
dynamic model. All 500 CROWN calls complete, backend errors and positive-
filter/witness conflicts are zero, the official no-license clone remains
clean, and the formal SAFE count is deliberately zero.

| epsilon | router positive filter | route-invariance filter | two-path filter | eta filter | prediction-flip witness | route-flip witness | both |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5/255 | 18/20 | 2/20 | 2/20 | 0/20 | 1/20 | 0/20 | 0/20 |
| 1/255 | 1/20 | 0/20 | 0/20 | 0/20 | 1/20 | 0/20 | 0/20 |
| 2/255 | 0/20 | 0/20 | 0/20 | 0/20 | 3/20 | 0/20 | 0/20 |
| 4/255 | 0/20 | 0/20 | 0/20 | 0/20 | 4/20 | 0/20 | 0/20 |
| 8/255 | 0/20 | 0/20 | 0/20 | 0/20 | 8/20 | 1/20 | 1/20 |

Every “filter” entry is numerical conformance only, not a formal certificate.
At `0.5/255`, the individual static paths filter 3/20 and 2/20 inputs and both
filter the same 2/20. Plain CROWN resolves no static path at larger radii; the
eta encoding resolves no complete two-branch obligation at any radius. This
shows that route-independent decomposition is executable on the third-party
learned-router architecture, but plain CROWN is too loose to establish the
desired official-scale certificate yield. It is not evidence against the
staged semantics, and it does not license replacing UNKNOWN with UNSAFE. The
only UNSAFE rows are independently replayed prediction flips.

The raw r3 summary used a legacy field name,
`route_attack_or_prediction_witnesses`, for a prediction-flip-only count. The
enhanced audit explicitly separates prediction, route, and joint witnesses;
future runner output uses the corrected three fields. The accepted audit is
`results/baseline/advmoe_two_path_seed0_compat_full_r3_audit_r2.json`.

## Post-r3 evidence-semantics hardening

Frozen r3 remains schema v1 and is not rewritten. The current runner emits
schema v2. It separates prediction-, route-, and joint-flip counts; adds an
explicit portfolio that accepts any of route invariance, two-path filtering,
or eta filtering without changing their numerical-only evidence class; and
records router-filter/route-witness and output-filter/prediction-witness
conflicts separately. A concrete replayed prediction flip still overrides
every positive filter and is the only route to `UNSAFE` in this runner.

The independent auditor is version-aware. It re-audits frozen schema-v1 r3
without changing its historical aggregate, while schema v2 is recomputed with
the portfolio endpoint and the three unambiguous witness fields. The auditor
derives both conflict types from raw rows rather than trusting the runner's
boolean. The full frozen r3 artifact passes this compatibility audit with zero
issues.

Optimized CROWN labels now pass a fail-fast configuration gate: alpha-CROWN or
another optimized method requires gradient tracking, explicit optimization
arguments, and a positive iteration count. Plain CROWN remains valid without
autograd. This closes a method-label ambiguity; it does not add outward
rounding or promote any numerical filter to formal SAFE.
