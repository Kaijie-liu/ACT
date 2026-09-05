# Evidence-semantics hardening, 2026-09-06

This code stage responds to the post-B3/AdvMoE review without changing a
frozen experiment, checkpoint, radius, or historical result directory.

## Closed implementation gaps

1. AdvMoE two-path result schema v2 separates prediction, route, and joint
   witnesses. The evidence-safe portfolio accepts any positive numerical
   route-invariance, two-path, or eta filter; a full-forward prediction witness
   overrides every filter.
2. Router-filter/route-witness and output-filter/prediction-witness conflicts
   are separate fields. The independent auditor recomputes both from raw rows.
   Frozen schema-v1 artifacts retain their original aggregate and field names.
3. Optimized CROWN requests fail closed unless autograd and explicit positive
   optimization iterations are enabled. The actual backend configuration is
   serialized with each bound.
4. Future AdvMoE compatibility runs bridge only the unique, source-validated
   router-KL target softmax callsite. The accepted historical checkpoint keeps
   its globally scoped compatibility identity.
5. `RouteAEngine` is explicitly the Tier-1 gate-elimination API. Its `run()`
   method remains a compatibility alias for `run_tier1()`; F0 remains a
   separately exposed and audited selected-softmax top-2 fallback rather than
   a silently invoked public stage.

## Soundness boundary

The real-arithmetic path decomposition and the numerical backend are distinct
layers. HZ/HiGHS SAFE decisions remain conditional on the pinned optimality,
tolerance, correction, and positive-margin policy. That policy is fail closed,
but is not described as a universal proof about native solver floating point.
The installed CROWN backend has no outward-rounding contract; all positive
CROWN margins remain numerical filters and formal SAFE stays zero.

## Verification performed

- `act-py312`: AdvMoE schema/portfolio tests, bridge tests, CROWN configuration
  tests, compile checks, and the MoE regression suite.
- isolated alpha-beta-CROWN environment: real plain-CROWN regression and a
  two-iteration alpha-CROWN smoke with gradients and optimization arguments.
- frozen AdvMoE full-r3 schema-v1 artifact: complete independent replay audit,
  `PASS`, zero issues.

## Scientific decision

The controlled bal010 result remains the formal core: exact candidate
reduction, retained-guard width/support effects, route-changing SAFE results,
and F0's incremental contribution are independently established there. The
official-scale RT-ER and AdvMoE results establish executable specialization and
numerical coverage shape, but not formal SAFE or a retained-guard advantage.

The next result worth buying is therefore not another unguarded full-box CROWN
table. It is a paired official-scale test in which the same expert backend and
budget see (a) no retained route relation and (b) a sound retained-route
relation. Any proposed Lagrangian or cut-based guard injection must first have
a real-arithmetic proof, a concrete-constraint consistency test, and an exact
small-model differential against retained HZ. It is related to existing
constraint-propagation ideas and must be positioned as a routed-program
specialization, not a generic invention. Until its numerical backend is
outward validated, its endpoint remains a filter even if coverage improves.
