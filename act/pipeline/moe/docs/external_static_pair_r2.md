# Separate real-model external static-pair adapter R2

Freeze BEFORE external calls: seed0/index3000, the exact float64 checkpoint
and materialized request of request-LP R1,epsilon2/255. Parent request.pt SHA256
2523f7b04070af1eb4ab5d8bef282828902abfa921d3128f5ff3b9206916867e;
parent manifest SHA25690aad930c9f8b1f7018c3a7ef30cee80ab4487e5fae16b231bc6432c7b3cf082.
Two predetermined static pairs{2,4},{4,5}, the complete parent feasible list.
No new route enumeration, input, checkpoint or accuracy selection.

StaticSelectedSoftmaxPair evaluates exactly the real-arithmetic F_S expression:
scores of a,b from the same input, selected softmax of those scores, both expert
outputs, variable weights and their sum. No dynamic TopK/gather or clean-route
trace. It rejects training mode,unsupported gates and shared experts. Validate
concrete branch expression and variable-weight gradients in unit controls.

The external input domain is the ENTIRE BOX, not the pair guard. Thus each
query is a stronger static sufficient obligation, not an equivalent complete
MoE benchmark. Route exclusions come from ACT and remain trusted. A negative
external lower bound is not a loss on the exact guarded problem. Neither tool
replaces ACT/HybridZ. R1 frontend and LP results are untouched.

Use existing alpha-beta-CROWN Python3.11 environment with the same pinned
source revisions as external R1, and existing centralized typing compatibility
shim. No install. CPU/float64, batch1, plain CROWN, conv_mode=matrix, nine
classification rows through C in one bound call per pair. No alpha/BaB,
optimization settings, dtype fallback or parameter search.120seconds outer
per pair INCLUDING imports,checkpoint loading,conversion,probes and bounds.
Sequential,OMP/BLAS1,CPU only. Failures/timeouts retained; no query replacement.

Before bounds, record tensor/model identity and compare on five fixed points:
center,lower,upper,and two alternating-coordinate box corners. Compare the
adapter against the independently expressed forced branch and lowered graph,
absolute tolerance1e-10. This is finite conformance only, not full-domain/native
floating-point equivalence. Save nine concrete margins and lower/upper bounds,
graph node types,environment/source identities,timings and failure phase.

Report per-pair positive numerical rows and ALL listed pair/property coverage.
No formal SAFE: external bounds are not independently outward-rounded. Even
if all18 are positive, strongest label is ALL_LISTED_STATIC_OBLIGATIONS_NUMERICALLY_POSITIVE,
conditional on supplied route coverage and exported semantics. Performance
against HZ's guarded LP and high-accuracy real-model claims are not licensed
by this small adapter control. No source edits during its two bound calls.
