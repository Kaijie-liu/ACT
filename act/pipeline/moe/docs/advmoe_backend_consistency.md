# AdvMoE fixed-obligation backend consistency check

## Decision boundary

The official-scale Lagrangian effect search is frozen.  On the 20-input,
five-radius development cohort, Lagrangian compilation, graph-matched
`mu=0`, unguarded two-path verification, and the separate-interval control
all accepted the same 2/100 numerical-filter endpoints.  Local property
bounds improved in 241/1800 comparisons, but no improvement completed every
necessary branch/property obligation.  The disjoint holdout remains locked.

The previous r2 attribution used stronger causal language than the evidence
supports.  Adding normalized coefficients 8, 16, and 32 did not close the five
selected residuals, but this does not exclude a useful continuous multiplier,
a point between registered values, or a smaller coefficient.  Stage B also
selected only still-negative properties whose recorded best multiplier was
zero, so observing five zero-multiplier blockers is partly a property of that
selection rule.  Finally, positive finite-point dual upper bounds are not
evidence that a certificate exists: every sampled safety value was positive
and the family includes `mu=0`.

Accordingly, the supported conclusion is only:

> Under the frozen development protocol and common budget, the registered
> Lagrangian configuration improved local bounds but added no complete
> endpoint.  Its finite attribution diagnostic did not identify a unique
> cause.

The compiler remains a sound sufficient reduction with toy and retained-HZ
controls.  It is not a default official-scale configuration.

## One fixed four-cell check

Exactly one further diagnostic is allowed on the already selected
`sample6:eps0.5`, route 1, property 2 obligation:

| Graph | Bound method |
| --- | --- |
| router-free static expert with the single property supplied through `C` | plain CROWN |
| graph-matched Lagrangian compiler with `mu=0` and the property in the graph | plain CROWN |
| router-free static expert | frozen sparse-alpha configuration |
| graph-matched `mu=0` compiler | frozen sparse-alpha configuration |

The checkpoint, input box, property row, float32 dtype, CUDA device, and
backend options are identical across the applicable cells.  The sparse-alpha
configuration is described as a **memory-reduced alternative parameterization**,
not a resource-only equivalent repair: shared alpha variables and sparse
intermediate/specification options can weaken bounds.

For each cell the result records:

- the full lowered-node list and operation histogram;
- the source-to-lowered mapping for every `router_logits` parameter and
  whether router parameters remain in the lowered graph;
- the first and last optimized lower-bound iterates observed at the backend's
  best-result comparison, the best observed iterate, and the returned
  `keep_best` result;
- graph construction, solve time, peak memory, dtype, device, and complete
  backend options.

The trace wrapper is process-local and serial.  It copies each lower tensor,
delegates to the original backend function unchanged, and restores that
function in `finally`.  It does not alter optimization parameters or select a
different result.  Plain CROWN has no trajectory, so its sole result is
recorded as initial, best, and final.

Concrete equivalence of the router-free scalar property and the `mu=0`
compiled output is checked at the center, both box corners, and eight frozen
random in-box points before any backend call.

## Frozen interpretation

- If the router-free expert is materially tighter than graph-matched `mu=0`,
  the next engineering issue is graph simplification or zero-coefficient
  branch elimination, not routing theory.
- If alpha optimization begins with a useful bound and its last iterate is
  worse, the trace reports whether `keep_best` preserved the observed best.
- If the frozen sparse-alpha configuration is much worse on both graph forms,
  that is a configuration/parameterization result on this case, not evidence
  that ordinary expert verification is intrinsically that loose.
- If both graph forms remain nonpositive under plain CROWN, the fixed case
  exposes a limitation of that backend call; the check stops without tuning.
- Any incomplete/OOM state is retained and ends the check.

No negative relaxation result is `UNSAFE`, no positive filter is promoted to
outward-rounded formal `SAFE`, and no causal or prevalence statement extends
beyond this one development obligation.  No retries, new multipliers, sample
substitutions, radii changes, or holdout queries are permitted.

## Result (2026-09-06)

All four calls completed and the replay-strengthened independent audit passed
with zero issues.  The concrete pure-expert scalar and `mu=0` compiled output
were bit exact at all 11 frozen points.

| Graph | Plain CROWN | Frozen sparse alpha | Nodes | Peak GiB (alpha) |
| --- | ---: | ---: | ---: | ---: |
| router-free expert | -3.805543 | -4.181559296e10 | 187 | 38.03 |
| compiled `mu=0` | -3.805856 | -4.181558886e10 | 403 | 59.67 |

The frozen `1e-6` classifier reports
`COMPILED_MU0_GRAPH_IS_WEAKER_THAN_ROUTER_FREE_EXPERT` because the plain-CROWN
difference is `3.13e-4` (about `8.23e-5` of the pure bound magnitude).  This is
a real graph-expression difference under the registered rule, but it is far
too small to explain the sparse-alpha result.

The sparse-alpha result is effectively the same on the two graph forms: their
absolute difference is 4096, only `9.80e-8` relative to the approximately
`4.18e10` magnitude.  More importantly, the 20-step trace starts near
`-8.864e10` on **both** graphs and steadily improves to the returned value.
The final iterate is the best observed iterate and `keep_best` returns it.
There is therefore no observed useful initial bound that alpha optimization
later overwrites.  The enormous degradation is associated with the frozen
sparse/shared-alpha and intermediate-bound configuration on this obligation,
not uniquely with the router subgraph.

The graph evidence still matters for cost: all 61 router parameter tensors
remain in the lowered compiled graph; node count rises from 187 to 403,
sparse-alpha solve time from 51.04 to 107.42 seconds, and peak allocation from
38.03 to 59.67 GiB.  Removing the zero branch is therefore a valid future
engineering simplification, but it cannot repair the same huge bound already
present on the router-free expert.

Plain CROWN remains negative on the router-free expert.  That says only that
this fixed call does not close this fixed property; it does not prove true
unsafety or general CNN-backend difficulty.  No further CROWN tuning follows.
The official-scale Lagrangian adapter remains a negative development result,
the holdout remains locked, and resources return to the ACT/HybridZ main line.
