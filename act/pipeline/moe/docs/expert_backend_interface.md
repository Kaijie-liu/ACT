# Route Conditioning as an Expert-Backend Interface

## Positioning

Route conditioning is an analysis layer, not a commitment to one expert
verifier. For each feasible route branch, it produces a guarded input domain,
one concrete expert network, and linear output properties. A backend proves the
expert property over that domain. HZ/HiGHS is the exactness reference for the
verification-scale model; a scalable CROWN-family backend is the intended
official-scale configuration once its adapter passes conformance tests.

This separation changes neither status semantics nor soundness:

- every feasible tie-inclusive route branch must be covered;
- a backend `SAFE` result is accepted only for the exact guarded domain or a
  sound superset;
- a backend relaxation violation remains `UNKNOWN`;
- `UNSAFE` requires a concrete input replayed through the full hard-dispatch
  model, including agreement with the claimed route.

## Backend contract

For expert `i`, the route layer supplies:

```text
input box:             lower <= x <= upper
route guard:           A_i x <= b_i
expert:                E_i(x)
output safety rows:    C E_i(x) + d >= 0
numerical policy:      feasibility, integrality, and positive-margin tolerances
```

A backend returns `SAFE`, `UNKNOWN`, `TIMEOUT`, or a candidate witness plus
timing and bound metadata. The route layer owns full-model witness replay and
the final status.

## α,β-CROWN source audit

The official repository was pinned read-only at commit
`e5c7e17bf0488843acb77b7519f59876717a49f4` on 2026-08-29. Its current source
supports CIFAR/ResNet models and advertises Clip-and-Verify for linear
constraints. However, the ordinary VNNLIB parser still asserts that input
constraints must be a box, and the expression API rejects an input predicate
that references more than one input variable. Therefore the route polytope
`A_i x <= b_i` cannot currently be passed directly as a general input
constraint through those front ends.

Three adapters remain scientifically distinct:

1. **Exact guarded HZ.** Preserve `A_i x <= b_i` in the shared generator
   domain. This is the exactness reference and the source of the existing guard
   accounting result.
2. **CROWN over a guarded box hull.** Use HiGHS to minimize and maximize every
   input coordinate over `box intersect route_guard`, then verify the expert on
   that coordinate box. This is sound because the box is a superset, but it may
   lose correlations. It is the first official-scale adapter candidate.
3. **CROWN with an augmented affine-router output.** Append the route-score
   differences to the expert model and express the route guard as output
   constraints for Clip-and-Verify. Current official documentation suggests
   this may preserve more guard information, but it remains
   `UNVALIDATED_ADAPTER` until an installed, pinned environment passes concrete
   and soundness smoke tests.

The guarded-box adapter's coordinate-hull layer is now executable as
`guarded_hz_box_hull_highs`. It lowers one guarded HZ domain once, then changes
only the coordinate objective across the support sweep. Its result retains
per-side solver status and falls back to the unconditioned generator bound for
every incomplete objective, so a partial sweep remains a sound outer box.
Binary HZ variables are continuously relaxed and such a result is never marked
exact. Telemetry distinguishes model builds, objective changes, solves,
iterations, and accepted basis submissions; it explicitly does not claim that
HiGHS internally used an accepted basis as a warm start.

A separately modelled SciPy path is retained for differential tests. Both
paths ultimately use HiGHS, so this validates the lowering and incremental API
rather than constituting independent-solver evidence. Random guarded domains,
affine shared-frame outputs, infeasible guards, binary relaxation, and
zero-budget fallback are covered by tests.

### Tie-safe implication warning

A tempting single-output compiler defines

```text
g(x) = max_{j != i} (r_j(x) - r_i(x))
s(x) = min_k (C_k E_i(x) + d_k)
t(x) = max(g(x), s(x))
```

and asks a verifier to prove `t(x) >= 0` on the original box. This is correct
for strict route interiors, but it is **not** semantically exact under
`ANY_LEGAL_TOPK`: at a tie, `g(x)=0` and an unsafe `s(x)<0` still gives
`t(x)=0`, so the compiled property passes even though expert `i` is a legal
route and must be safe.

Two admissible designs remain:

1. retain the exact non-strict guard as a separate constraint in a backend that
   supports constrained implication; or
2. use a disclosed conservative margin `eta>0` and compile
   `max(g(x)-eta, s(x)) >= 0`. For every legal route point `g(x)<=0`, the first
   term is strictly negative, so safety is required. The price is also checking
   some non-member points with `0<g(x)<eta`.

The second design is sound but not semantically exact. `eta` must dominate the
frozen route/numerical tolerance and be preregistered. Neither design is
implemented in the current stage; no augmented-output result may use the naive
zero-margin reduction.

The repository also contains a newer optimization API whose documentation
mentions linear input constraints, but its normal expression parser still
reduces input constraints to coordinate bounds, and the inspected primal path
does not populate a general input-constraint matrix. It is not evidence of a
supported complete-verification route-polytope interface.

## Required paired experiment

On identical route branches and expert properties, report:

| Variant | Guard representation | Claim allowed before smoke |
|---|---|---|
| HZ guarded | exact halfspaces in HZ | exactness reference |
| CROWN guarded-box | coordinate box hull of guarded cell | sound scalable relaxation |
| CROWN original-box | original perturbation box | sound guard-dropping control |
| CROWN augmented-output | affine guard as output constraints | none; adapter unvalidated |

The primary guard result is the paired transition table, binary/relaxation
tightness where available, solved coverage, and runtime. A faster runtime is an
experimental outcome, never implied by binary-width theory.

## Current gate

No α,β-CROWN dependency was installed and no verifier run was launched. A
separate environment under `/data1/Kane/MOE` and a pinned conformance smoke are
required before promoting either CROWN adapter to a main experiment. The
existing `act-py312` environment remains unchanged.
