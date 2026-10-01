# Shared HybridZ endpoint mapping contract

This specifies the new opt-in adapter, not a new abstract domain or a proof of
network-to-HZ lowering. It instantiates the existing
[gate endpoint identity](gate_interval_endpoint_support.md) on actual guarded
`SparseHZono` objects. Native floating-point execution is outside the guarantee.

## Shared and private factors

Let an entry HZ have shared continuous/binary coordinates s_c,s_b. Expert A
extends these with private coordinates a_c,a_b; expert B extends them with
b_c,b_b. Their shared constraint prefixes must equal the entry coefficients
exactly and contain no private-column terms. The joint coordinates are

    continuous = (s_c, a_c, b_c)
    binary     = (s_b, a_b, b_b).

The LP relaxation orders all continuous coordinates before all binary ones and
relaxes every binary factor from {-1,1} to [-1,1]. The binary shift is the TOTAL
continuous width, not either expert's original continuous width.

The joint constraints are the entry rows once, then A's additional rows, then
B's additional rows, with the above injective maps. Outputs are the stacked
expert affine forms under those maps. A feasible joint assignment projects to
each supplied expert HZ; conversely any two feasible assignments agreeing on
the shared coordinates concatenate into a feasible joint assignment. This is a
statement about the supplied HZ objects. That true executions share precisely
this prefix and satisfy these HZs remains an upstream premise.

The independent-input control instead keeps disjoint copies of every expert
factor and row. Duplicating a shared assignment embeds the joint domain into
this product domain. Its weaker bound is therefore a valid relational ablation,
not evidence of an actual unsafe network execution.

## Exact property projection and coverage

For expert output dimension C, property qᵀy+c and weight t on A, the adapter
passes the rational output vector

    (t q, (1-t) q), offset c

to the original HZ batch support interface. This creates B+t(A-B) including the
constant exactly once. All products and sums of stored binary64 coefficients
and rational property weights are interpreted and checked as rational numbers;
there is no intermediate floating scalar-HZ projection.

Both distinct interval endpoints are required, with the minimum of their valid
lower bounds covering the independent gate interval. Equal endpoints share one
query only after exact equality checking. Every unordered pair and every supplied
property remain mandatory, including infeasible pair domains. The initial API
does not consume unproved route exclusions or partial-property reuse.

The checker independently reconstructs the column maps, retained rows, centers,
endpoint vectors and original-factor projections. It then uses the unchanged
exact residual-compensated dual checker. Positive acceptance requires every
aggregate property bound to exceed the frozen rational threshold 1/10000000.
Nonpositive lower bounds are UNKNOWN, not counterexamples. Missing whole-pair
evidence is explicitly UNKNOWN; malformed or missing endpoint rosters are rejected.

## Trusted boundary

The caller anchors the complete request before proposal. A hash binds what was
asked; it does not prove the request's network/guard/gate premises. Result status
`CHECKED_POSITIVE_GIVEN_GUARDED_HZ_AND_GATE` is conditional on those premises.
All-pair coverage closes the supplied obligation inventory, not upstream network
conversion. The checker implementation and parser remain trusted. The API uses
one cooperative deadline, not a hard-budget process supervisor or portable kit.
