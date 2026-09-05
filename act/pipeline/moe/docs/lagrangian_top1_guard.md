# Tie-safe Lagrangian hard-top1 guard compiler

## Purpose

This compiler is a narrow bridge from a hard-top1 routed program to a static
network backend that cannot directly retain input-side route halfspaces. It is
intended to test whether preserving router--expert dependence inside one graph
recovers information lost by full-box path verification.

For branch `i`, the tie-inclusive guard is

```text
m_ij = r_i - r_j >= 0, for every j != i.
```

For property row `s_l >= 0` and fixed nonnegative multipliers, the compiler
emits

```text
phi_l = s_l - sum_j mu_lj * m_ij.
```

On the legal branch `phi_l <= s_l`; proving `phi_l >= 0` over the original box
therefore proves the original row on that branch. Ties are retained because a
tied margin is zero. The full proof is in
`act/back_end/moe/proofs/lagrangian_top1_guard_compilation.md`.

## Implementation and fail-closed rules

- `LagrangianTop1GuardedProperty` embeds the router and expert in a single
  PyTorch graph with a shared input.
- Multipliers must be finite, fixed, and nonnegative. They may differ by
  property row.
- AdvMoE's optional ablation evaluates a frozen scalar multiplier grid and
  takes the best complete lower bound independently for each property row.
- A backend error, an incomplete call, a malformed bound matrix, or a
  non-finite value makes the aggregate `UNKNOWN`/`ERROR`; it cannot filter a
  row.
- A negative relaxation result is `UNKNOWN`. Only a concrete full-model replay
  can establish `UNSAFE`.
- Positive CROWN results remain
  `CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE`, because the installed backend has
  no outward-rounding contract.

## Evidence available now

The analytic toy conformance result is
`act/pipeline/moe/results/crown/lagrangian_guard_toy_conformance_20260906.json`.
On `x in [-1,1]`, branch 0 is legal for `x >= 0`. With safety `s=x+0.1` and
`mu=1`, the unguarded lower bound is approximately `-0.9`, whereas the
compiled row is the constant `0.1`. Replacing the safety offset by `-0.1`
keeps the tie unsafe and the compiled lower bound negative. Concrete grid
evaluation confirms `phi <= s` on every legal point.

An exact retained-HZ differential is recorded in
`act/pipeline/moe/results/crown/lagrangian_guard_exact_hz_differential_r2_20260906.json`.
For both the safe `s=x+0.1` and unsafe-tie `s=x-0.1` controls, exact guarded HZ
and the compiled `mu=1` property agree on the analytic lower bound within the
explicit `5e-9` comparison tolerance. The first execution used `1e-9` and
failed by about `1.1e-9` because the complete HZ support result includes
conservative numerical padding. That failure is preserved as
`lagrangian_guard_exact_hz_differential_r1_failed_20260906.json`; r2 changes
only the disclosed comparison tolerance, which remains below the project's
`1e-7` positive-margin threshold.

Together these controls establish the real-arithmetic reduction, executable
CROWN conformance, and a verification-scale exact-HZ differential. They do not
establish official-scale benefit, a formal CROWN certificate, or generic
novelty.

## Remaining experiment gates

1. Freeze the multiplier-selection protocol without observing the official
   test endpoint.
2. Run a paired official-scale comparison with identical expert backend,
   budget, samples, radii, and preprocessing.
3. Keep formal and numerical-filter endpoints separate in every table.
