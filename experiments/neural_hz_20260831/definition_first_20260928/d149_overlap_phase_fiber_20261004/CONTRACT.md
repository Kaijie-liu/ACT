# Shared overlapping phase fibers mathematical component

This default-off reference implements the definition-first D148 replacement with one shared residual coordinate per new gate and multiple overlapping coordinate-norm constraints. It tests the proposed domain and its propagation, not a helper added to a retained exact neural graph. It is a mathematical component only: no model worker, production integration, GPU claim, source qualification or benchmark gain is registered here.

## Definition and sound transformation

Readouts are affine in original box sources, original Boolean views of signed phases, and uniquely owned residual coordinates. The state retains its original input decoder and linear EQ/LE predicates. A residual bank e has one coordinate identity and a family of constraints norm(P_C e)<=R_C(beta), with phase-affine nonnegative radii. Coordinate sets C may overlap; no independent copy is created for a window. All original phases, including both legal zero labels, remain integral in the native concretization.

For g=Wy+d, use one deterministic forward nominal reference. Original normalized source and residual references are zero; initial free HZ phases have reference zero. Each newly created phase gets reference one exactly when its own gbar is strictly positive, and zero otherwise. For later operators, gbar is the constant affine coefficient plus the old phase coefficients evaluated at those fixed references tau. This is neither phase search nor a solver-dependent choice; the reference need not satisfy the retained predicates. The fresh actual phases gamma and residual e_new satisfy

```text
q=(1/2)g+(diag(gamma)-(1/2)I)gbar+e_new.
```

For each target coordinate set T, the actual remainder is (1/2)diag(2gamma-1)P_T(g-gbar). A reliable norm bound B_T(beta) therefore licenses norm(P_T e_new)<=B_T(beta)/2. Retain sign guards, q>=0, q>=g and q<=U_positive*gamma with reliable preactivation bounds. Do not retain the new gate's exact active-value upper relation; this is a lossy representation replacement. The whole-bank constraint is always retained alongside additional local constraints, and all parent constraints remain unchanged.

For an old residual bank and a complete cover C, let deg_i count occurrences and D_C=diag(1/deg_i) on its selected coordinates. The identity sum_C P_C^T D_C P_C=I proves

```text
norm(M e) <= sum_C norm(M P_C^T D_C) R_C(beta).
```

Compute the original D148 whole-only bound for this new bank and the fixed-cover bound as separate certificates. Retain that original whole-only constraint, not merely the old parent's whole constraints. The local cover degree uses only the declared local groups, excluding the separate whole-bank certificate. Keep both when distinct; never replace two affine radii by an unrepresented pointwise minimum or assume cover queries dominate whole-bank queries. If no local cover is declared, the whole constraint alone covers the bank. Duplicate coordinate identities inside a group, omitted coordinates in a declared complete local cover, or unauthenticated cross-frame joins fail closed.

The source-box contribution uses a reliable Gram bound (or a separately proved looser bound) on normalized xi in [-1,1]; a non-unit physical box must first contribute its midpoint and half-width to the affine map. The reference implementation keeps physical source coordinates: gbar includes C*midpoint, and its source norm uses C*diag(halfwidth), which is the same centering without changing source identity or decoder. Phase columns use reliable Euclidean norm bounds d_j multiplying |beta_j-tau_j|=tau_j+(1-2tau_j)beta_j. Negative radius coefficients are allowed; the full radius must remain nonnegative on the Boolean box. Old bank maps use a certified matrix norm bound, not an unchecked SVD or a free optimization oracle. Rational square roots are enclosed outward and squared certificates checked. All arithmetic is exact rational with the inherited 512-bit cap; floats, nonfinite values and unproved numerical premises are rejected.

Affine/Add/Concat combine coefficients of the same latent before applying any norm inequality. A parent-frame readout may be extended to an authenticated descendant. Independently evolved branches without a common authenticated extension are rejected. Whole/local queries emit sound phase-affine support certificates; scalar upper bounds may maximize these certificates over the full Boolean box without claiming exact support of the intersected domain. No dual optimization, backward rescue, splitting or new solver is introduced.

## Interval coefficient bridge

Real saved BN-folded coefficients are intervals, not exact fixed parameters. For A in [Alo,Ahi], b in [blo,bhi], choose their rational midpoint Ahat,bhat only as a nominal affine map. Given reliable coordinate bounds |y_j|<=M_j, introduce a shared parameter-error bank zeta with

```text
|zeta_i| <= delta_b_i+sum_j delta_A_ij M_j
true g = Ahat*y+bhat+zeta.
```

Retain one bank across all subsequent consumers, plus its singleton and whole norm constraints; do not claim the independent error enclosure retains every exact BN correlation. This is sound outer modeling of coefficient uncertainty, not permission to change network parameters. The interval bridge is tested mathematically only; no real BN/model binding or framework floating-point equivalence is qualified.

## Scope and full cost

The implementation uses dense rational affine maps and small explicit dense matrices for norm certificates, subject to explicit API size/work limits. Such reference limits are not a redefinition of the full research objective or a real-network admission. The implemented input decoder is source-affine only; arbitrary decoders depending on phase or residual coordinates are not supported. This covers ordinary box inputs but is not a qualification of every possible inherited HZ decoder. Every source, old phase column, residual coordinate, norm incidence, predicate, supported decoder and output coefficient remains payable. Local constraints add incidences and support work; they do not remove residual dimensions. No fixed-width, net memory or measured speed advantage is claimed.

The motivation comes from saved real layouts: first ReLU fronts have 65536, 14400 and 46656 coordinates, while an interior Conv3 consumer has 576 canonical inputs. A global norm alone can dilute a local bound; localization by itself is established norm reasoning and not a novelty claim. Ellipsotope-style duplicated coefficient blocks with equalities can also express overlapping continuous constraints. The research question is useful nonconvex compositional propagation at full cost, not merely renaming that construction.

Ordinary terminal LP/MILP and independent concrete validation retain their existing boundary. Unsupported identities, bounds, arithmetic and resources fail closed. Native membership or a relaxed feasible point is not a validated ADV. The D148 same-parent integer comparison does not imply fractional relaxation dominance or independent multilayer dominance.

A fixed local-strength control uses four independent sources in [-1,1], g_i=x_i+1/2, and the three groups (0,1), (1,2), (2,3). At x=0 and all actual phases active, q=(7/5,1/2,1/2,1/2) has residual (9/10,0,0,0). It satisfies the whole-only radius 1 and all retained linear gate rows, but violates the first local radius sqrt(2)/2 (including the registered outward rational enclosure). Thus the new local intersection can strictly strengthen the native whole-only abstraction, while keeping the concrete q=(1/2,1/2,1/2,1/2). This is a small mathematical control, not a real-network result, and exact HZ already rejects the spurious point. The target advantage must be useful verification at an affordable complete cost, not greater set precision than the exact network relation.

## Qualification requirements

All source, tests, this contract, registration, runner and collection plugin must be frozen before import, AST, compilation or execution. Preserve the complete D136 population of 3965 tests across 208 files and append the new registered tests. Retain the 60-second startup/collection/pytest/JUnit gate, one CPU/thread, CUDA hidden, AS16GiB, the inherited supervisor memory observations and automatically saved evidence. Do not execute old consumed runners or modify frozen files.

No source worker or GPU stage is registered by this contract. Existing actual-network archives remain read-only. New artifacts stay under this fresh directory and its fresh result directory. Branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Formal 1870/2413 and separate E0 61/400 are unchanged; formal gain is zero and the goal remains active.
