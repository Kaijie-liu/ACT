# Shared causal fiber reference implementation

This component tests whether the [causal fiber definition](../d135_causal_fiber_candidate_20261003/DEFINITION.md) can preserve joint, nonconvex neural semantics in a compositional implementation. It is a small exact-rational mathematical reference, not a new solver, a production Neural-HZ, or an established novel domain. The user explicitly requires progress in Neural-HZ itself rather than helper algorithms or storage optimization.

Date 2026-10-03 Australia/Sydney; branch redu-hz; commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. The unchanged tracked diff has SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. The goal authority and all prior archives remain read-only.

## Mathematical scope

The reference retains a common source and amplitude registry, original integral phase identities, linear equality and inequality predicates, and an original-input affine decoder. All consumers refer to the same amplitudes. Exact affine composition, addition and concatenation must not duplicate shared ancestors. Independently evolved branches without an authenticated common extension are rejected, not silently joined.

Each ReLU appends an amplitude and keeps its original phase, including at zero and for stable gates. It retains the exact four-inequality graph and adds the uniform causal upper relation from the definition. The relation is valid on the current abstract state; it is not licensed solely by a bound on original concrete trajectories.

The implementation specializes the causal row when its own current-element bound certifies the preactivation upper bound is nonpositive: set the new cap and all new causal coefficients to zero. ReLU is identically zero on that state, so this preserves the exact concretization and makes the zero available to subsequent cheap queries. Keep the original binary identity and all graph rows; at preactivation zero, both labels remain legal. This is one uniform certified-bound rule, not an instance or terminal-status choice. Without it, the generic row in the global control would be q3 <= 2 q2, whose fiber-only support is 8/5 even though the retained exact predicates force q3=0. The strengthened test must demonstrate zero support after construction and downstream affine consumption, not just evaluate one concrete point.

For a nonnegative strictly lower triangular L and nonnegative c, the intrinsic fiber query is

```text
F = {e : 0 <= e <= c + L e}
k_i = max(0, w_i + sum_(j>i) L_ji k_j)
support_F(w) = k dot c
```

This is exact for the fiber alone. It is only an upper bound after additional predicates are restored, and source-box maximization may lose source correlations. The recurrence is transparently a certificate for this structured fiber LP, not a generic backward verifier, failed-query rescue, or alternative witness generator. The original source and phases remain in the domain even when the cheap query ignores some constraints. A query optimizer is never an ADV.

All arithmetic in this reference uses exact rationals with the inherited 512-bit limit. The deliberately small reference supports at most 16 source coordinates, 32 amplitudes, 48 phases, 64 readouts and 256 predicate rows. Its fiber caps are constant, not arbitrary source-affine c; source mixed predicates and original signed phases are preserved, but a general HZ importer is not implemented. No binary64 or CUDA arithmetic soundness is asserted. Unsupported inputs, mismatched identities, uncertified tighter bounds and absent opt-in must fail closed. The mathematical tests do not qualify unrestricted rational workloads or physical storage.

## Discriminating controls

The positive control is the entire two-input residual example from the definition, including its successor ReLU. Its shared relation implies J <= 1/4 globally and hence the successor is zero. The old four-row LP admits the independently checked rational point J = 32/47 with the same source and bounds. That point is a relaxation diagnostic, not a real adversarial input or a claim about a computed LP optimum. Old HZ supplied with the same causal relation can match the new bound; this comparison must not be omitted.

Joint cancellation, shared concatenation and repeated consumers test the improvement over independently copied scalar fibers. Integral graph membership rejects the midpoint of two ReLU graph points. Zero preactivation accepts both original labels. Source predicates and the original decoder must survive composition.

## Unmet research obligations

This exact carrier still has HZ-expressible PWA denotations and retains all graph rows plus causal rows. Passing its tests would establish implementation consistency, not the requested definition innovation or a useful compression theorem. The full live Conv/Add frontier, source-conditioned fiber generality, efficient convolution representation, GPU kernels and their accounting, smooth activations, attention, and fair same-backend real-network comparisons remain unqualified. Neither fewer free variables nor better full LP/MILP optima are claimed.

The next capability test must compare original HZ, HZ with the same structural relations, and the candidate on the same entire ordinary live frontier, with the same source, backend and budget. No helper portfolio, attack, split, backward rescue or status-triggered switch may supply its gain. Existing tests containing frozen ordinary LP controls remain unchanged; those controls are not a new inference path in this component.

Formal baseline remains 1870/2413 and independent E0 remains 61/400. The candidate is default-off, has no production integration, and cannot change either score. All original shadow, per-family, full replay, physical and four-concurrent-request gates remain necessary for later promotion.
