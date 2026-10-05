# What a bounded phase interface preserves and forgets

There are two different claims: retaining all earlier valid predicates gives a sound compositional graph, while replacing those predicates with only a small current capacity interface can lose precision. The local compiler in this directory does the former. It does not claim the latter replacement is exact.

## A latest bit alone cannot preserve the older capacity

Use the preceding two-layer D023 network and its last original bit alpha. A concrete point with alpha=0 has t=0; another with alpha=1 attains t=13/25. Therefore every affine function A(alpha) that upper-bounds t for all concrete points must satisfy A(0)>=0 and A(1)>=13/25. It follows that A(1/2)>=13/50.

The old relaxation point has alpha=1/2 and t=479/2000<13/50, so it passes every such latest-bit-only affine capacity. Similarly, the negative amplitude t-g attains 121/200 when alpha=0 and is zero when alpha=1. All valid latest-only negative capacities at alpha=1/2 are at least 121/400, admitting the old negative amplitude 377/2000.

The retained ancestor capacity t<=1/50+beta/2, in contrast, evaluates to 3/25 and rejects that same old point. This is a strict information-loss argument for that interface projection, not for the complete HZ model with its earlier rows retained. It is not a general impossibility theorem about Neural HZ.

## Why one affine capacity is not closed under keeping both bounds

A new activation can have both a direct phase cap u_i beta_i and an inherited cap C_i(beta_old). Keeping their strongest combined upper envelope gives

```text
P_i = min(u_i beta_i, C_i(beta_old)).
```

This is generally not affine. Subsequent positive weighted sums produce sums of minima. Flattening all combinations can generate exponentially many affine pieces. Keeping only one alternative is sound but can lose the other relation; treating this choice as exact simplification would be incorrect. No search or phase split follows from this observation.

There is an alternative mathematical interface: retain a monotone circuit with nonnegative affine leaves, nonnegative weighted sums with nonnegative constants, and minima. Its hypograph has a linear extended representation. For each circuit node add a nonnegative continuous capacity q:

- At an affine leaf, impose q<=A(beta).
- At a sum node, impose q<=c+sum w_j q_j, with c,w_j>=0.
- At a minimum node, impose q<=q_j for each child.
- For each observed nonnegative activation, impose r<=q_root.

For fixed beta, induction gives q_v<=C_v(beta). Conversely setting every q_v=C_v(beta) satisfies every node constraint. Thus existentially quantifying these capacity auxiliaries is exactly equivalent to the selected upper-capacity circuit, including shared subexpressions and multiple roots. This equivalence requires the stated nonnegative leaves and monotone consumers; it does not apply to arbitrary signed uses of q.

The circuit can avoid enumerating affine pieces, but costs one continuous quantity per node and constraints proportional to nodes, edges and affine-leaf support. Those quantities are not new original network bits, but they are additional solver storage and work. No such circuit has been implemented or qualified in this turn, and its extended formulation is not asserted to be a new convexification principle.

## The implemented choice and its research obligation

The present component avoids ancestor substitution into each new row. It keeps old predicates, reads the current shared affine frame, and emits at most two new capacity inequalities per receiver. This has a simple sound composition proof and a three-layer strict output control, with no new component variables or bits.

That row-count result does not bound the entire analysis by a fixed frontier width, nor make the old predicates or difference-bound certification free. The explicit common-source reference costs O(k*d) per receiver; a native sparse Conv implementation must account for actual supports and shared coefficients. The next research step must determine whether ordinary network structures yield useful certified pair capacities at acceptable complete cost, and whether a richer relation language adds more than this known-rule component.

This note is a paper proof and interface audit. It changes no benchmark result or runtime rule. It prevents the local compiler from being misreported as a complete, lossless fixed-width phase domain.
