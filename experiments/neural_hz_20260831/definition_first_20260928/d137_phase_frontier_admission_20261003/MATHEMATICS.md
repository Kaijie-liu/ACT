# Fiber novelty and phase coupling boundaries

These are paper results, not a new executable candidate. Throughout, an original signed HZ bit b is written as beta=(1+b)/2 in {0,1}; this is notation, not deletion or relaxation of its identity. Fractional beta values appear only when comparing terminal LP relaxations. Original source constraints, every retained EQ/LE predicate, shared readouts and input decoding remain part of the domain semantics.

## The current fiber and prior affine templates

The constant-cap component uses

```text
F(c,L) = { e : 0 <= e_i <= c_i + sum_{j<i} L_ij e_j },
c >= 0, L >= 0 strictly lower triangular.
k_i = max(0, w_i + sum_{j>i} L_ji k_j), in descending i.
h_F(w) = sum_i c_i k_i.
```

The support identity follows by eliminating the greatest-index amplitude. Its effective positive coefficient selects its affine upper endpoint; a negative coefficient selects zero. Coefficients of shared ancestors are combined before those ancestors are eliminated. Induction proves exactness for F, not for F intersected with arbitrary source/phase predicates.

DeepPoly defines lower/upper affine constraints referring to preceding variables, accompanied by valid concrete bounds. Substituting lower_i=0 and upper_i=c_i+sum L_ij e_j embeds this F exactly in that template; concrete upper bounds t=c+Lt satisfy its invariant. This is our specialization of the paper's definition. It does not identify the full nonconvex Neural-HZ element with DeepPoly. [DeepPoly, Section 4](https://ggndpsngh.github.io/files/DeepPoly.pdf)

Our descending, coefficient-coalescing elimination gives the recurrence above. Do not assert that unmodified DeepPoly's simultaneous substitution schedule always gives the same answer: with e1 in [0,1], e2 in [0,1+e1/2] and J=-e1+2e2, ordered elimination gives 2, whereas simultaneous replacement of both displayed terms can give 2+e1 and then 3. The embedding and the elimination derivation are enough to rule out claiming a new fiber set class solely from these inequalities.

## The current ReLU cap follows from an existing inequality

For g=h+sum a_j e_j, e_j>=0, h<=U_h and q=ReLU(g), the component installs

```text
q <= max(0,U_h) + sum_j max(0,a_j) e_j.
```

Anderson et al. give an input-aware family of ReLU inequalities, indexed by subsets of inputs, in Proposition 1, equation 6b. Our specialization treats h as one bounded aggregate, selects positive-coefficient amplitudes with lower bound zero, and leaves negative-coefficient amplitudes outside the subset. It yields the stronger inequality below, from which the current cap follows. [Strong mixed-integer programming formulations for trained neural networks](https://arxiv.org/pdf/1811.08359)

```text
q <= sum_j max(0,a_j) e_j + U_h alpha
  <= sum_j max(0,a_j) e_j + max(0,U_h),
where alpha is q's original activation bit.
```

Independent validity proof: if alpha=0, q=0 and the first RHS is nonnegative. If alpha=1, q=g and h<=U_h, while negative amplitude terms can only decrease g. The second inequality holds for alpha in [0,1]. Correlation between h and e does not invalidate this outer bound.

Thus beating a four-row ReLU LP with this cap is useful evidence of a stronger relaxation, but not evidence for a new inequality principle. No separation routine, split, backward rescue or new helper is proposed here. The independently retained mixed predicates still make the complete domain nonconvex.

## Phase gating alone cannot supply the missing gain

Suppose beta_i is certified to be the on/off bit for amplitude e_i. Consider

```text
F_beta = { e : 0 <= e_i <= beta_i (c_i + sum_{j<i} L_ij e_j) }.
p_i = beta_i max(0, w_i + sum_{j>i} L_ji p_j).
h_F_beta(w) = sum_i c_i p_i.
```

For fixed beta the same elimination proof applies. If c is independent of phase, F_beta is a subset of F_all_on and the all-on assignment is one of the unrestricted assignments. Therefore union_beta F_beta = F_all_on, and maximizing support over otherwise free phases gains nothing.

If the full predicates P already contain e_i<=U_i beta_i, the gated row and the ungated row define the same full set after intersection with P: an off bit already forces e_i=0, and an on bit leaves the cap unchanged. This is the current D136 situation. Moving a known relation into a query kernel can improve that query, but does not by itself change the full concretization or prove novelty.

The scope matters. In D124/D126 an error amplitude ReLU((1-2 tau_i) f_i) is gated by its parent's original alpha_i for tau_i=0, and by 1-alpha_i for tau_i=1. A target bit appearing in a cap is not automatically that amplitude's on/off bit. An unproved index-wise association is unsound. For phase-dependent c(z,beta), the fixed-phase support proof still applies when c is nonnegative at that fixed source and phase, but the all-on dominance argument above does not transfer.

A max/product expression of O(n+nnz(L)) size evaluates one fixed phase; it does not solve global optimization over constrained phases. Any proposed symbolic query must account for that remaining problem, without introducing prohibited enumeration or rescue paths.

## An ordinary control for genuinely using phase information

Consider two source coordinates x,y in [-1,1]:

```text
f1 = x + y/10 - 1/4,  f2 = -x + y/10 - 1/4,
q_i = ReLU(f_i),  J = q1 + q2 + y/20.
l_i = -27/20, u_i = 17/20.
```

Because f1+f2<=-3/10, the original bits satisfy beta1+beta2<=1, including all legal zero labels. The bound q_i<=17 beta_i/20 gives J<=9/10, attained at x=y=1. Hence ReLU(J-19/20) is zero.

The original four-row LP without the clique admits x=0,y=1, beta1=beta2=6/11 and q1=q2=51/110. It gives J=43/44 and positive successor input 3/110. Old HZ supplied with the same clique matches the improved bound. This illustrates the required role of retained phase relations, but is already the type of mechanism documented in [D014](../d014_guarded_amplitude_20260928/D014_EXACT_TRANSFER.md), not a new domain theorem claimed here.

## Exact radial factorization and its paid boundary

Suppose each derived amplitude has the certified relation 0<=e_i<=u_i beta_i to its ORIGINAL on/off bit, with u_i>0, and these bits satisfy a certified clique sum beta_i<=1. Replace those amplitudes by

```text
e_i = u_i beta_i t,  0 <= t <= 1.
```

This preserves the labeled source/output projection when substituted into every predicate and consumer. With one active k choose t=e_k/u_k; with all bits off choose t=0. Conversely the substitution satisfies the original amplitude bounds. The auxiliary gauge t<=sum beta_i preserves that projection.

Amplitude exclusivity is insufficient: legal zero labels may have beta1=beta2=1, e1>0 and e2=0, which no shared positive t can reproduce. The required original-bit clique cannot be inferred from a non-strict zero-amplitude conflict alone.

Generic consumers C e become sums of products C_i u_i beta_i t. A native linear/MILP lowering generally restores product variables or expands guarded rows. If every remaining amplitude consumer has identical normalized columns C_i u_i=v, its aggregate becomes v t under the gauge. The exact ReLU graph predicates for source-affine f_i can in this restricted case be replaced by

```text
f_i <= u_i beta_i,
f_i <= u_i t,
u_i t - f_i <= (u_i-l_i)(1-beta_i).
```

Here l_i<=f_i<=u_i must remain certified by the original retained source domain. At beta_i=1 the last two rows enforce f_i=u_i t. At beta_i=0 the first row enforces f_i<=0 and the last row is implied by t<=1 and f_i>=l_i. This proves exact integer ReLU semantics without deleting its bit. Arbitrary additional predicates involving individual e_i are still charged; this restricted graph lowering does not remove their products automatically.

For the two-channel source-affine control above with common sum consumer, count both formulations symmetrically:

| Quantity | Old HZ plus clique | Radial form |
| --- | --- | --- |
| Continuous coordinates including x,y | 4 | 3 |
| Original bits | 2 | 2 |
| Structural rows excluding scalar bounds | 7 | 8 |
| Structural nnz | 20 | 25 |
| Rows including all continuous scalar bounds | 15 | 14 |
| Corresponding nnz | 28 | 31 |

Old structural rows are q_i>=f_i, q_i<=u_i beta_i, q_i<=f_i-l_i(1-beta_i), plus the clique. Radial structural rows are the three displayed rows per gate, plus clique and gauge. Both count source coordinates explicitly. The last two table rows retain x,y box bounds and all amplitude/t lower and upper bounds, including redundant uppers. If redundant q_i<=u_i and t<=1 are dropped on both sides, totals are 13 rows/26 nnz and 13 rows/30 nnz respectively. Explicit continuous relaxation bounds on the two bits add four rows/four nnz to both. RHS, readouts, certificates and decoding storage are additional. There is no unconditional row, storage or runtime saving claim.

More importantly, the same ordinary example proves failure of old-LP inclusion. Set

```text
x=0, y=-1, beta1=beta2=1/2, t=4/5.
f1=f2=-7/20, u*t=17/25.
```

All radial rows, the clique and gauge hold. In particular u*t-f_i=103/100 <= 11/10. Its linear sum readout is 17/25. At that same source and those same original phases, old four-row constraints instead give each q_i<=-7/20+(27/20)/2=13/40, so their sum is at most 13/20, lower by 3/100. Integer exactness therefore does not imply retention of the old fractional proof power. This is a paper counterexample, not an executed benchmark regression.

The radial idea is retained only as a scoped hypothesis. It is not selected for implementation on the current real frontier: a large-model identity consumer violates the common-column condition, generic lowering may repay the removed variables, and the relaxation counterexample prevents a no-regression claim. Do not manufacture an equal-column network to avoid those conditions.
