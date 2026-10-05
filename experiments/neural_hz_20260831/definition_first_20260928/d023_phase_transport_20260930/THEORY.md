# Forward phase capacity transfer through a mixed neural block

This candidate transfers retained upstream phase information through a mixed affine map and a new ReLU using two linear inequalities, without new phase products. A strict two-layer scalar-output control survives considerably stronger old relaxations than ordinary big-M. This justifies a bounded component correctness experiment, not promotion as a new Neural-HZ domain. The mechanism is a statically constructed consequence of known subadditivity and RLT; the remaining research claim concerns useful neural composition and complete cost, not greater set expressiveness than HZ.

## Nonconvex carrier and transfer

Retain the entire original HZ source relation, all continuous factors and binary identities, every gate and EQ/LE predicate, shared frame identity and the original input decoder. Let f_i be predecessor preactivations, r_i=ReLU(f_i), and beta_i their original bits. Assume certified positive upper capacities u_i and difference bounds -A_ij<=f_i-f_j<=U_ij, with A_ij,U_ij nonnegative in this experiment.

For a receiver g=c+sum_i w_i r_i, choose a fixed nonnegative coefficient pairing lambda_ij between positive-weight i and negative-weight j, with row sums at most w_i and column sums at most -w_j. Let p_i and q_j be their nonnegative unmatched remainders. Then

```text
g = c + sum_ij lambda_ij(r_i-r_j)
      + sum_i p_i r_i - sum_j q_j r_j.
```

This is an algebraic identity, not an optimized transport problem. Our finite rule pairs consecutive canonical fan-in slots (0,1),(2,3),...; opposite-sign pairs use lambda=min(abs(w_i),abs(w_j)), same-sign pairs remain unmatched. Pairing never reads a solver state, label, property margin, learned multiplier or prior result.

Define effective capacities Ubar_ij=min(U_ij,u_i), Abar_ij=min(A_ij,u_j). On the exact predecessor graph,

```text
-Abar_ij beta_j <= r_i-r_j <= Ubar_ij beta_i.
```

For the upper inequality, beta_i=0 gives r_i-r_j<=0. When beta_i=1, monotonicity of ReLU gives r_i-r_j<=max(f_i-f_j,0)<=U_ij, and nonnegative r_j gives r_i-r_j<=u_i. The lower inequality is symmetric. At zero preactivation every original bit choice is still allowed. No phase is removed or fixed by this argument.

Set

```text
K_i = p_i u_i + sum_j lambda_ij Ubar_ij,
H_j = q_j u_j + sum_i lambda_ij Abar_ij.
```

Dropping nonpositive remainder terms gives

```text
c-sum_j H_j beta_j <= g <= c+sum_i K_i beta_i.
```

Since K,H are nonnegative, the exact new gate t=ReLU(g), m=t-g=ReLU(-g) satisfies

```text
t   <= max(c,0)  + sum_i K_i beta_i,
t-g <= max(-c,0) + sum_j H_j beta_j.            (1)
```

Both right sides are nonnegative even over the entire relaxed bit box. The new gate's original bit alpha and its old gate rows remain. Equation (1) is not a convex replacement of that gate. Adding it preserves the integer relation and can strengthen the LP. It discards some sign/bias information and is not an optimal transfer claim.

Negative or stable difference bounds can be conservatively widened to a zero endpoint before this formula. The initial prototype instead restricts itself to strictly crossing predecessor and selected difference bounds, returning failure on unsupported premises. That restricted qualification does not redefine the whole project around this subset.

## Monotonicity and bounded per receiver size

For every beta in [0,1]^n, K_i<=w_i^+ u_i and H_j<=(-w_j)^+ u_j. Thus (1) is no weaker than the corresponding two unpaired phase-subadditivity inequalities. A strict coefficient improvement needs a nonzero matched amount and an effective difference capacity smaller than its scalar capacity; real-network frequency is unmeasured.

For k fan-in slots there are at most floor(k/2) pairs. With retained g,t coordinates, the two new rows have at most 3+k coefficient nnz: one t coefficient, one t and one g coefficient, plus phase coefficients on disjoint positive/negative supports. No new continuous variable, bit or source copy is required for these two rows. Original source, affine and gate relations remain payable.

Difference-bound construction is not free. For d explicit common-source box coordinates, directly combining row differences before interval evaluation costs O(k*d) rational operations. A native CNN implementation needs tensor/operator-aware certified bounds instead of assuming every hidden source has a small d. Substituting g into the second row also incurs its actual support. Certificates, coefficient bit width, buffers, terminal lowering and input reconstruction count toward complete cost.

The intermediate pair rows need not all be stored: their graph validity and certified bounds justify generating (1) directly. This preserves concrete semantics but is NOT LP-equivalent to retaining every intermediate pair factor; it may lose other relations. A comparison must distinguish direct receiver summaries, persistent pair rows, unpaired summaries and HZ supplied identical final rows.

## A strict two layer terminal control

Take x,y in [-1,1] and

```text
f=x+y/4, h=x-y/4, r=ReLU(f), p=ReLU(h),
g=1/50+r-(11/10)p, t=ReLU(g),
z=t+(2/5)(r-f).
```

The predecessor scalar capacities are 5/4, difference capacities are 1/2. Use the same receiver input polytope Q={0<=r,p<=5/4, abs(r-p)<=1/2} and its exact g bounds [-121/200,13/25] in both comparisons.

The old comparison contains full retained-source ideal hulls of each predecessor gate, their phase-difference factor, the receiver's ideal hull on Q, and unpaired phase-subadditivity rows. It admits

```text
x=-11/20, y=0, r=6/25, p=19/100,
beta=eta=1/5, alpha=1/2,
g=51/1000, t=479/2000.
```

Explicit hull witnesses establish this membership without a numerical solver:

- For f, mix source (24/25,24/25) with weight 1/5 and (-371/400,-6/25) with weight 4/5. Their preactivations are 6/5 and -79/80, giving the displayed r, beta and source mean.
- For h, mix (19/20,0) with weight 1/5 and (-37/40,0) with weight 4/5. Their preactivations are 19/20 and -37/40, giving p, eta and the same source mean.
- For the receiver on Q, mix (r,p)=(47/100,1/100) and (1/100,37/100) equally. Their g values are 479/1000 and -377/1000; their differences 46/100 and -36/100 lie strictly within Q's difference limits. The mixture has the displayed r,p,g,t,alpha.

All these constituent inputs are interior to their respective input polytopes and relevant preactivations have strict signs. The predecessor difference factor passes at delta=0,d=1/20. Unpaired receiver capacities are 27/100 and 11/40; both permit t=479/2000 and t-g=377/2000.

The fixed pair has lambda=1 and negative remainder 1/10, so (1) instead gives

```text
t   <= 1/50+(1/2)beta = 3/25,
t-g <= (5/8)eta      = 1/8.
```

Both reject the old point. More importantly, combining the first with the original n_f=r-f<=(5/4)(1-beta) proves z<=13/25 for the full concrete network and the new LP. The old point has z=1111/2000=0.5555. Therefore the property z<=11/20=0.55 has a real safety margin of 3/100 but is not proved by that old combination. The true maximum 13/25 is attained at (x,y)=(1/4,1); none of the separating hull witnesses depends on that endpoint or zero phase.

This is an unequal-weight, nonzero-bias, two-ReLU control with a genuine scalar-output separation. It does not compare against a receiver ideal hull retaining every upstream bit and source condition: that stronger object would already imply (1). No executable test or benchmark result is asserted by this proof.

## Known mechanism and research boundary

[Sharp HZ section II-C](https://arxiv.org/pdf/2503.17483) supplies the relevant RLT comparison. Algebraically t=alpha*g on the exact gate; multiplying a nonnegative phase capacity by alpha, then using alpha*beta_i<=beta_i, yields (1). The proposed construction obtains these consequences without materializing those products, but does not exceed the general RLT framework. [Strong single-node formulations](https://www.columbia.edu/~wm2428/papers/mip_neural_networks.pdf) also explain why local ideal hulls need not preserve omitted source/phase conditions across composition.

Compared with the preceding phase-coherence block, the new result is a finite forward transfer that carries upstream phase information through another nonlinearity with two receiver rows. Ordinary HZ given exactly those rows is equivalent. A possible domain contribution still needs a useful compositional language and measured complete-cost advantage on ordinary structures, plus all qualification/replay gates. No novelty or promotion follows from the control alone.

The direct GPU form of these rows uses K beta and H beta, and transposes K^T and H^T. Static disjoint pairing uses absolute values, minima and masks on canonical coefficient pairs. Certification, source-difference bounds, packing, original constraints, all solver buffers and transpose accumulation remain part of the bill. No GPU implementation or speed claim is made in this document.

## Provenance and intended experiment

Date 2026-09-30, branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, pre-existing dirty worktree. Root and two independent read-only reviewers checked the transfer, cost and strict control. This file precedes a proposed isolated, default-off rational component experiment; its execution and outcomes will be recorded separately. It is neither a full Neural-HZ implementation nor a GPU fallback. Formal 1870/2413 and independent 61/400 remain unchanged.
