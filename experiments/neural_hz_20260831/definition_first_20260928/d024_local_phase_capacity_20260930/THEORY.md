# Local phase capacity composition with retained predicates

This is a reference component toward a Neural HZ definition, not an admitted new domain. It implements the already derived mixed affine and ReLU transfer in the [preceding theory](../d023_phase_transport_20260930/THEORY.md), and makes its layer interface and cost explicit. The full HZ carrier, original bits, gates, shared input identities and decoder remain outside this small compiler and must remain unchanged in any later integration.

## Interface and concrete semantics

A layer frame consists of one identified bounded source vector x, affine predecessor preactivations f_i=a_i x+b_i, and distinct original phase identities beta_i. The concrete predecessor values are r_i=ReLU(f_i). The component never treats separately stored affine rows as independent source vectors. Its bounds are certified from the supplied common box; the original source may have additional EQ/LE and binary predicates, which are retained but not used to tighten these box certificates.

The receiver is g=c+sum_i w_i r_i and t=ReLU(g), with its original new bit and gate constraints. Fixed consecutive fan-in positions are paired only when weights have opposite signs, with lambda=min(abs(w_i),abs(w_j)). Scalar and pair preactivation bounds are computed by exact interval evaluation after combining affine coefficients. The initial component accepts strictly crossing predecessor bounds and strictly crossing selected pair bounds; unsupported premises fail closed, not through a second path.

As proved in the preceding theory, each positive predecessor receives coefficient
K_i=p_i u_i+sum_j lambda_ij min(U_ij,u_i), and each negative predecessor receives
H_j=q_j u_j+sum_i lambda_ij min(A_ij,u_j). Here -A_ij<=f_i-f_j<=U_ij and p,q are unmatched positive magnitudes. The two valid rows are

```text
t   <= max(c,0)  + sum_i K_i beta_i
t-g <= max(-c,0) + sum_j H_j beta_j.
```

All bits remain binary in the concrete domain, including independent choices at zero. These rows are redundant for the exact graph and may strengthen its LP relaxation; they do not change the input decoder. The component outputs only coefficients and proof premises, not a verification verdict.

## Composition without substituting every ancestor phase

At the next affine and ReLU block, use its immediate predecessor preactivation rows over their common input activation frame. Generate another two receiver rows in the immediate predecessor bits. Retain all earlier constraints: no ancestor phase substitution is needed in a newly generated row. This construction preserves the exact graph inductively and makes each stage's relaxed feasible set a subset of the previous one with the same newly appended old gate constraints.

This is a monotone strengthening theorem, not a theorem of strict final-output improvement at every depth. Nor does it eliminate older dependencies: they remain in the terminal predicate system and its solve cost. Discarding old rows to maintain only a small current frontier is NOT covered. If the model importer cannot supply the shared affine frame or certified bounds, this component is not applicable.

For a receiver with k fan-in entries and d explicitly stored source coordinates, bounds and fixed pairing cost O(k*d) rational operations in this reference representation. With retained g,t, the added two rows have at most k+3 nonzero coefficients. Across multiple receivers, this pays two rows per receiver and the sum of these supports; original gates, source rows and all bits remain. A decoder, sparse/tensor frame importer, Conv handling without dense expansion, certification bit growth, shared certificate storage, terminal buffers and solver work are not implemented here and cannot be priced as zero.

## Definition and novelty boundary

The component's semantic premise is a common-frame nonconvex phase graph; the derived phase capacities are observations of that graph. HZ with the identical rows has identical semantics and relaxation. Known monotonicity, subadditivity, RLT and symbolic bounds remain necessary comparisons. This alone is not the requested domain innovation, and does not claim to outperform known formulations at equal complete cost.

The reference test uses the preceding unequal-weight, nonzero-bias two-layer control and the three-layer control below. It checks exact arithmetic and explicit old-hull mixture witnesses, not an LP solver or neural benchmark. Real-network usefulness and broader multi-block applicability remain separate research obligations; this experiment must not replace those obligations with an easier goal.

## A third layer with an additional strict gain

Use the same first layer f=x+y/4, h=x-y/4, r=ReLU(f), p=ReLU(h), on [-1,1]^2. The second layer is

```text
g1=1/50+r-(11/10)p, t1=ReLU(g1), bit alpha1
g2=3/100+r-(6/5)p,  t2=ReLU(g2), bit alpha2.
```

Retain BOTH second-layer paired-capacity results, as well as all old gates and the first-layer phase-difference rows. On the common parent polytope Q={0<=r,p<=5/4, abs(r-p)<=1/2}, shared scalar bounds are g1 in [-121/200,13/25], g2 in [-18/25,53/100]. Every comparison uses those same bounds.

Append k=-1/100+t1-t2 and v=ReLU(k). Its input difference comes directly from g1-g2=p/10-1/100, giving [-1/100,23/200] even from the common parent box alone. The newly repeated rule gives

```text
v   <= (23/200) alpha1
v-k <= 1/100 + (1/100) alpha2.
```

Together with the retained gate row t1-g1 <= (121/200)(1-alpha1), the first row proves the actual residual output z=v+(23/121)(t1-g1) <= 23/200. This true bound is attained at x=1,y=-1. Thus the property z<=29/250 has margin 1/1000.

The old comparison, including both preceding paired-capacity transfers AND the second-layer parent phase-difference factor, instead allows

```text
(x,y)=(3/4,-7/10), r=7/10, p=1, beta=7/8, eta=9/10
g1=-19/50, g2=-47/100, t1=1/10, t2=39/500, alpha1=alpha2=1/5
k=3/250, v=51/2000, final bit=3/10
z=28251/242000 > 29/250.
```

This membership holds even for the separate retained-source ideal hulls of the first gates, the second gates' ideal hulls over Q, and the third gate's ideal hull over its input scalar box plus -1/100<=t1-t2<=23/200. Exact strict-sign, interior mixture witnesses are recorded and checked in test_capacity.py. They are local hull witnesses, not witnesses in a full joint upstream-network hull. The shared third preactivation bounds are [-1/50,21/200]. Its unpaired phase capacities also allow this point. The second-layer difference factor passes at t1-t2=11/500, g1-g2=9/100. The new positive row rejects it because (23/200)(1/5)=23/1000<51/2000. The property separation is exact and modest; it is not an estimate of expected real-network gain.

This example establishes that freshly generated local rows can add strict terminal strength beyond retained earlier rows. It does not establish strictness for arbitrary depth or a complete capacity-language closure. The gate graphs remain nonconvex and all original bits remain present.

The intended native row operations are K beta, H beta and their transposes. No GPU execution, all-GPU solver, CPU fallback for a GPU candidate, physical-memory qualification or acceleration claim is made by a rational reference compiler.

## Provenance

2026-09-30, branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, existing dirty worktree preserved. Formal baseline 1870/2413 and independent external 61/400 are unchanged. All new work is isolated here; old source and results remain read-only. The preceding D023 theory was read with SHA256 541fd2073f25363beaf9b592e6ae4f624420c361288b9940f8cf3da0034176df.
