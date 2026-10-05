# Joint amplitude interfaces and their complete cost boundaries

This supporting record preserves the other completed paper investigations from the same continuation. They are not alternative production paths or a menu of promoted candidates. Their common question is whether joint amplitude information survives compression with all original bits and shared source intact.

## An exact projection for three overlapping min max gates

Consider the original three ReLU gates and their original bits beta1, beta2, gamma:

```text
f1=a-b, q1=R(f1),    f2=a-c, q2=R(f2),
r=R(q2-q1),         Y=a-q2+r,
w=a-Y=q2-r.
```

On the integer graph Y=max(min(a,b),min(a,c)). Let l_i<=0<=u_i be the original bounds for the three gate preactivations. Set L_i=max(0,f_i), U_i=min(u_i beta_i, f_i-l_i(1-beta_i)) for i=1,2; C1=-l3(1-gamma), C2=u3 gamma.

For the LP comparison all three bit views have bounds [0,1]; the domain itself retains their original integer values. The twelve-row theorem concerns these gate rows and retained-source constraints. Additional predicates must depend only on retained coordinates or have a separately proved retained-interface rewrite; arbitrary predicates or raw consumers of q1,q2,r cannot be discarded.

The third gate becomes q1 in [w,w+C1], q2 in [w,w+C2]. Each parent also requires q_i in [L_i,U_i]. Eliminating the two intervals gives the following twelve rows:

```text
 f1-u1 beta1 <= 0             -f1-l1 beta1 <= -l1
 f2-u2 beta2 <= 0             -f2-l2 beta2 <= -l2
 w-u1 beta1 <= 0               w-f1-l1 beta1 <= -l1
 w-u2 beta2 <= 0               w-f2-l2 beta2 <= -l2
-w-l3 gamma <= -l3             f1-w-l3 gamma <= -l3
-w-u3 gamma <= 0               f2-w-u3 gamma <= 0.
```

The two own-interval feasibility conditions L_i<=U_i reduce to the original two phase sign rows per parent. The other conditions are w<=U_i and L_i<=w+C_i. Conversely choose q_i=max(0,f_i,w), r=q2-w. The intervals ensure all old rows. This proves exact projection of the specified continuous LP as well as the labeled integer relation. It neither enumerates phases nor duplicates the source.

Treating a,b,c as three physical source columns, with strictly crossing bounds and no coefficient cancellations, the original formulation has 6 continuous coordinates, 3 original bits, 12 rows and 30 coefficient nnz. The projected one has 4 continuous coordinates, the same bits and rows, but 36 nnz. The separate output readout changes from three coefficients to two. Thus fewer amplitudes do not establish net cost reduction.

Do not silently impose w>=0 in this LP equivalence theorem. For example a=b=c=0, parent bounds [-2,2], all three relaxed bits 1/2, q1=q2=0 and r=1 satisfy the old rows with l3=-2,u3=2; w=-1. Nonnegativity is true on the integer min graph and can be added as known min-order information, but the fair old comparison must receive the same strengthening r<=q2.

## Composable min trees still have a nonzero fill bill

For max_i min(a,b_i)=a-min_i R(a-b_i), use a binary min tree. There are m leaf ReLUs and m-1 original comparison gates. Keep every internal min output w and every original bit; eliminate only leaf amplitudes used by a unique min parent. Reparameterizing comparator amplitudes by internal min outputs is invertible before projection.

For a min edge with inputs v,q and output w, the four comparator rows are w<=v, w<=q, v<=w-l(1-gamma), q<=w+u gamma. If q is a leaf, intersect its parent interval with [w,w+u gamma] and eliminate q as above. A leaf's parent guards and comparisons become six rows; each internal-child edge retains two rows. The total is 6m+2(m-2)=8m-4, matching the old 4(2m-1) rows. All leaf lifts are simultaneous because each leaf has only one consumer.

For m>=2 the continuous amplitude count changes from 2m-1 to m-1, while all 2m-1 bits remain. However, with d_i actual nonzero source coefficients in f_i, strictly crossing bounds and no accidental column overlap/cancellation, the old min-coordinate strong reference has

```text
sum_i (2d_i+6) + 10(m-1) nnz.
```

The projected formulation has sum_i(4d_i+10)+5(m-2) nnz. The increase is 2 sum_i d_i-m, at least m for nonconstant leaves. With independent a,b_i columns it is 3m. These counts are for the specified physical-coordinate representation; extra bounds, side predicates, expanded source coefficients and witness costs remain due. General raw consumers or shared leaf q_i invalidate the displayed deletion bill.

This is a specialized interval/Fourier-Motzkin projection, not a new lattice domain or a demonstrated MILP speed improvement. [Fast BATLLNN](https://arxiv.org/pdf/2111.09293) already exploits TLL min/max semantics and box-like output properties; its architecture/property specialization is not proof of a cheap arbitrary-CNN conversion. The current project will not switch its full objective to TLLs merely because this exact projection is available. [D004](../d004_observable_interface_20260928/D004_RESULTS.md) is the earlier aggregate-projection and source-consumer comparator.

## Small value errors do not preserve the original phase interface

For g=h+aR(f), a>0 and kappa>=a, convexity gives the quantitative bound

```text
0 <= E=R(g)-R(h) <= (a/kappa) R(h+kappa f).
```

When f<=0 the left side is zero. When f>0, express h+af as the convex combination of h and h+kappa f. A reliable B>=sup(h+kappa f) gives E<=(a/kappa) max(0,B). This is elementary convexity, not a newly discovered error calculus.

For multiple positive edges, choose t_i=a_i/kappa_i>0 with sum t_i<=1. The same argument gives

```text
0 <= R(h+sum_i a_i R(f_i))-R(h)
  <= sum_i t_i R(h+kappa_i f_i).
```

Common-source secant upper bounds can be combined before boxing, but their coefficients and uncertainty remain charged. Replacing the sum of rectifiers by one rectifier of the summed f_i is unsound even when sum t_i=1: h=-1/4, f=(1,-2), a_i=1/2 and kappa_i=1 give positive error 1/4, whereas R(h+sum_i kappa_i f_i)=0.

Even E=0 does not establish phase equivalence. For h=-1 and a=kappa=f=1, g=0 permits beta_g=1 while h<0 does not. Nor may the true-value error bound be reused in r=beta_g*h+e: h=-1/2,f=a=kappa=1 gives r=1/2, beta_g=1 and E=1/2, but 0<=e<=1/2 would imply r<=0 in that representation.

There is also an ordinary fixed-phase obstruction to making R(h) a free exact helper. Let x in [-1,1], f=1+x/10, h=x/2, a=1, so g=1+3x/5>0. All existing gate phases are fixed active, but v=R(h) has a kink at x=0. In a linear-readout HZ with only those fixed original bits, continuous EQ/LE fibers and their linear projections are convex. The exact graph of (x,v,E) is not: the midpoint of its x=-1 and x=1 states has v=1/4 while the exact x=0 value is zero. Thus an exact retained helper needs additional nonconvex representation, or its graph must be approximated. Eliminating the helper pair back to r=g removes this claimed benefit. This is a scope theorem for linear HZ fibers, not a ban on nonlinear generator definitions or additional factors.

## Total amplitude normalization and aggregate complementarity

The proposed interface q=s p, s=sum q, sum p=1 and p_i<=beta_i fails when all q_i=beta_i=0: it forces both p=0 and sum p=1. Masking amplitudes s p_i instead, or adding an explicit zero-mass convention, repairs inclusion but needs its own predicates/products. For s>0, (s,p) is only a bijective reparameterization with 1+(m-1)=m degrees of freedom. Dropping source/direction coupling is an approximation, not exact compression. No main-domain implementation is selected from this proposal.

There is an exact local comparison. For a bare masked cone with beta binary, y>=0, s=sum y<=U and beta_i=0 implying y_i=0, its convex hull is already

```text
0<=beta<=1, y>=0, s=sum y<=U, y_i<=U beta_i.
```

For U>0, put mass m_i=y_i/U on the state y=U e_i and mass 1-sum_i m_i on y=0. In state j force bit j to one. On all other components together, whose total mass is 1-m_j, give bit j conditional probability (beta_j-m_j)/(1-m_j). This is in [0,1] since m_j<=beta_j<=1. If m_j=1, beta_j=1 and there is no other component. These choices can be combined into a finite mixture of binary assignments, proving sufficiency; necessity is immediate. The other components include nonzero states i!=j, not only the zero-amplitude state. For U=0 the claim is just the binary cube hull at y=0. This is a proof, not a runtime enumeration. The set deliberately has no actual preactivation/source guards, so this is not the full neural graph hull. Normalizing it cannot yield stronger linear queries than these ordinary rows without additional source/direction information. [Günlük and Linderoth](https://optimization-online.org/wp-content/uploads/2008/06/2014.pdf) provide established perspective/indicator convex-hull context; the displayed short cone proof is given here rather than attributed to an unverified theorem in that paper.

With n=q-g, q>=0,n>=0, the single equation sum_i q_i n_i=0 is equivalent to every scalar complementarity equation, because all terms are nonnegative. The original phase guards must still be retained; otherwise original bits become free labels. This is the known neural linear-complementarity representation, not a new discovery; see [Aydinoglu et al.](https://arxiv.org/pdf/2011.07626), Lemmas 1 and 2. Aggregating into a dot product does not prove cheap global feasibility or GPU verification. Existing [D114](../d114_quadratic_relation_boundary_20261002/THEORY.md) already records its semantic/prior-art boundary.

No candidate code or execution follows from these rejected shortcuts. Date 2026-10-02 Australia/Sydney; redu-hz; HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Paper only, no new score, no old archive modification.
