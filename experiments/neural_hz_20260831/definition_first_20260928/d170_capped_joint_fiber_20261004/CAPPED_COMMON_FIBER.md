# A common witness for branch bounds and energy

This paper-only study produces a strict native refinement of the D157 consumer fiber. Branch boxes and energy must hold for the same proxy, not for two separately projected witnesses. A fixed linear consequence carries this improvement through a mixed source residual and the next ReLU on an entire input box. However, the strong old representation can prove the same property from the same information. This is a useful definition-level result, not an implementation candidate that has passed the comparative gate or a claimed Neural-HZ breakthrough.

## Native element and whole parent soundness

Retain the entire parent HZ state: continuous source identities, all original integer phase labels, EQ/LE predicates, shared lineage, readouts and the original input decoder. Let g be the new preactivation, b its fixed structural reference, d=g-b, and C the complete consumer carrier including the mass row as in D157. Let s=Cd. Every actual consumer must factor through this carrier; no live skip, predicate or decoder use may be silently omitted.

For reliable fixed bounds L_i<=g_i<=U_i over the entire current parent, use the same phase-refined displacement intervals

```text
I_i^1 = [max(L_i,0)-b_i, max(U_i,0)-b_i],
I_i^0 = [min(L_i,0)-b_i, min(U_i,0)-b_i].
```

The original sign guards remain. The harmless interval on an impossible branch does not authorize that branch; its original guard excludes it. At a zero preactivation both original labels remain legal. Indicators beta are the existing bits in 0/1 notation, not relaxed native phases.

Introduce p packet amplitudes a and the native predicate

```text
exists one t:
  C t = s,
  C D_beta t = a,
  t_i in I_i^(beta_i),
  sum_i t_i^2 <= E.

packet output = C D_beta b+a.                            (1)
```

E must bound ||d||^2 over the whole parent native domain, not merely original concrete trajectories. The true new gate image has the extension t=d, so (1) is sound for every parent member. Original HZ embeds when the bank list is empty. The source decoder is unchanged; a proxy witness is not a validated network input/output trajectory.

Affine, Conv and same-frame Add/Concat operate on the shared packet readouts. A later activation may use the same rule after certifying its bounds and energy over the entire enlarged parent. All previous banks, predicates and live dependencies persist. This establishes sound recursive semantics, not bounded-width or cheap-query closure. No phase split, attack, backward pass, dual rescue or instance-dependent choice is introduced.

The original binary nonconvex semantics is retained. Using boxes inside a phase-conditioned predicate does not replace the whole domain by a box, Zonotope or CZ. This bounded version refines the existing nonlinear fiber family; it does not claim a new general expressive class relative to all earlier nonlinear domains.

## Exact minimum energy characterization

Fix the original integer beta and write c_i=C[:,i]. Define

```text
phi_A(v) = min sum_(beta_i=1) t_i^2
           subject to sum_(beta_i=1) c_i t_i=v,
                      t_i in I_i^1;
phi_I(v) = min sum_(beta_i=0) t_i^2
           subject to sum_(beta_i=0) c_i t_i=v,
                      t_i in I_i^0.
```

An infeasible fiber has value +infinity. The compact nonempty feasible fibers attain their minima. Because the two coordinate groups are disjoint, (1) is equivalent to

```text
phi_A(a)+phi_I(s-a) <= E.                                (2)
```

The minimizers concatenate into one t; conversely any common t bounds the two minima. This differs from intersecting a projected box and a projected ball, whose witnesses need not coincide.

For C=c^T of rank one, each branch admits an explicit member check. First check the target v against its weighted interval sum. For nonzero c_i, set t_i(rho)=clip(rho*c_i,ell_i,u_i). The sum F(rho)=sum c_i*t_i(rho) is continuous, nondecreasing and piecewise affine, with at most two breakpoints per coordinate. Sort ell_i/c_i and u_i/c_i in their correct order; scan to the target segment and solve its affine equation. Zero coefficients take the interval point nearest zero; endpoints and flat segments cause no ambiguity in the unique minimizing t. Completing squares, or coordinate optimality plus the shared equality, proves minimum energy.

This costs O(m log m) exact arithmetic/comparison work and O(m) temporary storage before bit-length costs. With a certified precomputed full breakpoint order, a point query can scan it linearly. It checks a supplied source and integer phase assignment, not support over all sources and phases. The construction is a classical continuous quadratic knapsack calculation, not a novel solver; [Kiwiel's research report, sections 1–2](https://rcin.org.pl/Content/139441/PDF/RB-2002-77.pdf) gives the clipped scalar parameter and breakpoint formulation. No such algorithm was run or installed here. General p-dimensional membership is a bounded minimum-norm problem; D157's unrestricted Gram elimination cannot simply be reused as an exact checker.

## Fixed linear consumption without a new optimizer

For a fixed interval I define

```text
psi_I(z) = max_(v in I) (2zv-v^2)
         = z^2-dist(z,I)^2,
maximizer = clip(z,I).
```

Completing squares proves the formula. For any fixed lambda,mu in R^p, every native member satisfies

```text
2 lambda^T a + 2 mu^T(s-a)
 <= E
    + sum_i beta_i psi_(I_i^1)(c_i^T lambda)
    + sum_i (1-beta_i) psi_(I_i^0)(c_i^T mu).             (3)
```

Apply the scalar inequality to the single witness in (1) and sum. Endpoints and directions are fixed before terminal queries, so the RHS is affine in the new original bits, in addition to any already certified affine dependence of E on old bits. Using phase-dependent or source-dependent endpoints would require paying their products and cannot inherit this linear compilation claim.

Because psi_I(z)<=z^2, (3) with the same directions is no weaker than D157's energy row, including on the finite relaxation with beta in [0,1]. The six existing axis direction pairs can all be generated uniformly, with no new amplitude, bit, adaptive direction search or optimizer. This is direct scalar algebra; resemblance to a conjugate formula does not authorize optimization over multipliers or LP-state feedback.

One row scans all m columns and uses certified clamp/square arithmetic for both branch intervals. A zero direction coefficient cannot automatically be skipped: if its interval excludes zero, psi_I(0)<0 still contributes minimum energy. Omitting such a negative term is a sound weakening, not evaluation of the same exact RHS. Six axis directions per packet coordinate cost O(pm) arithmetic, plus exact or outward coefficients, source expansion and evidence. Finite rows remain an outer approximation of (1), not exact general bounded-energy membership.

## Strict native separation with an entire box successor proof

Let x,y in [-1,1] and

```text
d = (x/10, (x+y)/2, (x-y)/2),
b = (1/20,1/4,-1/4),
g = b+d,
B = C = (1,1,1),
s = C d = 11x/10,
E = 101/100.
```

All three preactivations cross zero over this ordinary two-source box. The energy is certified directly by ||d||^2=51x^2/100+y^2/2<=101/100. The complete declared packet is the mass Q=sum ReLU(g_i); the raw-source residual below is also retained.

At x=y=0, beta=(1,1,0), consider a=4/5. The unbounded minimum-energy proxy is (2/5,2/5,-4/5), with energy 24/25<E. The exact box projection also permits a: (1/10,7/10,-4/5) is a box witness. Both satisfy Ct=0 and CD_beta t=4/5. The aggregate branch caps and mass condition hold.

But a common boxed-energy witness cannot exist. The inactive coordinate is forced to -4/5. The active coordinates sum to 4/5 and the first is at most 1/10, so the minimum is attained at (1/10,7/10). Its total energy is

```text
1/100+49/100+16/25 = 57/50 > 101/100.
```

Thus (1) strictly refines even the intersection of the old energy projection and the exact box projection. It also strictly refines the strengthened D157 native domain on this control.

The improvement is consumable by a fixed row across the whole box, not only at this source. Choose lambda=3/5, mu=-3/5. For both labels the three relevant capped constants are respectively 11/100,9/25,9/25. Equation (3) gives

```text
(12/5)a-(6/5)s <= 46/25,
a-s/2 <= 23/30.
```

Since beta^T b<=3/10, the physical mixed-source readout obeys

```text
J = Q-11x/20 <= 16/15 < 27/25.
```

Hence ReLU(J-27/25) is zero over the whole new native domain, with preactivation at most -1/75. The old separately witnessed point gives Q=J=11/10 and a false successor amplitude 1/50. This is a paper stability control, never a benchmark CERT or an ADV.

The result does not require selecting directions from the false point. The existing structural normalization t=sqrt(E/||C||^2)=sqrt(101/300) lies in [29/50,3/5]. For any certified t in that interval, the opposite direction row yields

```text
a-s/2 <= 1/(4t)+1/20+t/2 <= 559/725,
J <= 1553/1450 < 27/25.
```

Thus the standard axis rule also suffices. A numerical implementation would still need to certify its outward t lies in that interval before inheriting this control; no rounding implementation ran.

## The strong old comparison already proves the property

For exact original gates on the same source,

```text
J = beta^T b + (1/2) sum_i (2 beta_i-1)d_i
  <= 3/10 + ||d||_1/2.
```

Using only the same coordinate bound and energy information,

```text
||d||_1 <= 1/10+sqrt(2E),
J <= 7/20+sqrt(2E)/2
  < 7/20+5/7 = 149/140 < 27/25.
```

Here 2E=101/50<100/49. The rational old bound 149/140 is even smaller than the fixed new bound 16/15 by 1/420. It is a known grouped absolute-value/energy consequence of the original graph, not a helper solver. On this particular source there is also the stronger identity |d2|+|d3|=max(|x|,|y|)<=1, giving J<=17/20. An old path supplied these certified consequences also proves the successor stable.

The new domain's strict native refinement is real, but this control does not show new capability over the strong same-information reference. Merely showing a result beyond D157's current finite or native approximation would not meet the full research objective.

## Full cost and disposition

Keeping D157's layout and replacing its six energy RHS formulas leaves 2m+10p+2r rows, where r is the number of declared B rows. It does not remove original guards, phases, sources, inherited predicates, packet readouts or evidence. General native membership becomes harder even though this finite row count is unchanged.

For p=r=1 this is 2m+12 rows, versus 4m+1 for the old four-row graph plus one same-information consequence. A row-count window begins at m>=6, but this is not a runtime or total-memory result. The three-gate control has 18 versus 13 rows, and one versus three new amplitudes. Coefficient fill, all consumer mappings, source and bank storage, scalar membership, rounding, terminal conversion and concrete witness verification remain payable.

There is no demonstrated low-dimensional complete carrier for the priority real networks here. The saved D152 audit reports terminal 100-by-100 and 200-by-200 consumers and no AveragePool, GlobalAveragePool or ReduceMean in its three fixed models. These metadata do not prove impossibility of another useful structure; they do prevent assuming a small pooling interface without evidence.

Keep this native refinement and its exact control as research evidence. Do not launch an implementation solely from this control: the strong-reference advantage and practical complete-interface/query cost remain unproved. In particular, do not hide general membership behind a QP/SOCP helper or recover all per-gate proxy variables without charging them. Next work must identify a non-redundant relation or paid representation advantage on complete ordinary interfaces, rather than changing norms or adding another wrapper.
