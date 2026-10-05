# Common source lifting as a strong comparison

Date 2026-09-30; `redu-hz` at
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This is a paper comparison with
known extended formulations, not an implemented candidate. It accompanies
[the anchored-order investigation](D018_THEORY.md).

## Exact and locally ideal ordered lift

Let P be a bounded nonempty source polytope in R^d, and f_i(x)=a_i*x+b_i.
Suppose f_i>=f_(i+1)+delta_i on ALL of P, with exact delta_i>0. The original
bits are then ordered beta_1>=...>=beta_k, including every legal zero choice:
beta_i=0 gives f_i<=0, forcing f_(i+1)<0. Non-strict order would not suffice
for this bit-order inference at simultaneous zeros.

Set beta_0=1,beta_(k+1)=0,t_0=x,t_(k+1)=0 and add k vectors t_i in R^d.
For r=0,...,k write lambda_r=beta_r-beta_(r+1) and v_r=t_r-t_(r+1).
The comparison formulation has

```text
lambda_r >= 0
v_r in lambda_r P
a_r*v_r+b_r*lambda_r >= 0             if r>=1
a_(r+1)*v_r+b_(r+1)*lambda_r <= 0     if r<k
y_i=a_i*t_i+b_i*beta_i.
```

For a box, the perspective source rows are L*lambda_r<=v_r<=U*lambda_r.
For P={x:Hx<=h}, they are H*v_r<=h*lambda_r. Boundedness makes v_r=0
when lambda_r=0; silently dropping this condition would break the proof.

Integral ordered bits give exactly one lambda=1. The surviving v equals x,
so t_i=beta_i*x and every original gate value and legal phase are retained.
With relaxed bits, nonzero v_r/lambda_r reconstructs a source in the
corresponding prefix region. Conversely any mixture of such source tuples
satisfies these rows. Thus the formulation gives the local joint convex hull
in retained input, phase and output coordinates.

This is classical disjunctive convexification in cumulative coordinates:
[Balas, Disjunctive Programming](https://lara.epfl.ch/w/_media/projects/disjunctive_programming.pdf)
and [Vielma, Proposition 1](https://juan-pablo-vielma.github.io/publications/Embedding-Formulations-and-Complexity.pdf)
are necessary comparators. It is NOT a newly proved general neural domain.

## Cost and scope cannot be hidden

After eliminating the common source-sum equation, k*d additional continuous
coordinates remain. One vector beta_i*x can serve many consumers with THAT
same mask; it cannot substitute for all beta_j*x with independent original
bits. For d=27,k=2 this is 54 extra coordinates, not a reduction of ordinary
HZ's two output amplitudes.

With outputs materialized, box bounds and scalar phase endpoints already
shared, the joint lift has 2d(k+1) source rows, 2k boundary sign rows, k-1
internal order rows and k readout equalities. An independent extended ideal
formulation has k(4d+3)+(k-1) rows under the same convention. The joint lift
saves 2d(k-1) rows against THAT extended comparator, but ordinary four-row
gates have only 4k rows. Nonextended ideal facets are another fair comparator;
see [Anderson et al.](https://arxiv.org/pdf/1811.01988).

General source predicates cost (k+1) copies of their scaled nonzeros. Row
RHS, variable bounds, certificates, source references, input/output decoders,
full terminal matrices and any EQ-only slack coordinates are additional.
No complete physical or runtime benefit has been measured.

Three limitations are substantive:

1. The affine order must hold on P USED INSIDE the lift, not merely on the
   true upstream nonconvex relation. Otherwise the two boundary signs per
   increment do not enforce all intermediate signs. Adding all scaled order
   predicates or signs can erase the stated small-row advantage.
2. Retaining the actual upstream relation preserves integer semantics, but
   does not make this local hull ideal after arbitrary nonlinear source
   gluing. The D014 incompatible-source-mixtures counterexample still matters.
3. If original l/u bounds are tighter than validity over P provides, retain
   their old rows or prove them for every lifted component before claiming
   containment in the original LP.

The formula has no recursive search or new bits, but it explicitly represents
k+1 phase regions. It must NOT be relabeled as entirely free of phase
decomposition. No implementation is authorized by this paper comparison;
the project's no-split boundary is unchanged.

## Disposition

This construction explains what common-source consistency can cost. It is a
strong known baseline, not a solution to the definition task. We do not start
GPU construction, enumerate phase regions, or add a new solver to pursue it.
An admissible innovation needs a new composition/elimination benefit beyond
these known coordinates and must pay all retained source and phase semantics.
