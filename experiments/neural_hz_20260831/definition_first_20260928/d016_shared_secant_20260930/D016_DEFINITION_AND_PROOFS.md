# Shared source secant generators for Neural HZ

This paper candidate studies exact compression of several unstable ReLUs sharing
a source-relative increment structure. Four shared secant-cone predicates imply
the lower gate rows of an entire output family. Retaining its original upper
rows preserves every original binary phase and gives a relaxation contained in
the original LP. A restricted residual composition theorem reuses the same
continuous factors across depth.

This is a mathematical result with independently reviewed proofs, not an
implemented or qualified Neural-HZ domain. The lifting primitives have known
prior art. Novelty of the complete neural compression/composition result and its
prevalence in ordinary trained networks remain unestablished.

## Scope and provenance

Research identifier D016; date 2026-09-30; branch `redu-hz`; commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, with the pre-existing dirty worktree
preserved. Configuration is paper derivation and read-only source/literature
inspection. No candidate import, test, model run, LP call or GPU kernel executed.

This follows the [cross-domain review](../../literature_cross_domain_20260929/REVIEW.md)
and [D015 checkpoint](../CHECKPOINT_D015_V2_20260928.md). The latter exposed a
limitation of independent baseline bounds, not a proof that cross-zero
correlation is useless. All goal restrictions and formal ledgers remain intact.

## Candidate elements and concrete semantics

Start with an owned HZ source frame: bounded continuous coordinates, all original
binary factors, EQ/LE predicates, and the original input decoder. A proposed
element additionally holds certified source-relative secant bundles and affine
readouts from their continuous factors. It does not replace its exact mixed
source by an independent convex envelope.

A bundle records shared source expressions h and v_j, constants c_j, the
ALREADY PRESENT exact source gates

    p = ReLU(h), with original bit eta,
    r_j = ReLU(v_j + c_j h), s_j = ReLU(v_j),
    d_j = r_j - s_j, n = p - h.

Here d_j and n may be inline expressions, not new coordinates. Every source
gate, source predicate, bound and phase remains. Creating missing anchor/source
gates would be a different construction with its whole cost payable; it is not
assumed free. Source expressions may depend on the same retained nonlinear
prefix, provided their stated equalities are certified on that exact frame.

Visible values are affine readouts of original coordinates and the new t_j.
Strong concretization returns the original input, ALL original binary values,
and the relevant source/output ports satisfying the predicates. Removed output
auxiliaries are reconstructed by their readouts. Semantic order is inclusion
after aligning source and original binary identities. No computable best
abstraction or lattice completeness is claimed. An ordinary HZ embeds as an
element with an empty bundle list and unchanged readouts.

The retained source ReLU graphs and integral phases keep the element genuinely
nonconvex. For the control below, the midpoint of the source tuples at h=-1 and
h=1 with v=0 has p=1/2 at h=0, and is not a source tuple. The cone inequalities
alone are not substituted for those exact source graphs.

This grammar alone is an extended HZ formulation. The contribution under
investigation is the following neural transfer and composition theorem, not
greater exact expressiveness, renamed products, or deferred matrix storage.

## Shared secant compression theorem

Suppose m original target preactivations have certified identities

    f_i = a_i h + sum_j A_ij d_j,
    alpha_i = a_i + sum_j min(0, A_ij c_j) >= 0.

Retain their original bits beta_i and valid original bounds l_i <= 0 <= u_i.
Use strict alpha_i > 0 when claiming that f_i shares both the sign and the zero
set of h; the weaker nonnegative condition suffices for the exact replacement.

Introduce k shared continuous factors t_j. With c_j^- = min(0,c_j) and
c_j^+ = max(0,c_j), add four linear inequalities per j:

    c_j^- p <= t_j <= c_j^+ p,
    c_j^- n <= t_j - d_j <= c_j^+ n.                 (C)

Replace each old output auxiliary by the affine readout

    q_i = a_i p + sum_j A_ij t_j.                    (Q)

Keep the two actual original upper gate rows, with this readout substituted:

    q_i <= u_i beta_i,
    q_i <= f_i - l_i (1 - beta_i).                  (U)

Do NOT substitute mere sign guards on f_i for (U). Do not identify beta_i with
eta, drop beta_i, or add separate McCormick rows to this construction.

### Integer exactness and every original phase choice

Monotonicity and the unit Lipschitz constant of ReLU imply, when h is nonzero,

    d_j / h = c_j theta_j,  0 <= theta_j <= 1.

When h=0, the two original source gates have identical arguments, so d_j=0.
Consequently f_i/h >= alpha_i for h nonzero.

In the integer source relation, h>0 gives p=h and n=0; (C) forces t_j=d_j.
For h<0, p=0, n=-h; (C) forces t_j=0. At h=0, p=n=d_j=t_j=0. In all cases the
other two inequalities in (C) follow from the true secant relation. Thus every
original source tuple has an extension, and t_j=eta*d_j independently of eta's
legal choice at zero. Formula (Q) equals ReLU(f_i).

With the correct q_i, rows (U) preserve precisely the original beta_i choices:
positive f_i forces beta_i=1, negative f_i forces beta_i=0, and zero f_i admits
both independently. This remains true when alpha_i=0 and f_i vanishes away
from h=0. There is no phase enumeration and no binary pivot or deletion.

### Containment of the old LP relaxation

The source p gate guarantees p>=0 and n=p-h>=0 even with relaxed bits. Choosing
the appropriate endpoint of each inequality in (C) according to the sign of
A_ij gives

    q_i >= alpha_i p >= 0,
    q_i - f_i >= alpha_i n >= 0.

These are exactly the two omitted lower gate rows. Together with (U), every new
LP tuple reconstructs an old LP tuple. Substitute (Q) consistently in EVERY
consumer, additional predicate, property, and decoder. Then the whole new LP,
mapped to the old variables, is a subset of the old LP. For an unchanged linear
property objective this cannot weaken the optimum bound in exact arithmetic.
It does not prove equal solver runtime, numerical behavior, or solved counts.

### Why simple shared products were insufficient

Using only t_j=eta*d_j with ordinary McCormick bounds gives integer exactness,
but does not by itself imply both lower rows for the relaxed source. Keeping
only sign guards in that version can weaken the old LP. The source-relative
cones and the ORIGINAL upper rows are essential to the simultaneous compression
and LP-containment result. That discarded construction is not a second runtime
path and has not been implemented.

## Residual composition without additional continuous factors

Suppose current values have the readout q=a p+A t. Consider original residual
preactivations, on the same source frame,

    z = u h + B q,  u >= 0 componentwise.

Define

    a' = u + B a,  A' = B A,
    alpha'_i = a'_i + sum_j min(0, A'_ij c_j).

If alpha' >= 0, set q' = a' p+A't = u p+Bq. The cone implies q'>=0, and the
algebraic identity q'-z=u(p-h)=un implies q'>=z, including in the relaxed
relation. Retain the original two upper rows of EVERY new target ReLU, using
its original bit and bounds. Both old lower rows are therefore redundant again.

For integer eta=0, p=t=q=0, so z=uh<=0 and q'=0. For eta=1, p=h and q'=z>=0.
At h=0 all these values vanish and all original target zero-phase choices
remain independent. Hence this residual transfer is exact, needs no new t,
and preserves LP containment by induction across any sequence satisfying the
stated coefficient conditions. If u and alpha' are strictly positive and both
signs of h occur, each such target really crosses zero; this is not stable-gate
pruning under an independent interval bound.

General affine/Conv readouts, shared Add and Concat remain exact by linear
combination on the aligned frame. Arbitrary biases, unrelated skips and arbitrary
subsequent ReLUs do NOT satisfy the specialized closure theorem automatically.
They need their ordinary exact HZ transfer unless another theorem applies.
The structural rule and its validity checks must be uniform; no model/instance,
label, margin, or LP-status dispatch is involved.

## Explicit mixed input control and a strict LP separation

Use h,v in [-1,1], with p=ReLU(h), r=ReLU(v+h), s=ReLU(v), d=r-s; k=1 and c=1.
For i=1,...,8 choose a_i=i+1, b_i=(-1)^i and f_i=a_i h+b_i d. Then alpha_i>=1.
Use u_i=a_i+max(0,b_i), l_i=-u_i. Both input signs and both signs of every f_i
are attainable. The two source arguments v+h and v are nonparallel.

The four new rows are

    t >= 0,  t <= p,  t >= r-s,  t <= r-s+p-h.

The readout is q_i=a_i p+b_i t. A strict old-LP-only point is

    h=v=1/4, p=s=1/4, r=1/2,
    eta=beta_r=beta_s=1, all beta_i=1/2,
    old q_i=(a_i+b_i)/4+1/4.

Every old source and output gate row holds. No source argument is at its zero
boundary. But p=h makes the new cone force t=d=1/4, so (Q) requires
q_i=(a_i+b_i)/4 and excludes this point. This proves strict relaxation
improvement on the specified formulation, not over all known lifted controls.

Add eight genuine residual gates z_i=3h-q_i/4. Their new readouts are

    q'_i=(3-a_i/4)p-(b_i/4)t,
    alpha'_i=3-a_i/4+min(0,-b_i/4) >= 1/2.

Valid same-comparator bounds are l'_i=-3 and
u'_i=3-a_i/4+max(0,-b_i/4). The second retained upper row simplifies exactly to
3(p-h)<=3(1-beta'_i), with no t coefficient. This is a two-layer, mixed-sign
residual instance of the composition theorem.

## Arithmetic size and its limits

The counts below are for explicit linear gate rows, bounded continuous
coordinates and final affine readouts. Sources p,r,s are included; d,n are
inlined. Variable bounds, RHS storage, indices, coefficient bit widths, phase
and source metadata, certificates and decoder ownership are NOT physical
memory measurements and are not hidden inside these nnz counts.

| Quantity | Original one layer | D016 one layer | Original two layers | D016 two layers |
| --- | ---: | ---: | ---: | ---: |
| Continuous coordinates | 13 | 6 | 21 | 6 |
| Original binary coordinates | 11 | 11 | 19 | 19 |
| Gate and cone rows | 44 | 32 | 76 | 48 |
| Predicate nnz | 122 | 109 | 202 | 157 |
| Final readout coefficients | 8 | 16 | 8 | 16 |
| Predicate plus final readout coefficients | 130 | 125 | 210 | 173 |

Derivation: source gates use 12 rows and 26 nnz. Each original first-layer
gate uses 4 rows and 12 nnz. The new cone has 4 rows with 1,2,3,5 nonzeros;
each retained first-layer pair has 3+6=9. Original residual gates each add
4 rows and 10 nnz; new pairs add 2 rows and 3+3=6 nnz after exact cancellation.

The one-layer and two-layer counts and the strict separation were independently
recomputed. The residual exactness/LP proof and upper-row cancellation were also
independently reviewed. All are paper calculations, not executable tests or
full-storage qualification.

Materializing all eliminated q_i again adds their coordinates and definition
equalities. A downstream Cq requires Ca and CA, whose multiplication and fill
must be paid; arbitrary consumers may erase the saving. Longer composition can
increase rational coefficient bit width even when factor count stays fixed.
If an EQ-only terminal introduces slacks for LE rows, their number, bounds,
storage and reconstruction must also be charged. These examples claim no
measured speed or complete native cost advantage.

## Prior art and the still open definition contribution

The scalar increment bound is established activation-sector mathematics;
see [Fazlyab, Morari and Pappas](https://arxiv.org/abs/1903.01287).
It does not require adopting an SDP solver. Source-relative zero anchoring is
essential: invertibility of a residual map alone does not preserve coordinate
signs. For example, x+1 is invertible but not sign preserving, and
(x1+x2/2,x2) is zero anchored globally but not on the x1=0 fiber.

The positive/negative cone lift is classical disjunctive/perspective geometry,
not a newly invented primitive; compare the traditional formulations discussed
by [Vielma](https://juan-pablo-vielma.github.io/publications/Embedding-Formulations-and-Complexity.pdf).
For an isolated secant factor, dividing (p,t) by eta and (-n,d-t) by 1-eta
constructs the two valid endpoints whenever 0<eta<1. This does not establish
ideality after attaching the actual shared v_j source gates.

The products eta*d_j are also special cases of the binary-continuous RLT
variables in [Sharp Hybrid Zonotopes](https://arxiv.org/html/2503.17483v1),
§II-C and §IV. Ordinary HZ with the SAME sector certificates, shared lift and
substitution can reproduce this representation. We therefore claim neither
strict dominance over that comparator nor a novel product algebra.

What remains a contribution candidate is the combination of a neural source
certificate, preservation of every original bit, removal of output coordinates
and lower rows, LP containment, and restricted residual closure with a paid
factor bound. No matching complete theorem was established by this limited
review, but absence from a small review is not proof of novelty. A new domain
claim still needs a substantive definition/invariant story and real utility.

The example deliberately has a common zero fiber. That structure must not be
assumed typical in CIFAR or Tiny networks. In a full-dimensional input region,
two affine forms which cross zero and share their entire zero hyperplane are
proportional; sharing signs is therefore a strong condition, not a generic
consequence of overlapping receptive fields. Nonlinear bundles can have richer
structure, as the example shows, but actual prevalence is still unmeasured.

## Next evidence required

Retain this as one candidate theorem, not a promoted path. Before numerical
implementation, freeze an exact structural matching rule against existing
source gates, certificate language, consumer accounting and a fixed population.
Do not manufacture a second anchor network or choose only known successful
instances. Reject the candidate as a main CNN direction if the necessary source
structure is absent, if only stable rows benefit, or if full costs exceed gains.

Only then implement a new isolated opt-in version and apply the complete
inherited qualification population plus its delta tests. Real-structure,
shadow, per-family, 2413-case and separate 400-case obligations are unchanged.
New mathematical evidence does not authorize a score or default update.
