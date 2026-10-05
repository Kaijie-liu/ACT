# Binding a shared neural decoder to its original source

This study advances the mathematical definition, not a helper implementation.
A strictly active set of original neurons that spans the layer's row space can
bind the entire shared nonlinear decoder to the actual input through its own
decoded equalities. This replaces D131's explicit phase-product source binding
at the semantic level. A direct feasible-direction proof also extends to a
specified family of monotone smooth activations.

The result does **not** yet provide a stronger or cheaper verifier. Fixing the
anchors inside the decoder optimization is unsound as a substitute; correct
elimination retains a coupled optimization whose optimality, after source
binding, recovers the original remaining gates. No QP wrapper, new solver or
numerical candidate is selected. All results below are exact-real paper reasoning
with independent manual review, not machine-checked or benchmark gains.

## The common decoder and the new source binding

Let the original affine bank be g=A*x+b, with m rows. Keep the original source
state theta, the actual readout x, every original continuous factor and signed
binary identity, predicates, shared frame and original input decoder. Write
beta=(signed_bit+1)/2 only as notation. The domain still requires beta in {0,1}.

The [D131 decoder](../d131_nonlinear_common_decoder_20261002/THEORY.md) is

```text
D(y) = argmin { J(q) : q>=0, A^T*q=y },
J(q) = ||q||^2/2 - b^T*q.
```

For any y with nonempty feasible set, a unique minimizer exists: the set is closed
and convex and J is strictly convex and coercive. Every actual q=ReLU(A*x+b)
is D(A^T*q), by the original difference-of-objectives proof. Defining D this way
does not authorize executing a QP or treating its solution as a free primitive.

Choose a structurally certified, fixed set B of original neurons with

```text
A_B*x+b_B > 0 throughout the current source relation,
rowspan(A_B) = rowspan(A).
```

A fixed independent subset of B suffices. The certificate is an exact row-space
identity, not a numerical rank threshold. The new binding is

```text
q denotes D(y),       q_B = A_B*x+b_B.
```

**Source-binding theorem.** Under these premises, q=ReLU(A*x+b) at every same
source state. No separate equation y=A^T*diag(beta)*(A*x+b) is needed.

Proof: for any i outside B write a_i=sum_j T_ij*a_j, j in B, and set
d_i=e_i-sum_j T_ij*e_j. Then A^T*d_i=0. Because every q_j for j in B is strictly
positive, sufficiently small changes in either direction on those coordinates
remain feasible in the decoder. Its directional derivative at q is

```text
gradient J(q)^T*d_i
  = (q_i-b_i)-sum_j T_ij*(q_j-b_j)
  = q_i-(a_i*x+b_i) = q_i-g_i.
```

If q_i>0 both directions are feasible, so optimality forces q_i=g_i. If q_i=0,
the positive direction is feasible and forces -g_i>=0. Thus every coordinate is
exactly ReLU(g_i). The anchor coordinates already agree. The argument needs no
Slater assumption, source convexity, solver, phase search or dual execution.

The converse follows from D131: every real bank state gives y=A^T*q and satisfies
the anchor equations. Retain the original guards (2*beta_i-1)*g_i>=0. Once the
values are recovered, these guards give the complete original labelled graph,
including both legal labels whenever g_i=0. Bits are not merged or deleted.

This is a sufficient uniform structural condition. It is not implied by many
stable gates alone. If A has full row rank, a proper subset of its rows cannot
span it. Restricting to a source affine hull x=x0+E*z requires consistently
redefining the decoder with A*E and A*x0+b; merely testing the smaller rank while
retaining the original decoder is not justified.

## An exact candidate element and its operator scope

An anchored element may retain the original source relation S(theta), the
certified A,b,B, one common statistic y, every original beta, and decoded q=D(y).
Its concretization is the set of the following simultaneous states:

```text
S(theta), x=X(theta), y in domain(D), q=D(y),
q_B=A_B*x+b_B, all original guards and retained P(theta,beta,q),
v=c+G*theta+C*q.
```

Original binary readouts can be included in G*theta with their discrete types
preserved. All consumers use the same q; a live residual uses the actual x or
theta, never a separately inferred source. By the theorem this representation
is exactly equivalent to the original labelled bank, not just its convex hull.
Here P retains all other source and external predicates after substituting the
common decoded q; the bank's replaced active-value equations are not silently
retained as a second complete encoding. Its original sign guards remain explicit.
Empty decoder banks embed the original HZ. Inclusion of concretizations in an
aligned frame defines precision; no computable best abstraction or complete
lattice is asserted.

Affine and Conv left-multiply the common readout and add the bias exactly.
Add and Concat share y, theta and original factor identities. A following
ReLU of C*q+S*x+c retains its own original bit and graph on that same value.
Because the bank replacement is pointwise exact, every such context preserves
the original values and original input reconstruction.

This compositional semantic statement does not make a cheap recursively closed
representation. The next layer's source may itself contain D(y); storing a chain
of such decoders without a better query or re-abstraction would be another
functional network graph. No bounded-width or GPU query theorem follows here.

The anchor equations and P belong **outside** the definition of argmin. Adding
them to the decoder's feasible set changes its optimality conditions, invalidating
the proof. They filter which decoder outputs represent the original source; they
must not prevent the internal perturbations used to characterize those outputs.
All existing source predicates remain on the same state, without convexifying or
dropping them. An implicit minimizer's source need not satisfy P independently;
the theorem establishes equality to the retained actual x before P is checked.

## An ordinary two-source four-gate control

Let x=(x1,x2) in [-1,1]^2 and use

```text
g1 = x1+x2/4+2,          g2 = -x1/3+x2+2,
g3 = x1-x2/2+1/8,        g4 = x1/4+x2-1/5.
```

The first two gates are strictly active with ranges [3/4,13/4] and [2/3,10/3].
Their row determinant is 13/12. The other two are nonparallel, biased crossing
gates with ranges [-11/8,13/8] and [-29/20,21/20]. Hence this is not D144's
at-most-one-unresolved-gate case, nor a duplicated/opposite-neuron construction.

At the interior source x=0, the actual q is (2,2,1/8,0). Consider instead

```text
q_fake=(2,2,1/4,0), beta=(1,1,1,0),
y_fake=A^T*q_fake=(19/12,19/8).
```

This passes the anchor equations, original sign guards, epigraph and off-mask
conditions, and statistic equality, but not the full decoder. Indeed

```text
d=(-10/13,9/13,1,0), A^T*d=0,
gradient J(q_fake)^T*d=1/8.
```

For small positive t, q_fake-t*d remains nonnegative and decreases J. All
preactivations here are nonzero. A mixed readout with a live source skip,

```text
next = ReLU(q3-q4/2+x1/10-3/16),
```

is zero at the real same source but 1/16 at the false assignment. This is a
fixed-source correctness diagnostic, not a global safety result or an ADV.

The original complete four-row HZ already excludes the false assignment:
beta3=1 forces q3<=g3=1/8. Thus the control proves that the decoder is essential
to this new coordinate contract; it does not show strength over the original
exact graph or the complete old four-row relaxation. No query relaxation for
the new element has yet been qualified.

If anchors are fixed *inside* the minimization, A_U for U={3,4} has determinant
9/8. Its nonnegative feasible fiber at y_fake becomes a singleton, so the false
assignment becomes a minimizer. This is precisely the missing cross-anchor
variation, not a numerical corner case.

## Correct elimination and the cost that remains

Choose B as a row basis of A, write A_U=T*A_B, and define v(y) by
A_B^T*v=y on range(A^T). Every point in the decoder fiber has

```text
q_U=u, q_B=v-T^T*u,
u>=0, v-T^T*u>=0.
```

Up to a constant independent of u, its objective is

```text
H=I+T*T^T,
J_reduced(u)=u^T*H*u/2 - [T*(v-b_B)+b_U]^T*u.
```

This is a correct reduced strictly convex QP. It may use |U| optimization
variables rather than m; claiming that every decoder necessarily needs m
physical QP variables would be false. However, it contains a generally coupled
quadratic metric, anchor nonnegativity constraints, range/basis transforms and
all original source/readout costs.

At an externally anchored state q_B=A_B*x+b_B>0, those anchor nonnegativity
constraints are locally inactive. Its remaining optimality gradient simplifies:

```text
H*u-T*(v-b_B)-b_U = u-A_U*x-b_U.
```

The resulting nonnegative-coordinate optimality conditions are exactly the
original U ReLU complementarity. This explains both the exact binding theorem
and why this elimination has not by itself supplied an easier terminal query.
It is not a lower bound against every possible encoding or useful future outer
approximation.

Fair comparison in the four-gate example starts with **two**, not four,
nonlinear gates: the old B outputs are already affine and their original bits
can remain fixed without deletion. Substituting each U preactivation into its
standard four gate inequalities uses two amplitude coordinates, eight LE and
twenty coefficient nonzeros, including q>=0; source bounds, fixed original bits,
downstream consumers and normalization are additional on both sides. If using
physical g_U and excluding the common q>=0 bounds instead, the same old bill is
six LE / fourteen nnz plus two preactivation EQ / six nnz. These conventions
must not be mixed.

Materializing y=A^T*q here adds two continuous coordinates and two EQ with ten
nonzeros before any elimination. Implicitly omitting q avoids that particular
bill, but requires the whole nonlinear decoder/query. Correct elimination has
two U variables, so no dimension advantage over the two old uncertain amplitudes
has been demonstrated. Exact rank, T and v construction can fill in coefficients;
their certification, arithmetic bit lengths, bounds, all skip/predicate consumers,
host/device coexistence, evidence and terminal lowering remain chargeable.

The old one-dimensional-kernel decoder is not a general multi-gate shortcut.
With a row basis of size rank(A)=m-1, only one coordinate remains outside it.
The more interesting multi-uncertain-gate case generally needs a coupled decoder;
neither its optimality nor the cost is removed by naming it an abstract generator.

## A monotone smooth version of the binding theorem

Let F_i be proper closed strictly convex scalar functions and define

```text
D_F(y)=argmin { sum_i F_i(q_i)-b^T*q : A^T*q=y }.
```

Assume the feasible set has a finite attained minimum; uniqueness follows from
strict convexity. Retain the same row-spanning B. Suppose each anchor q_j is in
the interior of dom(F_j), F_j is differentiable there, and externally enforce
F_j'(q_j)=a_j*x+b_j. Then for every i,

```text
a_i*x+b_i belongs to subgradient F_i(q_i).
```

For proof use the same d_i. For any t in dom(F_i), the perturbation
epsilon*(t-q_i)*d_i is feasible for sufficiently small positive epsilon; anchors
stay interior. Optimality gives
F_i'(q_i; t-q_i)>=(a_i*x+b_i)*(t-q_i). Convexity bounds this directional derivative
above by F_i(t)-F_i(q_i), yielding the subgradient inequality. Its value cannot be
positive infinity because the secant is finite; negative infinity would violate
optimality after adding the finite anchor derivatives. No differentiability is
needed at non-anchor coordinates.

Whenever phi_i=(subgradient F_i)^(-1) exists on the actual preactivation range,
the theorem recovers q_i=phi_i(a_i*x+b_i). That existence is an extra assumption,
not a consequence of strict convexity on all of R. ReLU uses
F_i(q)=q^2/2+indicator(q>=0). Sigmoid and tanh have the familiar entropy-based
potentials with derivative logit and atanh on their interior ranges. Their
anchor equalities are nonlinear, so this is not a cheap linear encoding or a
smooth-model verification result. It does not cover nonmonotone GELU, coupled
Attention, LayerNorm or complete Transformers automatically.

## Sparse current nonlinearity does not imply recursive small width

The D144 census only describes the saved current banks. A single crossing
amplitude can coexist with many independent live affine directions. For
x,y0,y1,...,yr in [-1,1], set

```text
q=ReLU(x+1/4),
v_i=2+y_i-y0/4+c_i*q,       c_i in [1/4,1/2],
r_i=ReLU(v_i-17/8).
```

Every v_i>=3/4 is strictly stable, yet at x=1/4, y0=0 choose
y_i=+/-1/2-c_i/2+1/8. These are strictly interior source values and independently
give every successor sign pattern, each with an open neighborhood. This is a
proof construction, not runtime phase enumeration. It rules out an inference
from a narrow old nonlinear frontier to a bounded-width recursive closure;
any useful condition must also account for the newly read live source directions.

## Prior art and research decision

Active-output spanning is established reconstruction mathematics, not a new
injectivity discovery. [Puthawala et al., Globally Injective ReLU Networks,
Definition 1 and Theorem 2](https://jmlr.org/papers/volume23/21-0282/21-0282.pdf)
use directed spanning sets and active rows to characterize global injectivity.
[Haider, Ehler and Balazs, Sections 3 and 4.4](https://proceedings.mlr.press/v202/haider23a/haider23a.pdf)
study restricted-domain injectivity and explicit frame-based reconstruction.
We do not adopt their facet enumeration, retraining or inversion algorithms.

Convex-potential activation graphs and smooth inverse-integral potentials also
have direct prior art: [Gu, Askari and El Ghaoui, Section 4.1](https://proceedings.mlr.press/v108/gu20a/gu20a.pdf).
Their training/dual optimization is not an authorized execution path here.
The project-level advance is the link between active-row source binding and the
existing common decoder, the direct exchange-direction proof, its smooth scope,
and the incorrect conditional-decoder counterexample. External novelty is not
established by these derivations or the limited literature review.

Keep this exact coordinate contract as a mathematical candidate, not a completed
Neural-HZ domain or a selected implementation. The unresolved substantive task is
a uniform, useful query or re-abstraction for the shared decoder and a following
mixed ReLU with live residual, without simply restoring the old graph or paying
an external optimizer. Restating it as a single QP, Fenchel gap, known RLT lift
or tensor storage does not solve that task. No new helper/test framework follows
from this record.

Formal baseline remains 1870/2413, separate CIFAR100 25 + TinyImageNet 36 remains
61/400, both gains zero. All 13-family retention, explicit original nonconvex
phases, source identity, fail-closed, GPU/smooth/new-family goals and full replay
requirements remain unchanged. Goal active; no candidate or numerical execution.
