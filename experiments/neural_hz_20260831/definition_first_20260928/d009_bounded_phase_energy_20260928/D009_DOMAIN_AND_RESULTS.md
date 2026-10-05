# D009: bounded phase-energy Neural-HZ calculus — mathematical candidate

2026-09-28; branch `redu-hz`; HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

Status: paper definitions, proofs, independent mathematical review and primary
literature inspection. No new numerical experiment, module import, test run,
model decoding, solver call or production integration. The last qualified
reference remains D003; none of its qualification transfers automatically.
Formal baseline remains 1870/2413 (1063 CERT + 807 validated ADV). Separate E0
remains CIFAR100 25 / TinyImageNet 36. Formal gain of this work is zero.

The preceding user-facing turn was NO PROGRESS under the goal's criterion:
it checked and restated alignment. This turn changes the next research action
with a strictly stronger bounded query invariant and explicit composition
theorems/limits. It does not establish the requested innovative domain or
complete the active Goal.

## 1. The question changed by this checkpoint

D007's exact phase/magnitude grammar is established mathematics, and ordinary
HZ supplied the same conservation rows has the same terminal relation. A
certificate-row compiler is therefore not, by itself, the requested domain
innovation. D008's two unexecuted drafts remain preserved, not qualified:

- `../d008_bundle_kernel_20260928/bundle_kernel_v1.py`;
- `../d008_bundle_kernel_20260928/test_bundle_kernel_v1.py`.

No additional D008 implementation or test population was created this turn.
The next main action is not expanding its conservation-selector infrastructure.

The new question is whether **joint bounded resources of nonconvex factors**
can be propagated forward through neural operators and converted into useful
linear terminal predicates without a quadratic solver, phase enumeration or
backward/dual reasoning. Unlike D007's homogeneous conservation cone, the
resource facts use the actual common input box and finite magnitudes.

The positive result in section 5 strictly strengthens, on one ordinary mixed
block, the intersection of independently ideal neuron hulls and ALL D007
source-box defect certificates, including all exact conservation relations.
A general lemma explains this separation. This changes what a prospective
calculus must reason about; merely finding more relations cannot reproduce
that comparison on this block.

The ingredients are classical Gram identities, Cauchy--Schwarz and known
activation contraction. Novelty of their bounded, proof-producing Neural-HZ
calculus, and actual CNN utility, remain unestablished. Do not relabel the
ingredients as new geometry or claim a PLDI contribution from a toy example.

## 2. Candidate elements and concretization

An element consists of an owned HZ latent frame H, a collection of exact neural
factor relations B, and a finite checked resource ledger E:

    D = (H, B, E, input_decoder, output_decoder).

H retains every original bounded continuous factor, free binary identity,
EQ/LE predicate and shared ancestor. Each activation factor retains its own
original binary beta and has

    beta in {0,1}, m = (2 beta - 1) f, m >= 0,
    r = (f + m)/2.

Thus m=|f| and r=ReLU(f), with BOTH original beta choices legal at f=0.
An affine bank refers to the same owned input or earlier neural coordinates;
it is never an independent copy of a patch or residual branch.

E contains certified statements about these SAME coordinates, for example

    sum_i w_i (v_i - v0_i)^2 <= B0,  w_i >= 0, B0 >= 0,

as well as their proved linear magnitude consequences. A ledger entry carries
the source map, bounds, exact arithmetic and derivation premises. It is not a
free extra assumption, an independent noise source, or an SDP oracle.

Strong concretization Gamma(D) retains original input, ALL original/free/ReLU
bits and visible outputs, existentially hiding only internal continuous values:
all of H, B and the resource statements must hold. Valid derivations make E
redundant for exact Gamma, so adding them is Gamma-preserving. Nonconvexity
remains in B; no norm ball or convex envelope replaces B or H. For example,
the retained scalar ReLU graph contains (-1,0) and (1,1), not (0,1/2).

Original HZ embeds with B and E empty and unchanged decoders/predicates.
The semantic preorder on a common frame is Gamma-inclusion; distinct ledgers may have
equal Gamma but different certified query envelopes. No canonical quotient,
best abstraction for the full domain, general join, widening or efficient
semantic-inclusion procedure is claimed.

This is currently an exact nonconvex HZ/neural base coupled to a resource
abstraction, i.e. a reduced-product candidate. Its mathematical distinction
from ordinary HZ plus an equivalent externally generated resource certificate
is NOT proved. A useful new inference/transformer calculus and full cost/real
benefit, not set expressiveness or a renamed factor record, must supply the
eventual contribution.

## 3. Shared-input Gram rule

Let f=z+A xi, with xi_j in [-1,1], using a faithful common frame. For fixed
nonnegative rational weights w, let

    W0 = sum_i w_i,
    G = A^T diag(w) A,
    B0 = sum_j G_jj + 2 sum_{j<l} |G_jl|,
    c0 = sum_i w_i |z_i|.

G is positive semidefinite by construction. For every admitted xi,

    sum_i w_i (f_i-z_i)^2 = xi^T G xi <= B0.

Triangle inequality followed by weighted Cauchy--Schwarz gives

    sum_i w_i |f_i| <= c0 + sqrt(W0 B0).                 (E)

A rational certificate q>=0 with q^2>=W0 B0 yields the sound row

    2 w^T r - w^T f <= c0 + q.                          (L)

No floating tolerance can stand in for the squared inequality. For W0=0 the
row is a tautology. An unchecked sign, bound, coefficient or arithmetic-size
premise rejects the certificate, not the underlying exact state, and never
produces SAFE. Every original native gate row, bound and bit remains.

For original patch coordinates p in [L,U], write t=(L+U)/2 and rho=(U-L)/2.
With f=F p+b, take z=F t+b and A=F diag(rho). Zero radii are legal. A box
enclosing correlated coordinates is sufficient: no independence assumption
is needed, because the bound is universal on that enclosing box.

When flattening (L) into the original p and r variables the correct form is

    2 w^T r - (w^T F) p <= c0 + q + w^T b.

In normalized xi coordinates it is instead

    2 w^T r - (w^T A) xi <= c0 + q + w^T z.

The bias/center compensation is essential. These are algebraic identities,
not permission to modify the original network weights or preprocessing.

The sum in B0 keeps cancellations INSIDE the Gram entries before taking
absolute values. Replacing it with the sum of absolute term products loses
that source of precision and must not be reported as the same rule.

## 4. Forward transformer calculus and its limits

The exact H/B component keeps the ordinary affine/Conv, guarded ReLU,
shared-frame Add/Concat and extra-predicate semantics of D007. Resource facts
are propagated soundly as follows; they do not replace those exact operations.

### 4.1 ReLU: diagonal weighted energy survives

If sum_i w_i (f_i-z_i)^2<=B0, define r0=ReLU(z) coordinatewise. The scalar
1-Lipschitz property gives

    sum_i w_i (r_i-r0_i)^2 <= B0.

This is a forward rule, valid with arbitrary z and unchanged nonnegative w.
For z=0 one can also use the exact identity m_i^2=f_i^2. It holds for every
integral phase including zero, and is precisely the link used by (E).

An arbitrary non-diagonal quadratic form does NOT contract through ReLU.
For Q=[[1,1],[1,1]] and f=(1,-1), f^T Q f=0 but ReLU(f)^T Q ReLU(f)=1.
Therefore keeping a Gram matrix numerically unchanged after a ReLU is invalid.
E records the proved diagonal-weight resource and derivation, not that claim.

### 4.2 Affine/Conv after an energy bound

Suppose delta=v-v0 satisfies delta^T D delta<=B0 with diagonal D>=0.
For u=Vv+b, u0=Vv0+b and a chosen nonnegative diagonal D_out, form
K=V^T D_out V. A nonnegative rational h satisfying

    h D_jj >= K_jj + sum_{l!=j}|K_jl| for every j

certifies hD-K positive semidefinite by symmetric diagonal dominance.
Consequently (u-u0)^T D_out (u-u0)<=h B0. If D_jj=0 and the corresponding
K row is not zero this sufficient certificate cannot pass; do not divide by
zero or silently discard the coordinate. The rule is conservative and can
accumulate large losses. It is not a complete energy transformer.

This calculation is a forward structural operation on a fixed next layer,
not property-directed backward propagation, a dual search, or an SDP solve.
Dense K can be prohibitively large; convolutional local use must pay for every
source port and overlap rather than pretend an entire CNN has 27 coordinates.

### 4.3 Shared residuals and concatenation

For two same-frame vectors with centered D-energies bounded by B1,B2,
Add has centered energy at most (sqrt(B1)+sqrt(B2))^2, or the always-rational
bound 2(B1+B2). This needs no independence assumption. It may be loose even
when exact shared H/B algebra cancels a branch; do not delete that algebra.
Concat with block-diagonal weights has energy at most B1+B2, again even with
shared latent factors. All certificates continue to reference those factors.

Together these rules are compositional SOUND resource propagation. They do
not prove that an optimal energy bound, a full convex hull, exact terminal
cost reduction, or a useful bound survives an arbitrary deep network.

## 5. Strict paper control beyond all D007 relations

Take x in [-1,1]^2, all three distinct mixed rows

    f1=x1+x2,  f2=x1-3x2/2,  f3=x1+x2/2,
    r_i=ReLU(f_i), m_i=2r_i-f_i.

Sound tight individual bounds are +/-2, +/-5/2, +/-3/2. With w=(1,1,1),

    G=diag(3,7/2), B0=13/2, W0=3,
    q=9/2, q^2=81/4 >= W0 B0=78/4.

Hence the shared resource rule gives

    m1+m2+m3 <= 9/2,
    equivalently 2(r1+r2+r3)-3x1 <= 9/2.                (P)

The old fractional point

    x=(0,0), r=(1,5/4,3/4), beta=(1/2,1/2,1/2)

has m=(2,5/2,3/2), whose sum is 6. It violates (P).

It satisfies all ordinary four-row gate relaxations. It also survives the
intersection of the individual IDEAL retained-phase neuron hulls: for each
row separately average its positive and negative maximizing input corners.
These pairs are +/- (1,1), +/- (1,-1), +/- (1,1), respectively. Each mixture
has the same mean x=0, beta=1/2 and r equal to half that row's positive bound.
These separate mixtures are not one common input mixture.

The row rank is two; the entire affine dependence space is spanned by

    4 f1 + f2 - 5 f3 = 0.

At the fractional point its weighted magnitudes are (8,5/2,15/2). Every one
is no larger than the sum of the other two. Thus every D007 homogeneous
single-dependence triangle (all dependences are scalar multiples) and its
nonnegative-magnitude anchor accept the point. There is no missing additional
independent relation for a better selector to find here.

In fact, this point survives ALL D007 box-defect certificates, not just exact
dependences. Generally, for any zero-centered bank f=A xi over the symmetric
unit box, set d_i=||A_i||_1, x=0, m_i=d_i, r_i=d_i/2, beta_i=1/2. This tuple
belongs to every independent ideal retained-phase neuron hull. For ANY a,
the D007 exact source-box residual is epsilon=||sum_j a_j A_j||_1, and

    |a_i| d_i <= sum_{j!=i}|a_j| d_j + ||sum_j a_j A_j||_1

is simply the vector triangle inequality applied to a_i A_i. Every member
row therefore holds; the zero-centered anchor is automatic. An arbitrary
constant c makes epsilon=|c|+||sum a_j A_j||_1, so its member rows are weaker
and its anchor still automatic. Wider, sound residual bounds do not help.

Consequently whenever the Gram rule yields q<sum_i w_i d_i, it strictly
strengthens the entire D007 certificate LANGUAGE on this unconstrained bank.
The control satisfies this with 9/2<6. Extra predicates in a real network may
already exclude the tuple, so the theorem is not a real-network gain claim.
It does not cover every inequality derivable with finite bounds, all group
hull methods, SDP or arbitrary HZ predicates. Ordinary HZ supplied (P) makes
exactly the same exclusion.

For an affine residual output y=2 sum r_i-3x1, (P) proves y<=9/2. The old
query point has y=6. The exact maximum is actually 4: the convex function
sum_i|f_i(x)| reaches its box maximum at a vertex, and its values at the two
opposite corner pairs are 4 and 3. This is an analytic paper check, not runtime
sampling, an attack, a benchmark CERT, or a validated ADV.

Direct-inequality native bill: 5 continuous variables and 3 original bits;
12 gate rows / 30 matrix nnz. The three D007 triangle rows add 13 nnz;
(P) adds one row / 4 nnz. Thus gates + three triangles + energy have
16 rows / 47 nnz. If the redundant homogeneous anchor is also emitted, add
one row / 5 nnz: 17 rows / 52 nnz. No claim relies on omitting an already
retained row. Bounds, RHS, source, ledger and witness costs are additional.
An equality-only native backend needs a bounded slack for each extra LE row;
its dimension and normalization cannot be hidden in this bill.

## 6. Ordinary convolution opportunity and full cost

D007's immutable Tiny/CIFAR saved descriptors identify an initial 64-filter
bank over 27 patch coordinates. This rule does NOT require rank deficiency,
near-duplicate filters or sparse nullspace relations. It applies to any
faithful affine bank with a common bounded source frame. Biases and asymmetric
patch boxes are explicitly supported by section 3.

For a fixed filter group F and weights w, cache only the exact coefficient
Gram H=F^T diag(w)F. At position t, the correct energy bound is

    B_t=sum_j H_jj rho_tj^2
        +2 sum_{j<l}|H_jl| rho_tj rho_tl,
    c_t=sum_i w_i |(F mid_t+b)_i|.

Each patch pays for its own center, radii, certificate, emitted row and evidence.
Patches still reference the same input; shared H does not make them independent.
A position-dependent subset/weight mask generally changes H, so one cannot
claim once-only Gram work for arbitrarily different unstable-neuron groups.

For k rows and d source coordinates, dense exact Gram construction costs
O(k d^2) rational arithmetic and O(d^2) Gram storage, in addition to source
O(kd). Centers, bounds and row flattening cost O(kd+d^2) per patch (unless a
specifically proved existing operation supplies a reusable value). One energy
row has at most k+d matrix entries in a direct source-coordinate encoding.
For the saved shape these symbolic sizes are 64x27 source, 27x27 full Gram
and at most 91 row entries per patch. They are not measured wall-time, bytes,
or proof that the complete source/terminal representation fits existing gates.

Keep every original 512-bit rational/native-range/physical-memory/work/test
gate in its original scope. Arithmetic-operation counts are not bit complexity.
If an intermediate exceeds the certified representation, decline this resource
certificate without altering H/B; no numerical epsilon may repair it. Source
expansion, duplicated storage, native slack, exact rounding, certificate checking,
four-concurrent cost and concrete-input reconstruction all remain payable.

The Tiny saved BN-edge map remains historically defective; this checkpoint
uses only authenticated descriptor SHAPES. It cannot certify effective F,b or
their bounds. No source-faithfulness claim or network decoding occurred here.

The inequality need not beat existing bounds: correlated rows, large centers,
wide shared-source dimension, low unstable population or repeated Lipschitz
loss can erase benefit. Applying it uniformly is sound, not automatically fast
or useful. Do not select its target using public sat labels or terminal margins.

## 7. Prior art and contribution boundary

[Quadratic Zonotopes](https://arxiv.org/html/1411.5847v1), section 2/equation 2,
already supplies the quadratic-box bound that specializes exactly to B0 here.
Thus even the no-SDP Gram energy calculation, not just Cauchy--Schwarz, has
direct prior art. No replacement by that domain is proposed.

Quadratic constraints and activation contraction are established NN verification
tools. [Fazlyab--Morari--Pappas](https://arxiv.org/html/1903.01287v3), sections
III-C/III-D and IV, use activation QCs and SDP-based reasoning. This checkpoint
uses elementary forward certificates, not that solver workflow, but avoiding
SDP does not make Cauchy--Schwarz or the energy semantics novel.

The full affine-consistent extension of the [Raghunathan--Steinhardt--Liang SDP](https://arxiv.org/pdf/1811.01057),
equation 4 (which omits biases for exposition), also implies this mass inequality
(our derivation in the companion note). This is a cheaply checkable fragment,
not a stronger convex geometry.

[Ellipsotopes](https://arxiv.org/html/2108.01750v4), definition 2, already combine
affine generators, grouped norm constraints and equalities. Thus adding a norm
budget to a factor set is not sufficient novelty. Here exact binary neural
relations must remain; replacing them with an ellipsotope is not authorized.

[Sharp HZ / RLT](https://arxiv.org/html/2503.17483v2), section IV/theorem 7,
already studies tightening HZ through lifted constraints while retaining HZ
representability. A new row or nonlinear notation cannot by itself distinguish
this work. No dominance over fixed-level RLT or SDP has been proved or tested.

Also retain D007's abs-normal, ImageStar, mixed-polynotope, PRIMA and Anderson
comparators. Standard HZ plus IDENTICAL resource rows must be a strong ablation;
it has identical query semantics. A legitimate contribution would concern a
nontrivial, bounded-cost forward factor calculus and useful neural propagation,
not a false expressiveness claim against that comparator.

## 8. Disposition and next admissible action

Retain the bounded phase-energy hypothesis. Preserve D008 as an unqualified
conservation-only supporting draft; do not resume its test/runner expansion as
the main innovation milestone. Additional paper findings and rejected shortcuts
are recorded in D009_PROOFS_AND_LIMITS.md.

Next: specify and preregister ONE default-off mathematical resource transformer,
including exact source/bound ownership, original EQ/LE/free-bit preservation,
asymmetric-box bias compensation, outward rational root, forward ReLU transfer,
and the complete control in section 5. Reuse the qualified semantic reference,
but retain the complete inherited test population and unchanged resource gates;
do not treat these unexecuted proofs as a numerical pass. Compare its generated
native relation against ordinary HZ with exactly the same resource row.

After mathematical qualification, inspect a faithful original affine/Conv
source under a new frozen protocol, not the defective saved Tiny map. Test the
uniform structural rule on real same-structure banks before shadow or broad
replay. No test count, physical budget or full-replay gate is changed here.

If it only reproduces a generic HZ+norm-bound wrapper without a useful further
calculus/real gain, archive that negative result and revise the definition
hypothesis. Neither a successful certificate kernel nor the paper separation
completes the requested abstract-domain innovation. Goal remains ACTIVE.
