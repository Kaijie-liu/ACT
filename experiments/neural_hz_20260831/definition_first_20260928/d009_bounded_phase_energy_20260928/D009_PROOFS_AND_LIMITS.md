# D009 supporting proofs, alternative hypotheses and limits

2026-09-28. Same provenance, unchanged baseline, no numerical execution and
read-only historical custody as D009_DOMAIN_AND_RESULTS.md. These are paper
results, not extra algorithms authorized to enumerate phases or run solvers.

## A. Affine conservation usually does not survive ReLU as affine conservation

Let V be the finite-dimensional span of 1 and the explicitly retained prefix
value functions. New preactivations f_j belong to V; let r_j=ReLU(f_j).
Suppose for each j there is an open neighborhood in which all old prefix
functions are affine, f_j crosses zero transversely, and no other new f_i
crosses the same zero surface. Different j may use different neighborhoods.
Then

    span(V,r_1,...,r_k) = V direct-sum span(r_1,...,r_k).

Indeed, in an identity v+sum c_i r_i=0, crossing the j-th surface along a
transverse line creates a derivative jump only from c_j r_j, so c_j=0.
All coefficients vanish. This is the familiar independence of distinct hinge
singularities, not a novelty claim. It needs the stated reachable-neighborhood
premise; neither saved matrix rank nor instability bounds alone proves it.

In particular on ker(a^T), k>=3 and every a_i nonzero, an open neighborhood
of zero has distinct coordinate hinge hyperplanes. The unbounded joint
(f,ReLU(f)) graph has affine span dimension 2k-1. There are generally no new
unconditional output-only linear equalities. This is a graph-span fact, not
a lower bound on arbitrary nonlinear representations or a license to sum
hidden-layer ranks after projecting most coordinates away.

## B. Why unbounded phase-hull strengthening cannot do the job

Let X be the exact homogeneous single-relation ReLU graph with all original
bits, allowing both bit choices at every zero activation. Let P be its D007
closed continuous (f,r) hull. Then

    closure(conv X) = P x [0,1]^k.

For any p in P choose a finite convex representation by exact graph points.
Scale their value coordinates by 1/epsilon and their mixture weights by
epsilon, preserving their respective legal bits. Put the remaining mass at
the zero graph, whose bits can have any chosen mean in the cube. Value
coordinates stay p; bit coordinates approach the desired cube point as
epsilon tends to zero. The opposite inclusion follows from closedness of
P and the cube. This is a closed-hull result, not necessarily an equality
for ordinary finite conv X at boundary bit values.

Finite magnitude bounds prevent that scaling argument. They are therefore
part of the desired mathematics, not an implementation detail to append only
after deriving an unbounded cone.

## C. A bounded phase-capacity projection with a complete small description

For the FULL relation

    sum_i g_i=0, |g_i|<=M, M>0,
    delta_i in {0,1}, g_i>=0 if delta_i=1, g_i<=0 otherwise,
    T=sum_i max(g_i,0), n=sum_i delta_i,

the exact projection on (delta,T) has

    0 <= T <= M min(n,k-n).

Both signs carry mass T. One side has n available coordinates, the other
k-n; every coordinate has capacity M. Conversely every T up to that minimum
can be distributed separately across the two sides and realizes the relation.
Zero mass leaves every individual original bit unrestricted.

Its CONTINUOUS convex hull, retaining all individual delta coordinates, is

    0<=delta<=1, T>=0,
    T<=M sum delta_i,
    T<=M (k-sum delta_i),
    T<=M floor(k/2).

Proof of sufficiency. For any fractional delta with sum s, the polytope
0<=u<=1, floor(s)<=sum u<=ceil(s) has integral vertices: if there are at
least two fractional coordinates one can perturb them oppositely; if exactly
one is fractional, neither active integer cardinality boundary can hold.
Thus delta is a mixture of bit vectors with only floor(s) or ceil(s) ones.
The piecewise-linear interpolation of min(n,k-n) on integer n is precisely
the minimum of the three displayed upper planes. Each integral bit vector
admits every T from zero to its own capacity. Mixing corresponding scaled
capacities realizes every requested T below that interpolated bound.

For odd k the last row strictly strengthens the first two; for even k it is
redundant. This is an exact projected query result, NOT a full (g,r,delta)
ideal formulation, nor the hull after adding a particular shared input map.
It uses classical cardinality integrality/capacity reasoning. No runtime
enumeration or deletion/identification of original bits is proposed.

With unequal symmetric capacities U_i the exact maximal mass is instead

    max_{S subset [k]} min(sum_{i in S}U_i, sum_{i not in S}U_i).

The upper bound for a fixed sign support is achievable by filling the smaller
side and continuously matching its mass on the larger side. It equals half
the total capacity exactly when the capacities admit an equal partition.
Thus an unrestricted exact mass oracle already includes the integer PARTITION
problem. Naming a factor after this oracle does not remove its computation.
The set-theoretic maximum is a proof, not an authorized subset-search routine.

The cheap uniform-capacity row remains a useful supporting hypothesis, but
does not solve arbitrary wide biased filters. It is not the next main target
instead of the general shared-input Gram rule.

## D. When continuous hulls really do compose

Let X1 and X2 be exact sets containing zero and closed under nonnegative
scaling. Their ONLY shared real coordinates are a scalar ReLU interface
(f,r) with r=ReLU(f); all remaining coordinates are private. Using ordinary
finite convex hulls, not an unspecified closure, one has

    conv(X1 fiber-product X2)
       = conv(X1) fiber-product conv(X2).

Proof. Positive homogeneity makes a finite convex-hull point a finite sum of
exact points (and vice versa, after scaling and averaging). The separator
graph has the two independent rays rho+=(1,1) and rho-=(-1,0). In a given
separator hull point, their total masses are uniquely

    T+=r, T-=r-f.

Group each local exact-sum decomposition by positive, negative or zero
separator. Normalize every nonzero separator to its ray, absorbing its size
into the coefficient. For one sign, if the two sides have coefficients t_i
and u_j, their sums are the same T. For T>0, weights t_i*u_j/T pair the
normalized exact points. Their row/column sums restore both local vectors;
each pair is an exact global point. For T=0 there are no such positive-mass
terms. Zero-separator terms can be matched with the other side's all-zero
point. This constructs the desired finite global conic/convex combination.
The reverse inclusion follows from projection and convexity.

Leaf elimination proves the analogous statement for a finite tree with the
running-intersection property, provided every edge's COMPLETE real separator
is precisely one such (f,r). No explicit phase-valuation enumeration is used
by this proof. A fast algorithm for more general separators does not follow.

Example: over free q in R^3, take

    f=(q1+q2+q3, q1, q2+q3, q2, q3).

Its two constraints f1=f2+f3 and f3=f4+f5 share just the third activation
interface. Combining their D007 exact local cone hulls gives the exact whole
continuous graph hull. An invertible dense q=Mx makes the rows mixed without
changing this argument. Such a circuit-tree is not a property of a generic
dense CNN merely because that CNN is feedforward.

Sharp boundaries:

- Sharing f alone is insufficient: two local r_i=ReLU(f) hulls can have
  f=0,r1=1,r2=0, while the global graph and its hull require r1=r2.
- Two shared ReLU interfaces already fail. One local block has
  f3=f1+f2, the other f4=f1-f2. At all mean f=0, both local hulls allow
  r1=r2=1/2, r3=r4=0, using respectively opposite-sign and same-sign pairs
  of (f1,f2). A global mixture with mean f3=f4=0 and r3=r4=0 must have
  f3=f4=0 at every component, forcing f1=f2=r1=r2=0: contradiction.
- Finite boxes, nonzero biases and unscaled Boolean phase coordinates do
  not satisfy the stated cone assumption. Homogenizing and then sharing a
  fixed scale adds an interface coordinate; it does not evade this condition.
- Shared original input, overlapping convolutional patches and residual
  bypasses are shared coordinates, not private variables one may drop to
  make a separator look scalar.

The theorem establishes a restricted compositional property. It does not
justify a tree-based main implementation for the saved Tiny/CIFAR CNNs, and
ordinary HZ with the same local rows has the same continuous envelope.

## E. Two additional directions checked and not mistaken for novelty

For a linear value space S, conformal elementary-vector decomposition gives

    conv{(f,|f|):f in S}
      = cone{(e,|e|):e an elementary direction of S, both orientations}.

Every f decomposes without sign cancellation, so absolute values add too;
the reverse inclusion is immediate. This uses existing mathematics:
[Muller--Regensburger, theorem 3](https://arxiv.org/html/1512.00267v2).
These directions belong to S, whereas D007's dependence circuits belong to
S-perp. Confusing them explains why all dependence triangles need not be a
complete hull. At rank r, selecting r-1 independent zero-coordinate equations
gives at most 2*binomial(k,r-1) candidate directions, still potentially enormous.
No generator enumeration or underapproximation by a few directions is proposed.

Tropical/support-function re-expression is also not an unexplored escape:
[Zhang--Naitzat--Lim](https://proceedings.mlr.press/v80/zhang18i/zhang18i.pdf)
connect ReLU networks with tropical rational maps; [Goubault et al.](https://arxiv.org/html/2108.00893v2)
already give a tropical-polyhedral neural abstraction with exact ReLU transfer
and approximated affine maps. This checkpoint does not replace HZ with that
domain, adopt subdivision, or implement a tropical candidate. The observation
only prevents declaring a max-plus rewrite a new Neural-HZ definition.

## F. Known SDP already implies the Gram-mass certificate

This is our comparison derivation using the full affine/bias-consistent extension
of [Raghunathan--Steinhardt--Liang, equation 4](https://arxiv.org/pdf/1811.01057)
(the displayed original omits biases), not a theorem quoted from that paper.
In its Gram-vector realization take a
unit constant vector e, input vectors q_j with ||q_j||<=1, preactivation
vectors F_i=z_i e+sum_j A_ij q_j and ReLU vectors R_i. The lifted ReLU
identity ||R_i||^2=<R_i,F_i> implies

    ||2R_i-F_i||^2=||F_i||^2.

The scalar m_i=<2R_i-F_i,e> is at most ||F_i||. Triangle and weighted
Cauchy--Schwarz give

    sum_i w_i m_i <= sum_i w_i |z_i|
      +sqrt(W0 sum_{jl}G_jl <q_j,q_l>)
      <= sum_i w_i |z_i|+sqrt(W0 B0).

Thus the analytic certificate extracts a valid consequence without solving
the SDP; it does not beat that complete moment relaxation. No SDP execution
or weaker unstated moment formulation is assumed here.

## G. Evidence and custody

Root derived and inspected the candidate/proofs and read the cited primary
sources. Independent agents checked the Gram rule/control, the single-port
gluing proof and counterexamples, the phase-capacity projection and the prior
art boundaries. This is mathematical review, not machine proof or numerical
qualification. No current result relies on the unexecuted D008 drafts.

Pre-document read-only identities, SHA256:

    goal amendment:
    0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c
    D007 SHA256SUMS:
    18c8696e51dfba11653c78de904cd0264c9b46d3f6aa0a57a474c575fae9fc17
    D008 bundle_kernel_v1.py (UNQUALIFIED draft):
    ab5d8190629a360420b61e70c5de7f144c2316f0a64c5c123942658059413713
    D008 test_bundle_kernel_v1.py (UNQUALIFIED draft):
    5c97534d3bf3acefae251a6e79b6d96f1e90d79d99f2ecce1380620dc51e4cb1
    saved CIFAR descriptor:
    03aff8b0cb3aaa80a15ff27774cb74dfef599f180849878a098c807b30f66ce4
    saved Tiny descriptor:
    f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274

All old files remain at their original paths; they were not rewritten or
reclassified as numerical passes. New paper files and their final hash manifest
are isolated under d009_bounded_phase_energy_20260928. CURRENT_RESEARCH.md is
the existing intentionally mutable navigation index, not a frozen score record.
No commit/push, baseline update, production edit or runtime default change.

The ordered name-plus-file-byte SHA256 of D003's 13 production/provenance files
was recomputed read-only and still equals
`15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75`.
D007's complete five-entry seal verified again; both D008 draft hashes above
are unchanged. The worktree retains its pre-existing nine tracked dirty files.
