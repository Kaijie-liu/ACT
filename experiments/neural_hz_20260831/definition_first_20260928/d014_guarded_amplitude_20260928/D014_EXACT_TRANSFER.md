# D014 — Guarded amplitudes and disjoint-sign exact neural transfer

2026-09-28; branch `redu-hz`; HEAD
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Configuration: paper mathematics, primary-source reading and independent
read-only review. No implementation, imports, tests, numerical qualification,
model decoding, solver, benchmark, production change, commit or push.
Authority: [definition-first goal](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md).
Formal baseline remains 1870/2413 (1063 CERT +807 validated ADV); separate E0
remains CIFAR10025 +TinyImageNet36 =61/400. Neither has been replayed here.

## 1. Outcome and definition-level question

D013's phase monomial expansion does not obtain a generic compact exact neural
calculus. This note instead treats certified nonnegative, phase-guarded
amplitudes and their incompatibilities as an interface for forward values.
There is a positive exact ReLU transfer with a complete retained-phase
correspondence and an LP-subset guarantee, not just an extra cut. A restricted
bias-free subnetwork remains in this representation without new amplitude
variables. An ordinary mixed two-gate control reduces native variables and
rows even after including its original source products.

This is a candidate neural calculus, NOT an established innovative abstract
domain. The scalar identity is classical; arbitrary biased/dense composition
is not covered. No ordinary CNN prevalence, native physical-cost qualification,
speed or new solved instance is established. General HZ with the same valid
identity and substitution can reproduce the result. Equal expressive power
is expected; usefulness and a substantive NN-specific contribution are still
required, not an impossible claim of greater expressiveness than general HZ.

## 2. Candidate elements and strong concretization

An element stores:

- Original continuous frame xi and ALL original free and ReLU bits B.
- All source EQ/LE predicates, original neural guards and exact input/value
  reconstruction records. An eliminated value is replaced consistently in
  every consumer, predicate, property and decoder, not silently forgotten.
- Nonnegative amplitudes e_i, their certified bounds and exact defining
  relations to the same source. A mask sigma_i is an alias of an original bit
  or its complement, not an independent clone.
- Optional certified group budgets sum_{i in G} sigma_i<=U_G and/or explicit
  incompatibility facts e_i*e_j=0, whose validity has already been proved
  on the complete original INTEGER relation, including original zero phases.
- Declared value forms y=s(xi,B)+C*e; the simple closed sublanguage below has
  s=0. General baseline terms are stored, never dropped to fit that language.

Gamma is the relation of (original input, ALL original bits, declared values)
obtained by ranging over original continuous latent assignments consistent
with the original-input decoder and auxiliary continuous amplitudes, subject
to those exact source/guard relations and the value forms. Certified
facts are consequences of the admitted relation, not additional assumptions.
Order is inclusion after aligning original identities. No best abstraction,
complete lattice, complete inference algorithm or general join is established.

Original HZ embeds by retaining its affine map c+Gc*xi+Gb*(2*zeta-1), original
predicates and bits, with an empty amplitude interface. Ordinary exact ReLU
relations can introduce amplitudes in the shared frame. Unsupported structural
claims leave the original relation intact; they do not authorize a second
solver route or a success-dependent representation menu. Binary bits are
never pivoted, deleted or relaxed as the represented semantics. A retained
input/ReLU graph is already nonconvex, so this is not a CZ/Zonotope replacement.

Crucial distinction: if e_i=sigma_i*t_i for an ORIGINAL source t_i, one must
keep that defining relation. Replacing it by 0<=e_i<=u_i*sigma_i alone loses
source correlation. The standalone factor in the companion note has an exact
hull only as a standalone object, not after arbitrary source gluing.

## 3. Exact disjoint-sign ReLU and LP inclusion

Let e>=0 and write one preactivation as

    f=P-N, P=sum_i max(w_i,0)*e_i, N=sum_i max(-w_i,0)*e_i.

Assume P*N=0 on the complete admitted integer relation. This may be certified
by pairwise incompatibilities between every positive and negative contributor;
same-sign amplitudes may coexist. Then

    ReLU(f)=P.

Proof: if P>0, N=0 and f=P; otherwise P=0 and f=-N<=0. No enumeration of
phase assignments is required by this identity.

Retain the gate's ORIGINAL phase beta and the SAME certified old bounds
l<0<u. Its exact new guard is

    P<=u*beta, N<=-l*(1-beta).

P>0 forces beta=1 and N>0 forces beta=0. At P=N=0 BOTH original choices are
legal. Bounds are valid since on the integer relation P=f when P>0 and
N=-f when N>0. No output-support count may be imposed on these bits merely
because at most one output is positive.

For the LP claim, the old native four rows are

    r>=0, r>=P-N, r<=u*beta, r<=P-N-l*(1-beta).

Add the valid identity r=P and substitute it everywhere. The first two rows
become tautologies from P,N>=0; the other two become the displayed guards.
Consequently every new LP point, lifted by r=P, is an old LP point. This is
an LP SUBSET theorem, not D005/D013's incomparable flat replacement. It
requires nonnegativity throughout the relaxed source formulation, the SAME
bounds, and consistent substitution of all old consumers. Replacing l,u by
arbitrary looser new M values does not inherit the proof.

For an unmaterialized eligible gate this saves one continuous variable and
two gate rows. If r remains an independent native variable, add r=P as an
equality: there is no continuous-variable saving and only one net row saved.
Coefficient fill, shared readout storage, proof records, native inequality
slacks and decoding must all be paid. This is an algebraic count, not a
physical-memory or speed qualification.

## 4. Strict two-gate control, with the complete source products

Keep ORIGINAL t1,t2 in [0,1], masks sigma1,sigma2 binary, and

    e_i=sigma_i*t_i, sigma1+sigma2<=1,
    W=[[2,-1],[-1,2]], r=ReLU(W*e)=(2*e1,2*e2).

The two original output bits beta1,beta2 remain. Tight bounds on both
preactivations are [-1,2]. The normalized new guards are

    e1<=beta1, e2<=1-beta1,
    e2<=beta2, e1<=1-beta2.

At e=0, beta=(1,1) is legal: do NOT infer beta1+beta2<=1. For each source
product, count the four exact rows e>=0, e<=t, e<=sigma,
e>=t+sigma-1, with 1+2+2+3=8 nonzeros.

| Formulation | Continuous | Original bits | Rows | Row nnz |
| --- | ---: | ---: | ---: | ---: |
| Original products, budget, two native ReLU gates | 6 | 4 | 17 | 38 |
| Products, budget, exact output readouts | 4 | 4 | 13 | 26 |
| New form with both outputs separately materialized | 6 | 4 | 15 | 30 |

Unmaterialized outputs also store two readout coefficients. Scalar source/bit
bounds, RHS, all evidence, metadata and decoder are additional. The last row
adds two equalities/four nnz, saving TWO rows total, not two per gate.

A strict old-LP-only point is

    t=(0,0), e=(0,0), sigma=(0,0), beta=(1/2,1/2), r=(1/2,1/2).

It satisfies the old products and four-row gates but violates r=2e. The shared
residual readout Y=r-2e is exactly zero in the true/new relation, whereas
the old fractional point makes it positive. This is paper separation only,
not an executed new CERT or a full input-output hull comparison.

## 5. Restricted exact composition through multiple layers

If the source amplitudes have at most one nonzero coordinate globally and
y=C*e, an entire bias-free affine/Conv, ReLU and shared Add/Concat subnetwork
has exact value propagation

    Affine/Conv: C <- W*C,
    ReLU:       C <- entrywise max(C,0),
    shared Add: C <- C1+C2,
    Concat:     C <- stack(C1,C2).

For each assignment either e=0 or e=t*unit_i, t>=0; positive-part homogeneity
proves the ReLU identity rowwise. Keep ALL original gate bits and their two
sign rows using the PREACTIVATION coefficient row. Keep the original source
relation and reconstruction throughout. This supplies an induction across
layers without inserting new ReLU amplitude variables or phase products.

It is not free: C may become dense, and q gates on k source amplitudes can
require 2q guard rows with O(q*k) nnz, exceeding the original sparse graph.
No complete runtime/byte benefit follows from avoiding q scalar variables.
Independent branch frames cannot be used when their original source is shared.

For only independent amplitudes and laminar budget predicates, the mixed-row
rule is narrow. Ignoring provably zero amplitudes WITHOUT deleting their bits,
two coordinates are incompatible iff they share a capacity-one ancestor.
Maximal capacity-one groups are disjoint. If every positive/negative pair in
a mixed row is incompatible, its whole support lies in one such group.
Higher capacities alone do not give a new generic mixed-row closure theorem.
Additional original HZ correlations may certify more general PN=0 relations.

## 6. Bias and a source-derived control

For a constant bias b, put P=b^++sum w_i^+ e_i and
N=b^-+sum w_i^- e_i. If b>0 and PN=0 then N is identically zero and f is
strictly active; if b<0 the gate is strictly inactive. Thus genuinely unstable
uses of THIS simple rule substantively need zero baseline. A nonzero source
baseline s(xi) cannot be discarded. Exact cancellation may prove it zero,
but such a certificate is itself required and must not use forbidden rescue.

A finite source-derived positive control demonstrates that the assumptions
can arise from mixed, distinct, nonparallel first-layer rows. It does NOT
establish their frequency in a trained CNN. For (x,y) in [-1,1]^2 let

    F=[[1,1],[-1,1/2],[1/2,-1],[-3/4,-5/4]],
    b=(3/2,1,1,8/5), f=F*(x,y)+b,
    r=ReLU(f), e=r-f=ReLU(-f), sigma_i=1-beta_i.

Box lower bounds of f_i+f_j for pairs12,13,14,23,24,34 are respectively
1,1,13/5,1,1/10,1/10. Hence at most one f_i<=0, so sum sigma<=1 including
ALL zero-phase choices. Error caps are (1/2,1/2,1/2,2/5), attained respectively
at (-1,-1),(1,-1),(-1,1),(1,1). Keep the original four bits and exact sources.

Set a=(7/12,-2/3,-1,1). Then a*F=0 and a*b=97/120, so

    g=a*r-97/120 = a*e,
    P=(7/12)*e1+e4, N=(2/3)*e2+e3,
    ReLU(g)=P, tight g bounds [-1/2,2/5].

The parent is genuinely unstable, and retains its fifth original bit using
P<=(2/5)*beta, N<=(1/2)*(1-beta). Choosing a in the left nullspace and cancelling
its bias was DELIBERATE. Ordinary-sized coefficients and nonparallel rows
do not make this an observed ordinary CNN motif. No numerical experiment
was performed, and this example cannot qualify real-network applicability.

## 7. Prior art, remaining obligation and next action

The scalar identity is classical disjoint-function/vector-lattice algebra:
nonnegative disjoint P,N obey |P-N|=P+N, whence (P-N)^+=P.
[Gluck, introduction](https://arxiv.org/html/2001.10941v3) explicitly states
the disjointness modulus identity. We do not claim that scalar result as new.

Joint unstable phase structure is also used in existing neural work. The
[MEAP construction, section2.1](https://arxiv.org/html/2605.17153v1) uses coupled
pairs with at least one active member and max/min aggregation. That is not
this exact two-guard lowering, but prevents a broad novelty claim based just
on using mutually constrained unstable phases. Its constructed stress cases
are NOT adopted here as a substitute for ordinary CIFAR/Tiny structures.

General HZ can encode the same integer relation and use the same substitution;
ImageStar with the corresponding exact regional constraints has no claimed
expressiveness disadvantage. Mixed-symbol/polynomial domains can express
the phase/source products. The open contribution is a useful NN-specific
calculus and complete-cost advantage, not the notation of guarded amplitudes.
Limited prior-art search has not established novelty of the retained-phase
normalizer or its composition. Absence of an exact search hit is not proof.

The [biased-extension note](D014_BIASED_EXTENSION.md) subsequently establishes
a weaker partial-negative shielding rule for varying affine baselines and
overlapping residual cores. It does not generalize this note's variable
elimination claim: its gate retains one output variable and four rows.

The companion note records the standalone factor theorem and two ordinary
counterexamples: individual ideal hulls cannot be glued for free over a shared
source, and independent-amplitude factors are not generally ReLU-closed.
These change the next action: test biased extension and faithful ordinary
source applicability before selecting an implementation, not expand a spline
or phase-enumeration engine around a narrow artificial example. Every later
candidate remains default-off with new preregistration and all original
math/real-structure/shadow/full2413+400 gates. The full Goal remains ACTIVE.
