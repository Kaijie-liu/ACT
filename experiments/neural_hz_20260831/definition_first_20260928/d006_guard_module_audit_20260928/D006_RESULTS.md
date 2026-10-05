# D006: guard-module definition audit, not a new-domain qualification

Date: 2026-09-28. Branch: `redu-hz`.
Commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Configuration: paper mathematics, primary-source review and read-only local
provenance checks; no numerical imports, model execution, solver run or tests.
Baseline: 1870/2413 (1063 CERT, 807 validated ADV); separate E0 CIFAR10025 and
TinyImageNet36. No new solves or default changes. All existing gates remain.

## Alignment and decision

The user's requirement is a nonconvex neural abstract domain derived from HZ's
mathematical definition, not merely storage, construction or compiler rewriting.
The active Goal and `../ALIGNMENT_D005_20260928.md` remain authoritative.
This audit rejects an unsupported upgrade of D001's guard quotient into a
claimed new domain. D003 remains a semantic reference; D004/D005 and the
results below remain supporting research. No D005 engineering census begins.

Three independent read-only reviews examined the algebraic scope, domain
definition and prior art. Their conclusions were checked and synthesized here.
This is a paper argument, not a machine-checked proof or benchmark qualification.

## 1. What the complete guard quotient actually contains

Use real coefficients for the mathematical statement. An implementation would
need separately specified exact rational data and bit-complexity bounds.
Let

    B = R[beta_1,...,beta_q] / (beta_i^2 - beta_i),
    M = B^(p+1),
    f(xi,beta) = a(beta) + g(beta) xi.

Let S be the SAME feasible prefix relation on continuous xi and all original
Boolean identities beta, retaining input constraints, EQ/LE predicates and
every phase guard. Define

    Ann(S) = { f in M : f(xi,beta)=0 for all (xi,beta) in S }.

For a bit assignment b, put S_b={xi:(xi,b) in S} and

    K_b = span{(1,xi): xi in S_b}^perp.

**Fiber theorem.** Boolean evaluation gives B isomorphic to a finite product of
copies of R, and Ann(S) isomorphic to the product of the spaces K_b.

Proof: the idempotents

    e_b(beta) = product_{i:b_i=1} beta_i
                product_{i:b_i=0} (1-beta_i)

are disjoint indicators with sum one. Evaluating a form at b gives an ordinary
affine coefficient vector. It vanishes on S_b exactly when this vector belongs
to K_b. Conversely, any tuple v_b in K_b is represented by sum_b e_b v_b.
This establishes an algebraic characterization, NOT a phase-enumeration
algorithm or authorization to enumerate phases.

For a nonempty fiber with affine hull xi_b+V_b,

    K_b = {(-g xi_b,g): g is orthogonal to V_b},
    dim K_b = p - dim V_b.

An empty fiber has K_b=R^(p+1); a nonempty full-affine-dimensional fiber has
K_b=0. Consequently a complete guard quotient only adds identities from
unreachable fibers or affine restrictions within reachable fibers. It does
not summarize all inequality geometry.

Ordinary example: for ReLU(xi), xi in [-1,1], the two fibers [-1,0] and [0,1]
both have full affine dimension; Ann(S)=0 even though the guards are essential.
The two sets [-1,0] and [0,1] also have the same affine annihilator, so replacing
the retained predicates with that annihilator would lose information.

A boundary example fixes scope, not a proposed optimization target: for
ReLU(xi+1), the inactive fiber is {-1}. The form (1-beta)(xi+1) vanishes, but
1-beta does not. The legal zero-phase assignment may not be deleted.

## 2. Complete normalization can conceal the verification problem

The constant-one form belongs to Ann(S) iff S is empty. More locally,
e_b belongs to Ann(S) iff S_b is empty. Therefore a complete normalizer that
can distinguish the zero and one classes for arbitrary retained predicates
already decides feasibility of those predicates.

For an elementary reduction, encode each Boolean clause by an affine
inequality saying its literal sum is at least one. Include an unused bounded
continuous xi if a continuous factor is required. The resulting S is empty
exactly when the CNF is unsatisfiable. Thus a complete membership oracle for
this candidate cannot be treated as a cheap independent simplification step.
This is not a proof that useful incomplete structural identities are impossible.

Finite generation is not an efficiency result either. One can choose at most
p+1 generators by padding and aligning bases of the K_b across fibers, but
their Boolean coefficients may require exponentially large truth tables.
No compact construction or normal-form bound follows from the algebra alone.

## 3. Exact contextual collapse is not a general value congruence

Suppose an exact raw-value quotient identifies a<b while allowing arbitrary
affine translation followed by ReLU and exact observation. Choose a<t<b.
The context ReLU(value-t) distinguishes the two values: its results are zero
and b-t. Hence such a quotient cannot identify unequal raw values and still
support all these contexts exactly.

This does not prohibit alternative representations of the SAME function,
set-valued abstractions, existential projection with adequate interfaces,
consumer-indexed semantics or certified overapproximation. It does prevent
promoting D005's ReLU-port contextual equivalence into an unrestricted
affine/Add/ReLU congruence without additional retained information.

## 4. The useful incomplete contract, and its limits

One may retain a finite certified submodule J=sum_i B*j_i, with each j_i in
Ann(S). A proposed rewrite f to h needs an explicit locally checkable derivation

    f-h = sum_i c_i(beta)*j_i.

Boolean-only multipliers preserve the continuously affine module. Multiplying
by another continuously dependent form is not justified by this contract.
Generator validity needs a sound specified proof rule; neither arbitrary
semantic implication nor Boolean-circuit identity checking is a free oracle.
Missing proof means unchanged representation, never a positive conclusion.

The identities persist when constraints are strengthened, or when an extension
projects into S. They need not persist after weakening S. Keep all predicates,
bits, guards, shared frame identity and input reconstruction.

On fixed S, M/J describes equivalent representations of raw functions. It is
not by itself a semantic inclusion order, abstraction map, precision gain,
canonical normalizer or novel neural abstract domain. Adding redundant
certificates alone leaves concretization unchanged.

## 5. Strong prior-art comparisons

- Mixed polynotopes already use shared typed continuous/Boolean/signed symbols
  and exact type-dependent polynomial rewrites. Our Boolean-affine language is
  a fragment of that representation, although its general guard-aware module
  is not supplied just by those type identities. See sections3.3/4.3,
  Definitions28/30 and Proposition31 of
  [Combastel](https://arxiv.org/html/2009.07387v2).
- CPZs combine polynomial readouts with polynomial equality constraints.
  Inference for this project: bounded Boolean identities and bounded guard
  inequalities can be encoded with polynomial equations and bounded slack,
  retaining inputs/bits/outputs in the readout. That is a representability
  comparison, not an authorized replacement by CPZ. See Definition5 of
  [Kochdumper and Althoff](https://link.springer.com/article/10.1007/s00236-023-00437-5).
- Algebraic consequence certificates and normalization modulo generated ideals
  are established machinery. This does not imply complete consequences of
  arbitrary real inequalities. See section2, Definitions2/4 and Theorems1-3 of
  [Sankaranarayanan, Sipma and Manna](https://theory.stanford.edu/~sipma/papers/popl04.pdf).
- Sharp HZ already applies reformulation-linearization to Boolean monomials
  and continuous-times-Boolean products. Its hierarchy preserves the original
  integral relation and tightens relaxations, with full-depth convex-hull
  exactness and potentially exponential lifting. Merely adding these products
  is therefore not a new Neural-HZ contribution. See sectionsII-C/IV,
  Theorems1/2/7 of [Glunt et al.](https://arxiv.org/html/2503.17483v2).
  This is a comparator only: no convex replacement, binary relaxation or new
  solver path is adopted here.

## 6. Alternative reviewed, not selected as an innovation

A bounded-nullity interface can retain all bits while projecting m gate values
r onto z=Hr, provided EVERY outside consumer and predicate factors through H.
For rank(H)=m-1, write r=Rz+v*t. Substituting into the standard4m ReLU rows and
pairing positive/negative t rows gives an exact Fourier-Motzkin projection,
including the full LP relation. With preactivations independent of this block's
r, each gate contributes two rows of each sign when v_i is nonzero. The pair
count is at most4m^2, plus zero-coefficient rows as applicable. Other retained
constraints/bounds must also be included, so this is not a universal total.
Binary/zero-phase identities remain intact. Shared branches require one
compatible complete interface; separate incompatible projections lose coupling.

This is exactly available to ordinary HZ/MILP projection. It can trade one
continuous variable for many rows, coefficients, slacks and reconstruction
costs; residual identity consumers can remove the rank deficit entirely.
It is recorded as a comparator/rejected standalone novelty claim, not a new
implementation route or evidence of an actual CIFAR/Tiny opportunity.

## 7. Consequence for the active research

The next mathematical hypothesis must specify domain elements, concretization
and order, original-HZ embedding, shared-frame NN transformers, and a concrete
compositional/precision/complexity advantage on an ordinary repeated neural
structure. Comparison must permit the ordinary baseline the same certificates
and reductions and include terminal solving and witness reconstruction costs.
Greater set expressivity than HZ is NOT required; a proved useful domain
invariant or analysis algorithm may suffice. A name, raw circuit wrapper,
one local rewrite, or an uncomputed complete quotient does not suffice.

No candidate meeting that test is established by D006. Its progress is the
explicit characterization and falsification of an unjustified definition
shortcut. Do not extend accounting/test infrastructure or run a large target
census merely to manufacture activity before a concrete hypothesis exists.
The overall Goal remains ACTIVE and incomplete; no pause/block/complete action
was taken. Historical data, production files and frozen sources were not edited.
