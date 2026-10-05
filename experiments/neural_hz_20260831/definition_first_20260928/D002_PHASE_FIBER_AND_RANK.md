# D002 — Full phase relations, generator-rank separation, and terminal limits

Date: 2026-09-28. Branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Scope: paper derivations, independent mathematical review, primary-source
comparison and read-only saved-topology evidence. No implementation, numerical
qualification, model/solver run, score gain or novelty claim.

## 1. A stronger semantic object than the output set alone

D001's concretization projects away latent variables. For safe compositional
rewriting, make the retained relation explicit:

    Gamma(D) = { (u,beta,y) : xi in [-1,1]^p, beta in {0,1}^q,
                  P(xi,beta), u=iota(xi,beta_input), y=v(xi,beta) }.

`iota` is the original input map, `P` contains all EQ/INEQ and phase guards,
and beta includes every original phase identity. The ordinary output set is
only the projection of Gamma onto y. A permitted exact local rewrite proves,
for each SAME latent assignment, equivalence of predicates and equality of
visible values; it does not just prove equality after existentially hiding
phase history. Shared branch environments are joined by identity, not copied.

Retaining a relation is not by itself a novelty claim: dependency-preserving
symbolic sets and exact network graph sets already exist. It is the correct
contract against which candidate simplifications must be evaluated.

Example: two equal ReLU preactivations have equal outputs, but at zero their
independent bits may differ. Enforcing bit equality preserves the output set
and loses Gamma witnesses. Consequently a functional-equivalence theorem is
not automatically a phase-preserving domain rewrite.

## 2. A definition-level rank barrier for constant-generator HZ

Lemma (connected-set affine-hull bound). Suppose a nonempty connected set Z
has a standard finite-binary HZ representation with constant continuous
generator matrix Gc and r continuous factors. Then

    affine_dimension(Z) <= rank(Gc) <= r.

Proof. Let V=col(Gc), and pi be the linear quotient map modulo V. Every point
of Z is c+Gc*xi+Gb*s for one of finitely many binary assignments s, regardless
of its equality/inequality predicates. Thus pi(Z) is a subset of the finite
set {pi(c+Gb*s)}. It is also connected, because pi is continuous and Z is
connected. A connected finite subset of a Euclidean quotient space is a
singleton. Therefore Z lies in one translate of V, proving the claim.
This finite-set reasoning is a proof, NOT a runtime phase-enumeration method.

Corollary (neural graph tangent rank). For a continuous piecewise-affine F
on a full-dimensional convex input box U, retain the JOINT graph
Z={(x,F(x)):x in U}. On each nonempty open affine cell s let J_s be the
Jacobian. Any exact constant-generator HZ with an affine joint input/output
decoder needs at least

    rank span { columns([I; J_s]) : open cells s }

continuous factors. The graph is connected and each listed tangent belongs
to the direction space of its affine hull. Alternatively the tangent
inclusion follows cell by cell and applies to disconnected input pieces.
The theorem does not require a computation or enumeration of all cells.
Here the connected graph projects away phase coordinates; Gamma itself need
not be connected. Preserving the stronger Gamma also preserves this projection,
so the bound still applies to a constant-affine representation of Gamma.

### Ordinary coordinatewise ReLU: p versus at least 2p

Take U=[-1,1]^p and F(x)=ReLU(x). On the all-negative open cell the graph
tangents include (e_i,0). Changing coordinate i's sign adds (0,e_i) to their
span. These 2p directions are independent, so affine_dimension(Z)=2p and
every constant-Gc HZ needs r>=2p.

D001 represents this SAME graph using p continuous factors and p retained
bits, via y_i=beta_i*x_i and (2 beta_i-1)x_i>=0. Thus changing the generator
language really can reduce continuous-factor count, without convexification
or binary deletion. This is not merely a different CSR layout.

### Ordinary mixing affine/ReLU layer: p versus at least p+m

Let F(x)=ReLU(Wx+b), with m nonzero rows. Assume each row's geometric zero
hyperplane cuts the interior of U, and the m geometric hyperplanes are
pairwise distinct (unequal coefficient rows alone are NOT sufficient).
For each i, pick a point on its hyperplane in the interior, off all other
hyperplanes. Such a point exists because finitely many proper intersections
cannot cover that hyperplane's relatively open portion. Neighboring cells
have Jacobians differing by e_i*w_i. Since w_i is nonzero, differences of
their graph tangents generate the vertical direction (0,e_i). Together with
one full-rank [I;J] these generate p+m independent directions. Hence

    constant-Gc joint graph: r >= p+m;
    D001 guarded dependent generators: p continuous factors, m new phase bits.

This covers generic crossing affine/ReLU rows, not only exact duplicate or
diagonal chains. Coincident hyperplanes, restricted input dimension and
noncrossing rows need their actual tangent span; no p+m claim is made there.
No target network has been screened for these coefficient premises in D002.

### Limits that prevent a false performance claim

These are paper derivations, not claims that the geometric observation is
previously unknown. Mixed polynotopes can also exploit variable generators.
The theorem concerns the retained input/output GRAPH with a constant-affine
joint decoder. Output-only sets can have a much smaller affine hull. It does
not apply unchanged to nonlinear decoders or only one projected query.

Moreover, any finite-binary LINEAR terminal formulation whose joint readout
is constant-affine in r continuous variables faces the same argument: after
fixing binaries, its readout lies in a translate of a fixed continuous column
space. Thus fully lowering the generic graph to a linear backend restores at
least the graph-rank requirement. Continuous variables may be unbounded in
that formulation; the geometric argument still applies. General unbounded
integer variables/nonlinear readouts are not covered by this statement.

Do NOT add the p+m bounds layer by layer to count all hidden neurons. They
refer to one specified observed input/output graph. If only k final outputs
are observed, that graph's affine dimension is at most p+k; the bound does
not require retaining every historic hidden value. Conversely p+k is only an
ambient-dimension upper bound, NOT a guarantee of a p+k-variable compact
linear formulation: predicate/extension complexity may still be much larger.
This makes the exact observable interface, including every predicate consumer,
a meaningful parameter for the next domain definition and cost comparison.

The factor-rank separation is therefore a real semantic-coordinate benefit,
but NOT a claim of fewer terminal variables, resident bytes, a faster solver,
or a new solved instance. Propagation, guard storage, certificates, terminal
lowering and witness recovery must all be paid. In particular D001 cannot
promise a generic full-graph linear backend using only p variables.

## 3. Complete threshold-chain guard theorem

To close D001's illustrative chain rather than leave an optimistic example,
consider r0=f and

    r_i=ReLU(a_i*r_(i-1)-theta_i), a_i>0, theta_i>0,
    A_i=product_(j<=i) a_j,
    T_i=sum_(j<=i) theta_j/A_j.

T_i strictly increase. The original values and ALL binary guards are exactly
equivalent, on the same (f,beta), to

    r_i=A_i*beta_i*(f-T_i),
    (2 beta_i-1)*(f-T_i)>=0, for all i.

Proof: the first gate is immediate. For i>=2, induction gives
r_(i-1)=A_(i-1)*ReLU(f-T_(i-1)). If f<=T_(i-1), the
next original preactivation is -theta_i<0, as is f-T_i. Otherwise it is
A_i*(f-T_i). Thus original and rewritten preactivations have identical signs
and zero sets. At f=T_i both original bit choices remain legal. The guards
also imply beta_1>=...>=beta_m without an added runtime case split.

Retired continuous intermediates are reconstructed from the displayed
formula. Every outside predicate or consumer of r_i counts as fanout; it
cannot disappear because a layer's ordinary output is dead.

The theta>0 premise is necessary for the full phase relation: with
r1=ReLU(f-1), r2=ReLU(r1), f=0, original bits (0,1) are legal but flattening
the second guard to f-1 wrongly forbids its active bit. With a negative
threshold, even value fusion can fail. These are ordinary proof-boundary
examples, not a new extreme-case optimization campaign.

## 4. Fair complete linear comparators

Assume certified L<T1<...<Tm<U, with boundary scalar t=f already represented.
Define Delta0=T1-L, Delta_i=T_(i+1)-T_i for 1<=i<m, Delta_m=U-Tm.
Counts below count nonnegativity as a row, and exclude only identical shared
input constraints present in EVERY arm. They are symbolic formulation counts,
not current-backend measurements or a completed physical-resource ledger.

Terminal-only formulation, h=r_m/A_m:

    beta_i>=beta_(i+1)                         (i<m)
    0<=h<=Delta_m*beta_m
    t-h>=L+sum_(i=1..m) Delta_(i-1)*beta_i
    t-h<=T1+sum_(i=1..m-1) Delta_i*beta_i.

Monotone bits have a prefix of k ones. For k<m, h=0 and the last two rows
place t in [T_k,T_(k+1)] with T0=L; for k=m they give t-h=Tm. Both prefix
choices at an interior threshold remain. This proves equivalence of the full
phase relation without an algorithm enumerating prefixes.

All-fanout optimized comparator, h_i=r_i/A_i:

    t-h1<=T1
    t-h1>=L+Delta0*beta1
    Delta_i*beta_(i+1)<=h_i-h_(i+1)<=Delta_i*beta_i   (1<=i<m)
    0<=h_m<=Delta_m*beta_m.

The successive differences fill their intervals in order; positive Delta_i
forbid a 0->1 bit inversion. They reconstruct exactly every hinge value and
its boundary phase alternatives. This is ordinary incremental PWL algebra,
not a special privilege of the proposed domain.

| Full formulation | Extra continuous values | Retained bits | Inequality rows | Constraint nnz |
| --- | ---: | ---: | ---: | ---: |
| Four-row gatewise lowering | m | m | 4m | 8m |
| Flat terminal plus separate guards | 1 | m | 2m+2 | 4m+4 |
| Terminal aggregate guards above | 1 | m | m+3 | 4m+4 |
| Optimized all-fanout incremental comparator | m | m | 2m+2 | 6m+2 |

If f has d nonzero coefficients, introducing t=f costs one common continuous
variable, one equality and d+1 nonzeros. Substituting f into all rows instead
duplicates its coefficients and must be counted. An equality-only HZ backend
also pays for required bounded slack factors; one slack per inequality is
only a mechanical upper bound, not an optimal count. Metadata A/T/Delta,
proofs, bit lengths, actual storage and inverse evaluation remain unmeasured.

Decision: a terminal-only chain can lose m-1 internal value variables while
retaining every bit/guard. But optimized ordinary HZ/MILP can use the SAME
rewrite, so it does not establish a new-domain advantage. If all intermediates
are observed, the graph directions span m+1; a constant-affine native decoder
cannot reduce that total continuous rank. No numerical candidate is admitted.

## 5. Relevance and prior-art decisions

[Saved topology audit](D002_SAVED_STRUCTURE_EVIDENCE.md) shows no direct
ReLU->positive-diagonal-affine->ReLU chain in the two recorded Tiny/CIFAR
graphs: their ReLU successors encounter Conv/Dense mixing (or a residual Add
followed by mixing). This does not exclude hidden coefficient identities,
which were not scanned. Failed early CNN sharing censuses are not evidence
that all CNN sharing opportunities are absent.

Independent literature review found these necessary comparators:

- [Mixed polynotopes](https://arxiv.org/html/2009.07387v2) already provide mixed
  typed symbols and neutral/inclusion-preserving rewriting. D001 notation and
  exact symbolic rewrites alone are not established novelty.
- [Neural-network lumping](https://arxiv.org/abs/2209.07475) gives exact
  proportional reductions and discusses co-sign linear combinations. The
  associated [journal version](https://doi.org/10.1016/j.neunet.2024.106411)
  is also a comparator. A same-sign identity alone is insufficient.
- [Stability-based exact compression](https://optimization-online.org/wp-content/uploads/2021/02/8256.pdf)
  covers rank-based stable-neuron simplification and stable-layer folding.
- [Zhang and Bolcskei, 2026 v2](https://arxiv.org/html/2602.00266v2), revised
  September 3, gives a compositional logical characterization of functional
  equivalence and explicitly illustrates a positive threshold chain in
  Example 4/Figure 10. Its conclusion leaves constructive rewrite-sequence
  synthesis open. Functional equivalence is not the same contract as Gamma
  preservation, but that distinction alone does not prove our novelty.

Close the scalar-threshold-chain idea as a MAIN implementation target: it is
not supported by these target topologies and offers no unique terminal
advantage over the strong ordinary comparator. Keep its exact full-phase
theorem as a reusable regression/example. Do not manufacture a benchmark
population of such chains or resume C131 cost tuning to avoid this finding.

## 6. Consequence for the next candidate

Keep the phase-dependent generator language as a REFERENCE candidate, not an
established breakthrough. Any next prototype must operate on ordinary mixing
affine/ReLU/residual blocks, retain Gamma, and compare against both the existing
HZ simplifications and an equally shared unnormalized circuit. A gain may be
in propagation or a proved observable interface reduction, but the full
terminal bill must be included; the rank theorem rules out promising generic
all-graph native-variable compression below p+m.

The next bounded deliverable is a reference definition/transformer and exact
phase-relation checker for those ordinary blocks, with a complete lowering
comparison. Its implementation/test plan must be frozen before numerical
execution and must preserve the inherited qualification population and caps
where applicable. This note does not authorize an easier promotion gate,
phase elimination, new solver rescue, full-model run or score change.
