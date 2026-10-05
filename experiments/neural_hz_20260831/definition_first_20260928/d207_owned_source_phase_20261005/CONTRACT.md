# Owned source and phase Neural HZ calculus

This default-off candidate implements the source-phase relation and query calculus from D204 and D205, addressing D206's exact-alias precision loss. It is one owned nonconvex state with intrinsic transfer and query operations, not a standalone bound helper. The mathematical question is whether shared-source relationships survive intermediate variables and actually stabilize a later ReLU through the same domain's API.

This is not an established new set class or a finished Neural-HZ innovation. Its integer semantics retain an exact piecewise-affine graph, which ordinary HZ can also represent. The potential contribution is a structurally generated relational calculus with usable, compositional precision at acceptable complete cost. Real-network gains, novel abstraction or simplification beyond this calculus, GPU execution, smooth activations and Transformers remain open parts of the full objective.

## Element and concretization

An element owns a source frame, an ordered sequence of continuous factors and original signed binary identities, all EQ/LE predicates, exact affine alias definitions, birth-ordered ReLU banks, visible affine readouts, and the original input decoder. A single assignment satisfies every factor bound and every predicate; all consumers read that same assignment. A signed binary has values -1 and +1 only. Writing beta=(sigma+1)/2 never changes its discrete type.

The concretization is the image of these common assignments under the chosen readout. Empty banks embed bounded HZ states with arbitrary supplied linear source and signed-phase predicates. Inclusion of concretizations on a common interface defines precision; no computable best abstraction or complete lattice is asserted. Different independent extensions are not silently joined. Existing prefix views remain usable only on their own descendants.

The supported constructor is explicit opt-in box, followed by the owned operations below. Direct unowned state construction is rejected; this is not a security claim against arbitrary Python private-state manipulation. The current original-input decoder is affine in the initial continuous source coordinates, matching the input-box setting. An importer needing additional decoder phase columns is not admitted by this interface.

Affine maps, Add, selection and Concat coalesce coefficients of identical factor identities. An alias creates a fresh continuous factor and the exact equation alias=producer. Its reliable bounds come from the owned parent. The alias remains in the physical mathematical state and in membership; only certificate expressions substitute its known definition. There is no search through arbitrary equations, no binary pivot and no removal of original variables or consumers.

Alias definitions refer strictly to earlier factors. Substituting them in reverse birth order is exact on every member. The canonical expression is therefore unchanged by inserting such intermediate variables, provided the same original producer expression and factor identities are retained. This is a local producer-identity property, not an invariance theorem for arbitrary alternative bases or all equivalent HZ descriptions.

## ReLU birth and uniform source-phase generation

Each bank is born from one complete parent view. Every gate retains its own original signed binary and nonnegative continuous output q. Reliable parent bounds L<=f<=U give the usual exact graph rows

```text
q>=0, q>=f, q<=max(0,U)*beta,
q<=f-min(0,L)*(1-beta).
```

Stable gates retain their phase identities, including both legal zero labels. Bounds are obtained internally; the caller does not supply caps, energy or a certificate chosen for a property. The original graph is retained even where a stronger query is unavailable.

Pair adjacent positions (0,1), (2,3), and so on, at the bank's original order, with fixed a=b=1/2. An unmatched gate remains represented. Stable gates use their exact stable transfer. For a pair of crossing gates, form r1=f1-f2/2 and r2=f2-f1/2 after expanding exact aliases and combining all shared coefficients. Bound the complete current parent factors by their reliable intervals, including older amplitudes and integer factors' interval envelopes without changing their types. Normalize nonfixed coordinates to [-1,1].

For ri=di+pi*xi let ci=di+sum(abs(pi)). When both ci are positive, on every common same-sign nonzero coordinate let

```text
vj=sign(p1j)*min(abs(p1j)/c1,abs(p2j)/c2),
tau=min(1,1/(4*sum(abs(v)))) if v is nonzero,
w=tau*v, omega=sum(abs(w)), s=1-omega+w*xi, l=1-2*omega.
```

Zero common support gives s=l=1. As proved in D205, ri<=ci*s throughout this reliable parent box, hence throughout the complete parent element, and 1/2<=l<=s<=1. If a positive-cap premise fails, no source-scale pair is asserted; the original graph and fixed single-gate certificate remain. This is a declared mathematical-premise condition, not a result-dependent verifier menu.

For a covered pair retain all ten D204 rows, where Ui=(ci+cj/2)/(1-1/4): nonnegativity, each of qi<=Ui*beta_i and qi<=Ui*(s-l+l*beta_i), and each of qi<=ci*beta_i+qj/2 and qi<=ci*(s-l+l*beta_i)+qj/2. No new binary, beta*s product or phase enumeration is used. These valid rows preserve integer graph semantics; they strengthen an outer query, not the exact HZ set. Their coefficients are expanded and counted, not treated as ten free scalar records.

## Intrinsic query and recursive soundness

Every query unconditionally computes three complete scalar upper certificates: independent single-gate triangles, constant-cap pair feedback, and source-dependent pair feedback. Taking their minimum is one fixed query rule. All three are evaluated and paid; an unsupported value or exhausted resource fails the operation, rather than activating a rescue path.

For a crossing gate set ell_i=-L_i>0. The source-phase row and an original graph row yield

```text
qi<=hi+Mij*qj,
hi=ci*(l*fi+ell_i*s)/(ci*l+ell_i),
Mij=ell_i/(2*(ci*l+ell_i)).
```

The constant-pair certificate uses s=l=1. On the entire legal parent, hi>=0 and M12*M21<1/4. The independent certificate uses qi<=U_i*(fi+ell_i)/(U_i+ell_i). Exact stable-positive outputs use fi and stable-inactive outputs use zero. A source expression can contain signed contributions from older gates; the proof only needs its value to be nonnegative on legal parents, not every coefficient to be nonnegative.

For each pair, collect the complete signed query coefficients before using D138's closed formula

```text
D=1-M12*M21,
k1=max(0,z1,(z1+M21*z2)/D),
k2=max(0,z2,(z2+M12*z1)/D).
```

Replace the current pair contribution by k1*h1+k2*h2, combine it with every retained parent and skip contribution, and continue through earlier births. The coefficients k are independent of the parent assignment. Thus the replacement is an upper bound pointwise on every legal parent; induction proves the final source-box certificate sound, even when earlier coefficients become negative. The next ReLU calls this same query. No original-network backward verifier, LP status, dual ray, infeasibility repair or second verification algorithm is consulted.

These formulas give exact support only for a selected two-coordinate feedback fiber with its parent fixed, not for the complete nonconvex element. Retaining old graph rows does not prove blanket query dominance across all earlier research domains. The three fixed certificates preserve the particular single-gate and constant-pair alternatives in this candidate; benchmark zero regression still requires its full original gates.

## Costs and scope

This is a new sparse implementation, not an increase of D136, D149 or D180's frozen dense-reference capacities and not a transfer of their qualification. Its accounting uses the existing native diagnostic ceilings: 256000000 total work, 200000000 branch work, and 64000000 logical entries. Values use the established 512-bit rational restriction. Optional caller caps may only tighten these ceilings. One lineage meter covers source construction, exact alias transport, all banks, predicates, readouts, decoder and all query certificates; it is never reset for a pair or a new layer.

Conservatively, the entire lineage's arithmetic is also accumulated against the branch ceiling; neither a new public query nor a sibling extension resets it. Retained-entry accounting also charges copied prefix metadata and transient readouts without reclaiming them. It is an upper logical ledger, not a measurement of resident objects or an exact machine-instruction count.

Logical storage and arithmetic accounting do not certify Python object memory, full native storage, execution time or GPU lowering. The mathematical gate retains its separate 16 GiB process address-space and 60-second complete pytest limits, and the inherited supervisor-only memory observations. A full 200-gate supplied structure must be represented in one state and counted with all its original graphs; it is not admission of the real 200-gate Tiny checkpoint or payment for its preceding network.

Exact rational operations are a proof-oriented CPU implementation. Independent pairs may eventually permit device batching, but reliable floating arithmetic, sparse alias expansion, shared storage and host/device coexistence remain unimplemented obligations. General convolution import, model binding, joins of independently extended states, terminal lowering and concrete-network ADV validation are outside this gate. The existing formal baseline and production defaults cannot change.

## Evidence boundary

D138, D204, D205 and D206 are mathematical anchors, not inherited execution qualifications. The same-information comparator is old HZ with these same valid relations; no superiority over that exact integer set or complete source-labelled joint hull is claimed. The fixed ordinary positive control tests improved consumption, not external novelty.

All new files and outputs stay in the new isolated D207 directories. The original formal 1870/2413 and independent E0 61/400 remain unchanged. Provenance is redu-hz at f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, with tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. The complete research goal remains active.
