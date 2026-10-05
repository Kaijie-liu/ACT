# Shared consumer fibers with a fixed mass coordinate

This default-off rational component implements the domain proposed in the frozen [D156 theory](../d156_two_branch_consumer_fiber_20261004/THEORY.md). It replaces fresh per-gate amplitudes by a complete declared consumer packet; it does not retain the full new ReLU graph and attach a helper. All original continuous source identities, signed binary phases, existing EQ/LE predicates, shared lineage and the source-affine input decoder remain. Actual network binding, GPU computation and benchmark qualification are not claimed.

## One structural carrier rule

For an m-wide preactivation g and the complete declared consumer matrix B, set C to B followed by the all-ones row. If an exactly equal all-ones row already occurs in B, reuse its first occurrence. Do not perform numerical-rank selection or choose rows by a property, model identity, margin or solver state. The resulting packet represents Bq and the shared mass Q=sum(q_i), q=ReLU(g). Additional B rows, including an identity skip when present, are never omitted.

This API defines a mathematical block whose consumer list is B. It does not authenticate a real graph's completeness. A future adapter must prove that every live use, predicate and decoder use is covered. The p rows of C need not be independent and need not satisfy p<m. No compression or speed claim follows from the API alone.

Use the existing D149 exact rational Form, Frame, Readout, Identity and Work primitives as supporting algebra, not its lossy ReLU transformer. Reuse the inherited 512-bit, small dimension, predicate, matrix-entry and work limits without raising them. A physical non-unit source box retains its midpoint, half-width and exact decoder; it is not treated as a unit perturbation box.

## Native predicate and sound replacement

Choose gbar by substituting the physical source midpoint, previously fixed nominal phase labels, and zero amplitudes into g. This is a structural reference, not a sampled network input or an assumed reachable state. Preserve all new original phase identities in gate order. Their fixed nominal references are 1 for positive gbar and 0 otherwise, without fixing their actual labels.

For d=g-gbar and s=Cd, introduce p common amplitudes a. The packet is

```text
Cq = C diag(beta) gbar + a,
T = C diag(beta) C^T,
U = C diag(1-beta) C^T,
a in range(T), s-a in range(U),
a^T T^dagger a+(s-a)^T U^dagger(s-a)<=E(beta_old).
```

E is certified over the entire current parent native domain. The exact true image has a=Cdiag(beta)d. Equivalently, the bare integer fiber permits one shared proxy dhat in d+ker(C) with squared norm at most E. It does not permit independent proxies for different consumers; it also does not preserve the original source component in ker(C). All-zero gate values retain both original phase labels.

Native membership evaluates a supplied assignment, including its integral signed bits, linear predicates and each bank. Exact rational elimination of Tz=a and Uw=s-a checks range and evaluates a^Tz+(s-a)^Tw. T and U are Gram matrices by construction; the value is independent of free kernel coordinates. This is evaluation of the declared predicate, not an optimization oracle, terminal verifier, phase search, ADV generator or solver rescue. Arithmetic or work-cap failures fail closed. Fractional phases may be used only to inspect the finite LP outer rows, not as native members.

Original HZ embeds with no consumer bank. Affine, matrix-form Conv, Add and Concat act on the same shared affine coordinates. Ancestor readouts may be aligned into descendants; independent siblings cannot be mixed. Every bank and inherited predicate persists into later blocks. A concrete input decoded from an abstract state is not a validated counterexample until the original network and property are checked separately.

## Fixed energy budget

The source contribution is bounded by the inherited outward Gram absolute-sum certificate with the physical box half-widths. Its outward norm is squared to obtain a squared-energy bound. Old phase-column j contributes its squared Euclidean coefficient norm times |beta_j-tau_j|, with tau_j in {0,1}. Each old bank contributes ||M_k C_k||_2^2 E_k, with the operator norm replaced by the inherited certified outward matrix-norm bound where needed.

Omit identically zero terms. If there are n remaining source, phase-column and bank terms, use the single fixed weighted-Cauchy rule

```text
E_new=n*sum(term_squared_energy).
```

If n=0, E_new=0. This is equal positive weights in D156 equation 7, not an optimized or status-dependent budget. The result is phase affine and nonnegative on the full Boolean box. Old references and negative phase coefficients are preserved, not clipped away. These bounds may be loose and their cross-layer growth is payable; this registration does not claim precision-preserving recursion.

Scalar support used to obtain reliable guards bounds source and phase contributions and each bank direction independently. A bank direction w obeys |w^Ta|<=||C^Tw||_2 sqrt(max E). Any use of existing single-row predicates must be explicitly implemented, deterministic and sound; no implicit optimization over the full predicate set is assumed. Bounds must cover abstract parent members, not just original network trajectories.

## One finite linear lowering

Every block emits 2m original sign guards, 4p coordinate branch bounds, two mass-dominance rows per declared B row, and 6p energy rows. Existing rows remain. Every new bit is retained even for stable or zero gates.

For reliable L_i<=g_i<=U_i, guards are g_i<=U_i beta_i and g_i>=L_i(1-beta_i). Phase-conditioned bounds for d use

```text
active:   [max(L_i,0)-gbar_i, max(U_i,0)-gbar_i],
inactive: [min(L_i,0)-gbar_i, min(U_i,0)-gbar_i].
```

The zero interval on an impossible branch is harmless because the original guards exclude that branch; zero labels remain legal. Apply each row C_j with correct coefficient signs to obtain lower and upper affine bounds for a_j and s_j-a_j. These four rows per coordinate are valid for the real graph and are explicitly part of the strengthened candidate. They are not asserted to follow from the bare energy fiber.

The common nonnegative mass gives, for every B row,

```text
min_i B_ji * Q <= (Bq)_j <= max_i B_ji * Q.
```

These two valid rows preserve elementary dependencies among packet coordinates. They are not a full projection of all per-gate epigraph or mask rows.

Let Emax be the reliable Boolean-box upper bound of E. For each packet coordinate j set t=sqrt_upper(Emax/sum_i C_ji^2) when numerator and denominator are positive; otherwise set t=1. Emit the D156 linear inequality

```text
2(lambda-mu)^T a+2 mu^T s
 <= E+sum_i beta_i(C_i^T lambda)^2
      +sum_i (1-beta_i)(C_i^T mu)^2
```

for the six pairs (t e_j,0),(-t e_j,0),(0,t e_j),(0,-t e_j),
(t e_j,-t e_j),(-t e_j,t e_j). All scales and directions are fixed by the same formula before querying. No caller-supplied cuts, adaptive direction search, backward pass, LP-state repair, QP, SOCP or SDP is introduced.

Finite rows are only an outer approximation of the native fiber. Coordinate branch bounds make all-on and all-off packets exact, but generally do not recover every mixed-phase range condition. Native recursive bounds may use native predicates; the larger finite outer polyhedron must not be substituted as if it were the native domain.

## Positive and negative ordinary controls

The D156 Householder control has g=(I-11^T/8)x-1/2, x in [-1,1]^16. Declare the actual mixed consumer B=(1,...,1,-1/4); the uniform rule adds the mass row, hence exactly two new amplitudes. The mass coordinate's t=1 energy row yields 2Q+N+sum(x)<=16. Its upper branch bound gives Q<=(9/4)N, and the generated dominance row gives Bq<=Q. Therefore, after the shared input skip, J=Bq+(9/22)sum(x)<=72/11<33/5. A positive rational combination of generated rows certifies the next ReLU is zero. This is the existing D148 energy principle carried by the new packet, not a newly invented energy inequality or a benchmark CERT.

The added mass row makes the old two-gate B=(1,-1) carrier full column rank; the D156 false amplitude must now be rejected by native membership. This improvement does not make the general rule exact. For g=x+(1/4,-1/4,1/4), x in [-1,1]^3, B=(1,-1,1/2), consider x=0 and beta=(1,0,1). C has kernel vector (3,1,-4). The proxy dhat=(3,1,-4)/32 produces a=(1,-1)/32, with squared energy 13/512 below the exact source bound 3. The reference implementation uses sqrt_upper(3) squared, a slightly larger outward rational certificate, rather than asserting that this stored coefficient equals 3. The resulting Bq=13/32 and Q=15/32 differ from the actual 3/8 and 1/2, yet retain the declared branch and mass bounds. The next ReLU(Bq-25/64) can have the false same-input value 1/64. This is an ordinary lossy-state control, not an ADV or a whole-box property verdict.

The tests must preserve both the repaired old case and the remaining loss. Other controls cover HZ embedding, zero labels, all-on/off finite exactness, shared skip and descendants, decoder, non-unit boxes, ownership, fractional outer/native distinction, finite caps and complete logical accounting.

The Householder test supplies the rational row-combination weights explicitly. This checks the strength of automatically generated relations, not an implemented automatic whole-box stability query. The current support query bounds source, phase and bank contributions separately and does not combine the predicates. Inherited predicates still retain their logical implications; the weak support query is not evidence that the native domain loses the demonstrated whole-box property. Automatic joint support and forward propagation, followed by real-network capability checks, remain necessary work. Increasing test counts, matching one handcrafted certificate or adding an external optimizer would not establish that advance.

## Costs and admission

Charge original source and bit maps, all old banks and predicates, C and B, s forms, energy coefficients, nominal references, bounds, row storage, output readouts, decoder, rational bit growth and temporary Gram/elimination work. The predicate count per new block is 2m+10p+2*rows(B), before inherited rows. The mass row can increase rank and state size, and native membership may form Gram matrices. This is not a performance result or complete physical memory qualification.

The reference remains limited to the inherited small dimensions. It is not admitted for 3072-dimensional CIFAR sources or full Tiny frontiers, and no patchwise independent-source approximation is authorized. Smooth activations, Transformer, GPU arithmetic and outward device error remain unimplemented requirements of the full goal.

Freeze this contract, registration, implementation, test file, runner and collection plugin before imports, AST parsing, compilation, collection or numerical execution. Keep all 4000 inherited tests/210 files and append 20 tests/one file. The once-only mathematical gate is not real-source qualification, shadow or full replay. Formal 1870/2413 and independent E0 61/400 remain unchanged. Branch redu-hz, HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5; all historical artifacts remain read-only.
