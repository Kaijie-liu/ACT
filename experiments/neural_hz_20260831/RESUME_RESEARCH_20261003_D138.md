# Neural HZ small feedback block candidate checkpoint

The full definition-first nonconvex Neural-HZ goal is active. Formal baseline 1870/2413 and independent CIFAR25 plus Tiny36 equal 61/400 are unchanged; both new gains are zero. No source/model/solver/GPU/test execution occurred in this paper turn. D136 remains the latest executed mathematical population, 3965 tests across 208 files under its original gates.

The [D138 definition](definition_first_20260928/d138_block_feedback_candidate_20261003/DEFINITION.md) proves a useful project-level extension of D135: retain nonnegative feedback cycles in the selected amplitude fiber instead of cutting them into a triangular graph. For c>=0, M>=0, rho(M)<1, support of F={e>=0:e<=c+Me} is c^T k*, with k*=(w+M^T k*)_+. All original source, mixed predicates P, original signed bits, zero labels, amplitude identities and input decoder remain. F is convex; the whole retained carrier remains nonconvex. This support is not generally the support of full Gamma.

The proposed implementation class is SCC blocks of size at most two, not a generic iterative optimizer. For normalized M_BB=[[0,a],[b,0]], a,b>=0 and D=1-ab>0:

```text
k1=max(0,w1,(w1+b*w2)/D)
k2=max(0,w2,(w2+a*w1)/D).
```

Coalesce every later consumer into the effective signed direction, solve blocks in reverse order, and propagate M_BA^T k_B. Exact query arithmetic is O(r(n+nnzM)), excluding all paid construction/readout/source costs. Raw self-loops require positive whole-row normalization; the final WHOLE selected M must satisfy the block contract. Do not confuse an upper approximation of k* with an automatically dual-feasible floating certificate.

The ordinary shared-source control uses q1=R(x+y/10+1/4), q2=R(x/2+y/10+1/4), x,y in [-1,1]. Its two valid feedback rows q1<=1/2+q2 and q2<=7/40+q1/2 simultaneously give J1=q1-q2<=1/2 and J2=q2-q1/2<=7/40. Both ReLU(J1-11/20) and ReLU(J2-1/5) are therefore zero. Either fixed triangular cutting direction loses one bound, even with exact coordinate caps. The archived paper also gives two distinct same-source/same-bound old-LP fractional points with positive successors 3/110 and 29/240. They are not a common witness or ADV. HZ with the same two feedback rows matches both improved bounds.

Source relation generation uses R(fi)<=eta R(fj)+max(0,support(fi-eta fj)), eta>=0, on the current state BEFORE installation. It must combine common source coefficients; independent scalar bounds make these relations redundant on crossing gates. A same-prefix disjoint bank pairing can provide two-coordinate blocks with only backward-to-earlier-bank dependencies. Pairing/eta policy, source binding and all costs are not yet frozen or source-qualified. Multiple upper rows do not fit one M: unselected old rows stay in P, and selected-row replacement does not guarantee old cheap-query dominance.

The Bellman/LCP optimization principle is established prior art, not a new invention. Kallenberg's optimal-stopping equations and Toda's contraction theorem are cited in the definition. The prospective contribution must be the useful, compositional nonconvex neural representation/query with paid real-network value. General fixed-step contraction tails were proved only as supplementary scope, not a second implementation or rescue path.

Next: implement a default-off SAME-FRAME block reference, preserve original nonconvex predicates and all shared consumers, and preregister/freeze before any import/AST/compile/numerical test. Keep all 3965 predecessor tests and original timing/resource/memory gates. A standalone matrix optimizer is insufficient. Then bind the rule to the full ordinary Conv/Add frontier from D137 and measure useful caps, full cost and downstream capability against old HZ and HZ with identical relations. Do not repeat circuit containers, standalone metrics, separate scalar adapters or another test framework as the main research result.

All full source, GPU, physical, shadow, per-family, 2413 and separate 400 replay requirements remain. Smooth/Transformer and new-family scope remain active and unqualified. Historical files and all nine tracked edits remain unchanged; new documentation only. Branch redu-hz, HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac, tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. No commit, push or process manipulation occurred. Status, paper review record and checksums are with D138.
