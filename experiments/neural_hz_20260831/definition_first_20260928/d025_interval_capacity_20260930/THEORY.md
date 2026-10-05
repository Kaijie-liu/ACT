# Certified phase capacities for ordinary convolution frames

The D024 reference proved a useful three-layer control but rejected stable predecessors, one-sided differences and interval-certified BatchNorm coefficients. Those are ordinary CNN cases, not numerical exceptions. This extension admits them by one uniform rule while retaining the entire nonconvex HZ carrier. It is still a known-relation component under evaluation, not a completed Neural HZ definition or a new convexification principle.

## Any finite predecessor and difference bounds

Let r_i=ReLU(f_i) and beta_i be its original binary phase, including independent legal choices when f_i=0. Suppose certified finite bounds give f_i<=u_i and L_ij<=f_i-f_j<=U_ij on the same source relation. Put c_i=max(u_i,0), D_ij=min(max(U_ij,0),c_i), and E_ij=min(max(-L_ij,0),c_j). Then

```text
-E_ij beta_j <= r_i-r_j <= D_ij beta_i.
```

When beta_i=0, r_i-r_j=-r_j<=0. When beta_i=1, its positive part is at most both max(f_i-f_j,0) and r_i=f_i, hence at most D_ij. Exchange i,j for the lower bound. The argument covers strictly active, strictly inactive, exactly zero, crossing and one-sided-difference cases without selecting another execution path.

Clipping is only for these new nonnegative capacities. Do not replace original signed gate or difference predicates with their clipped versions: those old predicates can be stronger and can constrain the original bits. Equal values at zero do not imply equal phase identities.

## Interval-certified fixed coefficients

The actual receiver coefficients are fixed by the original network, but a rigorous numeric certificate can provide intervals w_i in [a_i,b_i] and c in [c_lo,c_hi]. This includes outward enclosures of BatchNorm square roots; intervals describe uncertainty in fixed coefficients, not independent new network latent variables.

For every fixed consecutive canonical pair i,j, define both oriented guaranteed matched amounts

```text
lambda_ij = min(max(a_i,0), max(-b_j,0))
lambda_ji = min(max(a_j,0), max(-b_i,0)).
```

At most one is positive. Each amount is no larger than the corresponding actual positive and negative coefficient magnitudes. Start with positive magnitude bounds max(b_i,0) and negative magnitude bounds max(-a_i,0); subtract the respective guaranteed matched amount and multiply unmatched magnitudes by c_i. Replace each matched positive scalar capacity by lambda_ij D_ij and its negative capacity by lambda_ij E_ij. Call the resulting nonnegative constants K_i and H_i.

For every actual coefficient assignment consistent with the certificates and the original source,

```text
g = c + sum_i w_i r_i
ReLU(g)  <= max(c_hi,0)  + sum_i K_i beta_i
ReLU(-g) <= max(-c_lo,0) + sum_i H_i beta_i.
```

Proof: decompose the actual weighted sum into the guaranteed paired differences and the actual remaining signed magnitudes. Their positive and negative parts are bounded by the unpaired interval endpoint magnitudes minus the corresponding matched amounts. Apply the preceding difference inequalities, drop nonpositive contributions and use nonnegative upper bounds through ReLU. Coefficient correlations do not invalidate a universal enclosure argument. Ambiguous weight signs simply yield zero guaranteed matching; both residual signs remain represented. No midpoint coefficient substitution is allowed.

When intervals are points and all D024 premises hold, the coefficients agree with D024. For every relaxed beta in [0,1], the paired coefficient K_i or H_i is no larger than its unpaired counterpart from the same certified intervals. This coefficient-wise comparison is not a theorem that a final network property will be solved.

## Sparse common source and convolution semantics

Each preactivation certificate keeps its affine constant interval and a sparse map from ORIGINAL input coordinate IDs to coefficient intervals. Add maps on matching IDs before taking the source-box bound of f_i-f_j. An interval coefficient is an enclosure of a fixed original coefficient, not a replacement input variable. The full source predicates and parameter provenance remain unchanged.

Convolution slots use canonical NCHW channel/kernel order, including padding positions. A padding slot is identically zero after the relevant incoming affine or activation operation; it has no new original phase. Keep its place in the pair ordering, rather than removing it and changing all subsequent pairs. Real activation slots retain (original ReLU output port, channel, row, column) identities. A shared cache may reuse a certificate only for the same source/frame and source key.

First-layer input preprocessing must be applied only at valid input pixels: ONNX Conv padding is zero AFTER preprocessing. Post-Conv channel affine and BatchNorm operations retain their original order. Their parameters are read from original bytes; BatchNorm uses a proved outward square-root enclosure. Dynamic Add/Sub consumers and shortcut branches remain recorded and unchanged; this initial first-bank study does not claim to interpret through residual merges.

## Cost and study scope

For k canonical slots, there are at most floor(k/2) pair certificates. A sparse difference touches at most the union of its two actual source supports; no full-input dense row or all-positive/all-negative Cartesian pairing is needed. Per receiver, two capacity inequalities contain at most 2k+3 nonzeros when coefficient signs are uncertain, and at most k+3 when they are certified. Original gates, all bits, decoder, source predicates, coefficient certificates and solver storage remain payable.

For one source-model census, compute each channel's BN enclosure once, each used source form/bound once, each distinct canonical source pair bound once, and every receiving coefficient and its contribution. Count parsing, pair construction, interval work, stored provenance, evidence serialization and terminal-row support. A smaller front-end representation alone is not a gain.

The intended actual-network check uses the unchanged three source models, one fixed original property each, five original spatial anchors and every output channel of every admitted direct next-ReLU branch. It measures applicability and coefficient reductions versus the same unpaired capacities, not dataset prevalence, final CERT/ADV, real signs inside a merely crossing outer interval, or an actual speedup. No model/spec is selected by prior solve status, label, margin or solver result.

This mathematical extension is independent of the CPU reference used to validate it. GPU propagation and terminal work remain required by the goal; a CPU census is not a substitute GPU candidate. Matvec and transpose descriptions alone cannot qualify an all-GPU implementation.

## Provenance and invariants

2026-09-30, redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. D024 is frozen and remains unchanged. Formal 1870/2413 and independent 61/400 remain unchanged. All new source and outputs remain isolated. Preserve continuous factors, every original bit and zero choice, EQ/LE, shared latent/frame identity, reconstruction and fail-closed semantics. No attack, BaB, input/phase split, backward/dual rescue or runtime identity menu is introduced.
