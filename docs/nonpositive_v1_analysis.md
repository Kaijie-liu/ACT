# Saved nonpositive obligations — V1

Read-only diagnosis of the sealed reuse-ON arm, not a new verification run.
Parent archive: `docs/reuse_supervised_v1_execution_results.json` (SHA-256
`2d0c4dff8d150cae700461fbb01ea67158691992767699bf51cf8919bfffb3e0`). Exact per-row values and all inspected raw-file hashes
are in [nonpositive_v1_analysis.json](nonpositive_v1_analysis.json).

## Verdict

All **36/36 saved candidate lower bounds reproduce exactly** under the rational
checker: **6 positive, 30 nonpositive**, no missing obligations, no complete
positive requests. **30/30 nonpositive remain UNRESOLVED_CANDIDATE_VS_LP_RELAXATION.**
There were **zero model calls, zero solver calls and zero new candidates**.
This offline inspection took 40.845 seconds; it is not
charged retrospectively to the frozen experiment or claimed as verifier speed.

Certificates contain only LP identity, inequality/equality duals and the claimed
bound. No primal feasible vector, native objective/status record, bound marginals
or independently checked optimality/LP upper bound was retained. The proposal
code required floating solver success before exporting; that fact is NOT a saved
exact optimality proof. A nonpositive LOWER bound cannot show that the LP optimum
is nonpositive, let alone that the original MoE is unsafe.

## Per-request summary

Each request has exactly one legal pair and nine classification obligations.
The gate is the weight of the first expert in the ordered pair. Router intervals
below are derived from the two saved, previously checked order-support claims;
no sigmoid range tightening was performed.

| Input | Pair | Checked router-difference enclosure | Gate envelope | Positive / nonpositive | Most negative property / competitor | Lower bound |
| --- | --- | --- | --- | --- | --- | --- |
| 220 | [1,2] | [0.173190, 0.367837] | [0.5, 1] | 0 / 9 | p6 / class 6 | -10.555861 |
| 222 | [0,1] | [-0.409156, -0.212625] | [0, 0.5] | 2 / 7 | p7 / class 7 | -5.736547 |
| 230 | [0,3] | [0.216351, 0.411303] | [0.5, 1] | 4 / 5 | p3 / class 3 | -5.029986 |
| 232 | [0,1] | [-0.004075, 0.192680] | [0, 1] | 0 / 9 | p2 / class 2 | -11.822475 |

All 30 difference enclosures cross zero. Their widths and gate envelopes are
observable looseness candidates, **not established causes**: no feasible LP
point or optimality sandwich separates range loss, binary relaxation, McCormick
loss and candidate-dual strength. Input 232 retains [0,1] because neither checked
order lower bound is positive; this does not establish that every gate value in
[0,1] is attainable. No query-order motive follows from this batch: the ON arm
already supplied every required obligation.

## Exact dual accounting (not an error-slack ablation)

For min cᵀz+d with Az≤b, Ez=h, l≤z≤u and y≤0:

`B0 = d + bᵀy + hᵀv; r = c − Aᵀy − Eᵀv;`

`Rbox = Σ min(r_j l_j, r_j u_j); Lchecked = B0 + Rbox.`

The JSON separates base inequalities, the four McCormick rows, equalities,
continuous factors, relaxed binaries, gate and product variables. It also binds
the original coefficient identity and records the five largest residual terms.
The residual box term ranges **−13.729863 to −4.345765** across the 30 rows.
Nearly all this magnitude comes from continuous factors; aggregate remaining
binary/gate/product contributions are at roughly floating-roundoff scale.

**This is not evidence that numeric residual padding destroyed positive bounds.**
Bound-variable multipliers are not saved; reduced costs need not vanish for an
optimum at a box endpoint. Rbox is required exact finite-box dual accounting,
not solely a rounding correction. Although B0 is positive in 22/30 cases,
**B0 alone is not a valid bound**, and dropping Rbox is forbidden. Conversely,
the eight negative B0 values do not rule out better multipliers.

## All 30 nonpositive obligations

Property indices and competitor labels are zero-based. More negative indicates
a weaker recorded bound, not a proved causal bottleneck; **every** nonpositive
row blocks complete acceptance. Display decimals are descriptive only; exact
rational quantities and threshold remain in JSON and unchanged.

| Input | Property / competitor | Difference enclosure | B0 (NOT a bound) | Rbox | Checked lower |
| --- | --- | --- | --- | --- | --- |
| 220 | p0 / 0 | [-6.39136, 15.77468] | 6.50938 | -8.43490 | -1.92552 |
| 220 | p1 / 1 | [-9.61115, 18.61372] | 6.99118 | -10.91672 | -3.92554 |
| 220 | p2 / 2 | [-9.37931, 18.09818] | 2.46791 | -11.05347 | -8.58556 |
| 220 | p3 / 3 | [-6.21558, 18.42740] | 2.51479 | -9.18640 | -6.67161 |
| 220 | p4 / 4 | [-9.20635, 9.02800] | -0.00449 | -7.63493 | -7.63942 |
| 220 | p5 / 5 | [-7.27845, 16.55348] | 1.15838 | -9.05600 | -7.89763 |
| 220 | p6 / 6 | [-10.86134, 21.91770] | 3.17400 | -13.72986 | -10.55586 |
| 220 | p7 / 8 | [-6.28102, 24.52765] | 10.67305 | -11.36607 | -0.69302 |
| 220 | p8 / 9 | [-15.37087, 11.11818] | 4.22829 | -11.51977 | -7.29148 |
| 222 | p1 / 1 | [-5.48658, 10.76127] | 1.98709 | -5.51860 | -3.53151 |
| 222 | p2 / 2 | [-8.08523, 10.70340] | 3.77331 | -6.61194 | -2.83862 |
| 222 | p3 / 3 | [-9.34168, 10.43918] | 5.14444 | -6.55676 | -1.41233 |
| 222 | p4 / 4 | [-7.84545, 14.11063] | 4.67166 | -6.99674 | -2.32508 |
| 222 | p5 / 5 | [-10.80684, 9.38469] | 4.14648 | -6.59328 | -2.44680 |
| 222 | p6 / 6 | [-14.73791, 8.18633] | 6.23817 | -6.85359 | -0.61541 |
| 222 | p7 / 7 | [-4.95219, 15.14394] | 1.01988 | -6.75642 | -5.73655 |
| 230 | p2 / 2 | [-7.17413, 14.54275] | 5.74111 | -7.10661 | -1.36549 |
| 230 | p3 / 3 | [-4.65743, 11.48424] | 0.00754 | -5.03752 | -5.02999 |
| 230 | p4 / 4 | [-9.23854, 15.34542] | 6.71773 | -8.35900 | -1.64126 |
| 230 | p5 / 6 | [-11.84696, 13.62317] | 6.58871 | -9.11620 | -2.52749 |
| 230 | p6 / 7 | [-6.92796, 17.02958] | 5.26807 | -7.61593 | -2.34786 |
| 232 | p0 / 0 | [-11.95315, 15.54001] | -1.07343 | -8.63412 | -9.70755 |
| 232 | p1 / 1 | [-18.51309, 14.24089] | -0.89068 | -9.54260 | -10.43328 |
| 232 | p2 / 2 | [-12.34537, 14.06257] | -2.91389 | -8.90859 | -11.82248 |
| 232 | p3 / 3 | [-6.02552, 8.83828] | -2.14053 | -4.34577 | -6.48629 |
| 232 | p4 / 4 | [-12.47359, 13.24226] | -0.90417 | -8.38438 | -9.28855 |
| 232 | p5 / 6 | [-12.06079, 16.01364] | 3.10554 | -9.19524 | -6.08970 |
| 232 | p6 / 7 | [-14.93570, 11.97305] | -3.37895 | -7.79543 | -11.17437 |
| 232 | p7 / 8 | [-15.26675, 12.15090] | 2.93692 | -8.37115 | -5.43424 |
| 232 | p8 / 9 | [-14.54684, 15.15435] | -2.64295 | -9.04974 | -11.69269 |

## Research decision and evidence still needed

Do **not** change representation, ranges, numerical acceptance, sample size,
budget, training or order based on these data. Keep caching attribution in its
own frozen follow-up, not coupled to precision changes.

A future separately scoped diagnostic could retain a primal candidate and check
its exact feasibility and objective, together with a checked dual lower bound.
A feasible relaxed point with nonpositive objective would show that **that LP**
cannot prove a positive minimum, but not that the network is unsafe. A positive
primal objective alone would not establish safety or LP optimality. Isolating a
specific range or envelope would still require a bounded, separately registered
control. No such diagnostic solving was done here.

Tests include exact decomposition, invalid identity/sign/overclaim rejection,
zero-width product, and two analytic LPs with the same nonpositive candidate
bound but opposite true-optimum signs. Network→HZ, guard lowering and route
exclusions retain their original trust boundary. Sealed artifacts are unchanged.
