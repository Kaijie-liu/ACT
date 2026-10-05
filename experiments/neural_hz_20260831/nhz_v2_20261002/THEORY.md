# Neural-HZ v2: projection-aligned Hybrid Zonotopes with a concretization tower

Status: definitions and paper proofs, reviewed by the author only (not machine
checked). Implementation: `nhz_engine.py` (isolated, default-off, not imported
by ACT). Novelty is NOT established; related work is listed in Section 6.

## 1. Motivation from measurements

The exact HZ ReLU graph admits many parametrisations of the same set. The
current ACT encoding (`tf_mlp.sparse_hz_apply_relu_exact`) writes an unstable
output as `u/2 - (u/2) xi`, so the input/output coupling exists only in an
equality row. Any query that ignores predicates (fast bounds, phase
classification, GPU tensor propagation) then sees every unstable ReLU as an
independent box `[0,u]`. On the E0 CIFAR100/TinyImageNet UNKNOWN rows this
costs a factor 6-8 in robustness-margin bounds (LOG N001). The point of the
definition below is that the parametrisation is part of the abstract element:
one object should serve the exact query, the LP query and the predicate-free
query, each at its natural cost, with transformers that commute with all three
projections.

## 2. Domain elements

**Definition 1 (projection-aligned Neural-HZ state).** A state is a tuple
`S = (K, c, G, A, b, P)` with

- continuous latent factors `w in [-1,1]^K`;
- an affine value map `v(w) = c + G^T w` in `R^n` (no binary generators in the
  value map);
- convex predicate rows `A w <= b` (equalities are written as two rows);
- a finite phase list `P`; each phase `j` carries affine forms `x_j(w)`,
  `y_j(w)`, valid bounds `l_j < 0 < u_j` and a binary `d_j in {0,1}` with the
  two disjunctive rows
  `M_j: y_j(w) <= u_j d_j,  y_j(w) <= x_j(w) - l_j (1 - d_j)`.

Three concretisations are attached to the same element:

- exact: `gamma(S) = { v(w) : w in box, A w <= b, exists d: M_j(w, d_j) for all j }`;
- LP: `lambda(S) = { v(w) : w in box, A w <= b }`;
- shadow: `sigma(S) = { v(w) : w in box }`.

Obviously `gamma(S) subset lambda(S) subset sigma(S)` (Lemma 2 below shows that
the relaxed `M_j` rows are implied, so `lambda` is the LP relaxation of
`gamma`). Binary phases are retained in `gamma`; nothing is pivoted or relaxed
away in the exact semantics.

Embedding of an ordinary HZ. An ordinary HZ `(c, Gc, Gb, Ac, Ab, b)` with
binaries in the value map or predicates embeds by treating each binary
generator `beta in {-1,1}` as `beta = 2 d - 1` with `d` a phase-list binary
whose disjunctive rows are `beta <= 2d - 1 <= beta`; all predicates become rows.
Conversely, a Neural-HZ state is an ordinary HZ with continuous factors `w`,
binaries `d`, inequality predicates `A w <= b` and `M_j`. Hence the two
representations denote the same family of sets (finite unions of polytope
images); the contribution is the parametrisation discipline and the tower, not
new expressivity.

## 3. Transformers

**Affine / Conv / BN / Gemm / MatMul-with-constant / Add / Sub / Concat.**
`v -> W v + t` maps `(c, G) -> (W c + t, G W^T)` and leaves rows and phases
unchanged. Add/Sub of two states over the same factor space adds value maps;
shared factor ids are what make residual joins exact (latent coherence).

**Projection-aligned ReLU.** Let coordinate `i` of the value map be
`x(w) = c_i + g_i^T w` with bounds `l <= x <= u` valid on `gamma(S)`.

- `u <= 0`: `y = 0`; `l >= 0`: `y = x`.
- `l < 0 < u`: allocate one fresh factor `eta` and one binary `d`, and set
  `lam = u/(u - l)`, `mu = -lam l / 2`,

      y(w, eta) = lam x(w) + mu (1 + eta),
      rows:   -y <= 0,   x - y <= 0,
      phase:  M: y <= u d,  y <= x - l (1 - d).

**Lemma 1 (exactness).** If `[l,u]` bounds `x` on `gamma(S)`, then
`gamma(ReLU#(S)) = ReLU(gamma(S))` (coordinatewise on the transformed
coordinates, all other coordinates unchanged).

Proof. (supset) Take a feasible `(w, d)` of `S` and put `y* = ReLU(x(w))`.
Because `l <= x <= u`, `y* - lam x in [0, 2mu]` (it equals `-lam x` for
`x <= 0` and `(1-lam) x` for `x >= 0`, both at most `-lam l = 2mu`), so
`eta = (y* - lam x)/mu - 1 in [-1,1]`. The rows `y* >= 0`, `y* >= x` hold and
`d = [x >= 0]` satisfies `M`. (subset) If `d = 1`, `M` gives `y <= x`, the row
gives `y >= x`, so `y = x`, and `-y <= 0` gives `x >= 0`. If `d = 0`, `M` gives
`y <= 0`, so `y = 0`, and `x - y <= 0` gives `x <= 0`. Either way `y = ReLU(x)`.
Rows of earlier phases are untouched. QED.

**Lemma 2 (LP relaxation and row economy).** With `d` relaxed to `[0,1]` the
rows `M` are implied by `-y <= 0`, `x - y <= 0` and `eta <= 1`. Consequently
`lambda(ReLU#(S))` equals the triangle (Planet) relaxation of `lambda(S)`, and
the LP needs only two rows per unstable ReLU.

Proof. `eta <= 1` gives `y <= lam (x - l)`. A `d in [0,1]` satisfying
`y <= u d` and `y <= x - l + l d` exists iff `y/u <= 1 + (x - y)/(-l)`, which
rearranges to `y (u - l) <= u (x - l)`, i.e. `y <= lam (x - l)`. The two
endpoint conditions `y/u <= 1` and `1 + (x - y)/(-l) >= 0` follow from the
same inequality and `lam <= 1`. The feasible `(x, y)` set is therefore the
triangle `{max(0, x) <= y <= lam (x - l)}`. QED.

**Lemma 3 (shadow commutes with DeepZ).** `sigma(ReLU#(S))` is the DeepZ ReLU
image of `sigma(S)` for the same `[l,u]`; affine transformers commute with
`sigma` exactly. Hence for a ReLU network the shadow after `L` layers equals
DeepZ with the bounds used by the transformers.

**Theorem 1 (concretisation tower).** For networks of the operators above,
with bounds valid on `gamma` at every ReLU,

- `gamma(F#(S0)) = F(gamma(S0))` (exact),
- `lambda(F#(S0))` is the layerwise triangle LP relaxation with the same
  bounds, and
- `sigma(F#(S0))` is DeepZ with the same bounds.

Each projection can therefore be queried on the same object: closed form on
GPU (`sigma`), LP (`lambda`), MILP (`gamma`).

Proof: induction on the operator sequence with Lemmas 1-3. QED.

**Corollary (cost per unstable ReLU).** 1 continuous factor, 1 binary, 2 LP
rows, 2 MILP rows. For comparison: current ACT extended exact graph 2
continuous, 1 binary, 1 equality + 2 inequalities; ACT compact quotient
1 continuous, 1 binary, 3 inequalities but shadow `[0,u]`; Bird/Ortiz exact HZ
4 continuous, 1 binary.

**Monotone use of tighter bounds.** Bounds obtained from `lambda` queries are
valid on `gamma` (since `gamma subset lambda`), so they may be fed back into
later transformers without affecting exactness (Lemma 1 only needs validity).

## 4. LP queries on GPU (solver level, not a domain claim)

For an affine objective `f(w) = c + g^T w`, weak duality gives, for every
`nu >= 0`,

    max { f(w) : A w <= b, w in [-1,1]^K }  <=  c + nu^T b + || g - A^T nu ||_1.

`nu = 0` returns the shadow bound. The implementation batches all objectives of
a layer and improves `nu` by projected Adam; every evaluated `nu` is a valid
bound, so early stopping never affects soundness (only tightness). This is the
same LP query the frozen baseline's `tight_bounds=True` solves with CPU LPs;
it is reported as GPU acceleration of an existing component, never as a
Neural-HZ representation gain and never used to select a representation.

Floating point: the current probe evaluates bounds in float64 without
directed rounding. Certificates require the rigorous re-evaluation of
Section 5 (pending).

## 5. Rounding (pending, required before any CERT is counted)

Plan: carry a per-coordinate rounding radius `e` with the value map
(`|true value - (c + G^T w)| <= e` on the box), propagated by
`e' = |W| e + gamma_n |W| (|c| + sum_k |G_k|)` for affine maps with
`gamma_n = n u / (1 - n u)`, `u = 2^-53`, valid for any summation order. Rows
are relaxed by the corresponding radius, objectives enlarged by it, and the
final dual bound is re-evaluated with an explicit summation-error pad.

## 6. Related work that limits novelty claims

DeepZ (Singh et al. 2018) is the shadow. The triangle/Planet relaxation
(Ehlers 2017) is `lambda`. Big-M MILP encodings (Tjeng et al. 2019) and exact
HZ ReLU graphs (Ortiz et al. 2023, Bird et al.) give `gamma`. ImageStar /
Star sets (Tran et al.) keep an affine value map plus predicates, with the
triangle in the predicate, i.e. a `lambda`-level object without the binary
level. PyRAT combines several domains. The contribution claimed here is
narrower: one parametrisation discipline making the three classical levels
simultaneous projections of one HZ object with commuting transformers and a
2-row LP. Whether this is publishable novelty is an open question.

## 7. Selective exactness: a lattice between lambda and gamma

**Definition 2.** For a set `L` of phases, `gamma_L(S)` keeps the binary rows
`M_j` with integer `d_j` for `j in L` and relaxes `d_j in [0,1]` for `j not in L`
(where, by Lemma 2, the relaxed rows are implied).

**Lemma 4.** `gamma_emptyset = lambda`, `gamma_all = gamma`, and
`L1 subset L2 => gamma_L2 subset gamma_L1`. Every `gamma_L` contains `gamma`, so
an upper bound of a violation over `gamma_L` is sound for the exact state.

Proof: immediate from Lemma 2 (relaxed rows are implied) and from the fact that
integrality restrictions only remove points. QED.

The terminal query can therefore be answered on a chain
`lambda = gamma_0 supset gamma_1 supset ... supset gamma` where `gamma_k` keeps
the phases of the last `k` ReLU layers (a structural rule: layer position only,
never instance identity, margin or LP state). This is a single mixed-integer
query over a relaxation of one state; the input domain is never partitioned and
no phase is case-split outside the terminal solver. All binaries remain in the
representation; `gamma_L` is a query-time projection, not a representation
change.

Why the suffix matters (measured, LOG N009): on E0 CIFAR row 0 the worst
violation bound drops from 0.595 (exact LP, `gamma_0`) to 0.343 (`gamma_1`, 29
binaries). In the terminal objective `b - a^T W2 ReLU(z) - ...`, ReLUs with a
positive coefficient enter as convex terms; maximising a convex term is exactly
where the triangle relaxation pays its full vertical slack, while negative
coefficients are already tight in the LP.

## 8. Attention in the tower (float probe `nhz_attn.py`; rigorous version pending)

Bilinear products and Softmax are not piecewise affine, so `gamma` can only be
a sound over-approximation there; binaries stay where they are exact (ReLU in
the MLP blocks).

**Bilinear `z = x y`** with `x = cx + gx^T w`, `y = cy + gy^T w`:
`z = cx cy + cx gy^T w + cy gx^T w + q` with
`q = sum_k gx_k gy_k w_k^2 + sum_{k != l} gx_k gy_l w_k w_l`.
Since `w_k^2 in [0,1]`, `q in [m - r, m + r]` with `m = 1/2 sum_k gx_k gy_k`
and `r = |gx|_1 |gy|_1 - 1/2 sum_k |gx_k gy_k|`. The shadow replaces `q` by
`m + r eta` (fresh `eta`). For a matrix product the statistics are contracted
over the inner index (one fresh factor per output entry). This is the DeepT
style product; it is sound but loses the joint quadratic structure.

**Softmax.** With score differences `d_ij = s_j - s_i` (affine forms; shared
factors cancel exactly in the difference),
`p_i = 1/(1 + sum_{j != i} e^{d_ij})`. Shadow: tangent plane at the centre
`d0` plus a remainder `|R_i| <= 1/2 delta^T H_bar delta`, where `delta` is the
half-width of the box of `d` and `H_bar` bounds the Hessian entrywise over
that box (`|H_jk| <= 2 e_j e_k / T^3` off the diagonal and
`max(2 e_j^2/T^3, e_j/T^2)` on it, `T = 1 + sum e`, evaluated at the box
extremes). LP rows: `p_i in [1/(1 + sum e^{dhi}), 1/(1 + sum e^{dlo})]` and
`sum_i p_i = 1`; the latter holds exactly for the true probabilities, so it is
a valid equality on the represented forms.

Measured (LOG N018-N020): on the IBP ViT the attention terms contribute about
a fifth of the terminal relaxation slack; ReLU triangles dominate. On the PGD
ViT the product radii explode and no bound is useful at this input radius.

## 9. Sign-aware selective exactness on the last ReLU layer

**Proposition 1.** Let `L` be the last ReLU layer with phases, and let the
violation objective be `v(w) = cc + g^T w`. For a unit `k in L` with fresh
factor `eta_k`, suppose `eta_k` occurs only in `y_k`'s value map, its two LP
rows and its two binary rows (true for the last ReLU layer: no later ReLU
consumes it). If `g[eta_k] <= 0`, then dropping unit `k`'s binary rows does not
change `max v` over `gamma_L`.

Proof. Fix all variables except `eta_k`. The objective is non-increasing in
`eta_k`, and decreasing `eta_k` lowers `y_k = lam x_k + mu (1 + eta_k)`. The
only constraints that bound `eta_k` from below are the box `eta_k >= -1`
(i.e. `y_k >= lam x_k`) and the LP rows `y_k >= 0`, `y_k >= x_k`; together
they put `y_k`'s minimum at `max(0, x_k, lam x_k) = max(0, x_k) = ReLU(x_k)`
(for `x_k in [l_k, u_k]`, `lam x_k <= max(0, x_k)`). Hence an optimal solution
of the relaxation can take `y_k = ReLU(x_k)`, which satisfies the binary rows
with `d_k = [x_k >= 0]`. QED.

The selection depends only on the signs of the specification's objective
coefficients (`g` is the objective in latent coordinates), never on LP state,
margins or instance identity. Measured (LOG N024): on CIFAR E0 rows 76, 105,
113, 120 the sign-aware plan keeps 17/32, 5/12, 11/24, 17/32 binaries and
returns the same exclusion verdicts; row 105 has the same optimum (0.0265)
under both plans. HiGHS wall time did not improve consistently (68 vs 33 s,
50 vs 103 s, 10.6 vs 11.3 s, 6.1 vs 6.7 s), so it is not adopted as a speed
claim. Such sign arguments are standard in MILP verification folklore; the
statement here is for the record, not a novelty claim.

## 10. Rounding semantics actually implemented

`nhz_sound.py` (n004, float64) implements Section 5. `nhz_sound_mp.py` (n006)
and `nhz_sound_mp2.py` (n007) store generators in float32 and compute in
float64; for every operator the exact storage rounding
`sum_k |G64_k - float64(float32(G64_k))|` (exact subtraction by Sterbenz,
summation padded by `gamma`) is added to the radius, and eta/radius generators
are rounded up. Invariant INV of `nhz_sound.py` is preserved by each update.
n007 adds Sigmoid/Tanh: slope `lam = min(f'(l), f'(u)) (1 - 1e-9)` (strictly
below the minimum slope on `[l,u]` because `f'` is unimodal), range of
`f(x) - lam x` from the endpoints with a 1e-12 relative evaluation pad, and
LP lines shifted to validity by a 64-point grid maximum plus the Lipschitz
remainder `(max f' + |a|) h / 2`; every line row is relaxed by
`e_y + |a| e_x`. The terminal MILP is solved by HiGHS and accepted at margin
1e-4 below zero, the same solver standard as the frozen baseline.

Witness semantics: ADV is accepted under S1 (a real point inside the box,
ONNX Runtime evaluated on its float32 rounding, exact output check), which is
the baseline's `_is_cex` gate; S2 (the float32 vector itself satisfies every
input assertion) is reported separately (LOG N022).

## 11. Position against prior work (what is and is not new)

| component | closest prior work | status here |
|---|---|---|
| sigma level (constraint-free shadow) | DeepZ (Singh et al. 2018) | known; the point is that the HZ parametrisation makes it a projection of the exact state |
| lambda level (triangle LP) | Planet (Ehlers 2017), LP-all relaxations; NNV approx-star (Tran et al.) | known; only the 2-row form and its implied-row lemma are specific to the aligned encoding |
| gamma level (exact MILP) | MIPVerify (Tjeng et al. 2019); exact HZ ReLU graphs (Bird et al., Ortiz et al. 2023); PyRAT HZ | known |
| one state, three commuting projections | PyRAT combines several domains without a single latent system; VMCAI HyZor keeps one latent system but its ReLU encoding has a box shadow | the claimed contribution of this session (Theorem 1) |
| GPU batched weak-duality LP | alpha-CROWN optimises LP duals in backward form; PDLP-type GPU LP solvers | solver-level acceleration of the baseline's existing LP-tight query; not a domain claim |
| selective exactness lattice, last-layer MILP | partial MILP formulations and "convex barrier" analyses (Salman et al. 2019, Tjandraatmadja et al. 2020) | structural rule + measured effect on CIFAR/Tiny; modest novelty |
| sign-aware exactness | folklore of MILP verification | recorded with proof; not claimed as new |
| rigorous mixed-precision shadow | sound floating-point abstract interpretation (DeepZ, DeepPoly floating-point soundness) | the exact storage-rounding accounting is a practical design choice |
| attention transformers | DeepT (Bonaert et al. 2021), CROWN for transformers | float probe only; DeepT-style |
| smooth lambda level and segment unions | CROWN/DeepPoly sigmoid bounds; HyZor configurable selective K-seg | not yet shown to beat configurable selective; 55/63 dist_shift baseline CERTs with zero sigmoid binaries |

Novelty is therefore narrow and must be argued by the combination (one
GPU-resident object serving three levels, rigorous, with measured capability on
CIFAR100/TinyImageNet where the previous forward HZ had zero certificates),
not by any single transformer.

## 12. Query-level exactness: conjunctive sign rule and latent-box ideal cuts

Both statements below concern the terminal query `max v` over `gamma`, not the
domain element. The element keeps every factor, every binary phase, every
predicate and the input frame; only the MILP that answers one query is
smaller (12.1) or has more valid rows (12.2).

### 12.1 Proposition 1' (sign rule for epigraph objectives)

Setting as Proposition 1, with the epigraph objective of Section 3 of
`nhz_terminal_v3.py`: maximise `g^T [w; tau]` subject to the plan rows and
the atom rows `a_k^T [w; tau] <= b_k`. For a unit `j` of the last ReLU layer
with phases, suppose `g[eta_j] <= 0` and `a_k[eta_j] >= 0` for every atom row.
Then the plan without unit `j`'s binary rows has the same optimum and the same
feasibility under any objective cutoff.

Proof. The plan without the rows is a relaxation, so its optimum is at least
the original. Conversely, take any feasible point of the relaxation and lower
`eta_j` to the smallest value allowed by `eta_j >= -1` and unit `j`'s two LP
rows. That value puts `y_j` at `ReLU(x_j)` (Proposition 1). The objective does
not decrease because `g[eta_j] <= 0`. No atom row is violated because its
left side does not increase. No other row contains `eta_j`. The new point
satisfies `j`'s binary rows with `d_j = [x_j >= 0]`. QED.

Dropping the rows is sound unconditionally, since it only relaxes. The
proposition says it costs no precision. Implemented as `nhz_terminal_v5.py`.

Note on hard limit 2 ("do not delete binary factors or relax them
continuously"). The binary of unit `j` stays in the element. The query MILP
omits its integrality only where Proposition 1' proves the optimum is
unchanged. This is the same reasoning as dual fixing in MILP presolve. Whether
it is admissible under limit 2 is for the user to decide; every run that uses
it says so.

### 12.2 Lemma 5 (ideal unit cuts over the latent box)

Every pre-activation is affine in the latent vector, `x_k = c_k + a_k^T w`, and
`w` lies in `[-1, 1]^K`. For any subset `I` of latent coordinates,

    y_k <= sum_{i in I} (a_i w_i + |a_i| (1 - d_k)) + (c_k + sum_{i not in I} |a_i|) d_k

holds at every exact point `(w, ReLU(x_k), d_k = [x_k >= 0])` with `w` in the
box. This is the ideal formulation of one ReLU over a box (Anderson et al.,
2020) written in latent coordinates. `I` empty gives `y <= u_box d`, and `I`
full gives `y <= x - l_box (1 - d)`.

Rounding: adding `2 e_x + e_y` to the right-hand side covers the radii of `x`
and `y`. The perturbation of `x` acts as one more input with box
`[-e_x, e_x]`.

What is specific to the aligned HZ: the cut is available for every unit of
every layer. The earlier `eta` factors are latent box coordinates like the
inputs, so no per-layer input box has to be recomputed. What is not new: the
cut family and its linear-time separation are Anderson et al.'s.

### 12.3 Proposition 2 (monotone units on any layer)

Write every pre-activation of an unstable unit in unit coordinates,
`x_t = const + P_t eps + sum_{s<t} J_ts y_s`, by eliminating
`eta_s = (y_s - lam_s x_s)/mu_s - 1` layer by layer. Stable units are already
folded into the affine maps. Apply the same elimination to the query
functionals. An interval reverse pass with slopes in `[0, 1]` for unstable
units gives an enclosure `[lo_s, hi_s]` of `d f / d y_s`. The enclosure is valid
at every point whose upstream values lie in the lambda concretisation and
whose downstream layers are exact.

Claim: in a plan whose layers from `t` onward are all gamma, unit `s` (layer
`>= t`) can skip integrality without changing the optimum if `hi_s <= 0` for
the maximised objective and `lo_s >= 0` for every `<=` atom row.

Proof by backward induction over layers. Take an optimal point of the
relaxed plan. Process the last layer first, which is Proposition 1'. At
layer `s`, all later layers are already exact, so every functional is a
function of layer `s`'s outputs. Lowering a selected unit to `ReLU(x)` moves
along a segment that stays inside the region where the enclosure holds. On
that segment the objective does not decrease and no atom row increases. Then
recompute the later layers exactly. The result is an exact point that is at
least as good. QED.

Implementation: `nhz_monotone.py`. The test `test_n047_monotone.py` shows
the rewrite matches exact forward differences to 5e-14, and 1,950
finite-difference derivatives all lie in the enclosure. With plain interval
slopes the rule selects only last-layer units on ACAS Xu, so in practice it
adds nothing beyond Proposition 1'. Soundness never depends on it.

### 12.4 Disaggregated encoding of the gamma units (solver level)

Terminal v7 (`nhz_terminal_v7.py`) gives every gamma unit an explicit
pre-activation column `x_k` with one equality row. A binary unit uses the
disaggregated rows `x = x0 + x1`, `y = x1`, `x0 >= l(1-d)`, `x1 <= u d`. A
non-binary unit keeps its two LP rows. All rows carry the radii plus
`delta_k`, which bounds the float32 storage gap between the stored value
map and the aligned formula. The projection onto `(w, d)` matches the big-M
plan, so the LP strength is the same. HiGHS nevertheless needs half the
nodes (LOG N051). That is a solver observation, not a domain claim. It also
explains why the baseline's HZ encoding, which is disaggregated, was faster
per binary than the earlier dense big-M plans.

### 12.5 Witness rounding toward the interior

A decoded real point `x64` in the box (often a box corner) is rounded to
float32 toward the box interior, with one extra float32 step inward on every
coordinate whose box is wider than one float32 ulp. This closes the gap
between the float64 parse and the exact decimal bound, which is far smaller
than one float32 ulp. The resulting float32 point satisfies the input
assertions exactly (S2). If its output still meets the unsafe assertions, it
is a strict counterexample. This is fixed-point rounding, not a search step.
On all 81 Cora witnesses of N039 v3 it turns S1-only witnesses into S2-valid
ones (LOG N052). It cannot help when `lb = ub` is not a float32 value
(MetaRoom).

## 13. Sign/curvature exactness for smooth units (statement; not applicable on dist_shift)

Let `f` be sigmoid or tanh, and let a smooth unit `y = f(x)` have downstream
layers that are all exact. Suppose the derivative enclosure of the objective in
`y` (Section 12.3, with the smooth unit's output as the variable) satisfies
`sup <= 0`, and the unit lies in the convex region (`u <= 0`). Then lowering
`y` to `f(x)` never hurts. The constraint `y >= f(x)` is convex there, and
the tangent family describes it exactly. So the lambda level is exact for
the query and the unit needs no phase. The same holds symmetrically for
`inf >= 0` in the concave region (`l >= 0`). In the mismatched cases only one
side needs segments. The proof is the argument of Proposition 2 with `f` in
place of ReLU.

Measured (LOG N058): on dist_shift no sigmoid unit has a determined sign, so
the statement has no effect on that family.

## 14. Smooth units in the MILP plans (terminal v8)

For a smooth unit `y = f(x)`, the engine's shadow gives the value map
`y_hat = lam x_hat + c0 + nu eta`, with `c0 = yc - lam cx`. The true values
satisfy `|x - x_hat| <= ex` and `|y - y_hat| <= ye + delta`, where
`delta = 2^-24 lam sum|gx|` bounds the float32 storage of `lam * gx` under
round-to-nearest. Each rigorous line `(a, b)` that is valid on `[l, u]` gives a
row in `(x_hat, eta)`. For an upper line:

    (lam - a) x_hat + nu eta <= b - c0 + ye + delta + |a| ex

The lower lines are symmetric. With an explicit `x_hat` column (one equality
row, dense in `w`), each line row has two nonzeros.

Phase segments: split `[l, u]` at the breakpoints `b_0 < ... < b_S`, including
`0` when `l < 0 < u`. Introduce `x_k, y_k, d_k` with `sum_k d_k = 1`,
`b_{k-1} d_k <= x_k <= b_k d_k`, `sum x_k = x +- ex`, and
`sum y_k = y_hat +- (ye + delta)`. Add the perspective of each segment's
rigorous lines: `y_k <= a x_k + b d_k` and `y_k >= a x_k + b d_k`. Any true
`(x, f(x))` with `x` in segment `k*` satisfies all rows, with `d_{k*} = 1` and
the other pieces zero. So the plan contains the exact graph and every plan is
a relaxation. By Balas's theorem, the LP relaxation of the segment block is
the convex hull of the union of the segment hulls. Test: LOG N066.

## 15. What is claimed, what is not (status 2026-10-03)

Each item is graded as a proved statement, an engineering finding, or known
prior work.

| item | status | evidence |
|---|---|---|
| Projection-aligned exact ReLU with three concretisations and commuting transformers (Sections 2-3, Theorem 1) | proved; the alignment itself is our formulation | THEORY 2-3, LOG N001 |
| Selective-exactness lattice gamma_L (Section 7) | proved; a lattice view of known selective MILP encodings | LOG N009 |
| Proposition 1 / 1' (sign rule, conjunctive extension) | proved; folklore for single objectives, extension to epigraph objectives is ours | THEORY 9, 12.1; LOG N044 |
| Proposition 2 (monotone units, any layer) | proved; no practical gain with interval slopes | THEORY 12.3; LOG N047 |
| Lemma 5 (ideal unit cuts in latent coordinates) | proved; the cut family is Anderson et al. 2020 | THEORY 12.2; LOG N045, N048 (no gain) |
| Disaggregated encoding of gamma units | engineering finding (same LP strength, about half the B&B nodes) | THEORY 12.4; LOG N051 |
| Smooth units written into MILP plans; phase segments with Balas rows | soundness argument plus validity test; segment hulls are standard | THEORY 14; LOG N065, N066 |
| Sign/curvature exactness for smooth units | proved; not applicable on dist_shift | THEORY 13; LOG N058 |
| Softmax remainder rule (n011.2), difference-aware score bounds (n014) | sound by construction; score differences match the baseline's idea | LOG N054, N064 |
| Fused attention-mix, simplex knapsack cross term (n012, n013, n015) | sound; negative or mixed | LOG N056, N060, N064 |
| Inward float32 witness rounding and bound snapping | numeric clean-up of a MILP incumbent, not a search | THEORY 12.5; LOG N052, N057 |
| LP box maximiser, input-box centre, centre-seeded MIP start | **withdrawn**: attack or sampling helpers, not allowed as verdict sources (user rule, LOG N099) | removed in path v7.0 |
| Row-coupled softmax element: LP-tightened differences to the argmax token (Lemma 6) and ratio rows (Lemma 7) | proved sound; measured on PGD ViT (row 15 LP 0.30 -> -0.33; 62, 78 CERT); IBP unchanged | THEORY 17; LOG N116 |
| Numerical safety width on every unit row (NUM_TOL = 1e-5) | engineering finding after a false CERT from a spurious solver infeasibility; a relaxation | LOG N111-N114 |

Formal and E0 scores are unchanged until a single-path replay of the full
universe retains every baseline solve. As of 2026-10-04 the re-verified
domain-only replay (N039 v14 + N113) retains 1,746 of 1,870 with 38 gains and
0 conflicts; ViT (48 CERT), cold-start ADV incumbents (47 rows, pending the
user's decision on warm starts) and linearizenn (10) carry most of the gap.

## 16. Open research front: attention (status 2026-10-03)

Measured (LOG N063, N077): on the IBP ViT the engine's width over the sampled width is about
2.0 on affine nodes (the sampling deficit), 2.6 at the softmax and 1.75 at the attention
product, in every block; the residual adds return to about 2.3. The terminal LP is converged
(N062), so the loss is in the relaxation, not the solver. The baseline's relaxation keeps the
attention output affine in the Q, K, V generators with one error generator per output and
re-links the softmax Taylor term to Q and K (agent reading of ACT HybridZ); our composite keeps
a fresh factor per softmax coordinate and per product coordinate, which the LP cannot
correlate across coordinates.

Closed negatives: fused mix with span or weighted-variance remainder (n012), simplex-aware
cross term (n013), re-linked mix chosen by radius (n015), remainder in the value map (n018).

Candidate directions that stay inside the domain (none implemented):
1. Row-coupled softmax element: keep the per-row sum-to-one and ratio rows (they are in the
   state) but add coupling rows between the softmax fresh factors and the score generators,
   derived from the exact monotonicity of softmax in score differences (a row-wise lambda level
   for the softmax, like the triangle for ReLU), so the LP can trade softmax slack against
   score slack.
2. Exact phases for attention: a binary per (row, argmax token) with Balas rows for the regime
   "token j dominates", giving a gamma level for attention and a selective-exactness stage on
   the attention rows (analogous to stage D).
3. Score bounds through the LP: the score-difference radii (N077) are 1.4x wider than the
   zonotope ideal; the GPU LP tightening is applied to ReLU pre-activations only and could be
   applied to the score differences before the softmax enclosure.

## 17. Row-coupled softmax element (engine n021; LOG N116)

Setting. A softmax row has scores `s_1..s_n`, each an affine value map `s_j = c_j + g_j^T w`
with error `e_j`, over the latent set `W = box ∩ rows` of the element. Let `i*` be the token
with the largest centre score (an observable structural choice: it depends only on the element).

**Lemma 6 (LP-tightened differences).** For every true latent point `w ∈ W` and every `j`,
`s_j − s_{i*} ∈ [L_j, U_j]` where `L_j, U_j` are rigorous LP bounds of the affine functional
`(c_j − c_{i*}) + (g_j − g_{i*})^T w` over `W`, widened by `e_j + e_{i*}` and by the float32
storage error of the difference generators. Proof: the functional is affine; the weak-duality
bound of Section 4 is valid for every point of `W`; the true difference deviates from the
functional by at most the two errors; the stored float32 generators deviate from the exact
difference by the summed casting error. These bounds replace the box-only radii in the softmax
enclosure wherever they are tighter (symmetrically for the pair `(i*, j)`), so the Hessian
remainder and the output box of Section 8 only shrink.

**Lemma 7 (ratio rows).** For true softmax outputs, `p_j / p_{i*} = exp(s_j − s_{i*})`, hence
`p_j ≤ exp(U_j) p_{i*}` and `p_j ≥ exp(L_j) p_{i*}`. With value maps `p = p̂ + δ`, `|δ| ≤ e_p`,
the rows

    p̂_j − exp(U_j) p̂_{i*} ≤ e_{p,j} + exp(U_j) e_{p,i*} + pad,
    exp(L_j) p̂_{i*} − p̂_j ≤ e_{p,j} + exp(L_j) e_{p,i*} + pad,

hold at every true point, where `pad` covers the directed rounding of the exponentials and the
float64 evaluation of the row. They are added as predicate rows of the element over its factors
(the score factors and the softmax fresh factors).

Why this matters in this construction and not in the baseline's. Our attention product
`P @ V` is the DeepZ product of the softmax state with `V`, so its value map references the
softmax fresh factors; the ratio rows therefore constrain the factors that appear in the
terminal objective, and the LP can trade softmax slack against score slack. In the baseline's
construction the attention output is written on Q, K, V generators only, and its softmax rows
constrain generators the output never references (agent reading, LOG N064 note).

Measured: PGD ViT row 15 terminal LP bound 0.296 → −0.328 (CERT at the LP stage), rows 62 and
78 CERT end to end; IBP rows unchanged, because their first attention block precedes every
ReLU (no rows exist yet) and their softmax interval already equals its linear part (LOG N077).
Cost: `2 Q (n − 1)` dense rows and one batched LP of `Q n` functionals per softmax layer.
