# From Hybrid Zonotopes to Spec-Conditioned HZ
### A Self-Contained Briefing on Our Innovation in Forward-Only Neural-Network Verification

---

> ## ⚠ Post-experiment correction (2026-06-04 night)
>
> The PRUNE thesis described in §11-§13 below (d_L-driven per-rival generator pruning tightens LP UB) was **empirically falsified** by the Phase A continuation K ablation: on 40/40 Dense iids (acasxu + safenlp), LP UB monotonically **grows** as K shrinks. PRUNE is sound but produces **zero precision lift** on the tested benchmarks.
>
> What survived empirical testing — and produced the 358 NEW A on safenlp_2024 (924 → 1282 V/A combined, audited 368/368 STRICT-PASS) — is a different mechanism: **per-rival forward HZ + closed-form LP-maximizer at the input-box corner in the rival direction + strict ORT replay**.
>
> The correct framing for the contribution is:
>
> > **SC-HZ directional witness sidecar for wide-spec dense networks** — a forward-only, structured per-(y_true, rival) LP candidate generator with closed-form box-corner decoding and strict ONNX-runtime replay, principle-compliant (no backward, no gradients, no random/PGD, no BaB, no MILP), zero production-code modification.
>
> Not a "generic HZ tightening", not a "pruning method", not "approaches CROWN precision". Empirically verified on safenlp_2024 only (1080 instances); behavior on acasxu = 20/20 phantom; behavior on CIFAR/Tiny/VGG = not measured (ResNet shape impl gap).
>
> Read §11-§13 as **the original hypothesis we tested**. Read [research/sc_hz_phase_b_results_20260604.md](sc_hz_phase_b_results_20260604.md) for **what actually worked**.

---

**Date**: 2026-06-04
**Audience**: senior advisor unfamiliar with the day-to-day details of this project.
**Scope**: this document is intended to be read **linearly**, end to end, without consulting external references. It covers (a) what neural network verification is, (b) what Hybrid Zonotopes (HZ) are, (c) how forward-only HZ verifiers work today, (d) where standard HZ fails on robustness queries, and (e) our proposed innovation — Spec-Conditioned HZ (SC-HZ).

---

## Part I — Background

### 1. Neural network robustness verification: the problem

A modern image classifier `f` maps an input image `x ∈ R^n` to class scores `y = f(x) ∈ R^C`. Given an input `x*` (e.g. a CIFAR-100 image) and a perturbation budget `ε > 0`, the **L∞ robustness query** is:

> Is it true that for every input `x` with `||x − x*||_∞ ≤ ε`, the classifier still predicts the true class `y_true`?

Mathematically:
```
∀ x ∈ B_ε(x*):  argmax_k f(x)_k  =  y_true
```

For finite piecewise-linear networks with rational data, this problem is
**decidable** but computationally hard: ReLU robustness checking is NP-hard
and can be encoded as integer feasibility. The entire field of NN verification
studies how to decide or soundly approximate these queries efficiently.

A **verifier** is a program that takes (model, input box, output spec) and returns one of three verdicts:

| Verdict | Meaning |
|---|---|
| **V (Verified)** | The query holds — no x in the box can violate the spec. Equivalently: CERT. |
| **A (Falsified)** | The query fails — a specific witness x' ∈ box has been found that violates the spec. Equivalently: SAT. |
| **U (Unknown)** | The verifier could not decide within budget — neither could it prove safety nor find a witness. |

The **standard benchmark suite** for the field is VNN-COMP (the Verification of Neural Networks Competition), which in 2025 contained 26 benchmarks with a total of 3,453 instances spanning image classification, NLP, control, autonomous-driving perception, RL policies, and so on.

### 2. The four families of verifiers

Approximately four mechanistic families exist in NN verification:

1. **Forward abstract interpretation**. Propagate a SET (a box, zonotope, polytope, hybrid zonotope, star set, etc.) through the network layer by layer. At the output, check whether the output set lies entirely in the safe region. This is "forward-only", sound, but tends to be loose on deep networks.

2. **Backward bound refinement (CROWN / α-CROWN / β-CROWN)**. Compute the output bound by symbolically substituting layer-by-layer from output toward input. Choose ReLU relaxation slopes by gradient-based optimization to tighten bounds. Very tight but uses gradient information, optimizes bounds backward, and effectively a different mathematical object than a forward set.

3. **Branch-and-bound (BaB) on activation regions**. Recursively split the input box (or the activation pattern of a chosen ReLU) into sub-regions, verify each separately, combine. Eventually decides the question, but the search tree can be exponential.

4. **MILP / mixed-integer encodings**. Encode the network as a mixed-integer program and solve directly. Exact but exponentially expensive at scale.

Most leading verifiers combine 2 and 3: bound propagation (CROWN-style) gives a fast initial bound; BaB refines. **abcrown** is the canonical example of this combination and is the dominant winner of VNN-COMP year after year.

### 3. Why "forward-only" is a research category of its own

A verifier in family 1 — pure forward abstract interpretation — has a **single mathematical story**: a set propagates forward, the bound is what the propagation produces, and soundness is a direct consequence of the abstract interpretation's correctness. There is no auxiliary mechanism (backward iteration, search, MIP solver) that the verifier's claims depend on.

This makes the forward-only verifier's output **structurally auditable**: every claim is the direct output of one well-defined mathematical pipeline. By contrast, a hybrid verifier's output depends on the interplay of bound refinement + search + MILP, and external audit must verify each component.

Forward-only verifiers are therefore valuable for **safety-critical contexts** where the verification claim must be traceable. This is the niche the project ACT/HyZor pursues.

The price of staying forward-only is precision: we cannot use the bound-refinement and search machinery that gives abcrown and NeuralSAT their high resolve rates. The relevant question for our research is **how to make forward-only verification as competitive as possible within its principle set**.

---

## Part II — Hybrid Zonotopes

### 4. Zonotope: the simplest non-trivial set abstraction

Before HZ, we should understand its predecessor.

A **zonotope** in `R^n` is a set of the form

> `Z = { c + G ξ  |  ξ ∈ [-1, +1]^p }`

where `c ∈ R^n` is the **center** and `G ∈ R^{n×p}` is the **generator matrix** (each column of G is a "generator direction"). The set is the image of the hypercube `[-1, +1]^p` under the affine map `ξ → c + G ξ`.

A zonotope is **convex** and **centrally symmetric**. Its number of vertices can
grow polynomially in the number of generators for fixed dimension and
exponentially in the dimension in the worst case; the important point for this
brief is not the exact vertex formula but that the set is represented compactly
by `(c, G)` rather than by enumerating vertices.

**Concrete example**: in `R^2`, the zonotope with center `c = (0, 0)` and generators

```
G = [[1, 0.5],
     [0, 1]]
```

is the parallelogram

```
        ξ_1 ↑               ξ_1 = +1
              ●─────────●
           ╱     ╱
         ╱     ╱           ξ_2 ↗
      ●─────────●
        ξ_1 = -1
```

Zonotopes are closed under **affine transformations** (just multiply: `A · Z = c' + (A G) ξ`), which means they propagate exactly through linear layers (Conv2D, Dense, BN). The cost is that the generator count `p` only grows over time; for deep networks, `p` can reach tens of thousands.

### 4.1 Why zonotopes can't handle ReLU exactly

ReLU is the function `relu(z) = max(0, z)`. Its graph (in `z, relu(z)` space) is the bent line that follows `y = 0` for `z ≤ 0` then `y = z` for `z ≥ 0`. **The output of a zonotope under ReLU is no longer a zonotope** — the bent line introduces non-convex (or at least non-zonotopic) shape.

The standard relaxation is the **DeepZ triangle**: replace the ReLU output with the smallest triangle containing the graph segment over `z ∈ [l, u]`:

```
y ≥ 0,    y ≥ z,    y ≤ slope · (z − l)
```

where `slope = u / (u − l)`. This is the tightest single-neuron convex relaxation, and the relaxed output is again a zonotope (with one extra generator per unstable neuron, capturing the triangle slack).

But the relaxation is **lossy**: each ReLU introduces a small amount of imprecision (the triangle's gap above the bent line), and **the loss compounds over many layers**. After 30 conv layers each with ~100 unstable neurons, the output zonotope is a vast over-approximation of the true reachable set, and the verifier reports UNK on most queries.

### 5. Hybrid Zonotopes: adding integers to recover precision

T. Bird's 2022 PhD thesis introduced the **Hybrid Zonotope** (HZ): a generalization of the zonotope that adds **binary generators** and **affine constraints on the generators**.

**Definition** (Bird 2022 / ACT notation). A Hybrid Zonotope in `R^n` is a set
of the form:

> `Z = { c + Gc ξ_c + Gb ξ_b  |  ξ_c ∈ [-1, +1]^ng,  ξ_b ∈ {-1, +1}^nb,  Ac ξ_c + Ab ξ_b ≤ b }`

Some HZ papers write the affine constraints in equality form; ACT's verifier
uses an LP-facing polyhedral constraint notation. The distinction is not
important for this brief: the essential feature is a generator-space
polyhedron plus a subset of binary generators.

The 6-tuple `(Gc, Gb, c, Ac, Ab, b)` specifies the set:

| Component | Shape | Meaning |
|---|---|---|
| `c` | `(n, 1)` | center |
| `Gc` | `(n, ng)` | continuous generator matrix |
| `Gb` | `(n, nb)` | binary generator matrix |
| `Ac` | `(nc, ng)` | continuous side of the constraint matrix |
| `Ab` | `(nc, nb)` | binary side of the constraint matrix |
| `b` | `(nc, 1)` | constraint right-hand side |

The constraint `Ac ξ_c + Ab ξ_b ≤ b` is a **polytope** in the generator space (with continuous variables `ξ_c` and integer variables `ξ_b`).

**Why this is powerful**:

1. **Setting `nb = 0` and `nc = 0` gives a plain zonotope**. HZ is a strict generalization.

2. **Binary generators encode discrete choices exactly**. Because each `ξ_b^k ∈ {-1, +1}`, the binary part can express:
   - A union of two zonotopes: `A ∪ B  =  ((A + B)/2)  +  ((A − B)/2) · ξ_b` (set membership conditioned on the binary).
   - An exact piecewise-affine function. In particular, **exact ReLU is
     encodable with one new binary choice plus auxiliary affine constraints**
     in Bird's construction. The binary records which ReLU branch is active;
     the added affine constraints tie the pre-activation and post-activation
     consistently to that branch. The result is exact before relaxing the
     binary variable — no triangle relaxation is needed for that neuron.

3. **Intersection with a halfspace is closed-form**. Adding a constraint of the form `α^T y ≤ β` to an HZ produces another HZ (just append a row to `Ac, Ab, b`). This is critical for output specifications.

4. **Linear maps are closed-form**: `A · Z` is `(A c, A Gc, A Gb, Ac, Ab, b)`.

### 5.1 The HZ-vs-zonotope tradeoff

The cost of HZ's expressiveness is that the underlying optimization problem is now a **mixed-integer linear program (MILP)**. Bounds on `y = c + Gc ξ_c + Gb ξ_b` require solving:

```
maximize  y_i
subject to:  Ac ξ_c + Ab ξ_b ≤ b
             ξ_c ∈ [-1, +1]^ng
             ξ_b ∈ {-1, +1}^nb
```

For large `nb`, this is intractable. **The principle question for HZ-based verifiers is: when do we use binary generators (and pay the MILP cost), and when do we relax them to continuous variables (recovering a zonotope-style LP)?**

Two extremes:

- **"Triangle-only" HZ**: never use binary generators; relax every ReLU via DeepZ triangle. The HZ is then equivalent to a zonotope, and bounds are continuous LP. This is **fast but loose**.

- **"Exact HZ"**: encode every ReLU exactly with a binary generator. Bounds are MILP. **Tight but expensive**.

Real verifiers (including ours) sit in the middle. Most use triangle for the bulk of the network and reserve binary generators for select layers.

### 5.2 Why HZ is a richer abstraction than Star Sets

The Star Set abstraction (Bak et al.) is `{c + G α | P(α) ≤ q}` — a zonotope with an arbitrary polytope constraint on the generator coefficients. Star Sets and HZ are closely related; HZ is essentially a Star Set with the additional integer/binary structure on a subset of the generators.

When the binary generators are dormant (`nb = 0`), HZ ≡ Star Set. The structural advantage of HZ is the **option to activate binaries** for exact-ReLU encoding on specific neurons.

---

## Part III — Forward HZ Verifiers in Practice

### 6. The standard forward HZ pipeline

A forward HZ verifier for a robustness query processes the network layer by layer:

```
1. Input HZ:
     Given input box [lb, ub], encode as a zonotope:
         c_0 = (lb + ub) / 2          # center
         Gc_0 = diag((ub - lb) / 2)    # generators (one per input dim)
         Gb_0 = (n_input, 0) empty
         Ac_0, Ab_0, b_0 all empty
2. For each layer L = 1, ..., N:
     If layer L is linear (Conv, Dense, BN):
         h_L = c_L + Gc_L ξ_c + Gb_L ξ_b  =  W_L · h_{L-1} + bias_L
         (closed-form: Gc_L = W_L · Gc_{L-1}, Gb_L = W_L · Gb_{L-1}, c_L = W_L · c_{L-1} + bias)
     If layer L is ReLU:
         For each neuron i:
             Compute bounds (l_i, u_i) of pre-activation z_i = c_i + Gc[i,:] ξ_c + Gb[i,:] ξ_b
             If l_i >= 0 (always active):  h_i = z_i  (pass through)
             If u_i <= 0 (always inactive): h_i = 0  (zero out)
             Else (unstable):
                 Apply DeepZ triangle relaxation (1 new aux generator added to Gc)
                 OR
                 Apply Bird's exact-HZ ReLU (1 new binary generator added to Gb + 3 constraints)
     If layer L is MaxPool, AvgPool, Concat, etc.:
         Apply the corresponding HZ operator
3. Output query:
     Given output spec α^T y ≤ β:
         Solve LP (with continuous-relaxed binaries):
             max α^T y - β
             s.t.  y = c_N + Gc_N ξ_c + Gb_N ξ_b
                   Ac_N ξ_c + Ab_N ξ_b ≤ b_N
                   ξ_c ∈ [-1, +1]^ng,  ξ_b ∈ [-1, +1]^nb
         If max ≤ 0: CERT (the spec holds, query VERIFIED)
         If max > 0: extract candidate ξ*; decode to input x*; run x* through the model with onnxruntime;
                    if the model output actually violates the spec at x*: FAL
                    else: UNK (the LP candidate was a "phantom" — feasible in the abstract domain but not in the concrete model)
```

The receipt produced by an HZ verifier (under the project's principle set) includes the input box, the spec, the output verdict, the witness x* (if FAL), and SHA256 hashes of all artifacts. This receipt is independently auditable.

### 7. The two pain points of standard forward HZ on robustness queries

In our experiments (2026-04 through 2026-06), we systematically characterized where forward HZ verifiers fail on robustness benchmarks.

#### Pain Point 1 (PP1): HZ degenerates to Star Set on conv layers

In our project's default forward HZ verifier, the binary generator machinery is **dormant** during conv body propagation. For the entire conv body of cifar100, tinyimagenet, vggnet16, and yolo benchmarks (tens of layers), the HZ has `nb = 0` and the constraint set `(Ac, Ab, b)` is empty. **We are running Star Set, not HZ.** The exact-ReLU binary mechanism only activates at the network tail (in our code, called `large_cls_proof_mode`).

This is a structural waste: we carry the HZ representation cost (the `Ac, Ab, b` tuple) but get no HZ benefit because the binaries are never used during conv propagation.

#### Pain Point 2 (PP2): Triangle slack compounds multiplicatively

Each unstable ReLU adds one aux generator with magnitude `μ_i = (-l_i u_i) / (2(u_i - l_i))`. After `K` stacked ReLUs in a feedforward network, the cumulative aux mass scales as

```
Σ_{k=1..K} μ_k · ||W_subsequent||
```

For a typical CIFAR conv body (~30 unstable ReLU positions per layer × 10 layers), this is the **dominant source of LP UB looseness** at the final tail.

In our experiment §7 (the "CIFAR final-tail per-neuron hull" test), we showed that the production endcap LP is **bit-exact** with a spec-compliant clean LP on 20 sentinel iids. The per-neuron triangle is provably the tightest single-neuron convex relaxation; we cannot tighten further at the per-neuron level. Joint relaxation (k-ReLU cuts) is the natural next step but was closed-negative on acasxu (PRIMA k=2, k=3 cuts both showed 0 lift).

#### Pain Point 3 (PP3): Generator reduction is blind to the verification query

When the generator count `ng` becomes too large (typically > a memory budget cap), HZ verifiers apply **Girard reduction**: keep the top-K generators by **column norm**, merge the rest into a sound axis-aligned box.

But the column norm is a **query-independent** criterion. For robustness verification, the "right" criterion would be: keep the generators that contribute most to the **rival margin** `(W_out[r] − W_out[y_t]) · h` for the rival classes `r`. Column-norm reduction discards generators that may be column-light but rival-relevant, and keeps generators that are column-heavy but rival-irrelevant.

In a prior experiment (`spec_aware_girard_v2_20260530.md`), we tried adding output-direction information to Girard reduction. On conv layers, the lift was negligible because Conv operators **homogenize generator directions** — every column of Gc has similar "spread" across output positions, and the directional ranking is flat.

#### Pain Point 4 (PP4): The full output HZ is computed when only the rival projection matters

For a top-1 robustness query, the answer we need is `max over x in box of (W_out[r] - W_out[y_t]) · y(x)` for each rival `r`. This is **one scalar per rival**. Yet the verifier computes the full output HZ `y` (1000-dim for ImageNet-class problems) and projects at the end.

Carrying the full HZ through the network is **expensive and largely wasted**: most per-output-position information is irrelevant to any specific rival.

### 8. Our project context: ACT/HyZor

The ACT/HyZor verifier is the project that wrote this document. Under the principle set

| ID | Principle | Forbids |
|---|---|---|
| P1 | Forward-only | CROWN backward, gradient-based bound refinement |
| P2 | No gradients | PGD / FGSM / CW / AutoAttack falsification |
| P3 | Continuous LP only | MILP, integer reasoning |
| P4 | No fallback verifier; no BaB | search, input splitting |
| P5 | No random / corner falsification | "trial-and-error" witness generation |

ACT/HyZor achieves **924 V/A across 22 VNN-COMP-2025 benchmarks** (805 V + 119 A, 26.8% resolve rate, 109 tool errors). On the same sweep:

| Tool | V+A | Resolve | Engine |
|---|---:|---:|---|
| abcrown `--NOPGD` | 2460 | 71.2% | BaB + bound prop |
| NeuralSAT `--disable_attack` | 2065 | 59.8% | BaB + bound prop |
| nnenum | 1445 | 41.9% | exact-star splitting |
| PyRAT `[con_z]` | 1393 | 40.3% | forward constrained zonotope |
| **ACT (HZ) GPU** | **924** | **26.8%** | **forward HZ (ours)** |
| PyRAT `[hyb_z]` | 627 | 18.2% | forward HZ |
| NNV STRICT | 457 | 13.2% | forward approximate star |
| CORA TRUESTRICT | 2 | 0.06% | forward reachability |

ACT is **#1 among pure-forward verifiers** (the bottom 4 in this list — ACT, PyRAT [hyb_z], NNV, CORA), beating its only same-domain competitor (PyRAT [hyb_z]) by **+47.4%** with **9× fewer tool errors**. But on absolute scale, ACT is #5: there is a gap of 1536 V/A to abcrown.

This gap is the motivation for the redesign described in Part IV.

---

## Part IV — Our Innovation: Spec-Conditioned HZ (SC-HZ)

### 9. Intuition and core idea

The four pain points (PP1–PP4 above) suggest a single root cause: **standard forward HZ is "spec-blind"** — it propagates the same HZ regardless of what robustness query is being asked. For a CIFAR-100 query, it propagates a 100-dimensional output set; for a binary classification query, it propagates a 2-dimensional output set; for both, the inner HZ representation carries information about every output dimension.

But the verification question is **always 1-dimensional**: "for each rival class `r`, is `(W_out[r] − W_out[y_t]) · y < 0` for all `x` in the box?". For a 100-class problem there are 99 such 1-D questions.

**The insight**: if the verifier KNEW which output direction it was asking about, it could **spend its generator budget on the generators relevant to that direction** and **merge the rest into a sound tail**.

This is the basis for **Spec-Conditioned Hybrid Zonotope (SC-HZ)**. The full document containing the design lock is `research/dc_hz_phase_a_plan.md`; this section is the conceptual presentation.

### 10. The d_L direction — a pre-computed linear functional

Given a network with hidden layers `L = 1, ..., N` and weight matrices `W_1, ..., W_{N+1}` (where `W_{N+1}` is the output classifier), and a top-1 robustness query with true class `y_t` and rival set `R`, we precompute:

```
For each rival r ∈ R:
    d_N^r  =  W_{N+1}[r, :]  −  W_{N+1}[y_t, :]        # at output (vector of length = last hidden dim)
    for L = N-1, N-2, ..., 0:
        d_L^r  =  W_{L+1}^T · d_{L+1}^r                 # backward through weights ONLY
```

**Walking through what `d_L^r` is**:

`d_N^r` is the rival classifier's direction: the linear functional that maps the last hidden layer `h_N` to the rival margin `y[r] - y[y_t]`. If we replaced the network's final layer with `d_N^r · h_N`, the output would be the rival margin directly.

`d_L^r` (for `L < N`) is **what the rival direction would be at layer L if every ReLU in between were the identity**. Specifically, `d_L^r · h_L` is the value of the rival margin **under a linear approximation** of the network from layer L to output.

`d_L^r` is computed by chained linear-algebra backward multiplication through the weight matrices only. It does **not** use:
- The input box `[lb, ub]` of any iid.
- The actual ReLU activations of any iid.
- Any bound information at any layer.

It is a purely **architectural quantity** of the (model, rival) pair, computable once per model and cached forever.

Equivalently, `d_L^r` is the **identity-ReLU skeleton** of the rival
functional: it is what a CROWN-style backward symbolic coefficient would look
like if every ReLU slope were fixed to `1`, every intercept were fixed to `0`,
and no bounds, dual variables, or optimization were allowed. This is useful
but imperfect. It can overweight paths that are actually dead for a particular
input box. That hurts precision, not soundness, because `d_L^r` only orders
generators; the dropped part is always boxed soundly. Phase A must therefore
include ablations against simpler scores: linear-only `d_L`, forward-stable-mask
`d_L` (mask only ReLUs proven inactive by a forward interval pass), random
keep-K, and norm keep-K.

For non-Dense operators, Phase A should keep the same interpretation: use the
linear adjoint when the operator is linear and fail closed when it is not.
Conv uses the transpose convolution, ADD sends the same cotangent to both
branches, Concat slices the cotangent by output range, Flatten/Reshape invert
the shape transform, and AvgPool uses the transpose of the averaging operator.
MaxPool is not linear; Phase A should either stop `d_L` at MaxPool, use only a
forward-proved stable-max route, or mark that model unsupported for SC-HZ.

### 10.1 A worked example

Consider a small network:

```
Input x ∈ R⁴  →  Dense(W_1) + ReLU  →  h_1 ∈ R⁶  →  Dense(W_2) + ReLU  →  h_2 ∈ R³  →  Dense(W_3)  →  y ∈ R²
```

Suppose `W_3` is

```
W_3 = [[ 2,  1,  0],     ← class 0 weights
       [-1,  3,  2]]      ← class 1 weights
```

A robustness query asks: "is class 0 always greater than class 1 over the input box?". So `y_t = 0` and `r = 1`. We compute:

```
d_2^1  =  W_3[1, :] − W_3[0, :]  =  [-1, 3, 2] − [2, 1, 0]  =  [-3, 2, 2]
```

`d_2^1` is a vector in `R^3` (the space of `h_2`). Interpretation: at the L2 output, the rival margin is `d_2^1 · h_2 = -3 h_{2,0} + 2 h_{2,1} + 2 h_{2,2}`.

Next:
```
d_1^1  =  W_2^T · d_2^1
```

`W_2` is `R^{3×6}`; `W_2^T` is `R^{6×3}`; `d_1^1` is in `R^6`. This is the rival direction in the space of `h_1`, **assuming ReLU between L1 and L2 is the identity** (so `h_2 = W_2 h_1`, hence rival margin = `d_2^1 · W_2 · h_1 = (W_2^T d_2^1) · h_1 = d_1^1 · h_1`).

Finally:
```
d_0^1  =  W_1^T · d_1^1
```

is the rival direction in input space.

The entire chain is **pure linear algebra** — three matrix-vector multiplications. No bounds, no inputs, no autograd.

### 11. The PRUNE operator: per-rival generator selection

Inside the forward HZ, at any layer L, the HZ state is

```
h_L = c_L + Gc_L · ξ_c        (continuous generators only — focusing on the zonotope-like part)
```

`Gc_L` is `(hidden_dim, ng)`. Each column `Gc_L[:, j]` is a **direction in feature space**: as `ξ_j` varies in `[-1, +1]`, the state `h_L` moves along `Gc_L[:, j]`.

Now we have `d_L^r` (computed in §10) — a vector in feature space telling us "this is the direction the rival margin is sensitive to, under the linear approximation".

**The PRUNE operator**: at layer L, sort the generators by

```
relevance(j) = | d_L^r · Gc_L[:, j] |
```

The geometric meaning: this is **the projection of generator j onto the rival direction**. If `relevance(j)` is large, moving along generator j strongly affects the (linearized) rival margin; if `relevance(j)` is small, it does not.

PRUNE keeps the top-K generators by relevance and merges the rest into a
**sound interval tail**:

```
PRUNE(c, Gc, d_L, K):
    relevance = abs(d_L @ Gc)                  # (ng,) per-column scalar
    keep_indices = argsort(-relevance)[:K]
    drop_indices = argsort(-relevance)[K:]
    Gc_kept = Gc[:, keep_indices]
    r_tail = abs(Gc[:, drop_indices]).sum(axis=1)   # (hidden_dim,) row-wise L1 sum of the dropped columns
    Gc_tail = diag(r_tail)                           # conceptual; store sparsely / as BoxHZ
    return (c, concatenate([Gc_kept, Gc_tail], axis=1))
```

The result is a new HZ with `K + hidden_dim` continuous generators: K
"rival-relevant" ones plus an axis-aligned interval remainder that
conservatively absorbs all the discarded generators.

In implementation, `diag(r_tail)` must **not** be materialized as a dense
matrix for conv feature maps. It should be stored as a sparse diagonal,
BoxHZ, or interval-remainder sidecar and only expanded when an operator truly
requires explicit columns.

This detail is crucial. A **single** tail column `r_tail[:, None]` would only
create one line segment in the positive `r_tail` direction and would not
represent an axis-aligned box. The sound construction needs independent
tail degrees of freedom per coordinate, or an equivalent BoxHZ / interval
remainder representation.

### 12. Soundness of PRUNE

**Claim**: the pruned HZ is a sound over-approximation of the original.

**Proof sketch**: any point `c + Gc · ξ` with `ξ ∈ [-1, +1]^ng` decomposes as

```
c + Gc[:, keep] · ξ_keep  +  Gc[:, drop] · ξ_drop
```

The contribution of the dropped generators is bounded coordinate-wise by

```
| Σ_{j ∈ drop} Gc[i, j] · ξ_drop[j] |  ≤  Σ_{j ∈ drop} |Gc[i, j]|  =  r_tail[i]
```

So for each coordinate `i`, an independent `ξ_tail[i] ∈ [-1, +1]` can realize
the dropped contribution within the interval `[-r_tail[i], r_tail[i]]`. Hence
the original HZ is a subset of the pruned HZ. **Sound.**

### 13. The critical principle observation

PRUNE depends on `d_L^r`, which depends on the network weights and the rival classifier. But — and this is the key insight — **the SOUNDNESS of PRUNE does not depend on whether `d_L^r` was computed correctly**.

Even if we passed PRUNE a completely wrong `d_L^r` (e.g. a random vector, the zero vector, the negation of the true direction, an adversarial value), the proof in §12 still goes through: the dropped generators are STILL absorbed into a sound interval tail `r_tail` defined by their absolute-value row sums. The result is STILL an over-approximation of the original HZ.

A wrong `d_L^r` makes PRUNE less effective — the LP UB at the output will be looser than with the right `d_L^r` — but it cannot make PRUNE unsound.

This is the key to the principle question: **`d_L^r` is a heuristic for choosing which generators to keep; it is NOT bound information**.

### 14. Why this is not backward bound refinement

α-CROWN (the workhorse of abcrown) propagates backward through the network and computes BOUNDS at each hidden layer using OUTPUT bound information. The slope chosen for each ReLU's relaxation is then optimized to TIGHTEN those bounds. This is bound refinement: the L-layer bound is REFINED by L'-layer bound information for L' > L.

SC-HZ does not refine ANY bound. The bounds at each layer are computed PURELY FORWARD: input bound → Conv2D → ReLU triangle → ... → output bound. The d_L^r quantity is used SOLELY to decide which generators to KEEP during forward propagation. Whether d_L^r is "right" or "wrong" affects how many irrelevant generators we waste budget on, but does NOT affect what the bound at any layer represents.

The honest comparison is: `d_L^r` resembles the **linear coefficient skeleton**
of CROWN, but none of the CROWN machinery that makes CROWN powerful is present.
There are no optimized ReLU slopes, no backward-propagated bounds, no
dual-variable updates, and no gradient loop. The coefficient skeleton cannot
validate a bound; it can only prioritize which already-forward generators fit
inside the memory budget.

In symbols:
- α-CROWN: `bound_L = f_L(bound_{L+1}, ..., bound_N)` (backward bound refinement)
- SC-HZ:   `bound_L = f_L(bound_{L-1})` (purely forward), with `representation_L = g_L(bound_L, d_L)` (per-rival representation choice using d_L)

The bound is forward-only. The representation choice uses d_L. The two are separate axes.

### 15. The adopted definition of "forward-only"

After the principle review (recorded in `research/hz_redesign_for_robustness_20260604.md` §9), the project's definition of forward-only is:

> A verifier is **forward-only** iff for every layer L, the bound information at L is determined SOLELY by the bound information at L−1 and the operator at L. No bound at L′ > L can refine the bound at L.

Under this definition, SC-HZ is forward-only. The `d_L^r` quantity is a **representation-choice heuristic**, not a bound; soundness holds regardless of its value; therefore there is no bound information flowing backward.

### 16. Why we expect SC-HZ to lift V/A

The mechanism described above lets us answer the four pain points:

- **PP1 (HZ ≡ Star on conv)**: PRUNE applies inside the standard HZ pipeline, so the binary side is untouched. SC-HZ is fully compatible with binary activation at the tail (which we will add in a later phase).
- **PP2 (Triangle compounding)**: SC-HZ does not change the per-neuron triangle and it cannot beat a full unpruned HZ. Its target is narrower and more honest: under a fixed memory budget, it should be tighter than **query-blind budgeted reduction** because rival-relevant generators are kept and rival-irrelevant generators are absorbed into a sound interval tail. If production is already unpruned at the comparison point, PRUNE can only equal or loosen.
- **PP3 (Generator reduction blind to query)**: SC-HZ directly fixes this. PRUNE is **query-conditioned** by construction.
- **PP4 (Full output HZ is wasted)**: SC-HZ propagates ONE HZ per rival, each
  with a K-budgeted generator set plus a sound interval tail. At the output,
  we solve a 1-D LP on the rival margin, not a 100-D LP on the full output set.

The mechanism is principled and sound, but it is not magic. Its gain requires
two conditions at once: (1) generator budget pressure is active, and (2) the
current reduction order is misaligned with the rival direction. Where either
condition is false, SC-HZ should be neutral or looser.

The computational plan follows from this constraint. Share the ordinary
forward affine propagation as long as possible, branch only when per-rival
reduction starts to differ, batch rivals in small GPU groups, pre-filter rivals
with cheap interval bounds, compute `d_L @ G` as batched matrix products, and
store interval tails sparsely. A correct Phase A is allowed to be slow; a
production Phase B must prove that per-rival cost is bounded.

### 17. Honest expectations

The earlier draft of this analysis claimed SC-HZ + four engineering complementaries could lift ACT from 924 to ~1977 V/A. After advisor review, that target was walked back to a staged plan:

| Stage | Expected V/A | Gate |
|---|---|---|
| Current (no SC-HZ) | 924 | baseline |
| Phase A: SC-HZ generator budgeting only | +5 to +50 (on 80 sentinels) | proof-of-signal |
| Phase B: SC-HZ on 6-8 benchmarks | 924 → 1100/1300 | meaningful lift on multiple families |
| Phase C: SC-HZ + Selective binary activation + GPU batching | 924 → 1500/1700 | competitive with PyRAT con_z / nnenum-class totals |
| Phase D: + engineering cleanup (stable fastpath, exact MaxPool) | 924 → 1700/1900 | long-horizon |

We do not yet promise the long-horizon numbers. Phase A is the first gate; if it does not show signal, the entire SC-HZ direction closes.

The "selective binary activation" in Phase C means **continuous LP relaxation
of binary-derived HZ constraints only**. Any integer solve, MILP, or branch on
the binary variable remains excluded by the project principles.

### 18. The Phase A test plan

To test the SC-HZ hypothesis without committing weeks of engineering, we have selected 80 sentinel iids across 4 benchmarks. The composition is deliberate:

| Benchmark | iids | Role |
|---|---|---|
| **safenlp_2024** | 20 random UNK iids | **Primary positive signal target**: wide-spec disjunctive (many rivals per iid), where per-rival pruning has the most theoretical reach. |
| **acasxu_2023** | 20 random UNK iids | **Secondary positive signal target**: small-dense net with mid-network blowup in `ng`. |
| **tinyimagenet_2024** | 20 random UNK iids | **Dense-conv signal target outside CIFAR**. |
| **cifar100_2024** | 20 lowest-LP-margin UNK iids from atlas v3 | **Hard negative-control target**: §7 showed production endcap LP is already at the per-neuron triangle math ceiling at the tail, and atlas data suggests CIFAR has little/no production reduction to undo. SC-HZ should therefore be equal or looser on CIFAR unless the comparison path differs. Any CIFAR improvement is a strong bug signal and must be attributed before being counted. |

### 19. The Phase A gate

The test PASSES if:

- All 80 iids complete fail-closed (no crashes, no silent drops).
- On the **positive-signal group** (`safenlp_2024`, `acasxu_2023`,
  `tinyimagenet_2024`), new V/A ≥ **5**, OR at least two of the three
  benchmarks show per-benchmark median LP UB reduction ≥ **25%** on completed
  UNKNOWN rows.
- `cifar100_2024` behaves as a negative control: no unexplained new V/A and no
  unexplained LP UB tightening relative to production.

The test FAILS if:

- Positive-signal new V/A = **0** AND every positive benchmark has median LP UB
  reduction < **10%**.
- CIFAR shows unexplained tightening or new V/A. That is an implementation
  investigation, not a success.
- Or any FAL witness fails strict ORT replay, or any CERT proof fails the
  independent LP/provenance/no-lost audit. ORT replay validates witnesses; it
  is not by itself a CERT audit.

INCONCLUSIVE in between → expand K-cap from 256 to 512 and re-run the 10 weakest sentinels.

The gate reports three separate counters: `new_cert`, `new_fal_strict_replay`,
and `phantom_lp_sat`. LP-feasible counterexamples that depend on interval-tail
variables and cannot be decoded to an original input are **phantom LP SAT** and
must remain UNKNOWN. SC-HZ is expected to help CERT more directly than FAL
unless the witness lives in retained root variables.

### 20. What SC-HZ contributes to the verification literature

To our knowledge, SC-HZ is a **new mechanism** in the verifier-design space. The closest prior art:

- **α-CROWN** (Salman et al., Wang et al.) optimizes ReLU slopes backward — fundamentally different mechanism, not forward-only.
- **β-CROWN** (Wang et al.) adds BaB on top — not forward-only.
- **PRIMA / Singh et al. k-ReLU cuts** — joint-relaxation on the FORWARD side, but per-neuron K-ReLU, not per-rival.
- **Generator reduction with output sensitivity** (Mirman et al., spec_aware_girard_v2 in our prior work) — tried this on forward HZ; reported limited lift on conv. SC-HZ differs by computing the d_L at **each layer for each rival** (not a single output-projected metric).

SC-HZ in one sentence: **forward-only propagation with per-rival, layerwise,
weight-only generator budgeting, where every discarded direction is absorbed
into a sound interval tail**.

If the Phase A signal is positive, this becomes a paper section on its own, framed as:

> **"Spec-Conditioned Hybrid Zonotopes: a forward-only abstraction that budgets
> generator capacity per output query while preserving soundness by boxing every
> discarded direction."**

---

## Part V — Reading guide

This document is the **conceptual presentation** of the SC-HZ work. For deeper engagement:

- **Design lock for Phase A (the implementation contract)**:
  `research/dc_hz_phase_a_plan.md` — mathematical specification, soundness proofs, hard gate, unit-test matrix.
- **Strategic context (the redesign analysis)**:
  `research/hz_redesign_for_robustness_20260604.md` — the 5 proposed mechanisms (SC-HZ is R1), the principle question, the principle decision.
- **Project state (the 924 V/A result)**:
  `research/paper_skeleton_20260604.md` — paper-shaped narrative including positive/negative results.
- **Cross-tool comparison**:
  `research/tool_comparison_20260604.md` — same numbers as §8 above, but with detailed per-benchmark breakdowns.
- **Profile honesty document**:
  `research/profile_matrix_20260604.md` — which 924 V/A came from which profile (benchmark-name vs structural).
- **Bird's thesis on HZ**:
  `/data1/Kane/HyZor/HZ/PhD_Trevor_Bird_2022.pdf` (sections 3–4 for the definition, section 5 for the exact-ReLU encoding).
- **α-CROWN paper** (for the natural comparison):
  see references in `paper_skeleton_20260604.md`.

---

## Part VI — Where this fits in the broader research arc

The project's overall trajectory has been:

1. **2026-03/04**: Establish ACT/HyZor as a forward-only HZ verifier with provenance + strict ORT replay.
2. **2026-04/05**: Multiple per-benchmark optimizations (small-dense witness profiles, CIFAR endcap profile, nn4sys lindex fastpath). Reach 924 V/A.
3. **2026-06**: Confront the "pure-forward #1 but #5 overall" positioning problem. Three closure analyses (CIFAR-ImageHZ, VGG-ImageHZ, CIFAR final-tail per-neuron hull) showed that per-neuron and per-layer levers are exhausted under standard HZ.
4. **2026-06-04 (now)**: Propose SC-HZ as the structural change that addresses what the previous mechanisms could not — query-aware generator allocation.

The SC-HZ proposal is the project's first principled attempt to redesign the HZ
representation itself for robustness verification, as distinct from
per-benchmark profile work. If Phase A signal is positive, it represents the
project's contribution beyond the current 924 V/A delivery.

---

## End of brief

For questions about specific sections, or to extend the explanation in any direction, contact the project author.

The author of this document is the same engineer who:
- ran the 22-benchmark cross-tool sweep referenced in §8,
- discovered the §6b denominator bug and the §6c iid-15 dense forensic,
- ran the §7 CIFAR final-tail hull pilot that established the per-neuron triangle math ceiling,
- co-designed (with the senior advisor) the principle set in §8,
- authored the design lock for SC-HZ Phase A,
- and is awaiting the advisor's review of this brief before proceeding to Phase A implementation.
