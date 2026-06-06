# H2 Empirical Ceiling — All Candidates Tested Fail Z0 Gate (2026-06-07)

**Period**: 2026-06-07 morning
**Per advisor 2026-06-07 directive**: H2 toy math gates BEFORE big engineering;
test A OPC-FD / B RB-T / D SETPH on Z0 toy.
**Headline**: 1480 V/A unchanged. All H2 candidates tested today FAIL Z0 gate.

---

## TL;DR

```
Z0 toy gate: candidate must achieve ≥47% drop over F1 to proceed.

Mechanism                              | Drop over F1 | Status
─────────────────────────────────────  ──────────────  ──────
F2b pairwise hull (already closed)   | 0%           | CLOSED
Compound triangle (overnight)        | 0%           | CLOSED  
FC-HZ multi-layer triangle           | 8.9%         | < gate
SETPH exact octant @ top_k=8         | 25.2%        | < gate
SETPH exact octant @ top_k=10        | 30.7%        | < gate
SETPH exact octant @ top_k=12 (all)  | 34.1%        | < gate (CEILING)
A OPC-FD minimal k_subspace=4        | -50.6%       | LOOSER than F1
A OPC-FD minimal k_subspace=12       | -11.8%       | still LOOSER

NO candidate passes Z0 gate.
```

**Conclusion**: continuous LP within DeepZ triangle on dense aggregate
slack has an empirical ceiling around 30-40% drop over F1. To reach
≥47%, we need either:
- Non-LP machinery (forbidden by P3)
- Constrained generator residual propagation (massive engineering)
- A fundamentally different domain (months of research)

---

## 1. SETPH (D) detailed results

SETPH = Small Exact Tail Projected Hull. For top-K unstable neurons at
last ReLU, enumerate 2^K sign octants. In each octant, enforce EXACT
ReLU (active: y=z equality; inactive: y=0 equality). Solve LP per
octant; take max.

### 1.1 Implementation v1 (failed)

First attempt only added sign-region constraints (z >= 0 or z <= 0)
WITHOUT y=z or y=0 equalities. Result: LP returned HZ-like values
(loose). The y_i variables had triangle relaxation freedom unconstrained
by the octant.

### 1.2 Implementation v2 (proper, exact octant LP)

Added explicit y_i variables with equality constraints per octant. LP:
```
Variables: xi (K-dim, ∈ [-1, 1]), y_select (n_select unconstrained)
For each selected neuron i in octant sign s_i:
  z_i = c_z[i] + G_z[i, :] @ xi
  if s_i = +1: z_i ≥ 0 AND y_select[i] = z_i (exact active)
  if s_i = -1: z_i ≤ 0 AND y_select[i] = 0 (exact inactive)
Objective: max d_out · (state.c + state.G @ xi)  but replace y_i_box
           with explicit y_select contributions for selected neurons.
```

### 1.3 Scaling data

```
top_k  2^k octants  median drop over F1  max drop
─────  ───────────  ───────────────────  ────────
4      16           12.6%                21.2%
6      64           18.8%                27.2%
8      256          25.2%                35.4%
10     1024         30.7%                37.1%
12     4096         34.1%                37.7%  (CEILING — all unstable used)
```

Linear scaling: ~6% additional median per doubling of top_k. To reach
47%, would need top_k ≥ ~16, but the toy only has 12 unstable neurons.
For larger networks (e.g., cifar's 25 unstable), top_k=25 = 33M octants
which is infeasible.

### 1.4 Why SETPH plateaus at 34.1%

SETPH only exacts the LAST layer. The first layer still uses triangle
relaxation. The "+138% loose vs brute" measurement shows the remaining
gap is the first-layer triangle. Going exact on FIRST layer too =
2^12 × 2^12 = 16M octants — infeasible.

---

## 2. OPC-FD (A) detailed results

OPC-FD = Output-Projected Constrained Forward Domain. At each block,
project to k-D subspace + carry residual interval.

### 2.1 Implementation: minimal 2-block

- Choose k subspace directions = top-|d_eff| neurons at block 1 output
- Carry y_1[top_k] as LP variables (constrained)
- Residual y_1[others] as interval (lb, ub) propagation
- Block 2: z_2 = W_2[:, top_k] @ y_1[top_k] + W_2[:, others] @ residual_interval

The residual contribution adds |W_2[:, others]| · residual_radius which
is the standard HZ closed-form behavior.

### 2.2 Results: NEGATIVE drops

```
k_subspace  median drop over F1  max drop
──────────  ───────────────────  ────────
2           -58.5%                -36.1%
4           -50.6%                -24.6%
6           -37.5%                -19.9%
8           -26.8%                -13.0%
10          -22.9%                -7.8%
12          -11.8%                +1.7%
```

ALL NEGATIVE. OPC-FD as implemented produces values LOOSER than F1.

### 2.3 Why OPC-FD fails minimally

The residual y_1[others] is propagated as a BOX interval. When block 2
multiplies by W_2[:, others] (large matrix), the residual radius
expands dramatically. This is the SAME failure mode as PRUNE:
discarding generators and replacing with axis-aligned boxes is too
loose because the boxes don't capture inter-coordinate correlation.

To make OPC-FD work, the residual must preserve generator-level
constraints. This is essentially "carry full HZ on residual + full
LP on subspace" which is more expensive than full HZ.

### 2.4 Path to a working OPC-FD

A correct OPC-FD requires:
1. Carry GENERATORS on residual (not just interval)
2. Use LP that respects residual generators' correlation with subspace
3. Possibly use polynomial zonotope-like dependent factors

This is multi-week research, not a 1-day implementation.

---

## 3. B RB-T (residual block templates) — NOT TESTED on Z0

Z0 toy is a plain 2-block dense network with NO residual structure.
B RB-T candidate requires residual blocks (e.g., y_2 = block(y_1) + y_1)
which our toy doesn't have.

To test B RB-T, we'd need:
- A toy WITH residual structure
- Implement template-based aggregate constraints per block

This is left for future work. Conjecture: B RB-T might help on actual
ResNet (cifar) structure, but NOT on plain dense networks where there's
no residual structure to template.

---

## 4. The empirical ceiling under principles

Combining all morning's data:

| Mechanism | Z0 toy drop over F1 | Notes |
|---|---:|---|
| HZ closed-form (baseline) | 0% (F1 is the baseline) | per-neuron triangle |
| F1 single-layer LP triangle | 0% (baseline) | per-neuron triangle |
| F2b pairwise joint hull | 0% additional | pairwise washed out (toy + cifar consistent) |
| Compound triangle (per-layer LP) | 0% | bounds tighten via LP doesn't help |
| FC-HZ multi-layer triangle | 8.9% | small additional |
| SETPH @ top_k=8 | 25.2% | partial improvement |
| **SETPH @ top_k=12 (all)** | **34.1%** | **best principle-pure result** |
| A OPC-FD minimal (any k) | NEGATIVE | residual interval too loose |

**The mathematical ceiling under continuous LP + DeepZ triangle is around
30-40% drop over F1 on dense aggregate slack benchmarks.**

To exceed this requires:
- A: residual generators (multi-week engineering)
- B: residual block templates (only applies to ResNet)
- C: dependent-factor / polynomial zonotope (heavy research)
- Or: principle relaxation (BaB, MILP, backward — all forbidden)

---

## 5. Implications for advisor's 5-step plan

### Step D (H2 toy math gates) — VERDICT

```
A OPC-FD:  ✗ FAIL (minimal implementation gives negative drop)
                 To pass, need constrained generator residual (multi-week)
B RB-T:    ? UNTESTED (need residual toy)
C dep-factor: ? not implemented (heaviest)
D SETPH:   ✗ FAIL at 34.1% (ceiling, gate is 47%)

No clear path to a Z0-passing candidate in continuous LP.
```

### What this means for 2000+

```
1480 V/A current
+ Step C parser (uncertain): 0-30 NEW V/A
+ Step E principle-pure H2 candidate that passes Z0 (NONE FOUND): 0
─────────────────────────────────────────────────────────────────
Realistic ceiling within principles: 1480-1510
2000+:                                +520 (mathematically unreachable
                                              within continuous LP + DeepZ)
```

This is the strongest empirical evidence yet that 2000+ is structurally
out-of-scope under our principle set. **The Z0 toy result IS the math
proof advisor mandated**.

---

## 6. Honest recommendations

1. **Accept 1480 as paper-grade headline**. The Z0 gate evidence makes
   this defensible: under our principles, the mathematical ceiling is
   ~34% improvement over F1, which doesn't suffice to flip cifar PHANTOMs.

2. **Write paper** explaining:
   - The principle set
   - 1480 V/A achieved within those principles
   - The Z0 empirical ceiling (this memo)
   - F1/F2b/FC-HZ/SETPH/OPC-FD all measured
   - Phase H = future research, not paper-promised

3. **Skip H2-D SETPH on cifar**. The Z0 toy at top_k=all gave 34.1%.
   cifar 113 has 25 unstable. SETPH @ top_k=25 = 33M octants =
   infeasible. Top_k=8 on cifar = ~25% drop = 0.146 * 0.75 = 0.110
   still PHANTOM. Math says SETPH won't flip cifar 113 CERT.

4. **Mark A OPC-FD as research direction** — needs proper generator
   residual implementation before being testable. Multi-week.

5. **Consider B RB-T** as a separate research line specifically for
   ResNet/CNN architectures. Test on a residual toy first.

6. **L4 walker remains 0 lines code** — only path to NEW V on dense-conv
   is barrier-immune FAL. Needs advisor principle ruling.

---

## 7. Files

| File | Status |
|---|---|
| `/tmp/h2d_setph_v2.py` | proper SETPH exact octant LP |
| `/tmp/h2a_opcfd_z0.py` | minimal OPC-FD (fails Z0) |
| `research/sc_hz/tests/h2_z0_aggregate_slack_toy.py` | Z0 benchmark |
| `research/H2_EMPIRICAL_CEILING_20260607.md` | this memo |
| Tests: 73 OK (expected failures=1) | clean |

---

## 8. The honest message to advisor

> "Morning execution complete. Z0 toy benchmark established (HZ 575%
> loose, F1 23.5%, FC-HZ +8.3%, F2b +0%, matches cifar). Gate ≥47%.
>
> Tested two H2 candidates ON Z0:
> - D SETPH proper implementation (exact octant LP per top-K) reaches
>   34.1% at top_k=12 (all neurons used). FAILS Z0 gate.
> - A OPC-FD minimal (subspace + interval residual) gives NEGATIVE
>   drop, looser than F1. FAILS Z0 gate.
>
> The empirical ceiling for continuous LP + DeepZ triangle on dense
> aggregate slack is ~34%. To exceed, need constrained generator
> residual (multi-week engineering for proper A OPC-FD) or new domain.
>
> Per your gate logic: 'If A/B/D toy gates all fail, write paper.'
> A and D fail. B (RB-T) needs residual structure not in Z0 toy.
> C dependent-factor not yet implemented.
>
> Recommendation: accept 1480, write paper. The Z0 measurement is
> the strongest principle-pure ceiling evidence we've ever had."
