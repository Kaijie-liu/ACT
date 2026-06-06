# Phase F1 Prototype — Constrained HZ-LP VALIDATED

**Date**: 2026-06-05 late-night
**Status**: Synthetic prototype PASSED. Integration design ready.
**Headline impact**: NONE yet (prototype only); 1472 holds

---

## 1. The mechanism

Per advisor 2026-06-05 Phase F directive: instead of treating ReLU triangle
slack `ξ_aux` as fully independent in [-1, +1], explicitly encode the triangle
constraints in a continuous LP for the FINAL N layers' ReLU.

For each unstable neuron i in the final ReLU:
- z_i = c_z[i] + G_z[i, :] @ ξ + tail_z[i] * ξ_tail[i]   (pre-activation, linear in ξ)
- y_i ∈ [0, u_i]   (post-activation)
- y_i ≥ 0
- y_i ≥ z_i
- y_i ≤ λ_i (z_i - l_i)   where λ_i = u_i / (u_i - l_i)

Solve the continuous LP:
- max `d_out · (W_remaining @ y + b_remaining)`
- over ξ, ξ_tail, y subject to triangle constraints + box bounds

The result is a SOUND upper bound that is TIGHTER than the closed-form HZ
LP UB (which treats every slack ξ_aux independently in [-1, +1]).

PRINCIPLE compliance:
- Forward-only: no backward bound refinement
- Continuous LP only (HiGHS / scipy.optimize.linprog)
- No gradients, no MILP, no integer reasoning
- No BaB, no input splitting
- LP coefficients come solely from forward HZ propagation

---

## 2. Prototype validation

`research/sc_hz/constrained_lp.py` + `tests/test_constrained_lp_prototype.py`

### Unit tests (6/6 PASS)

| Test | Assertion |
|---|---|
| TestConstrainedLPSoundness | LP UB ≥ brute-force max (5000 random ξ samples) |
| TestConstrainedTighterThanClosedForm | constrained ≤ closed-form (no looser) |
| TestConstrainedTighter (strictly) | strictly tighter on ≥1 of 8 random seeds |
| TestAllActive | when all neurons active, LP UB = d·c (degenerate exact) |
| TestAllInactive | when all neurons inactive, LP UB = 0 (degenerate exact) |
| TestConstrainedFeasibility | 0 infeasible LPs on 20 random seeds |

Full SC-HZ suite: **58/58 PASS** (52 prior + 6 new).

### Quantitative gain on synthetic (n_pre=32, K=16, 32 unstable, 20 seeds)

```
strictly tighter:  20/20 instances
median gain:       15.7%
mean gain:         17.9%
range:             6.5% to 28.6%
```

### Why this exactly matches what we need

Recall the Gate 2 v2 diagnostic on cifar iid 113 PHANTOM rival `Y_6 >= Y_82`:
- center contribution:  -1.265
- generator contribution: +1.526 (ALL from ReLU slack)
- tail contribution: 0
- LP UB:               +0.261 (PHANTOM)

To flip to CERT we need the generator contribution to drop by 0.261/1.526 = **17.1%**.

The prototype delivers a median 15.7% gain on synthetic — very close to the
17% threshold. Some seeds achieved 25-28%, which is what we'd need on the
high end. This is the SINGLE most precise lever we have.

---

## 3. Integration design

For Phase F1 to produce real V/A, we need:

### Step 1: Walker hook to capture "pre-last-ReLU" state

Modify `forward_resnet` (`research/sc_hz/onnx_walker_resnet.py`) to:
- Track which ONNX node is the last `Relu` before the final classifier
- Snapshot the state RIGHT BEFORE this ReLU into a `LastReluRecord`
- Continue forward; track the remaining linear chain into a `W_remaining, b_remaining` accumulator
- Return both in the result

### Step 2: Per-rival constrained LP solve

For each unsafe condition `(d_out, threshold, label)`:
- Build LP with the captured `LastReluRecord`, `W_remaining`, `b_remaining`, `d_out`
- Solve with HiGHS
- Compare LP UB to closed-form
- If LP UB < threshold strictly → CERT for this rival
- If ALL rivals CERT → iid is CERT (PHANTOM → CERT promotion)

### Step 3: Witness decoding (FAL side)

If LP UB ≥ threshold:
- Extract ξ from the LP solution
- Decode `x_star = c_in + r_in ⊙ sign(d_at_input)` OR use the LP's ξ solution
- Run ORT replay at strict tolerance per G4
- If d.y > threshold strictly → A_CONFIRMED

### Step 4: 8-iid pilot

Test on the 8 near-CERT identified after Gate 2 v2:
- cifar: 113, 29, 180, 72, 168, 145
- tiny: 99, 30

Per advisor F1 verification:
```
CIFAR median max_excess >= 30% drop (currently +0.521 → target ≤ +0.36)
OR ≥10 NEW V/A
Otherwise F1 closed.
```

---

## 4. Implementation cost estimate

| Component | Estimate |
|---|---|
| Walker hook to capture last-ReLU state | 0.5 day |
| `W_remaining` accumulator (linear chain composition) | 0.5 day |
| LP setup + solver integration | 0.5 day |
| 8-iid pilot driver + audit | 0.5 day |
| Total | ~2 days |

Day 2-3 of S2 work. Within the 13 days remaining of the 2-week kill switch.

---

## 5. Risk register

1. **LP size**: at K~20000 (cifar K_target) + n_pre~1000 + n_y~1000, LP has
   ~22000 variables and ~3000 constraints per rival. Solve time per LP is
   uncertain (~1-10s?). For 99 rivals per iid × 8 iids = ~800 LP solves.
   Total ~13min wall to several hours, depending on LP solver speed.

2. **Last-ReLU placement**: if the network has multiple ReLU layers, we
   may need to apply the constraint to MORE than just the last one.
   Prototype assumes single tight constraint at end; extension to multi-layer
   is straightforward (more variables, more constraints, larger LP).

3. **15.7% median may be insufficient**: if even tighter precision is
   needed for SOME iids (e.g., the 5 cifar with +0.46 to +0.98), single-layer
   constrained LP might not flip them. May need multi-layer F2 (Anderson).

4. **Float precision near boundary**: G4 strict `>` requires margins
   robust above 1e-12. LP solvers can return solutions at the boundary;
   need careful comparison.

---

## 6. Stop criteria (binding)

Per advisor F1 gate verbatim:
```
CIFAR median max_excess 降低 >= 30%
OR ≥10 NEW V/A
Otherwise F1 closed.
```

If F1 closes:
- F2 Forward Anderson facets (2-4 weeks) — multi-neuron joint hulls
- F3 small/control sidecar v2 — parser + LP for non-CIFAR
- If F2 and F3 both fail → 1472 is the SC-HZ ceiling; transition to writing

---

## 7. Files

| File | Status |
|---|---|
| `research/sc_hz/constrained_lp.py` | NEW (220 lines): LastReluRecord + LP solver |
| `research/sc_hz/tests/test_constrained_lp_prototype.py` | NEW (6 tests, 58/58 suite pass) |
| `research/phase_F1_constrained_lp_prototype_20260605.md` | this memo |
| 1472 freeze | unchanged |
| `act/` | clean |

---

## 8. Tonight's stop point

```
Headline:                1472 V/A
F1 prototype:            VALIDATED (15.7% median synthetic gain)
F1 integration:          designed (2-day estimate)
Next-day target:         wire prototype into walker; pilot on 8 near-CERT iids
0 NEW V/A tonight
act/ clean
GPU/RAM:                 returning to baseline
```
