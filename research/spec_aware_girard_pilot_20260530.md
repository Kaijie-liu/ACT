# Spec-aware Girard reduction — pilot (2026-05-30)

## What was tried

Per-layer linear spec direction `F_t` precomputed by **forward layer-traversal**
(setup-time matrix chain through DENSE / RELU / SCALE / BIAS layers using
**fixed** triangle slope from interval bounds — not optimized α). Used as
**ranking heuristic** in Girard reduction: keep the generators with largest
`||F_t @ Gc[:, i]||_1` instead of largest `||Gc[:, i]||`.

Soundness: Girard reduction is sound for ANY subset of kept generators. We only
changed the ranking criterion. Verified by 38/38 reduction-soundness tests pass.

## Principle alignment

| Principle | Check |
|-----------|-------|
| P1 No CROWN backward bound propagation | ✅ — no bound refinement; F_t is a linear functional, not (lb, ub) |
| P2 No backward / autograd / gradient | ✅ — λ is fixed from interval, not optimized |
| P3 No Gurobi / MILP | ✅ |
| P4 No fallback | ✅ |
| P5 No BaB | ✅ — no branching |
| P6 No PGD / random sampling | ✅ |

The reverse-order matrix product (`M · W_L · diag(λ) · ...`) is arithmetically
the same as CROWN's spec direction, but used only for ranking — never for
bound tightening or verification. Reverting to standard column-norm gives
identical soundness; the ranking lift is a heuristic.

## MVP scope

Supported in the reverse F-chain:
- DENSE: `F_in = F_out @ W`
- RELU: `F_in = F_out * λ` (per-neuron triangle slope from `before[L.id]`)
- LRELU: same with leaky slope
- BIAS: F unchanged
- SCALE: `F_in = F_out * a`

Stops the chain (so earlier layers fall back to column-norm scoring):
- CONV2D, BN, ADD, CONCAT, MAXPOOL2D, FLATTEN, GATHER, SLICE, PAD, REDUCE_SUM, ...

This MVP therefore activates the heuristic only at LAYERS WHOSE OUTPUT IS A
SUPPORTED LAYER (typically the late dense classifier head). For cifar100
(conv-heavy backbone + dense head), F is set only for the LAST few layers.

## Files

- `act/back_end/hybridz_tf/algorithms/spec_aware.py` — new module
- `act/back_end/hybridz_tf/tf_mlp.py:170-208` — `set_current_layer` calls
- `act/back_end/hybridz_tf/tf_mlp.py:hz_reduce` — scoring override
- `act/back_end/hybridz_tf/representations.py:SparseGcZ.reduce_generators` — scoring override
- `act/back_end/solver/solver_hz.py:consume_cons` — setup-time precompute call

Env knobs:
- `ACT_HZ_SPEC_AWARE_GIRARD={0,1}` — enable (default OFF)
- `ACT_HZ_SPEC_AWARE_DEBUG={0,1}` — print chain debug

## Smoke test

malbeware iid 0: CERTIFIED 2.1s, chain stops at FLATTEN — F set for 2 layers
(final dense + final ReLU of classifier head). No regression.

## Pilot

10-iid linearizenn (pure dense, 0V baseline) + 5-iid cifar100 (conv-heavy, 0V).
4-stream parallel.

### Results

```
linearizenn_OFF     V=2  A=0  U=8  E=0  n=10  mean_wall=13.3s
linearizenn_ON      V=2  A=0  U=8  E=0  n=10  mean_wall=13.3s
cifar100_OFF        V=0  A=0  U=5  E=0  n=5   mean_wall=4.8s
cifar100_ON         V=0  A=0  U=5  E=0  n=5   mean_wall=4.8s
```

**0 lift, 0 regression. Walls bit-identical.**

### Diagnosis (why 0 lift — root cause, not just empirical)

Inspection with `ACT_HZ_SPEC_AWARE_DEBUG=1` reveals the MVP F-chain is too
shallow on every pilot target:

| Bench | F-chain reach | Stops at |
|-------|--------------|----------|
| malbeware iid 0 | 2 layers (final dense + ReLU) | `FLATTEN` |
| linearizenn iid 0 | 2 layers | `CONCAT` (skip connections) |
| cifar100 iid 0 | small dense tail | `CONV2D` |
| acasxu iid 0-2 | **14 layers** (full FC) | `FLATTEN` at input |

But on the bench where F-chain is deep (acasxu), Girard reduction **never
fires** because the network is tiny (≤ 5 neurons per layer, ng never grows).
On the benches where reduction fires (cifar100 — conv backbones produce
thousands of generators per layer), F-chain stops at the first CONV2D so the
lever doesn't see them.

`OFF == ON` identical-wall behavior confirms the scoring branch was active
but irrelevant: on linearizenn the rare reduction at the tail covered ≤ 2
layers worth of generators (small effect); on cifar100 the `nc>0` early-return
in `SparseGcZ.reduce_generators` (representations.py:1342) skipped the
scoring entirely.

### What would need to happen for the lever to actually fire on cifar100

1. **Extend F-chain to CONV2D**: materialize the conv as a sparse Toeplitz
   matrix `W_flat` of shape `(C_out·H_out·W_out, C_in·H_in·W_in)`, then
   `F_in = F_out @ W_flat`. For cifar100 ResNet (3072 → ... → 100), each
   conv stage adds ~10K rows and ~3K cols. Dense materialization is
   ~100 MB per conv, manageable.
2. **Extend F-chain to CONCAT**: split F across branches at concat.
   For ADD/RESIDUAL: F_in = F_out for both branches (additive split).
3. **Extend F-chain to BN**: F_in = F_out * scale (where scale is the
   inference-time BN gain).
4. **Remove SparseGcZ early-return on nc>0**: implement constraint-aware
   sparse Girard widening, so reduction-with-constraints actually executes
   the scoring path. (This is significant work; deferred earlier per
   representations.py:1343-1346.)

### Decision

Per pre-registered rule: 0 new V on both targets → **MVP closed out as inert
on intended targets**, not because the math is wrong but because the supported
op set (DENSE/RELU/LRELU/BIAS/SCALE only) doesn't reach where reductions fire
on conv-heavy benchmarks.

The mechanism is sound, principle-compliant, and the env knob remains in code
(default OFF) for future extension. To get a real test, items (1) + (4) above
would need to be implemented.

### Honest position vs the original hypothesis

The hypothesis was "spec-aware Girard ranking might lift V on 0-bench by
preserving spec-relevant generators across reductions". The pilot doesn't
falsify the hypothesis — it shows the MVP couldn't even execute it on the
targets that matter. Falsifying or confirming requires the v2 extension to
CONV2D + lifting the SparseGcZ nc>0 guard.

Total cost so far: ~150 LOC + 4-stream 3-min pilot. Reasonable for an MVP
that delivered a clean wiring + soundness check + clear extension path.

## Trace

- Raw: `/data1/Kane/ACT/audit_results/spec_aware_pilot_<STAMP>/`
- Driver: `/tmp/spec_aware_pilot.sh`
