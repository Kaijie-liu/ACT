# act/back_end/fchz_tf — FCHZ Transfer Function

ACT-integrated Forward Constrained Hybrid Zonotope transfer function for
strict P1-P5 forward neural network verification.

## Principles (strict P1-P5)

- **P1**: Forward propagation only — no backward, no CROWN
- **P2**: No gradient — no PGD, no autograd, no helper attacks
- **P3**: Open-source LP/MILP only — HiGHS/scipy or equivalent open solver;
  no Gurobi or commercial MILP solver
- **P4**: No input split — no Branch-and-Bound on input
- **P5**: No random certify — deterministic walker; ORT only for post-hoc audit

## Module structure

```
act/back_end/fchz_tf/
├── __init__.py             — exports FchzTF
├── representations.py       — FCHZState + initial_state + apply_dense + compress_g_to_tail
├── sigmoid_chord.py         — analytical Sigmoid/Tanh chord (no sampling)
├── tf_cnn.py                — CONV2D, BN, MAXPOOL2D ops
├── fchz_tf.py               — FchzTF: TransferFunction (DENSE/RELU/SIGMOID/TANH/CONV2D/BN/MAXPOOL2D)
├── tests/                   — unit tests
│   └── test_fchz_tf.py     — 10 tests (invariants + parity)
└── README.md               — this file
```

## Mathematical representation

```
R(s) = { c + G·ξ + δ :
         ξ ∈ [-1, 1]^K,
         δ_i ∈ [-tail_radius_i, +tail_radius_i] (PER-ROW INDEPENDENT)
       }

HZ closed-form upper bound:
    max d·x = d·c + Σ_k |d·G_k| + Σ_i |d_i| · tail_radius_i
```

The **per-row independence of δ** is the key soundness fix vs naive single-column
slack pool. See `TAIL_RADIUS_SOUNDNESS_PROOF.md` for full proof.

## Three soundness mechanisms

### 1. tail_radius (per-row independent box error)

Propagation rules:
- Dense (W·x + b):  `tail' = |W| @ tail`
- BN (α·x + β):     `tail' = |α| * tail` (per channel)
- Conv2D:           `tail' = conv(|W|, tail)` (treat as image)
- Add bias:         `tail` unchanged
- Mul const:        `tail' = |const| * tail`
- Residual Add:     `tail' = tail_a + tail_b` (independent box sum)
- ReLU:             `tail' = λ·tail + |μ|` where (λ, μ) are DeepZ triangle params
- Sigmoid/Tanh:     `tail' = |α|·tail + radius_chord` (per-row analytical)

### 2. sparse-slack compression (deep CNN memory)

`compress_g_to_tail(s, K_max)`: keep top-K_max G columns by L∞ norm; absorb
dropped columns into per-row tail_radius. Sound: `R(s_new) ⊇ R(s_old)`.

Enables cifar 200/200 + tinyimagenet verification at bounded memory.

### 3. Sigmoid/Tanh analytical chord

For each row [l_i, u_i]:
1. Chord through endpoints `(l_i, σ(l_i))` and `(u_i, σ(u_i))`
2. Find critical x* where `σ'(x*) = chord_slope` (analytical):
   - Sigmoid: `σ(x*) = (1 ± √(1-4α))/2`, `x* = logit(σ)`
   - Tanh:    `σ(x*) = ±√(1-α)`, `x* = atanh(σ)`
3. Compute sound radius = max(|σ(x*) - chord(x*)|, 0)
4. Recenter β at midpoint of deviation

Verified sound on 200k-sample fine grid for both Sigmoid and Tanh.

## Usage

### Register as ACT transfer function

```python
from act.back_end.transfer_functions import set_transfer_function
from act.back_end.fchz_tf import FchzTF

# Register FchzTF
tf = FchzTF(G_max_cols=128)  # sparse-slack for deep CNN
set_transfer_function(tf)

# All subsequent ACT verifications use FchzTF
```

### With ACT's `verify_once`

```python
from act.back_end.verifier import verify_once
# Net is built from ONNX via standard ACT pipeline:
#   ONNX → convert_onnx_to_pytorch → VerifiableModel → TorchToACT → Net
result = verify_once(net)
```

### Standalone (lower-level)

```python
from act.back_end.fchz_tf import FchzTF
from act.back_end.core import Bounds

tf = FchzTF()
fact = tf.apply(layer, input_bounds, net, before, after)
# fact.bounds: output bounds
# fact.cons:   constraints (empty for FCHZ; ConSet via interval_tf if combined)
```

## Supported layer kinds (current)

| LayerKind | Mechanism |
|-----------|-----------|
| INPUT, INPUT_SPEC, ASSERT | passthrough |
| DENSE | `c' = Wc + b`, `G' = WG`, `tail' = |W| tail` |
| BIAS | bias-only Dense |
| SCALE | per-element scale |
| RELU | DeepZ triangle + tail_radius update |
| SIGMOID | analytical chord (see above) |
| TANH | analytical chord (see above) |
| CONV2D | PyTorch F.conv2d (CPU; GPU via HYZOR_FCHZ_USE_CUDA=1) |
| BN | per-channel `c' = αc + β` |
| MAXPOOL2D | sound box relaxation (G linkage destroyed) |

Unsupported layers fall back to interval bounds (sound but loose).

## Testing

```bash
cd /data1/Kane/ACT
python -m unittest act.back_end.fchz_tf.tests.test_fchz_tf
```

Current: 10 tests, all PASS. Tests cover:
- FCHZState invariants (sampling-based check)
- Dense propagation correctness
- tail_radius preservation in linear ops
- compress_g_to_tail soundness (UB monotone)
- Sigmoid/Tanh chord soundness (200k fine grid)
- TransferFunction interface compliance
- End-to-end MLP via ACT pipeline
- **Parity vs raw walker (bit-identical)**

## Provenance

This module is a clean refactor of `research/sc_hz/fchz_walker.py` and
`research/sc_hz/fc_hz_state.py` to integrate with ACT's `TransferFunction`
interface. Mathematical correctness is preserved (parity test confirms
bit-identical results to the raw walker).

Original walker remains in `research/sc_hz/` as a reference implementation.

## Future work

- [ ] CONVTRANSPOSE2D handler
- [ ] AVGPOOL2D + ADAPTIVEAVGPOOL2D handlers
- [ ] ADD/MUL/SUB layer kinds (residuals, broadcasts)
- [ ] Optional Gurobi-free F1_LP refinement layer
- [ ] Real cifar/tinyimagenet ONNX → ACT pipeline → FchzTF verify_once full test
