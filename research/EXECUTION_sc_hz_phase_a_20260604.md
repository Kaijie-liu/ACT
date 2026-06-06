lkjPxj478739690# SC-HZ Phase A — Execution Playbook

**Date created**: 2026-06-04 night
**Purpose**: ordered, command-level execution sequence for Phase A. Each step has a **goal**, **commands**, **expected output**, and **acceptance check**. This document is the operational complement to the design lock (`dc_hz_phase_a_plan.md`) and the brief (`INNOVATION_BRIEF_sc_hz_20260604.md`).

**This is not a discussion document.** Each numbered step is to be executed in order; do not skip. If any step's acceptance check fails, stop and resolve before proceeding — do not "patch around" failures.

---

## 0. Pre-flight (1 hour total)

### 0.1 Confirm baseline 924 V/A is intact

Goal: prove the Phase A work has not yet touched any production code.

```bash
cd /data1/Kane/ACT
git status -uall | head -40
git log --oneline -5
```

**Expected**: no uncommitted changes to `act/`, only changes (if any) under `research/`. `git log` shows recent commits unrelated to SC-HZ implementation.

**Acceptance check**: `act/pipeline/cli.py`, `act/back_end/hybridz_tf/*`, `act/pipeline/verification/torch2act.py` are all unmodified vs the 924 V/A baseline commit.

If anything in `act/` is dirty, STOP. The Phase A work must not depend on production-side mutations.

Note: the working tree does **not** need to be globally clean. This project
often has research notes and audit artifacts in flight. The hard condition is
"no production `act/` mutation", not "no dirty files anywhere".

### 0.2 Confirm conda env

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python --version
/data1/Kane/miniconda3/envs/act-py312/bin/python -c "import torch, highspy, onnx, onnxruntime, numpy, scipy; print('OK')"
```

**Acceptance check**: prints `OK`. Any ImportError aborts here — fix env first.

### 0.3 Confirm sentinel JSON exists

```bash
ls -la /data1/Kane/ACT/audit_results/sc_hz_phase_a_sentinels_20260604.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -c "
import json
d = json.load(open('/data1/Kane/ACT/audit_results/sc_hz_phase_a_sentinels_20260604.json'))
for b in ['cifar100_2024', 'safenlp_2024', 'acasxu_2023', 'tinyimagenet_2024']:
    iids = d[b]['iids']
    print(f'{b}: {len(iids)} iids — {iids[:5]}...')
"
```

**Expected**: each benchmark prints exactly 20 iids.

**Acceptance check**: 80 total. If any benchmark has <20, regenerate via the script in §11 below.

### 0.4 Create the `research/sc_hz/` skeleton

```bash
mkdir -p /data1/Kane/ACT/research/sc_hz/tests
touch /data1/Kane/ACT/research/sc_hz/__init__.py
touch /data1/Kane/ACT/research/sc_hz/tests/__init__.py
```

**Acceptance check**: directory exists, empty package.

---

## 1. TDD entry: write unit tests BEFORE implementation (1 day)

Per design lock §7, the unit tests are the entry gate. Implementation only begins after tests are written (they will fail until implementation lands — that is expected).

### 1.1 Write `tests/test_prune_soundness.py`

Goal: pin the core soundness contract — PRUNE must over-approximate the original HZ for ANY value of `d_L`.

Required content (sketch):

```python
"""PRUNE soundness regression: brute-force containment.

Covers I6 of dc_hz_phase_a_plan.md. Must FAIL on the single-column
r_tail[:, None] (unsound) construction. Must PASS on the independent
interval-tail / sparse-diagonal construction.
"""
import unittest, numpy as np
from research.sc_hz.prune import prune

class TestPruneSoundness(unittest.TestCase):
    def _sample_xi(self, ng, n_samples=1000, seed=20260604):
        rng = np.random.default_rng(seed)
        return rng.uniform(-1, 1, size=(n_samples, ng))

    def test_pruned_contains_original_on_random_d(self):
        rng = np.random.default_rng(20260604)
        n, ng, K = 8, 20, 6
        c = rng.normal(size=n)
        G = rng.normal(size=(n, ng))
        for d_label, d in [
            ('zero',  np.zeros(n)),
            ('rand',  rng.normal(size=n)),
            ('flip',  -rng.normal(size=n)),
            ('orth',  self._make_orthogonal_to_max_norm_col(G)),
        ]:
            state_p = prune(c, G, d, K, return_metadata=True)
            xi_orig = self._sample_xi(ng)
            for xi in xi_orig:
                p_orig = c + G @ xi
                # Construct a feasible pruned coefficient directly:
                # xi_keep = xi[keep],
                # xi_tail_i = dropped_contribution_i / r_tail_i if r_tail_i > 0.
                # This checks containment without turning the unit test into
                # another LP solver problem.
                ok = self._is_contained_by_metadata(p_orig, state_p, xi)
                self.assertTrue(ok, msg=f'd={d_label}: original point not in pruned set')
    # ... etc
```

**Acceptance check**: test file exists and is syntactically valid. Run will fail (expected — `prune` not yet implemented).
The final committed test must implement all helpers shown in the sketch
(`_make_orthogonal_to_max_norm_col`, `_is_contained_by_metadata`). Do not leave
ellipsis / placeholder helpers in the test file.

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest -v research.sc_hz.tests.test_prune_soundness 2>&1 | tail -5
```

**Expected output**: `ImportError` / `ModuleNotFoundError` for the missing
research module. Do not proceed if it errors for any other reason (e.g. syntax
error in the test).

### 1.2 Write `tests/test_direction_chain.py`

Goal: pin the `d_L^r` computation correctness.

```python
"""d_L^r chain on a synthetic 4-layer linear-only net."""
import unittest, numpy as np
from research.sc_hz.precompute_direction import precompute_d_per_layer

class TestDirectionChain(unittest.TestCase):
    def test_dchain_on_4layer_linear(self):
        rng = np.random.default_rng(20260604)
        W1 = rng.normal(size=(6, 4))
        W2 = rng.normal(size=(3, 6))
        W3 = rng.normal(size=(2, 3))
        weights = [W1, W2, W3]
        y_true, rival = 0, 1
        d_per_layer = precompute_d_per_layer(weights, rival, y_true)
        # check d_3 (output direction)
        np.testing.assert_allclose(d_per_layer[3], W3[rival] - W3[y_true])
        # check d_2 = W3^T (W3[rival]-W3[y_true])
        np.testing.assert_allclose(d_per_layer[2], W3.T @ (W3[rival]-W3[y_true]))
        # check d_1 = W2^T d_2
        np.testing.assert_allclose(d_per_layer[1], W2.T @ d_per_layer[2])
        # check d_0 = W1^T d_1
        np.testing.assert_allclose(d_per_layer[0], W1.T @ d_per_layer[1])
```

**Acceptance check**: file syntactically valid; `import precompute_d_per_layer` failure expected.

### 1.3 Write `tests/test_forward_parity.py`

Goal: pin K=ng equivalence to baseline (no pruning ⇒ no change).

```python
"""When K=ng, PRUNE is the identity; output LP UB equals baseline."""
import unittest, numpy as np
from research.sc_hz.prune import prune

class TestForwardParity(unittest.TestCase):
    def test_prune_at_K_equals_ng_is_identity(self):
        rng = np.random.default_rng(20260604)
        n, ng = 8, 12
        c = rng.normal(size=n)
        G = rng.normal(size=(n, ng))
        d = rng.normal(size=n)
        state_p = prune(c, G, d, K=ng, return_metadata=True)
        np.testing.assert_allclose(state_p.c, c)
        np.testing.assert_allclose(state_p.G_kept, G)  # exactly identical
        self.assertEqual(state_p.metadata["drop"].size, 0)
        self.assertTrue(state_p.tail_radius is None or np.all(state_p.tail_radius == 0))
```

### 1.4 Write `tests/test_adversarial_d_soundness.py`

Goal: I10 observable. PRUNE must be sound for ALL d_L choices.

```python
"""I10: soundness independent of d_L choice.

Per design lock §1.4, PRUNE must over-approximate for any d_L,
including (a) all-zero, (b) random, (c) sign-flipped, (d) orthogonal to true.
"""
import unittest, numpy as np
from research.sc_hz.prune import prune

class TestAdversarialDSoundness(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260604)
        self.n, self.ng, self.K = 8, 20, 6
        self.c = rng.normal(size=self.n)
        self.G = rng.normal(size=(self.n, self.ng))
        self.xi_samples = rng.uniform(-1, 1, size=(1000, self.ng))

    def _check_containment(self, d, label):
        state_p = prune(self.c, self.G, d, self.K, return_metadata=True)
        for xi in self.xi_samples:
            p = self.c + self.G @ xi
            self.assertTrue(self._is_contained_by_metadata(p, state_p, xi),
                            msg=f'd={label}: original point not in pruned set')

    def test_d_zero(self):
        self._check_containment(np.zeros(self.n), 'zero')
    def test_d_random(self):
        rng = np.random.default_rng(20260605)
        self._check_containment(rng.normal(size=self.n), 'random')
    def test_d_sign_flipped(self):
        rng = np.random.default_rng(20260606)
        d_true = rng.normal(size=self.n)
        self._check_containment(-d_true, 'sign_flipped')
    def test_d_orthogonal_to_max_col(self):
        col_norms = np.linalg.norm(self.G, axis=0)
        m = int(np.argmax(col_norms))
        g = self.G[:, m]
        rng = np.random.default_rng(20260607)
        v = rng.normal(size=self.n)
        d = v - (v @ g) / max(g @ g, 1e-12) * g
        self._check_containment(d, 'orth_to_max_col')
```

### 1.5 Write `tests/test_relevance_score_ablations.py`

Goal: signal sanity — the chosen relevance score must beat trivial baselines
on constructed states before we trust it on benchmarks. This is a diagnostic
gate, not the soundness gate.

```python
"""Signal sanity for relevance scoring.

Soundness must not depend on the score. This test only checks that the score
has measurable value over random/norm baselines on constructed states where
the relevant columns are known.
"""
import unittest, numpy as np
from research.sc_hz.prune import prune
from scipy.optimize import linprog

class TestRelevanceScoreAblations(unittest.TestCase):
    def test_true_d_beats_random_and_norm_on_constructed_shapes(self):
        rng = np.random.default_rng(20260604)
        # Construct G so a small set of columns is clearly relevant to d while
        # several larger-norm columns are orthogonal distractors.
        # Compare LP UB after keep-K by: true d, norm, random, sign-flipped d.
        # Require true-d UB <= every baseline UB + tolerance.
```

### 1.6 Verify tests fail with the right errors

```bash
cd /data1/Kane/ACT
for t in test_prune_soundness test_direction_chain test_forward_parity \
         test_adversarial_d_soundness test_relevance_score_ablations; do
  echo "==$t=="
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest -v research.sc_hz.tests.$t 2>&1 | tail -5
done
```

**Expected output**: every test fails with `ImportError` / `ModuleNotFoundError`
(modules not yet implemented). NO test should fail with a syntax error.

**Acceptance check**: 5 missing-module failures, 0 syntax errors. Proceed to §2.

---

## 2. Implement core modules (3 days)

Per design lock §6, the 4 modules: `precompute_direction.py`, `prune.py`, `pruned_forward.py`, `run_sentinels.py`. Implement in TDD order: write code until each test in §1 passes.

### 2.1 Implement `research/sc_hz/prune.py`

Goal: `prune(c, G, d, K) -> (c_pruned, G_pruned)` that satisfies all soundness tests.

Implementation requirements (from design lock §1.3):

```python
def prune(c, G, d, K):
    """
    Args:
      c: (n,) center
      G: (n, ng) continuous generator matrix
      d: (n,) rival direction (heuristic only — soundness independent of d)
      K: int — generator budget

    Returns:
      PrunedState(c, G_kept, tail_radius, metadata), where:
        G_kept: (n, K) retained explicit generators
        tail_radius: (n,) independent interval remainder
        metadata: keep/drop indices for tests and receipts

    MUST:
      - If ng <= K: return (c, G) unchanged (no tail).
      - Else: keep top-K columns by |d @ G[:, j]|, build tail r_tail
        = abs(G[:, drop]).sum(axis=1), store it as independent interval tail
        — NOT r_tail[:, None].
    """
```

**Implementation note**: do **not** implement Phase A with a dense
`diag(r_tail)` fallback. That would make the dense-conv targets
(`tinyimagenet`, `cifar`) unusable and would hide the real memory question.
Represent the tail as sidecar metadata from day one. Toy tests may call a
`materialize_dense_for_test()` helper, but production sentinel code must keep
the interval tail symbolic/sparse.

**Acceptance check**: all PRUNE/relevance tests pass:

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest -v \
    research.sc_hz.tests.test_prune_soundness \
    research.sc_hz.tests.test_forward_parity \
    research.sc_hz.tests.test_adversarial_d_soundness \
    research.sc_hz.tests.test_relevance_score_ablations 2>&1 | tail -15
```

**Expected**: PRUNE soundness/parity/adversarial-d tests pass, and the
relevance ablation diagnostic passes on constructed states. **STOP if any
soundness/parity test fails — debug PRUNE before continuing.**

### 2.2 Implement `research/sc_hz/precompute_direction.py`

Goal: `precompute_d_per_layer(weights, rival, y_true) -> [d_0, d_1, ..., d_N]` per §1.1 of design lock.

```python
def precompute_d_per_layer(weights, rival, y_true):
    """
    weights: list of weight tensors [W_1, W_2, ..., W_{N+1}] in forward order
    Returns: list [d_0, d_1, ..., d_N] of len(weights) entries
    """
```

For Conv2D layers, the "transpose" is `conv_transpose2d` (adjoint of conv). For Dense, plain matrix transpose. The implementation should handle a list of generic "linear operators" each with a `.adjoint(x)` method.

Phase A scope: Dense, Conv2D, AvgPool2d, Flatten/Reshape, Add, and Concat.
ADD sends the same cotangent to both parents; Concat slices the cotangent by
the forward output ranges. MaxPool2d is NOT in Phase A scope unless the forward
interval pass proves a stable max index; otherwise raise `NotImplementedError`
and mark the iid unsupported/fail-closed. This keeps tiny/cifar residual paths
testable without pretending MaxPool has a linear adjoint.

**Acceptance check**:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest -v \
    research.sc_hz.tests.test_direction_chain 2>&1 | tail -5
```

**Expected**: pass. STOP if fails.

### 2.3 Implement `research/sc_hz/pruned_forward.py`

Goal: per-rival forward propagation that calls existing HZ ops and inserts PRUNE between layers.

This is the largest module. It must:
1. Take an initial input HZ (from VNNLIB box parsing).
2. For each layer L = 1..N:
   - Call the existing HZ Conv2D / Dense / ReLU triangle / etc.
   - Call `prune(c, G, d_L, K)` after the layer op.
   - Propagate the interval-tail sidecar soundly. For a linear op, either keep
     the tail as a sparse linear image or over-approximate by
     `new_tail_radius = abs(W) @ tail_radius`; never materialize a dense
     conv-sized diagonal unless the layer dimension is toy-sized.
3. At output, solve the per-rival LP `max d_N · y - threshold`.
4. Return `(verdict, lp_ub, xi_star_if_fal)`.

**Phase A scope**: reuse `act/back_end/hybridz_tf` HZ ops by importing them as library functions. Do NOT modify them.

If a model uses an operator not in {Conv2D, Dense, ReLU-triangle, AvgPool,
Flatten/Reshape, Add, Concat}, raise `UnsupportedModelError` and the iid stays
UNKNOWN in the receipt. No silent fallback.

If the LP optimizer returns a violating solution, classify it before replay:
- If the solution depends only on original root variables and retained
  decodable generators, decode to input and run strict ORT replay.
- If the violation requires interval-tail variables or non-root aux variables,
  record `phantom_lp_sat += 1` and return UNKNOWN for that rival. Do not count
  it as FAL.

**Acceptance check**: write one more test:

```bash
# In research/sc_hz/tests/test_pruned_forward.py:
# Test on a synthetic 2-layer Dense+ReLU+Dense model with known answer.
# Compare to a brute-force enumeration on a tiny input box.

/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest -v \
    research.sc_hz.tests.test_pruned_forward 2>&1 | tail -5
```

**Expected**: pass.

### 2.4 Implement `research/sc_hz/run_sentinels.py`

Goal: driver that runs 80 sentinels and emits per-iid receipts.

Required CLI:

```bash
python research/sc_hz/run_sentinels.py \
    --sentinels audit_results/sc_hz_phase_a_sentinels_20260604.json \
    --K 256 \
    --out audit_results/sc_hz_phase_a_<STAMP> \
    --wall-per-iid-s 600 \
    --rss-cap-gb 40 \
    --workers 2 \
    [--bench cifar100_2024]  # optional: restrict to one benchmark
```

Per-iid actions:
1. Load model + vnnlib via the existing `canonical_provenance.py` loader.
2. Read the `_confirmed` sentinel file from §3 and assert the iid is marked
   production-UNK under the matching comparison budget. Do not silently run on
   an unconfirmed iid.
3. If UNK confirmed: precompute `d_L^r` for all rivals.
4. For each rival r: run pruned_forward with K and PRUNE per layer.
5. Aggregate verdicts: CERT only if all rivals' LP UB < 0 and CERT audit later
   confirms the independent LP result. If any rival LP UB ≥ 0, attempt FAL only
   when the optimizer solution is decodable to original input variables; strict
   ORT replay must then confirm the spec violation. Tail-dependent LP SAT is
   recorded as `phantom_lp_sat` and remains UNKNOWN.
6. Emit receipt with provenance bundle + the three counters (`new_cert`, `new_fal_strict_replay`, `phantom_lp_sat`).

**Acceptance check**:

```bash
# Dry run with 1 iid on safenlp (no expensive forward yet):
python research/sc_hz/run_sentinels.py \
    --sentinels audit_results/sc_hz_phase_a_sentinels_20260604.json \
    --K 256 --out /tmp/sc_hz_dryrun_$(date -u +%Y%m%dT%H%M%SZ) \
    --bench safenlp_2024 --max-iids 1 --wall-per-iid-s 60
```

**Expected**: produces a per-iid JSON receipt; no crashes; either a verdict or a clear fail-closed reason.

---

## 3. Pre-flight: verify sentinels are actually UNK under production (4-8 hours)

Goal: confirm the 80 sentinel iids are currently UNKNOWN under the **same
production comparison budget** used for the Phase A gate; substitute any that
are already V/A. A short 60s run is not a canonical baseline and must not be
used to claim lift.

This is the §5 pre-flight step from the design lock. Without it, Phase A might "lift" iids that production already decides — false success.

### 3.1 Run production on each sentinel with the comparison budget

First try to read the current canonical / 924-sweep per-instance CSV from
`audit_results/`. If the iid already has a production V/A there, substitute it
without rerunning. Only run production for missing or stale rows.

```bash
cd /data1/Kane/ACT
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=audit_results/sc_hz_phase_a_preflight_${STAMP}
mkdir -p $OUT

/data1/Kane/miniconda3/envs/act-py312/bin/python research/sc_hz/preflight_confirm_unk.py \
    --sentinels audit_results/sc_hz_phase_a_sentinels_20260604.json \
    --canonical-production-root audit_results \
    --out audit_results/sc_hz_phase_a_sentinels_confirmed_20260604.json \
    --audit-out $OUT \
    --wall-s 600 \
    --workers 2 \
    --rss-cap-gb 32 \
    --canonical-root /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
```

`preflight_confirm_unk.py` must:
1. Prefer existing canonical production CSV rows from `audit_results/`.
2. Rerun production only for missing/stale rows, with `--wall-s` equal to the
   comparison budget.
3. Substitute any iid that production already resolves to V/A.
4. Emit `confirmed_iids`, `substitutions`, `production_verdict_source`, and
   `production_wall_s` per benchmark.

**Expected**: per benchmark, 20 UNK iids confirmed. Time budget depends on how
many rows are missing from the canonical production CSV. If all 80 need rerun,
run with bounded parallelism (2-4 jobs depending on GPU memory), not blind
80-way parallelism.

**Acceptance check**: each benchmark in the output JSON has `confirmed_iids` of length 20. If <20 (because too many iids resolved as V/A under production), expand the candidate pool until 20 UNK are found. Document the substitutions.

### 3.2 Write the confirmed sentinel JSON

After §3.1 completes, write:

```text
audit_results/sc_hz_phase_a_sentinels_confirmed_20260604.json
```

with the verified-UNK iids. From this point on, the driver uses the `_confirmed` file.

---

## 4. Phase A sentinel sweep (1 day wall-clock)

Run order is deliberately conservative:
1. One serial smoke iid per positive benchmark.
2. If memory is stable, 2-way parallel.
3. If peak GPU memory remains below 70 GB on the 96 GB GPU, 3-4-way parallel.
Do not start with 80-way or benchmark-wide parallelism; Phase A has new
per-rival state and interval-tail metadata, so memory must be measured first.

### 4.1 Run the full 80 sentinels

```bash
cd /data1/Kane/ACT
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=audit_results/sc_hz_phase_a_${STAMP}
export OUT
mkdir -p "$OUT"

nohup /data1/Kane/miniconda3/envs/act-py312/bin/python research/sc_hz/run_sentinels.py \
    --sentinels audit_results/sc_hz_phase_a_sentinels_confirmed_20260604.json \
    --K 256 \
    --out $OUT \
    --wall-per-iid-s 600 \
    --rss-cap-gb 40 \
    --workers 2 \
    > $OUT/run.log 2>&1 &

echo "PID: $!"
echo "OUT: $OUT"
```

**Expected wall**: initial estimate 80 iids × ~10 min per iid if serial
per-rival = ~13 hours. With 2-4 bounded workers and measured GPU memory, target
wall is 4-8 hours. If OOM appears, reduce workers immediately and resume only
missing iids.

**Acceptance check**: process completes; `$OUT/per_iid/*.json` has 80 entries; no crashes.

### 4.2 Real-time monitoring (optional)

```bash
# In a separate shell:
watch -n 30 "ls /data1/Kane/ACT/audit_results/sc_hz_phase_a_*/per_iid/*.json 2>/dev/null | wc -l"
```

Or use the Monitor tool to track HZ-PROGRESS lines.

---

## 5. Gate evaluation (2-3 hours)

### 5.1 Run gate evaluation script

```bash
cd /data1/Kane/ACT
: "${OUT:?Set OUT to the audit_results/sc_hz_phase_a_<STAMP> directory from §4}"
/data1/Kane/miniconda3/envs/act-py312/bin/python research/sc_hz/metrics.py \
    --in $OUT \
    --out $OUT/gate.json
```

This script must:
1. Aggregate per-iid receipts.
2. Compute three counters per benchmark: `new_cert`, `new_fal_strict_replay`, `phantom_lp_sat`.
3. Compute median LP UB reduction per benchmark.
4. Apply the §19 gate criteria from the brief:
   - **PASS**: on the positive group (safenlp + acasxu + tinyimagenet), new V/A ≥ 5 OR median LP UB reduction ≥ 25% on at least 2 of 3 benches.
   - **FAIL**: positive new V/A = 0 AND every positive bench median < 10%; OR CIFAR shows unexplained tightening; OR any FAL fails strict ORT replay.
   - **INCONCLUSIVE** otherwise.

### 5.2 Inspect gate.json

```bash
cat $OUT/gate.json | /data1/Kane/miniconda3/envs/act-py312/bin/python -m json.tool
```

**Expected fields**:

```json
{
  "overall_verdict": "PASS" | "FAIL" | "INCONCLUSIVE",
  "per_bench": {
    "safenlp_2024": {"new_cert": N, "new_fal_strict_replay": N, "phantom_lp_sat": N, "median_lp_ub_reduction_pct": F},
    "acasxu_2023": {...},
    "tinyimagenet_2024": {...},
    "cifar100_2024": {...}
  },
  "audit_notes": [...]
}
```

### 5.3 CIFAR sanity check

Per design lock §5, CIFAR is the negative control. Inspect explicitly:

```bash
: "${OUT:?Set OUT to the audit_results/sc_hz_phase_a_<STAMP> directory from §4}"
/data1/Kane/miniconda3/envs/act-py312/bin/python << 'PY'
import json, os
out = os.environ['OUT']
g = json.load(open(f'{out}/gate.json'))
cifar = g['per_bench']['cifar100_2024']
new_va = cifar['new_cert'] + cifar['new_fal_strict_replay']
red = cifar['median_lp_ub_reduction_pct']
print(f'CIFAR new V/A: {new_va} (expected: 0)')
print(f'CIFAR median LP UB reduction: {red}% (expected: ≤0%, i.e. no tightening)')
if new_va > 0 or red > 1.0:
    print('*** ALERT *** CIFAR shows lift. This is a BUG SIGNAL, not credit. INVESTIGATE BEFORE COUNTING.')
PY
```

**Acceptance check**: CIFAR new V/A = 0 AND median LP UB reduction ≤ 1%. If either is violated, **investigate the cause before any further interpretation** — see §6.

---

## 6. Decision tree

Based on gate.json:

### 6.1 If PASS

Goal: prepare Phase B (6-8 benchmark pilot).

Actions:
1. Write `research/sc_hz_phase_b_plan.md` (the Phase B design lock — TBD spec).
2. Update [VERIFICATION_IMPROVEMENT_ROADMAP.md](VERIFICATION_IMPROVEMENT_ROADMAP.md) §11 Phase B section with the Phase A results.
3. Update [paper_skeleton_20260604.md](research/paper_skeleton_20260604.md) §4.4 (negative results) — note that SC-HZ has lifted Phase A, mark as "in progress" not closed.
4. Brief advisor with `gate.json` + recommend Phase B kickoff.

### 6.2 If FAIL

Goal: clean closure of SC-HZ direction.

Actions:
1. Write `research/sc_hz_phase_a_closure_memo.md`:
   - Cite gate.json.
   - List which specific failure mode triggered (V/A 0 + median<10%, or CIFAR bug, or FAL replay fail).
   - Conclude that SC-HZ as defined is not viable.
   - Recommend reverting to paper's 924 V/A claim + engineering-only complementaries (R3/R4/R5 of the redesign).
2. Update [paper_skeleton_20260604.md](research/paper_skeleton_20260604.md) §4 to add SC-HZ as a fourth closed-negative direction (alongside CIFAR-ImageHZ, VGG-ImageHZ, CIFAR final-tail hull).
3. **Do not retry SC-HZ without new evidence.**

### 6.3 If INCONCLUSIVE

Goal: targeted K-cap widening on the weakest sentinels.

Actions:
1. From gate.json, identify the 10 sentinels with the smallest LP UB reduction.
2. Re-run those 10 with K=512:

```bash
python research/sc_hz/run_sentinels.py \
    --sentinels audit_results/sc_hz_phase_a_sentinels_confirmed_20260604.json \
    --restrict-to-iids <10 iids from gate.json> \
    --K 512 \
    --out audit_results/sc_hz_phase_a_kwiden_${STAMP} \
    --wall-per-iid-s 1200
```

3. Re-evaluate gate with the K=512 results merged in.
4. If still INCONCLUSIVE after K=512 expansion → escalate to advisor for ruling.

---

## 7. Soundness audit on every claimed CERT (must run regardless of gate outcome)

Goal: even if the gate is FAIL or INCONCLUSIVE, audit every claimed CERT for soundness.

Per the brief §19: "ORT replay validates witnesses; it is not by itself a CERT audit."

For every CERT receipt produced by SC-HZ:
1. Audit **all rivals** if the count is small; otherwise audit the maximum-UB
   rival plus the top-5 closest-to-zero rivals and record why full audit was
   skipped.
2. Independently rebuild the LP from the serialized SC-HZ state
   `(c, G_kept, tail_radius, constraints, spec row)` and solve it through a
   separate audit path. Do not call the same helper that produced the original
   SC-HZ LP number.
3. If every audited rival independently returns `< 0` with matching tolerance,
   CERT is confirmed.
4. If any audited rival returns `>= 0`, or if the audit cannot reconstruct the
   LP from receipt fields, the CERT is unaudited/invalid. **STOP all Phase A
   interpretation.**

```bash
: "${OUT:?Set OUT to the audit_results/sc_hz_phase_a_<STAMP> directory from §4}"
/data1/Kane/miniconda3/envs/act-py312/bin/python research/sc_hz/audit_certs.py \
    --in $OUT \
    --out $OUT/cert_audit.json
```

**Acceptance check**: 100% of CERT receipts confirmed by independent LP solve.
If only top-k rival audit is used for a very wide spec, `cert_audit.json` must
say `"audit_scope": "top_k"` and the paper/roadmap must not call it a full
independent CERT audit.

---

## 8. Provenance check on every emitted receipt

Goal: every Phase A receipt must carry the full provenance bundle (canonical_root + 3 SHA256).

```bash
: "${OUT:?Set OUT to the audit_results/sc_hz_phase_a_<STAMP> directory from §4}"
/data1/Kane/miniconda3/envs/act-py312/bin/python << 'PY'
import json, glob, os
out = os.environ['OUT']
missing = 0
for f in glob.glob(f'{out}/per_iid/*.json'):
    d = json.load(open(f))
    for k in ['canonical_root', 'instances_csv_sha256', 'onnx_sha256', 'vnnlib_sha256']:
        if not d.get(k):
            missing += 1
            print(f'MISSING {k}: {f}')
print(f'Provenance audit: {missing} missing fields across all receipts')
PY
```

**Acceptance check**: 0 missing fields.

---

## 9. Timeline summary

| Day | Step | What's happening |
|---|---|---|
| 0 | §0 | Pre-flight: env, baseline check, sentinel JSON sanity |
| 1 | §1 | TDD entry: write 5 unit-test files; verify ImportError failures |
| 2-3 | §2.1, §2.2 | Implement `prune.py` + `precompute_direction.py`; 5 unit tests pass |
| 4-5 | §2.3 | Implement `pruned_forward.py`; toy-model parity test passes |
| 6 | §2.4 + §3 | Implement `run_sentinels.py`; do production-budget pre-flight UNK confirmation on 80 iids |
| 7 (overnight) | §4 | Full 80-sentinel sweep, ~4-13 h wall depending on safe worker count |
| 8 | §5, §7, §8 | Gate evaluation + CERT audit + provenance audit |
| 8 | §6 | Decision tree action: write Phase B plan / closure memo / K-widening |

**Total wall**: ~8 working days from §0 to §6 decision.

---

## 10. Files this playbook will create

```text
research/sc_hz/
  __init__.py
  prune.py
  precompute_direction.py
  pruned_forward.py
  preflight_confirm_unk.py
  run_sentinels.py
  metrics.py
  audit_certs.py
  tests/
    __init__.py
    test_prune_soundness.py
    test_direction_chain.py
    test_forward_parity.py
    test_adversarial_d_soundness.py
    test_relevance_score_ablations.py
    test_pruned_forward.py
audit_results/
  sc_hz_phase_a_sentinels_confirmed_20260604.json   # from §3
  sc_hz_phase_a_preflight_<STAMP>/                  # from §3.1
  sc_hz_phase_a_<STAMP>/                            # from §4
    per_iid/*.json
    gate.json                                       # from §5
    cert_audit.json                                 # from §7
    run.log
  [if INCONCLUSIVE: sc_hz_phase_a_kwiden_<STAMP>/]  # from §6.3
research/
  [if PASS: sc_hz_phase_b_plan.md]                  # from §6.1
  [if FAIL: sc_hz_phase_a_closure_memo.md]          # from §6.2
```

---

## 11. Recovery — sentinel JSON regeneration if needed

If `audit_results/sc_hz_phase_a_sentinels_20260604.json` is lost or §0.3 fails:

```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python << 'PY'
# Regenerate from atlas v3 + §9 sweep UNK pool
# (full script lives in §11 of the design-lock predecessor; re-run that
# script with the same seed=20260604 to reproduce.)
import csv, json, random
random.seed(20260604)
# ... see design-lock plan for full code ...
PY
```

**Acceptance check**: regenerated JSON matches the original byte-for-byte (deterministic seed).

---

## 12. Hard rules — do not break

| Rule | Consequence if broken |
|---|---|
| `act/` production code NOT modified | Phase A loses its "zero risk to 924 V/A" property; entire experiment is suspect. STOP and revert. |
| PRUNE soundness tests pass before forward implementation | A PRUNE bug propagates silently into every sentinel result. Mandatory order. |
| CIFAR sanity check before counting any positive V/A | A CIFAR lift or tightening is an investigation signal, not credit. It may be a comparison-path mismatch or a PRUNE bug; CIFAR audit comes FIRST. |
| 100% of CERT receipts audited via independent LP | Without it, an unsound CERT cannot be distinguished from a real one. |
| 100% of receipts carry provenance bundle | Without it, paper claim cannot survive peer review. |
| Any FAL with `xi_star` in interval-tail variables = phantom_lp_sat, must remain UNK | Counting phantom witnesses as FAL is unsound. |
| Failed soundness or provenance audit → Phase A is INVALIDATED entirely | Re-run from §1, do not "patch around". |

---

## 13. Endpoints — what this playbook produces

If executed successfully (independent of PASS/FAIL/INCONCLUSIVE), the playbook produces:

1. **A reproducible audit_results directory** with 80 per-iid receipts, provenance bundles, gate.json, cert_audit.json.
2. **A documented decision** (Phase B plan / closure memo / K-widening) under `research/`.
3. **An updated brief and roadmap** reflecting the Phase A outcome.
4. **Zero modification to production 924 V/A baseline.**

The playbook is **complete when** the advisor signs off on the post-gate action document (§6).

---

## 14. Who to involve at each step

| Step | Person |
|---|---|
| §0 (pre-flight) | implementer (engineer) |
| §1 (TDD tests) | implementer; review by advisor optional |
| §2 (implementation) | implementer |
| §3 (UNK confirmation) | implementer, no advisor needed |
| §4 (sentinel sweep) | implementer, advisor for spot-check |
| §5 (gate evaluation) | implementer reports to advisor |
| §6 (decision) | **advisor decides; implementer prepares the action document** |
| §7-8 (soundness + provenance audits) | implementer; results to advisor |

The advisor must explicitly approve the §6 decision before the implementer writes Phase B plan or closure memo.
