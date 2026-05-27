# NNV STRICT no-helper patch — scientific-integrity notes

Target file: `/data1/Kane/nnv/code/nnv/examples/Submission/VNN_COMP2025/run_vnncomp_instance.m`

Backup will be saved as `run_vnncomp_instance.m.orig` before applying.

## Purpose

The NNV competition entry script bundles **three distinct helpers** that violate
the "pure sound verifier" protocol used for this paper's cross-tool comparison.
We patch the entry script to gate all three behind an environment variable
`NNV_STRICT_NO_HELPER=1`, leaving competition default behavior intact when the
variable is unset.

## What each helper is, and why we disable it

### Helper 1 — `falsify_single` (random sampling)

- Location: lines 41-60, 858-894 of upstream `run_vnncomp_instance.m`.
- What it does: draws `nRand` (100 or 500) random points from the input box,
  evaluates `predict(net, x)` for each, and returns `sat` if any point lies in
  the unsafe halfspace.
- Why it is a helper: this is exactly the random/PGD falsification family that
  we disable on every other tool (abcrown `--pgd_order=skip`, NeuralSAT
  `--disable_attack`, CORA `falsification_method='none'`).
- STRICT behavior: skip the entire block, set `counterEx = nan` so the
  downstream `iscell(counterEx)` check fails and reachability proceeds.

### Helper 2 — explicit `lb`/`ub` corner evaluation

- Location: `create_random_examples`, line 786: `xRand = [lb, ub, xRand];`.
- What it does: prepends the box lower and upper bound corners to the random
  sample, so the first two falsification trials are always at deterministic
  corner points.
- Why it is a helper: equivalent to CORA's `center-of-box` falsification mode.
  We already rejected this in CORA (the user explicitly demanded `'none'`).
- STRICT behavior: gated together with Helper 1 (skipping the whole
  `falsify_single` call removes the corner evaluation too).

### Helper 3 — `cp-star` reachability method (unsound)

- Location: `Prob_reach` (called when `reachOptions.reachMethod == 'cp-star'`).
- What it does: trains a linear surrogate model and computes a probabilistic
  bound with `coverage = 0.999, confidence = 0.999`. With probability up to
  10^-3 the bound is **wrong** (i.e. the true reachable set escapes it).
- Why it is the most serious helper: this is not a sound verifier at all —
  it is conformal prediction. A `unsat` from `cp-star` is a statistical claim,
  not a formal proof. It would be scientifically dishonest to put `cp-star`
  results in a table that also contains formally sound `abcrown` / `CORA` /
  `NeuralSAT` / `nnenum` verdicts.
- STRICT behavior: when STRICT is on and the only configured method for a
  benchmark is `cp-star`, the runner refuses to compute and writes
  `unsupported_strict` to the result file. The benchmark is reported as
  unsupported, not as `unknown` (so it doesn't poison the unknown rate).

### Benchmarks affected by Helper 3 (only cp-star configured)

These benchmarks will all return `unsupported_strict` and the runner will not
spend wall time on them:

- `cersyve`
- `cifar100_2024`
- `collins_aerospace_benchmark`
- `cgan_2023` (only when ONNX filename contains `transformer`)
- `cora_2024` (only the `-set` ONNX variants)
- `ml4acopf_2024`
- `nn4sys` (only the non-`lindex` ONNX variants — pensieve etc.)
- `soundnessbench`
- `tinyimagenet_2024`
- `vggnet16_2022`
- `vit_2023`
- `yolo_2023`

### Benchmarks NNV cannot run at all (upstream `error()`)

These are independent of STRICT mode; NNV's own code refuses them:

- `cctsdb_yolo_2023` ("Working on supporting this one")
- `lsnc_relu` ("IR and opset not yet supported in MATLAB")
- `traffic_signs_recognition_2023` ("IR and opset not yet supported in MATLAB")
- `nn4sys` `mscn` variants ("These are not supported yet.")

### Benchmarks runnable in STRICT mode (sound reachability only)

Each uses one of `approx-star`, `exact-star`, `relax-star-area`:

- `acasxu_2023` (exact-star, approx-star)
- `cgan_2023` (non-transformer ONNX: relax-star-area, approx-star)
- `collins_rul_cnn_2022` (approx-star)
- `cora_2024` (non `-set` ONNX: relax-star-area)
- `dist_shift_2023` (exact-star)
- `linearizenn_2024` (approx-star, exact-star; falls back to cp-star on
  `matlab2nnv` failure — under STRICT we reject the fallback)
- `malbeware` (exact-star)
- `metaroom_2023` (approx-star)
- `nn4sys` (lindex ONNX: approx-star)
- `relusplitter` (relax-star-area)
- `safenlp_2024` (approx-star, exact-star)
- `sat_relu` (approx-star, exact-star)
- `tllverifybench_2023` (relax-star-area, approx-star)

## Expected vs competition behavior

Compared to the competition NNV submission:
- **STRICTER on falsification**: NNV competition relies heavily on `falsify_single`
  for SAT verdicts. STRICT yields 0 SAT for every benchmark (no falsification
  mechanism remains).
- **STRICTER on method**: NNV competition uses `cp-star` for many heavy
  benchmarks. STRICT refuses these and reports `unsupported_strict`.
- **Same on sound reachability**: all `approx-star` / `exact-star` /
  `relax-star-area` UNSAT verdicts are unchanged.

So in the paper table:

| Tool | Reported V (verified) | Reported A (falsified) |
|---|---|---|
| NNV competition | high (mix of sound UNSAT + probabilistic UNSAT) | high (random falsification) |
| NNV STRICT (ours) | only sound UNSAT (subset of above) | **0** (no falsifier) |

This is honest and comparable to CORA TRUESTRICT (also patched to expose pure
reachability), abcrown `--pgd_order=skip`, NeuralSAT `--disable_attack`, and
nnenum (natively helper-free).

## R2026a known incompatibility (separate issue)

`importNetworkFromONNX` on R2026a sometimes emits an `nnet.cnn.layer.ScalingLayer`
that `matlab2nnv` does not know about. This is an upstream NNV bug, not a STRICT
issue. We do not patch around it — affected instances will appear as `error` in
the STRICT results, which is honest reporting of a tool limitation.
