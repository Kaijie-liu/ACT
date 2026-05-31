# Trade-off ablation: ReLU encoding (eq_lagr_v8 ↔ triangle) and S-shape K-piece

**Question.** For a previously-unseen architecture, how many tail ReLU layers should use eq_lagr_v8 versus triangle? And what value of K for Sigmoid/Tanh K-piece relaxation?

**Why this is needed.** The session-default `large_cls_proof_mode` uses eq_lagr_v8 on the last 3 ReLU and triangle on earlier ReLU, controlled by `HYZOR_LARGE_CLS_EQ_LAYERS`. The choice was based on an intuition (tail layers matter most for output-spec precision) but had no measured empirical floor. This document is the empirical floor.

---

## Experiment A — eq_layers ablation

### Setup
- 6 representative benchmarks spanning small-dense, conv, sequence:
  `linearizenn_2024`, `collins_rul_cnn_2022`, `malbeware`, `ml4acopf_2024`,
  `cifar100_2024`, `tinyimagenet_2024`.
- `HYZOR_LARGE_CLS_EQ_LAYERS ∈ {0, 1, 3, 5, 10}`.
- 10 iids per (bench, setting) cell. Wall 120-180s. RSS cap 5-8 GB depending on bench.
- 5-stream parallel within each wave (4 small benches in wave 1, cifar in wave 2,
  tinyimagenet in wave 3).

### Result table (format `V+A/n (mean_wall, OOM=#)`)

```
Benchmark            eq=0              eq=1              eq=3              eq=5              eq=10            best
linearizenn_2024     0/10 (12s,0)      0/10 (13s,0)      0/10 (12s,0)      0/10 (12s,0)      0/11 (11s,2)     INSENSITIVE
collins_rul_cnn_2022 8/11 (3s,2)       8/11 (2s,2)       8/10 (3s,0)       8/11 (3s,2)       8/10 (3s,0)      INSENSITIVE
malbeware            10/10 (5s,0)      10/10 (5s,0)      10/10 (5s,0)      10/10 (5s,0)      10/10 (5s,0)     INSENSITIVE
ml4acopf_2024        1/10 (34s,0)      1/10 (34s,0)      1/10 (33s,0)      1/10 (34s,0)      1/11 (31s,2)     INSENSITIVE
cifar100_2024        0/10 (8s,10)      0/10 (8s,10)      0/10 (8s,10)      0/9 (7s,8)        0/9 (7s,8)       OOM-BOUND
tinyimagenet_2024    0/8 (9s,6)        0/9 (9s,8)        0/10 (9s,10)      0/9 (9s,8)        0/9 (9s,8)       OOM-BOUND
```

### Reading the table

**4 of 6 benches are eq_layers-insensitive within the 120-180s budget.**
- `linearizenn_2024`: 0 V across all settings. Small-dense bounds-collapse pathology;
  eq_layers does not unblock.
- `collins_rul_cnn_2022`, `malbeware`, `ml4acopf_2024`: saturated at a fixed V count.
  Tail tightness is not the marginal lever — the structural difficulty class of each
  iid is already determined by upstream geometry.

**2 of 6 benches (cifar100, tinyimagenet) are OOM-bound, not eq_layers-bound,
under 5-stream parallel × 8 GB RSS cap.** Almost every instance OOMed within seconds.
This is a budget artifact, NOT a precision claim about eq_layers. To measure
eq_layers precision on cifar100/tinyimagenet you would need single-instance,
≥32 GB RSS, ≥240 s wall.

### Interpretation

**On benchmarks where HZ already verifies anything (collins, malbeware, ml4acopf),
the V count is a property of input-set geometry, not of eq_layers placement.**
The session default `eq_layers = 3` is therefore a safe choice — no benchmark
shows a measurable lift from higher values, and higher values increase OOM risk
without offsetting decisions.

**On heavy CNNs (cifar100, tinyimagenet), eq_layers is the wrong knob.** The
structural ceiling demonstrated in `star_vs_hz_analysis_20260530.md` (forward
triangle + LP output relaxation cannot tighten the 100-dim output enough for a
spec-direction LP to declare CERTIFIED) is not lifted by adding more eq_lagr_v8.
The ALL_EQ probe (`HYZOR_LARGE_CLS_EQ_LAYERS = 999`) reached the same conclusion
on the 8 zero-verdict benchmarks: 0 V + OOM.

### Rule for a new architecture

```
Layer count L, output dim D, conv count C.

if  L ≤ 8  and  D ≤ 64  and  C ≤ 1 :
    eq_layers = L     (run eq_lagr_v8 on EVERY ReLU — small enough to afford)
elif  D ≤ 256  and  C ≤ 4 :
    eq_layers = 3     (large_cls_proof_mode default — tail-only)
elif  C ≥ 4  and  D ≥ 100 :
    eq_layers = 1     (heavy CNN — minimise OOM risk; structural ceiling
                       not lifted by more tail eq anyway)
else:
    eq_layers = 3     (safe default)

# Sigmoid/Tanh networks: see Experiment B below.
```

---

## Experiment B — S-shape K-piece ablation

### Setup
- Sigmoid K on `dist_shift_2023`, iids 0,7,14,…,63 (10 iids), wall 120 s, RSS 8 GB.
- Tanh K on `cgan_2023`, iids 19,20, wall 300 s, RSS 16 GB.
- `K ∈ {1, 2, 4, 8, 16}`. Knobs `ACT_HZ_SIGMOID_K` / `ACT_HZ_TANH_K`.
- 5-stream parallel.

### Result table

```
=== Sigmoid K ablation on dist_shift_2023 (10 iids) ===
  K   V /  A /  U /  n    mean_wall
  1  10 /  0 /  0 / 10       72.0 s
  2  10 /  0 /  0 / 10       11.5 s   ⭐ sweet spot
  4   0 /  0 /  0 / 10       12.8 s   <-- regression
  8   9 /  0 /  1 / 10       10.6 s
 16   0 /  0 / 10 / 10        1.5 s   <-- fast bail to UNKNOWN

=== Tanh K ablation on cgan_2023 iids 19,20 ===
  K   V /  A /  U /  n    mean_wall
  1   0 /  0 /  0 /  4       24.1 s
  2   0 /  0 /  0 /  4       24.1 s
  4   0 /  0 /  0 /  4       23.1 s
  8   0 /  0 /  0 /  4       23.1 s
 16   0 /  0 /  0 /  4       24.1 s
```

### Reading

**Sigmoid: K = 2 is the production sweet spot.** Same V count as K = 1 at
~6 × the speed (11.5 s vs 72 s mean wall). K = 4 and K = 16 *regress* below K = 1.

**The non-monotone V curve on K is the surprise.** Naïve theory predicts a
monotone-tighter relaxation as K grows. We see the opposite at K = 4 and K = 16.
Two plausible mechanisms:

1. **PWL constraint blow-up triggers an early-abandon path.** K = 16 mean wall
   is 1.5 s with all UNKNOWN, which is too fast for genuine LP work. Some
   constraint-count threshold appears to be reverting to a fail-fast verdict
   instead of running the LP. Worth instrumenting.
2. **Inconsistent equality elimination at large K.** `project_eq_elim` is the
   measured 32-43% precision lever (memory: `project_eq_elim_hero_20260515`).
   At larger K it may leave residual constraints that the downstream LP can't
   exploit, producing a strict-spec LP that admits phantoms.

Both mechanisms need follow-up. Production setting is **K = 2** until then.

**Tanh: this slice (n = 4) is insensitive to K — every cell is 0 V / 0 A.**
The cgan iids selected are structurally hard for tanh-K alone. To get a real
tanh-K signal we would need to (a) widen the iid slice to ≥ 30, (b) try
different cgan iids that are within an order of magnitude of the V boundary,
or (c) test on a non-cgan tanh benchmark like vggnet16's tanh variants if any
exist. Not a useful default-K recommendation from this slice.

### Rule for new architectures with smooth activations

```
Sigmoid:  K = 2     (measured sweet spot on dist_shift_2023; same V as K=1 at 6× speed)
Tanh:     K = 2     (provisional — defaulting to the Sigmoid finding;
                     no measured signal yet on cgan_2023 iids 19/20.
                     If a future benchmark shows V(tanh K=1) < V(tanh K=2),
                     promote K to the next value that recovers full V.)
```

Do NOT default to K ≥ 4 for Sigmoid until the K = 4 / K = 16 regression is
understood. The fast-bail at K = 16 is currently a soundness-adjacent footgun:
UNKNOWN-without-LP-attempt could mask precision losses elsewhere.

---

## How this maps to the paper

- **§ Method.** State `eq_layers = 3` as the production default for benchmarks
  in the (D ≤ 256, C ≤ 4) class.
- **§ Method.** State `eq_layers = 1` for heavy CNNs as an OOM mitigation,
  with the caveat that this does not lift the structural ceiling (cite the
  ALL_EQ negative result).
- **§ Ablation.** Reproduce the table above. Be explicit that cifar100/tinyimagenet
  cells are budget-bound and would need a sequential rerun to disentangle precision
  from memory.
- **§ Threats to validity.** Acknowledge that the eq_layers parallel-budget cell
  for cifar100/tinyimagenet conflates OOM with precision; what we measured is
  that under realistic batch budgets eq_layers offers no marginal lift.

## Tracability

- Raw data: `/data1/Kane/ACT/audit_results/eq_layers_ablation_20260530T015205Z/`
- K-piece raw: `/data1/Kane/ACT/audit_results/sigmoid_K_ablation_20260530T0217*`
  (first run buggy, second run after env-prefix fix)
- Driver: `/tmp/eq_layers_ablation.sh`, `/tmp/sigmoid_K_ablation.sh`
- CLI env knobs added: `ACT_HZ_SIGMOID_K`, `ACT_HZ_TANH_K`
  (`/data1/Kane/ACT/act/pipeline/cli.py` near HZVerifier construction)
- Related: `star_vs_hz_analysis_20260530.md` (structural ceiling),
  `project_v100_v101_v102_cifar100_final_20260519.md` (V=154 cifar100 with
  workers ≤ floor(GPU_GB/32), eq_layers=1 — consistent with the eq_layers=1
  recommendation above for heavy CNNs).
