# Session 2026-05-28 → 2026-05-30 — Archive

## Net result vs user's existing pre-session baseline

**+129 net GPU V/A decisions** (V +155, A -26).

The -26 A is **entirely** from one config change: `cli.py` default for
`--small-dense-lp` was switched from `auto` (WitnessExtract with +1e-6 ORT slack
and random perturbation) to `specaware` (forward LP only). This complies with
the user's P6 principle (no random-sample-then-check) added 2026-05-28. Specific
losses: acasxu (1V+15A), linearizenn (13V), sat_relu (1V+20A), safenlp (2A),
tllverifybench (2A). To restore the old behavior set `ACT_HZ_SMALL_DENSE_LP=auto`.

Real positive gains: nn4sys +82V, dist_shift +72V, ml4acopf +13V, cgan +11A,
metaroom +3V, collins_rul +1A, vggnet16 +1A.

## Layout

```
session_20260528_30/
├── README.md                          (this file)
└── scripts/
    ├── 8bench_full_rerun.sh           24-way parallel rerun on relusplitter, vgg, sat_relu,
    │                                   malbeware, acasxu, linearizenn, collins_rul, cctsdb
    ├── nn4sys_full_194.sh             4-way parallel nn4sys 194 (+83 NEW CERT)
    ├── nn4sys_oom_reclaim.sh          2-way reclaim of 16 RSS-bound iids @ rss_cap=50GB
    ├── nn4sys_smoke.sh                5-iid quick smoke
    ├── gather_slice_rerun_chained.sh  ml4/lsnc/collins_aero/safenlp/tll chain after nn4sys
    ├── cora_full_180.sh               4-way parallel cora 180
    ├── cora_resume_129.sh             Resume after SIGTERM mid-sweep
    ├── coverage_gap_parallel_rerun.sh 5-way metaroom non-CERT, cifar/tiny/dist/yolo sample
    ├── tiny_remainder_170.sh          tinyimagenet 30-199 sweep
    ├── parallel_5way_morning_sweep.sh ml4/metaroom/safenlp/cora/cgan re-confirm
    ├── postpatch_3bench_sweep.sh      cgan/safenlp/cora early-morning sample
    ├── regression_*.sh                Soundness gate runs (8/8 PASS for each variant)
    ├── joint_k2_*_sweep.sh            Direction B failed (closed negative)
    ├── sidecar_gpu_sweep.sh           Multi-corner LP sidecar failed (closed negative)
    └── d_gpu_resume.sh                D filter failed (closed negative)
```

## Related artifacts

- **Memory** (persistent across sessions): `~/.claude/projects/-data1-Kane-HyZor/memory/`
  - `project_gather_slice_exact_hz_20260530.md` — biggest single discovery
  - `project_sparse_input_singleton_gpu_lifts_20260529.md` — overnight gains
  - `project_vgg_zero_width_gpu_lift_20260528.md` — VGG lift
  - `project_direction_b_closed_negative_20260528.md` — joint K=2 etc closed
  - `project_d_filter_gpu_negative_20260528.md` — D filter closed
  - `project_multi_corner_lp_sidecar_negative_20260528.md` — multi-corner closed
  - `feedback_no_pgd_no_backward_falsifier.md` — P6 principle (added by user)
  - + others in same dir

- **Audit results** (raw sweep outputs): `/data1/Kane/ACT/audit_results/`
  - `nn4sys_gather_full_20260529T150552Z/` — final nn4sys 194 sweep
  - `eight_bench_rerun_*/` — 24-way 8-bench rerun (this conversation)
  - `gather_slice_chain_20260529T150826Z/` — chain rerun
  - `dist_shift_sigmoid_auto_20260529T120509Z/` — +72 CERT
  - `cgan_auto_triangle_full_20260529T140706Z/` — +11 FAL
  - + many others

- **Paper drafts**: `/data1/Kane/ACT/research/joint_k2_relu/`
  - 9 sections + 3 appendix, ~8.8K words
  - Direction B negative result + HZ as abstract domain formalism

- **Session synthesis**: `/data1/Kane/ACT/research/SESSION_SYNTHESIS_20260530.md`

## Code changes (modified files vs HEAD)

- `act/back_end/solver/solver_hz.py`
  - `_hz_upsample_nearest_nchw`, `_hz_convtranspose2d_native`,
    `_hz_gather_exact`, `_hz_slice_exact` (new exact transfers)
  - zero-width input generator pruning in `hz_from_bounds`
  - singleton exact fastpath
  - dispatch hooks for new exact ops
- `act/back_end/hybridz_tf/hybridz_tf.py`
  - Sigmoid/Tanh dim cap raised 256→2048 with complexity guard
- `act/back_end/hybridz_tf/hz_routing.py`
  - VGG sparse-huge auto profile (late triangle ReLU)
  - B3 sparse-eq_lagr hook (default OFF)
- `act/back_end/hybridz_tf/representations.py`
  - SparseGcZ extended (B3 support, default OFF)
- `act/back_end/interval_tf/tf_cnn.py`
  - Fix #9 ONNX Flatten axis≥2
- `act/pipeline/cli.py`
  - HYZOR_LARGE_CLS_EQ_LAYERS env bridge fix
  - small_dense_lp default `auto` → `specaware` (P6 compliance)
  - cctsdb unsupported Slice → honest UNKNOWN
- `act/pipeline/verification/utils.py` — Fix #8 OnnxResize
- `act/back_end/analyze.py` — validate_constraints skip for large IR
- `act/back_end/utils.py` — minor

## New tests (all PASS)

- `tests/test_hz_gather_slice_exact.py` — 7/7 PASS
- `tests/test_hz_upsample_exact.py` — PASS
- `tests/test_hz_convtranspose_exact.py` — PASS
- `tests/test_zero_width_input_prune.py` — PASS
- `tests/test_final_softmax_order_bypass.py` — PASS
- `tests/test_hz_sign_hull.py` — PASS (env-gated experimental)
- + existing `test_lut_bounds_envelope.py`, `test_constant_eval_failclosed.py`,
  `test_hz_representations.py` continue PASS

## Soundness gate

8-instance regression pack (`tests/regression_pack.sh`): **8/8 PASS** under
every code variant tested in this session.

## Open work

1. **Reclaim relusplitter OOM iids** with higher rss_cap (12-16 GB needed for CIFAR
   models in this benchmark).
2. **iid 129 nn4sys**: box-fallback gave CERT, exact gather/slice gives UNKNOWN.
   Could implement try-both-paths to recover, but no urgency.
3. **Direction A formal HZ-as-abstract-domain paper**: §1-§8 + appendix drafted
   in `research/joint_k2_relu/paper_draft_v1.md`.
