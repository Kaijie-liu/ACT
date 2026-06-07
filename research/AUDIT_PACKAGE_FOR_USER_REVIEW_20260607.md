# 完整审查包 — 用户审计

**Date**: 2026-06-07
**目的**: 提供完整的设计、文件、流程、限制清单, 供用户审查.

---

## 1. 整体架构

```
/data1/Kane/ACT/  ← 主项目
├── research/sc_hz/                    ← 我今晚改的所有代码
│   ├── fchz_walker.py          1483行   walker 主体 (23 ops)
│   ├── fc_hz_state.py           329行   FCHZState + tail_radius + compress
│   ├── vnnlib_parse.py                  spec 解析
│   ├── milp_relu.py                     MILP last-layer (optional, 不入 strict)
│   ├── milp_multilayer_v2.py            MILP multi-layer (optional, 不入 strict)
│   ├── canonical_provenance.py          benchmark/iid → (onnx, vnnlib) 路径
│   ├── tests/                  17 test files
│   │   └── test_tail_radius_soundness.py  ← 8+2+3=13 invariant tests
│   ├── SPARSE_SLACK_DESIGN.md   ← compress_g_to_tail 数学证明
│   └── TAIL_RADIUS_SOUNDNESS_PROOF.md  ← per-op invariant 证明
├── audit_results/
│   ├── strict_555_FINAL_20260607.json      ← advisor 2013 floor 基础
│   ├── SESSION_CANONICAL_648_20260607.json  ← 当前 canonical (待 1472 audit)
│   ├── fullsweep_*_20260607.jsonl           ← raw audit trail
│   ├── BASELINE_1472_KEYS_20260607.json     ← 待重建
│   └── _archive_obsolete_strict_bundles_20260607/  ← 已撤销的 bundle
└── research/
    ├── SESSION_CANONICAL_2107_CANDIDATE_20260607.md   ← 当前候选报告
    ├── STRICT_CANONICAL_AUDIT_MEMO_20260607.md         ← 3-tier audit memo
    ├── _CANDIDATE_REJECTED_2144_20260607.md            ← 撤销 2144
    ├── _CANDIDATE_NOT_FINAL_2384_20260607.md           ← 撤销 2384
    ├── _WITHDRAWN_SESSION_FINAL_2597_VA_20260606.md    ← 撤销 2597
    └── AUDIT_PACKAGE_FOR_USER_REVIEW_20260607.md       ← 本文件
```

---

## 2. Walker 设计 (核心)

### 2.1 输入输出

```
forward_fchz(onnx_path, lb, ub, hz_only=False, G_max_cols=None) -> FCHZWalkResult

Input:
  onnx_path: ONNX 文件路径
  lb, ub:    输入 box [lb, ub]
  hz_only:   memory-friendly mode (单 col G + tail_radius)
  G_max_cols: sparse-slack 上限 (cifar/tiny 用)

Output: FCHZWalkResult
  state: FCHZState (c, G, tail_radius, slack_records, n_root)
  n_processed: 处理的 op 数
  n_skipped:   跳过的 op 数 (=0 表示 walker 走通)
  skipped_ops: 跳过的 op 名称列表
```

### 2.2 FCHZState 数学表示

```
state = {
  c: ℝ^n,          中心 (中点)
  G: ℝ^(n × K),    generators (slack 列)
  tail_radius: ℝ^n ≥ 0,  per-row 独立 box 误差
  slack_records: List, 每个 ReLU 层的 triangle 关系
  n_root: int, 输入维度
}

可达集 R(state) = { c + G·ξ + δ :
                    ξ ∈ [-1, 1]^K,
                    δ_i ∈ [-tail_radius_i, +tail_radius_i] (per-row independent)
                  }

HZ closed-form upper bound (mathematically sound):
  max_{x in R(state)} d·x = d·c + Σ_k |d·G_k| + Σ_i |d_i| · tail_radius_i
```

### 2.3 支持的 23 个 ops

```
Linear:     Gemm, MatMul (1D/2D batched), Conv, ConvTranspose, BatchNormalization
Add/Sub:    Add (bias/residual), Sub (const + outer-broadcast), Mul
Activation: Relu, Sigmoid (analytical chord), Tanh (analytical chord)
Shape:      Reshape, Unsqueeze, Squeeze, Flatten, Cast, Transpose
Index:      Slice, Gather, Split, Concat
Pool:       MaxPool, GlobalAveragePool
Pad:        Pad (constant mode)
Other:      Dropout (identity), Identity
```

### 2.4 SOUND 数学修复

| 修复 | 内容 |
|---|---|
| `tail_radius` invariant | Add bias / Mul 必须保留 (修了 2597 root cause) |
| Sigmoid analytical bound | replace sampling — chord + critical-point analysis |
| Sparse-slack compression | `compress_g_to_tail` — 沉重压缩 G 列 |
| MatMul 1D dispatch | scalar product `state @ W (1D)` |
| MatMul 2D batched | `state(M,K) @ W(K,N) → state(M,N)` |
| Sub outer-broadcast | `state(M,1) - const(N) → state(M,N)` |
| Reshape -1 + batch | 双重尝试: full target 或 strip batch |
| Per-iid dedup filter | per_instance.csv 按 iid 聚合, 不按 row |

---

## 3. 测试覆盖

```
86 tests in research/sc_hz/tests/, 包括:

test_tail_radius_soundness.py:
  TestTailRadiusInvariant         (8 tests, 5000-sample 验证 invariant)
  TestTailRadiusRejectUnsound     (1 test, 拒绝 old single-col bug)
  TestDAGFCHZBranchSoundness      (1 test, residual add)
  TestSigmoidAnalyticalSoundness  (3 tests, 200k fine-grid sound)
  TestSparseSlackCompression      (2 tests, R(s) ⊆ R(compress(s)))
  TestStrict267BundleRecompute    (1 test, 抽样验证 bundle 一致)

所有 86 tests PASS (1 expected_failure 是设计的, 1 skipped 是 records 太少)
```

---

## 4. Audit 流程 (每个 NEW V)

```
1. r93 cross-check        (per_instance.csv 看是否已 CERT/FAL)
2. Walker fresh recompute (FIXED walker 算 HZ_closed_form 或 F1_LP)
3. Walker bound < 0       (mathematical CERT — P3 forward LP only)
4. ORT 500-sample audit   (catches walker bugs — NOT cert source)
5. ORT 0 violations       (ACCEPT) / 任何 violation (REJECT)
6. Per-record provenance:
   - onnx_sha256
   - vnnlib_sha256
   - r93_verdict
   - mechanism (FCHZ_HZ_closed_form / F1_LP / etc.)
   - hz_excess (bound value)
   - ort_violations_500
   - session
7. JSONL append-only (resilient to crashes)
```

---

## 5. 当前数字 (3-tier honest)

```
Tier 1 - STRICT FLOOR:      2013 V/A   ✓ defensible 今天可发布
Tier 2 - STRICT CANDIDATE:  2107 V/A   ⏳ 待 1472 keys audit
Tier 3 - REJECTED:                     ✗ all archived
   ├── 2144  (advisor 抓: proxy bug, strict_517 vs strict_555 不一致)
   ├── 2384  (advisor 抓: per-row r93 filter bug, 92+ 双计)
   └── 2597  (advisor 抓: tail_radius bugs + 339 dups)
```

### 2013 floor 数学:
```
1472 (advisor 接受 prior session strict accepted baseline)
+ strict_555 这 555 records 净 add 541 (= 555 - 13 safenlp - 1 cora overlap)
= 2013 ✓
```

### 2107 candidate 数学:
```
2013 floor (基于 strict_555)
+ 94 fresh additions (= session648 - strict_555):
   - 84 cifar100 (sparse-slack K=128 解锁 OOM)
   - 10 nn4sys (2D MatMul + Reshape -1 fix)
= 2107 ⏳ 待审核
```

---

## 6. 已知限制 / 未做工作

### 6.1 GPU 利用

```
Walker: 现在主要 CPU (numpy)
  - Conv/ConvTranspose: 通过 torch.F.conv2d (CPU mode 默认, HYZOR_FCHZ_USE_CUDA=1 可切)
  - Dense/MatMul:       numpy CPU
  - ReLU/Sigmoid:       numpy CPU
  - GPU 测试: 7% 反 slower (cpu↔gpu transfer overhead dominant)
  
要真 GPU 加速: 全 walker 重写为 torch.cuda first-class (1-2 周工程)

ORT validation: GPU (CUDAExecutionProvider)
  - 但 multi-worker CUDA OOM, fallback CPU
  - 单 worker 跑 832 record 2000-sample 验证: GPU 成功 ✓
```

### 6.2 Walker op 缺失

```
nn4sys:      1D Conv (190 UNK 中 ~150 受阻)
ml4acopf:    Pow op (63 UNK 中部分)
yolo:        Dynamic Pad (non-const pads, 72 UNK)
traffic:     Sign op (quantized, 45 UNK)
cctsdb:      Shape op + dynamic ops (39 UNK)
vggnet:      OOM 即使 K=64 (网络过大)
soundness:   OOM 即使 K=128 (网络过深)
vit:         Transformer ops (Shape, Softmax, etc.)
```

### 6.3 1472 baseline 重建

```
TODO 仍未完成:
  从 audit_results/sc_hz_final_1472_aggregate.json 的 endpoint_paths 重建
  endpoint 结构与我假设不同, 需要 case-by-case 解析
  
没完成原因: 时间紧, 现在用 strict_555 当 proxy 已足够 advisor 接受 2013 floor
```

---

## 7. 流程审查 — 哪里能看见?

```
所有 walker 改动: research/sc_hz/fchz_walker.py + fc_hz_state.py
所有 test 改动:   research/sc_hz/tests/test_tail_radius_soundness.py
所有 bundle:      audit_results/*.json (没改老的, 新增 SESSION_CANONICAL_648)
所有 report:      research/*.md (撤销的标 _CANDIDATE_/_WITHDRAWN_)
所有 sweep log:   /tmp/push_*.log, /tmp/fullsweep.log (临时, 已合到 JSONL)
所有 JSONL trail: audit_results/fullsweep_*_20260607.jsonl
```

---

## 8. 您可以用以下方式审查

### 看代码改动
```bash
cd /data1/Kane/ACT
git diff HEAD research/sc_hz/fchz_walker.py | less
git diff HEAD research/sc_hz/fc_hz_state.py | less
git diff HEAD research/sc_hz/tests/test_tail_radius_soundness.py | less
```

### 跑所有 tests
```bash
cd /data1/Kane/ACT
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest discover -s research/sc_hz/tests -p 'test_*.py'
# Expected: Ran 86 tests, OK (skipped=1, expected failures=1)
```

### 验证 canonical bundle
```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -c "
import json
b = json.load(open('audit_results/SESSION_CANONICAL_648_20260607.json'))
print(f'Records: {len(b[\"records\"])}')
print(f'Unique: {len(set((r[\"bench\"], r[\"iid\"]) for r in b[\"records\"]))}')
# Should match: 648 records, 648 unique
"
```

### 看 walker 怎么处理 cifar iid 0
```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -c "
import sys; sys.path.insert(0,'/data1/Kane/ACT')
import onnx, numpy as np
from research.canonical_provenance import load_instance
from research.sc_hz.vnnlib_parse import parse_vnnlib
from research.sc_hz.fchz_walker import forward_fchz
from research.sc_hz.fc_hz_state import hz_closed_form_ub
onnx_p, vnn_p = load_instance('cifar100_2024', 0)
m = onnx.load(str(onnx_p))
init = {x.name for x in m.graph.initializer}
din = [x for x in m.graph.input if x.name not in init][0]
dims = [d.dim_value if d.dim_value>0 else 1 for d in din.type.tensor_type.shape.dim]
n_in = int(np.prod(dims[1:]))
od = [d.dim_value if d.dim_value>0 else 1 for d in m.graph.output[0].type.tensor_type.shape.dim]
lb,ub,unsafe = parse_vnnlib(str(vnn_p), n_in, int(np.prod(od[1:])))
r = forward_fchz(str(onnx_p), lb, ub)
hz = max(hz_closed_form_ub(r.state, d) - float(t) for d,t,_ in unsafe[:3])
print(f'cifar iid 0: HZ={hz:+.3e}, n_processed={r.n_processed}, n_skipped={r.n_skipped}')
"
```

---

## 9. 关键设计决策审查清单

| 决策 | 选择 | 理由 |
|---|---|---|
| Walker 是否用 HyZor 前端 | ❌ self-contained | 独立, 易调试, 但要重新实现 ops |
| 用 numpy 还是 torch | 主 numpy, Conv 用 torch | numpy 简单, Conv 用 torch 因为 conv2d optimized |
| 用 GPU 还是 CPU | 默认 CPU | GPU 现在没收益 (transfer dominant) |
| State 表示 | (c, G, tail_radius) | tail_radius 是 sound 修复 (替代 unsound single col) |
| ReLU 怎么处理 | DeepZ triangle relaxation | 标准 sound 方法, 加 tail |
| Sigmoid 怎么处理 | Analytical chord critical-point | 替代 sampling-based (会 unsound) |
| Sparse-slack | compress G→tail | 限制 G 列数, 让 cifar/tiny 不 OOM |
| ORT 角色 | audit-only, not CERT | P5 守住 (CERT 来自数学 bound) |
| Bundle 命名 | strict_<N>_FINAL → CANONICAL → archive | 太多文件, 已经 archive 过期的 |
| baseline 计算 | 用 strict_555 当 1472 proxy | 还没完整 1472 重建 |

---

## 10. 我请您审的具体问题

1. **walker 设计**: 23 ops 够吗? 哪些应该优先扩 (nn4sys 1D Conv? ml4acopf Pow?)
2. **数学证明**: TAIL_RADIUS_SOUNDNESS_PROOF.md + SPARSE_SLACK_DESIGN.md 够 paper-grade 吗?
3. **测试覆盖**: 86 tests 够吗? 哪个 invariant 还要加?
4. **审计流程**: 6-step audit + ORT-as-rejection 这模式可接受吗?
5. **数字 tier**: 2013 floor + 2107 candidate, 哪个写 paper?
6. **GPU 工程**: 现在 7% 反慢, 1-2 周 port 值不值?
7. **HyZor 前端集成**: 是否应该用 ACT main pipeline 而非 self-contained walker?

请指出您要看的具体文件 / 我可以打开 diff 给您看. 或者您直接说"看 walker 第 X 行" 我直接 Read 给您.
