# Experiment Scripts - 数据获取指南

## 数据获取流程

论文表格中的数据通过以下流程获取：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据获取流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│  1. NetFactory 生成网络                                              │
│     ↓                                                                │
│  2. ACT 验证器运行抽象解释 (interval/hybridz/dual)                    │
│     ↓                                                                │
│  3. 收集各层边界 (layer_bounds)                                       │
│     ↓                                                                │
│  4. TFMutator 注入变异 (M1-M6)                                       │
│     ↓                                                                │
│  5. Level 1 (SCC): 采样检查输出一致性                                 │
│     Level 2 (BCA): 检查边界包含不变量                                 │
│     ↓                                                                │
│  6. 收集检测率、定位准确率、时间开销                                   │
│     ↓                                                                │
│  7. 生成 LaTeX 表格                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 运行实验

### 方式 1: Mock 模式 (快速测试)

```bash
# 使用模拟数据快速测试脚本
python experiments/rq1_detection.py --seed 42 --mode mock

# 运行所有实验 (mock 模式)
python experiments/run_all.py --seed 42
```

### 方式 2: Real 模式 (获取真实数据)

```bash
# 使用真实 ACT 验证器收集数据
python experiments/rq1_detection.py --seed 42 --mode real

# 单独运行各 RQ 实验
python experiments/rq2_scc_effectiveness.py --seed 42
python experiments/rq3_localization.py --seed 42
python experiments/rq4_coverage.py --seed 42
python experiments/rq5_cross_domain.py --seed 42
python experiments/rq6_overhead.py --seed 42
```

## 数据来源说明

### RQ1: 检测率数据 (Table 1)

| 数据项 | 来源 |
|--------|------|
| SCC Only | `scc_result.status == SCCStatus.FAIL` |
| BCA Only | `bca_result.status == BCAStatus.FAIL` |
| Combined | `soundness_violated` (SCC或BCA任一检测到) |
| Localized | `violation_localized` (BCA检测到且有违规记录) |

**运行:**
```bash
python experiments/rq1_detection.py --seed 42 --mode real -v
```

**输出:** `results/rq1/results.json`, `results/rq1/table_rq1.tex`

### RQ2: SCC 有效性数据 (Table 2)

| 数据项 | 来源 |
|--------|------|
| Discovery Rate | 找到反例的采样比例 |
| Inconclusive | `scc_result.status == SCCStatus.INCONCLUSIVE` |
| Avg Time | SCC 运行时间 (ms) |

**注意:** LIN_POLY 规范无法直接采样，因此 Discovery Rate = 0%

### RQ3: 定位准确率数据 (Table 3)

| 数据项 | 来源 |
|--------|------|
| Top-1 Hit | `violations[0].layer_id == target_layer_id` |
| Top-5 Hit | `target_layer_id in [v.layer_id for v in violations[:5]]` |
| Error Rate | `bca_result.status == BCAStatus.ERROR` |

### RQ4: 覆盖率数据 (Table 4)

| 数据项 | 来源 |
|--------|------|
| Op Coverage | `covered_layers / total_trackable_layers` |
| Bug Yield | 检测到的 bug 数量 |

### RQ5: 跨域比较数据 (Table 5)

| 数据项 | 来源 |
|--------|------|
| BCA Fail Rate | BCA 检测失败率 |
| Bound Width | `avg(ub - lb)` 平均边界宽度 |
| Disagreement | 不同域结果不一致的比例 |

### RQ6: 开销数据 (Table 6)

| 数据项 | 来源 |
|--------|------|
| Params | 模型参数数量 |
| SCC (ms) | SCC 运行时间 |
| BCA (ms) | BCA 运行时间 |
| Overhead | `(scc + bca) / analysis_time` |

## 输出文件结构

```
results/
├── rq1/
│   ├── metadata.json      # 实验元数据 (种子、配置)
│   ├── results.json       # 完整实验结果
│   └── table_rq1.tex      # LaTeX 表格
├── rq2/
│   ├── results.json
│   └── ...
└── experiment_summary.json # 汇总统计
```

## 核心代码路径

- **数据收集器:** `experiments/data_collector.py`
- **Level 1 (SCC):** `act/back_end/validation/scc.py`
- **Level 2 (BCA):** `act/back_end/validation/bca.py`
- **变异操作:** `act/back_end/validation/mutations.py`
- **抽象分析:** `act/back_end/analyze.py`
- **传递函数:** `act/back_end/interval_tf/`, `hybridz_tf/`, `dual_tf/`

## 可复现性验证

```bash
# 生成基准结果
python experiments/verify_reproducibility.py --seed 42 --generate-baseline

# 验证复现
python experiments/verify_reproducibility.py --seed 42 --verify
```

## 完整实验执行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行所有实验
python experiments/run_all.py --seed 42 --mode real

# 3. 查看结果
cat results/experiment_summary.json

# 4. 验证可复现性
python experiments/verify_reproducibility.py --seed 42 --verify
```

## 常见问题

### Q: 为什么 mock 模式和 real 模式结果不同？
A: Mock 模式使用模拟的检测概率，Real 模式使用实际的 ACT 验证器。论文应使用 Real 模式的数据。

### Q: 实验需要多长时间？
A: 取决于网络数量和大小。100 个小网络大约需要 10-30 分钟。

### Q: 如何只运行特定的 RQ？
A: 使用 `--experiments` 参数:
```bash
python experiments/run_all.py --experiments rq1 rq3
```
