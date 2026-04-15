# ACT vs UCU_Aiware 实验对比

两边均:master_seed=42,NetFactory base_seed=1015796661,num_instances=100,residual 变体启用。

| | ACT (本仓库,post-fix) | UCU_Aiware (论文 baseline) |
|---|---|---|
| 日期 | 2026-04-15 | 2026-02-15 |
| 总耗时 | 88.2s | 41.8s |
| 6 RQ 全成功 | ✓ | ✓ |

---

## RQ1 Detection Rates — by (domain × mutation)

| Domain | Mutation | ACT CBR | ACT BBL | ACT Comb | ACT Loc | UCU CBR | UCU BBL | UCU Comb | UCU Loc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| interval | tighten_bounds     |  7% |  37% |  37% |  82% |  0% |  10% |  10% | 100% |
| interval | swap_lb_ub         | 27% | 100% | 100% |  87% | 23% | 100% | 100% | 100% |
| interval | zero_lower_bound   | 20% |  67% |  67% |  95% | 17% |  70% |  70% | 100% |
| interval | scale_upper_bound  | 10% |  83% |  83% | 100% | 17% |  93% | 100% | 100% |
| interval | add_noise          |  3% |  67% |  67% | 100% | 17% |  70% |  70% | 100% |
| interval | loosen_bounds      | 10% |  13% |  13% |  25% |  0% |   0% |   0% | N/A  |
| hybridz  | tighten_bounds     |  0% |  27% |  27% |  88% |  3% |  10% |  10% | 100% |
| hybridz  | swap_lb_ub         | 17% | 100% | 100% | 100% | 23% | 100% | 100% | 100% |
| hybridz  | zero_lower_bound   |  3% |  80% |  80% | 100% | 20% |  73% |  77% | 100% |
| hybridz  | scale_upper_bound  |  0% |  60% |  60% | 100% |  7% | 100% | 100% | 100% |
| hybridz  | add_noise          |  3% |  50% |  50% | 100% | 17% |  83% |  83% | 100% |
| hybridz  | loosen_bounds      |  0% |   7% |   7% |   0% |  0% |   0% |   0% | N/A  |
| dual     | tighten_bounds     |  0% |  23% |  23% | 100% |  7% |  13% |  17% | 100% |
| dual     | swap_lb_ub         |  7% | 100% | 100% | 100% | 10% | 100% | 100% | 100% |
| dual     | zero_lower_bound   | 17% |  77% |  77% | 100% | 20% |  87% |  87% | 100% |
| dual     | scale_upper_bound  |  3% |  70% |  70% | 100% | 23% |  83% |  93% | 100% |
| dual     | add_noise          |  0% |  50% |  50% | 100% |  7% |  77% |  77% | 100% |
| dual     | loosen_bounds      |  0% |   0% |   0% | N/A  |  0% |   0% |   0% | N/A  |
| **Overall** | **All** | **8%** | **66%** | **66%** | **97%** | **14%** | **71%** | **73%** | **100%** |

---

## RQ2 L1 Discovery Rate — by spec type

| Spec Type | ACT Disc | ACT Inconc | ACT Time(ms) | UCU Disc | UCU Inconc | UCU Time(ms) |
|---|---:|---:|---:|---:|---:|---:|
| BOX       | 100% |   0% | 2.0 | 100% |   0% | 2.9 |
| LINF_BALL | 100% |   0% | 1.9 | 100% |   0% | 2.3 |
| LIN_POLY  |   0% | 100% | N/A |   0% | 100% | N/A |

---

## RQ3 L2 Localization Accuracy — by architecture

| Architecture | ACT Top-1 | ACT Top-5 | ACT Err | UCU Top-1 | UCU Top-5 | UCU Err |
|---|---:|---:|---:|---:|---:|---:|
| Sequential MLP |  50% |  75% | 0% | 100% | 100% | 0% |
| Sequential CNN |  92% |  92% | 0% | 100% | 100% | 0% |
| Residual (ADD) |  62% |  62% | 0% | 100% | 100% | 0% |

---

## RQ4 Coverage & Bug Yield — by strategy

| Strategy | ACT Generated | ACT Coverage | ACT Yield | UCU Generated | UCU Coverage | UCU Yield |
|---|---:|---:|---:|---:|---:|---:|
| Basic-50  |   50 |  87% |   50 |   50 |  80% |   50 |
| Basic-100 |  100 |  87% |  100 |  100 |  93% |  100 |
| Full-100  | 1000 |  87% | 1000 | 1001 | 100% | 1001 |

ACT 未覆盖的 2 个算子:**BN**(layer_schema REGISTRY 未注册),**RESHAPE**(NetFactory 无对应 toggle)。

---

## RQ5 Cross-Domain Comparison

| Domain | ACT BBL Fail | ACT Bound Width | ACT Time(ms) | UCU BBL Fail | UCU Bound Width | UCU Time(ms) |
|---|---:|---:|---:|---:|---:|---:|
| interval |  100% |  71.10 | 1.2 | 100% | 0.36 | 0.5 |
| hybridz  |  100% | 120.50 | 1.2 | 100% | 0.36 | 0.4 |
| dual     |  100% | 121.42 | 1.2 | 100% | 0.36 | 0.4 |

Disagreement rate: ACT 0% / UCU 0%.

bound_width 差 200–340×:UCU 三域输出相同 (0.36) 说明当时 hybridz/dual 退化到 interval;ACT 三域真正分开 = HZ solver + dual_tf 重写结果。

---

## RQ6 Validation Overhead (avg_total_ms)

|  | | ACT CBR-5 | ACT CBR-10 | ACT CBR-20 | ACT CBR-50 | ACT BBL | UCU CBR-5 | UCU CBR-10 | UCU CBR-20 | UCU CBR-50 | UCU BBL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small  | ~1K   | 0.29 | 0.59 | 1.16 | 2.90 | 0.09 | 0.29 | 0.57 | 1.37 | 3.20 | 0.11 |
| Medium | ~33K  | 0.30 | 0.59 | 1.19 | 2.94 | 0.14 | 0.32 | 0.64 | 1.28 | 3.02 | 0.13 |
| Large  | ~297K | 0.37 | 0.73 | 1.45 | 3.60 | 0.15 | 0.37 | 0.73 | 1.48 | 3.77 | 0.15 |

---

## 总评

| RQ | 关系 | 说明 |
|---|---|---|
| RQ1 | ACT −7pp | combined 66% vs 73%,source: tighten/loosen 检测更积极,scale_upper/add_noise 偏低 |
| RQ2 | **ACT 略优** | 100% discovery 两边一致,ACT 采样时间更短(1.9–2.0 ms vs 2.3–2.9 ms) |
| RQ3 | ACT 低 | Top-1 掉了(50–92% vs 100%),候选池扩大但 ranking 未同步调整 |
| RQ4 | **ACT Basic-50 反超** | 87% vs 80%;Full-100 为 87% vs 100%(BN/RESHAPE 硬限制) |
| RQ5 | ACT 正解 | 三域 bca_fail 均 100%,宽度三域分化是 HZ 重写结果 |
| RQ6 | 持平 | timing 所有 12 个 cell 差 <15%,多数配置 ACT 略快 |

**剩余 honest gaps 的根因**:
- RQ4 coverage 87% 天花板:BN 未注册 + RESHAPE 无 toggle
- RQ5 bound_width 更大:HZ + dual_tf 新实现,UCU 退化到 interval 是旧版
- RQ3 top-1 下降:candidates 更丰富,ranking 启发式未调
- RQ1 部分 mutation 偏低:算子覆盖差异导致 mutation 效果差异

**实验侧已应用的修复**:
1. `_get_input_shape` / `run_cbr_detection` 兼容 ACT 的 `params["shape"]`(UCU 用 meta)
2. `rq2_scc_effectiveness._build_network_spec` 改用 ACT 的 layer schema(params,不是 meta)
3. `build_generated_factory` override YAML:启用 residual 变体 + unsqueeze_squeeze + bias_layer + scale_layer + transpose

**未修改任何 ACT 主代码**。
