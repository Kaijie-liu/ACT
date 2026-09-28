# 工程目录分类（路径保持原位）

由 `scripts/check_project_layout.py --render` 根据 [layout.json](layout.json) 生成。
这是导航，不是目录迁移、删除清单或新的执行授权。

## ACT 仓库顶层目录

### ACT 主实现

当前方法及通用 ACT；支持范围以具体入口合同为准。

- [act/](../../act)
- [modules/](../../modules)
- [data/](../../data)
- [vnncomp/](../../vnncomp)
- [ipynb/](../../ipynb)

### 协议、论文、文档与测试

维护入口；日期化协议/结果不可覆盖。

- [configs/](../../configs)
- [docs/](../../docs)
- [paper/](../../paper)
- [scripts/](../../scripts)
- [tests/](../../tests)
- [.github/](../../.github)
- [.vscode/](../../.vscode)

### 证据格式、LP 检查与通用请求

已存在的分阶段证据研究；不等于完整网络证明。

- [moe_evidence/](../../moe_evidence)
- [portable_proof/](../../portable_proof)
- [checked_gate/](../../checked_gate)
- [exact_matrix_cache/](../../exact_matrix_cache)
- [exact_primal/](../../exact_primal)
- [lp_sandwich/](../../lp_sandwich)
- [lp_diagnostic/](../../lp_diagnostic)
- [lp_diagnostic_archive/](../../lp_diagnostic_archive)
- [nonpositive_analysis/](../../nonpositive_analysis)
- [property_diagnosis/](../../property_diagnosis)
- [property_ranges/](../../property_ranges)
- [range_diagnosis/](../../range_diagnosis)
- [range_pipeline/](../../range_pipeline)
- [evidence_cohort/](../../evidence_cohort)
- [cohort_analysis/](../../cohort_analysis)
- [batched_evidence/](../../batched_evidence)
- [bounded_evidence/](../../bounded_evidence)
- [cached_portable/](../../cached_portable)
- [single_check_portable/](../../single_check_portable)
- [evidence_handoff/](../../evidence_handoff)

### 上游、缓存与证据模式集成

保留旧执行/比较；不是自动扩样队列。

- [upstream_portable/](../../upstream_portable)
- [upstream_archive/](../../upstream_archive)
- [upstream_cost_analysis/](../../upstream_cost_analysis)
- [upstream_reuse/](../../upstream_reuse)
- [reuse_supervised/](../../reuse_supervised)
- [reuse_archive/](../../reuse_archive)
- [source_cache_ablation/](../../source_cache_ablation)
- [source_cache_archive/](../../source_cache_archive)

### 来源与变换检查

检查范围必须连接到同一对象的新输出义务。

- [router_source/](../../router_source)
- [source_enclosure/](../../source_enclosure)
- [upstream_source/](../../upstream_source)
- [source_ranges/](../../source_ranges)
- [full_source/](../../full_source)
- [full_bounds/](../../full_bounds)

### 真实请求证明研究

旧输入封存；来源到完整正输出仍未闭合。

- [frontier_proof/](../../frontier_proof)
- [checked_route_frontier/](../../checked_route_frontier)
- [residual_proof/](../../residual_proof)
- [shared_route_residual/](../../shared_route_residual)
- [scoped_parse_proof/](../../scoped_parse_proof)
- [scoped_proof/](../../scoped_proof)
- [scoped_source/](../../scoped_source)

### 来源构造与解析成本研究

本轮研究已停止；可选项保留、默认关闭。

- [source_construction_lab/](../../source_construction_lab)
- [source_cost_controls/](../../source_cost_controls)
- [source_cost_supervised/](../../source_cost_supervised)
- [parsed_source_reuse/](../../parsed_source_reuse)
- [readonly_source/](../../readonly_source)
- [readonly_upstream/](../../readonly_upstream)

### 精确基、稀疏消元与原生工具对照

历史有限研究，保留控制和失败；不自动继续自研消元。

- [amortized_basis/](../../amortized_basis)
- [basis_compatibility/](../../basis_compatibility)
- [basis_supervised/](../../basis_supervised)
- [exact_basis/](../../exact_basis)
- [fidelity_diagnostic_archive/](../../fidelity_diagnostic_archive)
- [fidelity_supervised/](../../fidelity_supervised)
- [modular_basis/](../../modular_basis)
- [modular_diagnostic/](../../modular_diagnostic)
- [modular_supervised/](../../modular_supervised)
- [native_basis/](../../native_basis)
- [native_import_audit/](../../native_import_audit)
- [plan_basis/](../../plan_basis)
- [primitive_basis/](../../primitive_basis)
- [primitive_diagnostic/](../../primitive_diagnostic)
- [primitive_supervised/](../../primitive_supervised)
- [soplex_compat/](../../soplex_compat)
- [soplex_detached/](../../soplex_detached)
- [soplex_execution/](../../soplex_execution)
- [soplex_fidelity/](../../soplex_fidelity)
- [sparse_basis/](../../sparse_basis)
- [sparse_diagnostic_archive/](../../sparse_diagnostic_archive)
- [sparse_supervised/](../../sparse_supervised)

## MOE 工作区目录与资产

下表按名称分类，不扫描环境、原始数据或权重内容。
`--workspace` 会列出实际条目及数量，并拒绝未分类的新目录；不会清理任何条目。

| 类别 | 原位名称 / 模式 | 保留规则 |
|---|---|---|
| 主仓库与导航 | `ACT`、`README.md` | 代码与导航；仅在指定分支修改 |
| 研究指导 | `Advice` | 保留原文；晚期审计可修正较早指导的事实判断 |
| 作者制品与环境 | `baselines`、`envs` | 不移动，不直接升级，不提交到 ACT 仓库 |
| 数据与权重 | `baseline_data`、`baseline_weights`、`datasets` | 不删除/去重；身份、许可、引用和恢复方案须先核实 |
| 作者运行与本地服务 | `baseline_runs`、`run` | 保留失败、终态、日志与运行路径；socket 不代表任务在运行 |
| 审阅与可搬迁制品 | `competition_claims_review_*`、`submission-review-*`、`proof-closure-review-*`、`review-artifact-*`、`portable_conv98_*` | 冻结/评审来源，原位保留；不是临时垃圾 |
| 缓存与工具状态 | `cache`、`.pycache`、`.tmp`、`.tmux`、`.vscode`、`codex-stage2-cache.*`、`tmp` | 仅列为核查候选；不能据名称删除 |
| 控制测试遗留目录 | `test_*`、`parsed_source_controls_*`、`soplex_life_*`、`mainline_integration_*` | 可能含失败证据；单独检查引用和进程后才可提出回收清单 |

## 查实验不要只看目录名

- 当前状态以 [当前交接](../CODEX_HANDOFF.md) 指向的对应审计为准。
- `*_archive` 可能是审计代码包，不是可删除的旧结果。
- `test_*` 可能保留超时、错误和部分证据，不能按前缀直接删除。
- 作者仓库/环境/数据/权重与 ACT 不同仓，不应递归提交。
- 顶层根文件包括 Git 元数据、README、环境声明、许可证和工作区配置；
  分类表主要覆盖目录，保留这些根文件原位。

[返回工程总导航](../PROJECT_INDEX.md)
