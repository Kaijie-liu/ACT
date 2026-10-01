# ACT / MoE 工程总导航

整理日期：2026-09-28。**本页负责找东西，不是新的实验协议或结果表。**
主仓库 `/data1/Kane/MOE/ACT`；唯一工作分支 `feat/moe-route-verification`。

## 先看哪一份？

| 你要做什么 | 第一入口 | 注意 |
|---|---|---|
| 接手工程、决定下一步 | [当前交接](CODEX_HANDOFF.md) | 已压缩；封存边界优先于旧 NEXT |
| 看算法改进目标与下一门 | [研究目标合同](ALGORITHM_RESEARCH_GOAL.md) | HybridZ 算法、GPU 全流程收益、同源证明及外部竞争分别验收；不是新实验冻结 |
| 看实际 HybridZ 多目标支持 | [CPU 控制与保证](hz_batch_support_controls_20261001.md)、[先行冻结合同](hz_batch_support_design_20261001.md)、[硬预算监督控制](hz_batch_supervision_20261001.md) | 显式启用支持接口；CPU 16 项、监督 14 项控制通过，GPU／生产收益仍未完成 |
| 看支持界如何进入实际 HybridZ 传播 | [先行冻结](hz_checked_propagation_design_20261001.md)、[接线控制与保证](hz_checked_propagation_20261001.md)、[完整 CPU 监督](hz_propagation_supervision_20261001.md) | 真实 analyzer/ReLU 接线及有限监督通过；native 比较、真实请求与 GPU 收益未完成 |
| 看 HybridZ GPU 候选路线 | [设备候选合同](hz_device_candidates_design_20261001.md)、[实现与 CPU 控制](hz_device_candidates_20261001.md)、[设备监督](hz_device_supervision_20261001.md)、[有界生命周期](hz_device_lifecycle_20261001.md) | 新有界清理与归还模拟控制通过；实际 CUDA 清理、实机及端到端收益尚未验证 |
| 看小型实机执行冻结 | [接线控制与准入](hz_physical_device_20261001.md)、[执行配置](../configs/hz_physical_execution_20261001.json) | 八项控制及六次 CPU/模拟调用通过；实机准入拒绝，八个位置未启动，无 CUDA 或加速结果 |
| 看 H2 与实际 HybridZ 接入 | [端点接入结果](hz_endpoint_controls_20261001.md)、[CPU 完整监督](hz_endpoint_supervision_20261001.md)、[映射合同](../act/back_end/moe/proofs/hz_endpoint_mapping.md)、[先行设计](hz_endpoint_design_20261001.md) | 18 项数学控制、12 项监督控制通过；给定 HZ/gate，不是来源完整或生产性能结果 |
| 看声明来源怎样接入实际 HybridZ | [同源控制与边界](hz_source_connection_20261001.md)、[冻结合同](hz_source_connection_design_20261001.md)、[完整 CPU 监督](hz_source_supervision_20261001.md)、[隔离搬迁检查](hz_source_portable_20261001.md) | 来源、完整监督和八组搬迁控制通过；有限保存对象可重查，尚非真实模型或原生浮点证明 |
| 看同源 HybridZ 表示是否真有差异 | [端点与 MC 对照](hz_source_representation_20261001.md)、[先行冻结](hz_source_representation_design_20261001.md) | 四源八臂和 13 项控制通过；同一新 HZ 的正端点界与精确负 MC 可行点形成有限分离，不是真实模型或外部收益 |
| 看新 HybridZ 路径为何尚不能接实模 | [接入诊断](hz_real_intake_20261001.md)、[下一机制设计](hz_binary64_enclosure_design_20261001.md) | 维度／算子／包限制及精确系数不可无损回存已分开；8 项元数据控制，无实模运行，不直接提高上限 |
| 看 H1 代码与合成证明 | [来源依赖控制](h1_sparse_source_controls_20260930.md)、[完整监督与搬迁](h1_supervision_20260930.md)、[对象适配及停止裁决](h1_model_intake_20260930.md) | 适配控制完成；稠密隐藏依赖未缩小，不自动推进真实比较 |
| 看 H2 加权义务机制 | [gate 端点支持](h2_gate_endpoint_20260930.md)、[声明来源控制](h2_source_controls_20260930.md)、[监督与搬迁](h2_supervision_20260930.md)、[对象接入与容量](h2_model_intake_20260930.md)、[分块来源与 pair 表示](h2_factored_representation_20260930.md)、[分块监督与搬迁](h2_factored_supervision_20260930.md)、[全路径容量审计](h2_factored_capacity_20260930.md)、[逐行检查控制](h2_rowwise_controls_20260930.md)、[逐行完整监督](h2_rowwise_supervision_20260930.md)、[数学证明](../act/back_end/moe/proofs/gate_interval_endpoint_support.md) | 新核完整监督控制通过；全尺寸容量未准入，不是新增真实证书 |
| 看完整工程目录 | [目录分类](organization/DIRECTORY_CATALOG.md) | 目录还在原位；分类不是删除许可 |
| 看 H2 全尺寸容量状态 | [来源身份准备与结果](h2_capacity_preparation_20261001.md)、[完整容量监督与结果](h2_capacity_execution_20261001.md) | 两臂输出提议阶段超时；完整 pair 与正证明均为零，真实准入仍关闭 |
| 清理磁盘与防止缓存堆积 | [存储维护规则](organization/STORAGE_MAINTENANCE.md) | 只清经过核对的可重建缓存；实验、失败证据和权重保留 |
| 看当前实验对比 | [比较与保证裁决](competition_guarantee_disposition_20260923.md) | 内部优势不等于外部优势，策略接受不等于严格证书 |
| 看六篇作者基线跑到了哪 | [作者基线实际状态](author_baselines_status_20260924.md) | 分开部署、控制、训练与完整匹配比较 |
| 看最新证明研究结论 | [证明闭合决策](proof_closure_decision_20260925_r1.md) | 停止自动缓存/计时循环，不再重开封存输入 |
| 看论文 | [论文阅读顺序](../paper/README.md)、[短稿](../paper/review_main.tex) | 短稿不是已完成的匿名投稿版本 |
| 做独立评审 | [人类技术审阅与复现地图](SUBMISSION_REVIEW.md) | AI 自检和 JSON 审计不替代独立人类审阅 |
| 查旧讨论和执行历史 | [原始交接全文](CODEX_HANDOFF_HISTORY_20260928.md) | 原样保留 4,632 行，不是待执行队列 |

## 工作区的逻辑分层

```text
/data1/Kane/MOE/
├── README.md                  本机总入口
├── ACT/                       唯一主仓库
│   ├── MOE.code-workspace     VS Code 分区视图
│   ├── act/                   ACT 主实现、MoE 入口、历史管线
│   ├── configs/               顶层冻结协议；MoE 内部另有 configs/
│   ├── scripts/ + tests/      执行、审计、控制（启动前先看冻结协议）
│   ├── docs/                  当前导航、日期化报告、紧凑归档
│   ├── paper/                 论文正文、主表、附录、审阅材料
│   ├── data/moe/results/      大型本地证据，保持原位
│   └── 各证明研究包/           按目录分类表索引，不批量改 import
├── Advice/                    PI 原始指导
├── baselines/                 外部作者仓库与显式兼容版本
├── baseline_runs/             作者训练、认证、攻击与比较运行
├── baseline_weights/          公开与本地产生权重
├── baseline_data/ + datasets/  数据集
├── envs/                      隔离环境；ACT 使用既有 act-py312
├── run/                       本地服务端点等运行资产
└── review / cache / test_*     分别是审阅制品、缓存与控制遗留，不能混删
```

这次采用**导航分层，不做冻结路径迁移**。协议、候选证据、审计和包导入绑定了现有路径；
把几十个研究包简单搬进 `archive/` 会破坏可执行性和来源绑定。详细目录逐项列在分类表中。

## 代码从哪里读？

| 层 | 入口 | 作用 |
|---|---|---|
| ACT / HybridZ | [HybridZ 变换](../act/back_end/hybridz_tf)、[MoE 后端说明](../act/back_end/moe/README.md) | 传播、路由域、加权松弛等基本语义 |
| 主验证请求 | [staged_verifier.py](../act/pipeline/moe/staged_verifier.py) | `verify_staged_linf` 与请求证据包 |
| 调度 | [route_complexity_schedule.py](../act/pipeline/moe/route_complexity_schedule.py) | 按路由复杂度组织剩余预算 |
| 作用域复用 | [scoped_f0_proofs.py](../act/pipeline/moe/scoped_f0_proofs.py) | 有身份与域绑定的逐性质事实 |
| 加权输出 / monolithic | [weighted_top2.py](../act/back_end/moe/weighted_top2.py)、[monolithic_f0.py](../act/back_end/moe/monolithic_f0.py) | 两种求解组织，不能把差异全归因于关系保留 |
| 外部加权路径 | [external_pair_comparison.py](../act/pipeline/moe/external_pair_comparison.py) | ACT 前端＋plain CROWN，不是完整独立动态 MoE 工具 |
| 作者基线 | [近期基线设计](recent_moe_baselines_20260921.md)、[实际完成状态](author_baselines_status_20260924.md) | 先看实际状态，再按协议找 runner |
| 来源 / LP / 精确证据 | [证明模块分类](organization/DIRECTORY_CATALOG.md) | 不把所有实验性模块当作生产入口能力 |

主要已评估入口的范围仍是 eval、CPU/float64、输出层 selected-softmax top-2。
存在 top-1、多层或作者适配模块，不等于这些范围已有完整严格认证或论文规模比较。

## 实验证据如何找、如何读？

按 **协议/选择 → 执行终态 → 独立审计 → 派生分析 → 论文主张** 读取。
不要只看一个 `PASS`，也不要在失败后修改冻结的阈值、分母或对象。

- 当前主结果：[主表](../paper/results/main_tables.md)、[论文证据表](../paper/evidence_table.md)。
- 数值保证修正：[主表输入来源审计](main_table_source_applicability_20260921.md)。
- 最新真实证明终态：[闭合账本](proof_closure_20260925_r1.json)。
- 顶层协议：[configs/](../configs)；原 MoE 管线协议：[act/pipeline/moe/configs/](../act/pipeline/moe/configs)。
- 紧凑报告：[docs/](.) 与 [MoE 管线文档](../act/pipeline/moe/docs)。
- 大型本地证据：`data/moe/results/`、`baseline_runs/`；权重/数据不提交到 Git。

**状态优先级：对应执行的审计终态 > 较早的冻结/计划 > 聊天中的意向。**
同一工作线用当前交接定位最新裁决；旧日期文档的“下一步”不会自动复活。

## 日常操作与维护

在 VS Code 用“从文件打开工作区”打开 [MOE.code-workspace](../MOE.code-workspace)。
这只提供分区视图，隐藏 Python 缓存，并从默认搜索排除部分大型生成目录；
没有改变本机全局设置，没有隐藏失败状态或删除证据。需要搜原始结果时可关闭搜索排除。

```sh
cd /data1/Kane/MOE/ACT
git status --short --branch
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/check_project_layout.py --workspace
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S scripts/summarize_proof_closure.py --check
```

第一项检查目录分类、导航、历史交接字节和 619 个已记录实现绑定；第二项重算已归档账本。
它们不加载模型、不训练、不求解，也不是完整数学证明检查。
新增顶层研究目录时，更新 [layout.json](organization/layout.json)，再重生成并检查目录分类表。
生成命令为 `scripts/check_project_layout.py --render`（仅输出文本，不自动写文件）。

## 保留与清理边界

| 类型 | 本次处理 / 后续规则 |
|---|---|
| 冻结配置、原始结果、失败、证据和 checkpoint | **原位保留**；禁止凭文件名/年龄批量删除 |
| 外部仓库、隔离环境、数据、权重 | 原位保留；迁移或去重需单独核对引用、许可、恢复方案 |
| `test_*`、`parsed_source_controls_*` 等遗留 | 仅分类；先查引用和运行进程，可能包含失败现场 |
| pip/解析/Python 缓存、巨大日志 | 按[存储维护规则](organization/STORAGE_MAINTENANCE.md)逐项核对；仅 allowlist 下载/字节码缓存可计划删除，数据缓存和日志不混删 |
| `act/pipeline/log/pipeline_tests.log` | 用户确认有意删除；已单独提交 `711dae7a8`，可从 Git 历史恢复 |
| 4632 行旧交接 | 原样存入同目录历史文件，SHA-256 绑定；当前交接只保留有效状态 |

新控制应使用自动清理的专用临时目录；确需保留的失败必须进入有身份与说明的运行目录。
这只是今后的目录约定，不修改旧测试、冻结脚本或历史路径。整理不会升级任何科学主张。

本次变更范围与验证结果见 [整理记录](organization/ORGANIZATION_20260928.md)。
