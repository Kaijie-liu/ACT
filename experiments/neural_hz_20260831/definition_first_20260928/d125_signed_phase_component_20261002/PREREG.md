# 原相位条件外包的单次数学资格实验

本候选的前置研究完成了新的纸面定理、严格正反例和真实结构入口；紧接本次数值推进之前的用户回合仅状态核查，按 Goal 口径为 no progress。本轮依次完成默认关闭的 PWA 抽象变换、冻结和完整数学测试。本文件注册的只有数学阶段，不将通过小例改称真实网络收益，不取消后续实际消费者、GPU、shadow 或完整 replay。

## 冻结与唯一执行

任何新候选 AST/import/compile、数学运算脚本、pytest 收集或执行前，冻结这八个文件：CONTRACT.md、PREREG.md、phase_envelope.py、multiplier.py、test_phase_envelope.py、test_multiplier.py、run_math.py、collection_contract.py。可以在冻结前只读审核及编辑尚未消费的新源码，不运行它。冻结后不修改或重跑消费版本。

唯一 RUN 为 experiments/neural_hz_20260831/results/d125_signed_phase_component_20261002_v1；独占创建即消费本版本。只运行一次 `/data1/Kane/miniconda3/bin/python -B .../run_math.py --enabled`。runner 在候选导入前认证冻结源码、所有继承身份、解释器、当前分支/commit/diff、模型/性质身份及完整项目 import closure；然后单个 pytest 进程完成收集、全部执行和 JUnit。没有先行 filtered/collect-only 试跑。

schema 为 d125_signed_phase_component_v1；freeze 明确 mathematical_stage_only=true、worker_stage_registered=false。父 manifest SHA 使用 NEURAL_HZ_D125_MANIFEST_SHA256。NEURAL_HZ_ACTIVE_COMPONENT_RUN 指向本次新 RUN；D112 的四项旧证据控制只在认证后将 in-memory RUN 重定位到本次 inherited_d112_controls，不修改旧测试文件、断言或输入。

## 完整测试人口

完整、有序继承 D120 的 3869 项/194 文件，添加以下十二个无参普通函数，无 decorator 或参数化，合计 3881 项/196 文件：

1. test_phase_envelope.py::test_phase_positive_forward_control
2. test_phase_envelope.py::test_phase_zero_rho_labels
3. test_phase_envelope.py::test_phase_signed_reference_decomposition
4. test_phase_envelope.py::test_phase_decoder_rejects_lossy_point
5. test_phase_envelope.py::test_phase_preserves_shared_hz_state
6. test_phase_envelope.py::test_phase_fail_closed_guards_and_budgets
7. test_multiplier.py::test_multiplier_genuine_interior_optimum
8. test_multiplier.py::test_multiplier_boundary_infima
9. test_multiplier.py::test_multiplier_constant_plateau_witness
10. test_multiplier.py::test_multiplier_zero_weight
11. test_multiplier.py::test_multiplier_constant_and_zero_coefficients
12. test_multiplier.py::test_multiplier_general_source_box

前六项验证 D124 普通有偏置混权正控、旧 LP 明确可行点、前向后继关闭、零误差全部合法原标签及全系统精确性作用域、非零误差伪点拒绝、tau/有符号组合、全部旧 bit/source身份及未替换谓词/删除变量界保存；缺守卫、活跃外部消费者、错误 frame、非严格 opt-in、资源与位宽失败均不部分提交。后六项验证 common-kappa 内部最优、边界下确界与有限 witness、常值段、无正权、退化系数及非对称源箱。直接 Fraction 代数是数学测试，不是原网络执行或新 solver 查询。

collection plugin 在测试执行前核实全部有序 node IDs 和源路径，继承项不得跳过、筛选或重排。保留原固定组件 LP 对照，不新增 attack/PGD、BaB、输入/相位 split、backward/dual rescue。

## 原门与来源资格

pytest 包括进程启动、收集、执行、JUnit 的合计上限仍为 60 秒；CPU1、单线程、CUDA 空、AS16GiB。独立新 tmp/cache/evidence 路径，禁止写历史 RUN。监督器保留 1GiB 观察内存门和 65536 字节 summary reserve，原数学测试中的既有预算原样保留。没有提高测试门、减少人口或用 GPU 后备绕过失败。

只读继承并核验 D120 完成的 source diagnostic、complete_0..2 和对应来源/依赖/失败记录，但本次不启动 source worker。数学通过不授予新的 source/native/model/phase/GPU/full-physical 资格。256M whole work、200M/model、40M evidence、64M entries、512-bit rational、240秒来源 worker 等后续来源门不因本数学阶段而改变，也不在没有来源运行时虚报通过。

新 run 必须自动保留 preregistered.json、完整 inventory、tests.log、JUnit、继承控制证据和 exit.json；失败或 timeout 同样终态保存。观察超时不是重新启动的理由。source/input/provenance 漂移、跳过/遗漏测试、资源或数值失败使本版本资格失败；不在该版本上补丁重试。

## 判断与下一阶段

数学阶段必须全部通过，才能把组件标为数学资格通过。确认它确实删除发出系统的父连续列、保持原 bit 身份和所有相关限制；同时如实记录 Projection.original 仍保留旧系统，不能据此宣布物理节省。

真实来源必须另行冻结实际联合消费者和下一激活，不只统计局部证书命中。旧 D120 的 1600 个局部消费者或本轮十二项控制都不能替代全 2413/400 回放。任何正式收益仍为零，默认关闭，原 1870 和独立 61 基线只读。无生产改动、commit 或 push；完整 Goal 未完成。
