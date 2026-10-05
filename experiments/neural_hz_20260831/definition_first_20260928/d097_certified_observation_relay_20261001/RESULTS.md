# 实际出生证书与跨层共同观察的组合结果

D097 已一次通过 3841 项数学测试、187 个文件。它将 D096 的实际 ReLU 出生证书同时用于基础关系和后继共同观察，保留了此前普通多层块上的严格松弛收益。该结果是非凸 Neural HZ 定义研究的组件进展，不是新抽象域已经完成、预训练网络能力提升或 GPU 加速成绩。

## 单次执行证据

六个候选文件先由 [freeze.json](freeze.json) 冻结，再执行唯一版本。合同和人口见 [CONTRACT.md](CONTRACT.md)、[PREREG.md](PREREG.md)。本归档只读取既有证据，没有重新导入、收集或执行候选。

唯一结果目录为 `results/d097_certified_observation_relay_20261001_v1`。原会话 55968 已确认退出码 0；[退出回执](../../results/d097_certified_observation_relay_20261001_v1/exit.json)、[日志](../../results/d097_certified_observation_relay_20261001_v1/tests.log)、[JUnit](../../results/d097_certified_observation_relay_20261001_v1/tests.xml)、[测试清单](../../results/d097_certified_observation_relay_20261001_v1/inventory.json)及[运行登记](../../results/d097_certified_observation_relay_20261001_v1/preregistered.json)均保留。

完整继承 D096 的 3837 项测试、186 个文件，新增四项固定测试；结果为 3841 passed、13 warnings，没有失败、错误或跳过。测试子进程含启动、导入、收集、测试与 JUnit 共 48.54383327625692 秒，pytest 内部报告 47.28 秒；监督器含前后身份核查共 59.451119020581245 秒。原 CPU 1、单线程、CUDA 空、AS 16 GiB 和测试子进程 60 秒门没有改变。

`component_tests_passed`、`mathematical_component_gate_passed`、`inventory_validated_before_execution` 和 `all_stages_passed` 为 true；后者只指本次预注册的数学阶段，不包括未登记的模型或回放阶段。`source_drift=[]`、`input_drift=[]`、`provenance_drift=false`。监督器 traced peak 为 14224519 bytes、tracer metadata 为 4860784 bytes、RSS highwater 增长 0；这些仅属监督器范围，不是候选完整物理资格。

## 已获测试支持的结论

每个 BornGate 恰好一次读取实际出生 EQ、guard 和 carrier EQ 进行认证。基础关系与共同观察消费同一已认证预激活，而不是一处使用规范表达式、另一处继续把固定常量列当作自由盒变量。普通控制确认原提取器的余项半径为 7/8，而实际等式等价表达式的相应非恒定半径为 0。原连续因子、全部原 signed 二元相位、EQ/LE、输出、共享 frame 和输入前缀均保留；没有删列、松弛二元因子或改写旧谓词。

普通块为 `u=ReLU(x)`、`v=ReLU(y)`、`r=ReLU(u+v-3/4)`、`t=ReLU(u-v+1/4)`、`w=ReLU(r+t-1/2)`。实际出生给 w 的界为 [-1/2, 2]，没有采用旧控制中更紧的外给上界。原松弛和基础关系可以满足的一个点，在新增共同观察下对所有辅助扩展均被排除：理想间隙为 1/24，计入实际系数和 RHS 外向补偿后仍大于 1/48。该结构间隙沿用 D091/D092，本次贡献是验证它能在实际出生证书驱动的组合中保留，不重复记为新发现。

同一组原状态具有共同辅助解释，零点的两种原相位标签均保留。混号权重、普通偏置、原连续和二元余项均有控制。测试专用参考对象用于比较旧公式与新组合；生产候选没有执行该参考改写。

宽结构控制包含全部 64 个不同后继、68 个出生证书；每证书认证一次，raw graph 提取为 0。新增 195 个连续辅助量和 590 条 LE，原 68 个二元因子全部保留。这是完整所给结构组件的人口，不是整张真实网络的自动结构发现，也不是存储压缩声明。硬 pool 的局部收费核对通过，但不能替代全生命周期物理成本证明。

默认关闭、错误证书集合、实际行被改动、坏稀疏参数、支撑或有理位宽超限、预算不足均拒绝，输入保持不变。源码中的四项新增测试分别为 `test_born_relations_match_zero_constant_reference`、`test_common_extension_and_strict_next_layer_gain`、`test_full_bank_reuses_authenticated_relations` 和 `test_certified_binding_and_resource_fail_closed`。

## 尚未获得的资格

`actual_model_binding_qualified`、`actual_phase_column_binding_verified`、`source_component_qualified`、`source_census_qualified`、`native_HZ_admitted`、`complete_physical_qualification` 和 `gpu_computation_completed` 全为 false。`candidate_physical_gate_evaluated=false`、`worker_launched=false`。没有执行真实模型、终端 LP/MILP、GPU、shadow、逐家族或 2413 例完整回放。

所给证书必须覆盖本次结构的全部角色，但尚未建立真实网络上的完整选择、来源认证和在线生命周期。生产 frame 高水位、已有槽复用、deferred 部分物化、rebase 后的身份运输及完整成本仍待验证。D096 的单位盒界可能扩大真实不稳定人口，不能从小块证明推断 13 家族零回退。

下一步应做新预注册的真实同结构组合与完整代价检查，而非继续把常量编码、包装器或测试数量当作域创新。D088/D090 已消费失败、D095 未执行草稿均保持原状，不重跑旧版本。

## 记账与归档范围

2026 年 10 月 1 日，分支 `redu-hz`，commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`。正式 1870/2413（1063 CERT、807 validated ADV）与独立 CIFAR100 25、TinyImageNet 36，共 61/400 不变，`formal_gain=0`。候选默认关闭，Goal 未完成。

本次补归档只新增此结果说明、实验根目录的续接入口及 SHA256 清单；未改冻结源码、原始回执、生产代码和历史模型，没有 commit、push 或异机备份。文档技能用于将执行证据、数学意义、既有机制和未获资格分开记录。[最新续接入口](../../RESUME_RESEARCH_20261001_D097.md)串联旧支撑成果、文献和后续义务。
