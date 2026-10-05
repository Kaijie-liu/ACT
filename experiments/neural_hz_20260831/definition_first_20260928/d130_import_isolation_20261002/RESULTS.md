# 真实 Attention 局部改善与完整来源超时

本候选完成数学组件测试，并在原始 ViT 模型系数上保存了部分中间层改善；完整来源实验未通过。不能将这些局部结果算作网络 CERT、validated ADV、完整 Neural-HZ 域、新颖性或 GPU 成果。唯一会话 22072 已终态退出 1，不修改或重跑冻结候选。

## 数学组件与实际来源分开记账

[退出回执](../../results/d130_import_isolation_20261002_v1/exit.json)记录 3953 项测试、207 个文件全部通过，零失败、错误或跳过。包含启动、收集和 JUnit 的时间为 48.93076469562948 秒，pytest 自报 47.75 秒。完整保留 D129 登记的 3933 项，并添加本版 20 项；importlib 模式只解决同名模块收集冲突，没有删减测试。

D128 的 protobuf 容器接口失败和 D129 的收集失败均保留原状。本版数学资格不转授给它们，也不把组件测试数量当作实际网络验证数量。[D128 结果](../d128_bounded_attention_source_20261002/RESULTS.md)及[D129 结果](../d129_source_container_compat_20261002/RESULTS.md)说明各自停止原因。

预注册人口为两个模型首块 CLS 的全部 96 加 96 个后继前激活，共 192 行、1152 个有符号单头 root 查询。实际只完成第一个模型 pgd_2_3_16 的第 0 至 37 行，即 38 行。第 38 个方向只完成五个 root，没有完整读出；累计 roots=233 不能换算为 39 个完成行。IBP 模型未开始，完整模型 ledger 没有发布。

这里的 PGD 是原模型文件名称，不是本候选执行了攻击。来源 worker 的 model_forward_calls、diagnostic_solver_calls、new_benchmark_solves 均为 0。

## 部分精度证据及其比较范围

对已保存 38 个完整 row JSON 作只读聚合，相对同一认证系数信息的独立 score/value 矩形参考，38 行的激活前下界、上界均严格改善。11 行的后继 ReLU 下界改善，13 行的上界改善；没有新增稳定激活门或稳定关闭门。这是部分样本结果，不能当作完整 192 行的通过率。

例如[第 0 行](../../results/d130_import_isolation_20261002_v1/row_0_0.json)的候选区间约为 [5.5792054557, 6.4117923779]，参考约为 [5.5312636308, 6.4517050093]。精确数值以文件中的有理数为准。该行原本就为正，幅值收紧不是新相位判定。

参考不是完整旧 HZ，也不是全部多神经元或 Attention 方法。当前证据使用原网络 FLOAT 常量的实数解释和可靠系数区间，不是具体框架浮点 Softmax 的全网认证。不同方向与 head 的界不能拼成一个实际输入；查询的 root 下端点不是实际 Attention 值的下界，更不是 ADV 见证。

## 完整来源门失败

[来源回执](../../results/d130_import_isolation_20261002_v1/diagnostic.json)记录内部 235 秒定时器触发 TimeoutError，给原 240 秒外部门保留终态证据时间。worker 自报 235.00021431222558 秒，监督器观察 235.17014246806502 秒。不是工作量上限触发，也不是仍在后台运行。

累计 whole work 为 225216543／256000000，模型阶段为 182531801／200000000；evidence 实耗 719298，40M evidence 已预付包含在 whole 内，不能再次扣除后宣称便宜。部分 retained entry upper 为 1338166。RSS 高水位增长 166408192 字节，traced peak 50560667，tracer metadata 39176768。部分观察没有超出对应上限，不等于完整物理成本资格通过。

source_census_completed、source_census_qualified、source_component_qualified、native_HZ_admitted、actual_model_binding_qualified、actual_phase_column_binding_verified、complete_physical_qualification、gpu_computation_completed 均为 false。数学组件过门不能覆盖上述缺失；shadow 和完整回放没有执行。

## 费用定位不等于下一候选过门

对最先保存的 21 行作只读费用审计：总增量 76568890，查询侧 52519585。每行另有固定方向绑定费 1137013 和记录预付 8192。六个 root 的 work 是同一个查询预算的累计值，不能当六项独立费用相加。

同一方向和 head 的 patch 生成元只相差非负输入宽度，因此可研究共享角序；已认证循环边界也允许研究线性上下弧提取，避免通用 hull 重建。但当前没有保存具体比较次数，不能声称节省比例或完整 256M／240s 能通过。负方向反射已经在本版执行，不能重复记作未来收益。后续算法改变需要新证明、新预注册和新冻结，本记录不授权重跑。

## 正式口径与归档

正式成绩仍为 1870／2413，即 1063 CERT 与 807 validated ADV；独立 E0 仍为 CIFAR100 25 加 TinyImageNet 36，共 61／400。两边新增均为 0。旧账未改不等于新候选已证明保住全部旧解。候选默认关闭，完整 Goal 保持 active。

2026-10-02 Australia/Sydney；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。仅新增隔离总结，历史模型、冻结来源和生产默认未改；无 commit 或 push。

freeze SHA256 为 01a893685f9994a879315e69aecf59d5e4e75f20c0bb8eb7dd582e1201c6bb01；exit 为 39d9029d448bffe6b25b34246b5da20756df78a80c6c151993f48a1ed49b0514；diagnostic 为 62592228b34ad225cc3cb0783f03c936af8ad41a1483129e71da9d1f5dcc1d75。执行结束时 52 个工件与 12 个冻结文件已核查；本总结另作事后文档封存，不冒充执行前冻结。

write-page 技能用于把部分正证、完整失败和资格边界分开记录，并遵循现有本地 Markdown 归档格式；未创建外部 Page，网页排版未验证。
