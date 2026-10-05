# 共同源生成组件的一次数学资格实验

本版本实现 D052 的静态系数匹配、真实区间支持和单消费者完整投影。它只取得候选数学资格的机会，不能授予新域、真实 source、native、GPU 或正式回放资格。所有候选默认关闭，原 HZ、全部 bits、源谓词和 decoder 不变。

## 固定人口与执行前冻结

完整继承 D049 已通过的 3785 tests、173 files 及精确 node IDs，不减少人口、skip 或替换旧文件。新增 test_static_source.py 的四个普通顶层、无参数化测试，顺序和名称固定：

1. test_exact_matching_and_projection
2. test_nonpoint_parameters_and_zero_phases
3. test_strict_control_and_forward_use
4. test_disabled_identity_and_rejection

合计必须为 3789 tests、174 files。新增内容覆盖 THEORY.md 与 CONTROL.md 指定数学范围，包括 interval 非点参数而非仅点参数实现。测试中的少量有限真值 fixture 不迁移为运行时相位/输入枚举。

执行前冻结本文件、THEORY.md、CONTROL.md、static_source.py、test_static_source.py、run_math.py 六文件及上述四名称，schema=d053_frozen_v1。冻结前仅静态阅读或 AST 检查；不先导入、collect、单跑或调参后再冻结。

唯一新 RUN 为 experiments/neural_hz_20260831/results/d053_static_source_component_20260930_v1。首次 --enabled 独占创建并消耗版本；失败也保存，不重跑、不降低人口、不放宽预算。所有实验 cache、temp、日志、JUnit、人口与身份清单只写该新目录。

## 保留的原门

解释器固定 /data1/Kane/miniconda3/bin/python，assertions 开启，禁止 bytecode。CPU1、AS16GiB、库单线程、CUDA_VISIBLE_DEVICES 为空。collection 加 execution 的总窗口仍为 60 秒；没有数值 worker，不能把新测试转移到 240 秒 worker 窗口。

监督器保留 1GiB RSS high-water growth 和 tracemalloc peak+metadata+65536 两项观察。pytest 子进程的资源限制不等于完整聚合物理峰值认证。所有异常保留已有日志、工件、终态与最终身份检查。

认证 D049 的 preregistered、inventory、exit、freeze、runner，继承它的全部 source/input/provenance、原三模型/性质身份、decoder 和 GPU dependency 清单。可复用已认证 D038、D015、D017 的只读帮助函数，不调用旧 main、worker、writer，不修改旧 globals。production provenance 须执行前后相同，分支为 redu-hz。

D049 的 nested prior_attempt 中 D047 三源 census 失败状态原样保留；不能将数学通过升级为 source 成功。原三源人口没有缩成两个，本轮不补跑 D047。原 whole_work_cap=256000000、branch_work_cap=200000000、evidence_prepaid_work=40000000、retained_entry_cap=64000000 保留给之后完整物理候选，不声称本数学运行支付或通过这些门。

## 数学接口与拒绝边界

所有参数端点和逐步有理结果仍限 512 位；稀疏输入出现次数及支持运算的实际组合仍限 65536。统一静态数学规则不读取实例、家族、标签、history、margin 或 LP 状态。默认关闭/坏输入/资源/身份/前提失败不产生验证结论、不移除旧状态。

原 phase tokens 必须绑定各自不同的输出，但这只是形式身份检查，不认证真实 native 列。原真实模型、active 方向、source/参数 enclosure、frame 生命周期和 decoder 尚待集成认证。非点参数不通过中点替换为另一个网络。

完整 source 支持、排序、临时映射、P/N 与输入证据、返回两行复制、真实 latent 展开、旧谓词、终端与见证成本均须后续支付；当前未测 GPU 或端到端加速。不会仅凭数组规模授予物理资格。

## 晋级与记账

本轮无 worker；all_stages_passed 只表示全部预注册数学阶段通过。始终 source_census_completed=false、source_census_qualified=false、actual_phase_column_binding_verified=false、native_HZ_admitted=false、gpu_computation_completed=false、complete_physical_qualification=false、formal_gain=0。

即使数学通过，仍要真实同结构、shadow、逐家族与同候选全部 2413 回放，保住每个旧 CERT/validated ADV、13 家族零回退与 invalid ADV=0 后才可能更新正式成绩。独立 E0 400 回放、保住61及两家族零回退也不省略。能力四并发不回退门保留，纯速度 1.5x/2.0x/1.8x 门不混入能力门。

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式 baseline 为 1870/2413=1063 CERT+807 validated ADV，独立 E0 CIFAR100 25、TinyImageNet 36，共61/400；两者不相加。只写新隔离目录与唯一 RUN，历史模型、结果、冻结来源和 /data1/Kane/HyZor 只读。生产默认、旧 dirty changes、commit/push 均不改。

按 write-page 技能将已知机制、数学证明、执行结果和正式资格分开保存。整体目标不会因本组件通过而完成或缩小。
