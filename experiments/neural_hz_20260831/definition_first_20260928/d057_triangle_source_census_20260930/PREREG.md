# 三个真实模型的共享相位结构实验

本次检验已过数学门的共享相位组件是否适用于原三个真实模型的固定卷积结构。重点是实际门关系，不再增加独立的小型能力故事。它不是新抽象域资格或 benchmark 回放：当前表示仍是原非凸 HZ 加已证明的共同源后果，经典三角机制本身不具备新颖性。

## 不变的人口与统一规则

模型及性质、原字节和 decoder 依赖完整继承 D056 的 selected_sources，固定三个来源：CIFAR100 large、CIFAR100 medium、TinyImageNet medium。全部 admitted 第一 ReLU 的 immediate Conv 到下一 ReLU 分支，仍用各分支四角及中心的原固定 anchors、全部接收通道、全部 canonical slots 与 padding；不是整个模型的所有层或所有空间位置，更不是两个家族全性质。

每个窗口先计算所有接收门的可靠预激活界。源是 first-bank 激活后的原坐标，先由真实 first Conv/BN/input property 计算其预激活外包，再取非负部分，不能直接把预激活盒作为接收 Conv 的源。实际第二 bank 权重、bias 和 post-BN 仍用原可靠区间复合，不使用中点网络。

若 n 个接收门中 m 个不能由 lo>0 或 hi<0 严格证明稳定，则统一记录 C(n,3) 个总组、C(n,3)-C(m,3) 个稳定省略组，以及其余全部 C(m,3) 个不同原门三元组。触零门属于后者。此规则按数学结构执行，不按模型/实例、历史标签、margin、求解状态或上轮记录中的通道列表选组。Large 即使无组也必须完整处理原人口；Tiny 不得省略。

每窗口按原 phase ordinal 固定方向，对全部所需边各生成一次 D053 证书，局部共享这些证书，再对全部剩余三元组使用与冻结 D056.generate 相同的合成公式。pair证据每窗口只保存一次，所有三元组引用其原边并保留状态和全部返回行；偶号/零边无增强组同样记录。新数学测试检查与完整重算D056逐行等价，不因复用而继承未证明的组合资格。没有原二元因子的 pivot、删除、连续放松或相位改写。

冻结前的静态设计审查曾考虑每组三门完整重算三条证书；该方案会重复支付共同边的支持计算，故在任何导入、测试或模型执行前改为上述局部共享。缓存的真实typed存储、身份、plain证据及合成工作全部计费，不把缓存当免费，也没有调高预算或缩小人口。这是新组合实现的支撑步骤，不宣称缓存本身为域创新。

## 数学门与唯一执行

完整继承 D056 的3793项、175文件及精确node IDs。新增 test_window_relation.py 的四个普通无参数化顶层测试：test_complete_window_control、test_full_population_and_zero_phases、test_plain_identity_and_intervals、test_fail_closed_budget_and_default_off。总人口3797项、176文件；禁止skip或缩小旧人口。

freeze.json schema=d057_frozen_v1，首次候选导入前绑定本文件、THEORY.md、window_relation.py、test_window_relation.py、census.py、run_census.py六文件及四测试名。冻结前只允许静态检查，不能先试跑再选参数。唯一新RUN为 experiments/neural_hz_20260831/results/d057_triangle_source_census_20260930_v1，独占创建；成功或失败均消耗版本，不重跑，不覆盖旧结果。

解释器仍为/data1/Kane/miniconda3/bin/python，assertions开启、禁止bytecode；CPU1、AS16GiB、库单线程、无CUDA。collection加execution总限60秒；只有完整数学门成功才允许启动新的census.py，worker仍限240秒。两阶段时间不可互借。所有cache/temp/日志/JUnit/原始证据只写新RUN。

## 完整资源与证据

原whole-work 256M、per-model 200M、三模型共用40M evidence、64M retained numeric entries、512位有理数与65536 summary reserve不变。新bridge显式支付所有通道/slots检查、普通界、typed frame/context/forms、共享边构造和缓存、全部三角合成、排序、序列化和返回行。调用前按实际支持支付冻结未计量组件的工作；不得将未计量调用当零。具体公式及临时存储边界写于window_relation.py与THEORY.md。

原reader/parse amortized收费合约不变，新桥接工作另计。实际raw模型、spec、packet、box、接收参数、source bound cache、认证roots、plain证据及报告全部进入旧bounded ledger；typed窗口temporaries另有保守上界，不能直接交给不支持dataclass/MappingProxy/token的旧ledger或记为零。每个process的RSS high-water growth+reserve与tracemalloc peak+metadata+reserve各不超过1GiB，不宣称完整主机设备聚合物理认证。

Supervisor继承并核对全部source/input/provenance、解释器、decoder及GPU依赖，执行前后检查漂移。Worker在导入前认证全部实际项目依赖、冻结参数和完整测试人口。每模型完整证据独占写partial并成功后发布为complete；失败保留已经完成的证据和终态diagnostic，不能把部分成功当作三源通过。Supervisor独立核对几何、全部窗口/通道/slots/三元组、原身份引用、有限有理行和总计。

## 允许报告与禁止记分

本轮可以报告完整结构人口、严格稳定省略数、已生成奇负三角数、新行及nnz、明确限定的相位盒代数差、实际资源与完成状态。相位盒代数差不是满足网络guard/source的点，不证明真实非冗余或性质变强。哪怕三源成功，仍不能宣称新增CERT/ADV、真实下一混权层收益、native HZ资格、GPU速度或定义新颖性。

source_census_qualified只允许在完整三源、全部预算、全部证据核对成功时为true。native_HZ_admitted、actual_phase_column_binding_verified、gpu_computation_completed、complete_physical_qualification均必须false；formal_gain、diagnostic_solver_calls、model_forward_calls、new_benchmark_solves必须0。原D047失败、D049/D053/D056既有运行原样只读；本次仅新组合执行，不调用任何旧main/worker/writer或改旧globals。

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式基线1870/2413=1063 CERT+807 validated ADV与独立E0 61/400不变、不可相加。后续仍需真实同结构收益、shadow、13家族与同候选全部2413，以及独立全部400回放；能力四并发不回退和纯速度门解耦不变。本次不修改生产默认、现有dirty changes、旧模型或历史日志/结果，不commit/push；全部新工作只在隔离实验树。

write-page技能用于分开保存数学依据、真实适用性、资源资格与正式结果。整体目标未完成，局部成功不能替代定义创新、GPU和满分目标。
