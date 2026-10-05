# 前向关系组件通过完整数学测试

默认关闭的通用种子、差值加伴随幅值、固定混权组合和 ReLU 关系传播已实现，并通过一次完整 3769 项测试。新增的无激活支配正控也被精确有理数测试复现：不要求 q>=p、p>=q 或共同关断，仍能排除通过强局部组合松弛与六对全局差分行的点。结果是数学组件里程碑，不是已完成的新抽象域或真实网络新增解。

## 单次执行结果

唯一执行为 run_math.py --enabled，结果保存在 [d044_relational_generator_20260930_v1](../../results/d044_relational_generator_20260930_v1/exit.json)。session 18715 已返回 exit_code=0；没有独立 worker 或仍在运行的本轮测试，不重跑本版本。

完整继承 3759 tests、169 files，加预先冻结的十项新测试，共 3769 tests、170 files。collection 和 execution 联合时长为 58.65831170603633 秒，保持原 60 秒门；pytest 自报执行 45.13 秒、3769 passed、13 条既有 warnings。无 failure/error/skip，collection 和 JUnit 的全部精确 node IDs 均核对。根代理在终态后另外只读核对 JUnit，确认十项新增测试全部实际执行且无问题节点。

所有来源和输入漂移列表为空，生产 provenance 未变。运行清单继承原模型、性质、decoder 和 GPU 依赖身份，再绑定六份新冻结文件及 D042 两份只读证明。绑定这些文件不等于执行三模型验证或建立实际 HZ 列映射。

监督流程总时长 67.2221535295248 秒。监督器 tracemalloc peak 为 14128101 bytes，metadata 为 5054576 bytes；RSS high-water growth 记录为零，最终 VmRSS 为 48963584 bytes。零 growth 只是相对于起始 high-water 的增量，不是零内存。相应 1GiB 监督观测门通过。pytest 子进程保持 CPU1、AS16GiB、单线程及完整时间门，但没有测得整个候选的聚合物理峰值。

## 实现了什么关系

[relational.py](relational.py) 仅依赖 dataclasses 和 Fraction。所有公开生成操作默认关闭，显式 enabled=True 才工作。它接受调用方已认证的共同 frame、原相位绑定和真实范围；形式 token 不是模型证书。

该组件保存同一原相位下的差值 d=q-p 及伴随 p，按固定左基 q=d+p 组合不同的实际权重，先合并相同读出再求界。ReLU 同时更新差值和伴随幅值，条件锚仍是原相位。两条输出不等式可以编译成普通终端行的形式元数据，没有新建相位、复制输入或调用 LP、搜索、攻击及 backward/dual rescue。

十项测试检查通用种子的有限有理网格包含性、混权和同源抵消、伴随量不可省、跨零及零点的合法原相位、frame/anchor 错配拒绝、正控端点、强比较见证、下一混权块、默认关闭和部分错误前提、终端行含 bias 的符号。512 位与 MAX_SUPPORT 上限在代码中存在并经静态审查；本轮没有直接触发这两个边界的测试，不能说这些边界已获得完整运行或物理资格。

## 纸面与可执行正控的范围

[有序前缀证明](PROOF.md) 现在由通用种子直接闭合，不再依赖 D042 CONTROL 的两条额外源不等式。第一跳生成界为 19/40，旧点为 191/400；追加普通混权块后的界为 19/25，旧点为 191/250。两个物理读出均有预激活非零的真实达到点；此处的分数旧点不是 ADV。

[无支配证明](NONORDERED_PROOF.md) 更进一步取消原激活大小支配。其 g-f 范围跨零，既存在 q<p 也存在 q>p，alpha=0 时 p 可以严格为正。统一规则仍得到

```text
t-r <= 17/200+(14/25)*alpha,
Z=t-r+(14/25)*(q-x) <= 129/200.
```

预注册旧点满足完整真实前缀凸包、以该连续前缀为源的最后单门完整图凸包，以及全部六对全局 D020 行，却有 Z=163/250，严格缺口 7/1000。测试独立检查了两层凸组合见证、双向不支配见证和六对四行不等式。129/200 只是充分界，没有宣称真实取等或整个网络的理想凸包。

这项推进改变了研究前提：不再只能寻找相等、成比例或共同关断的源对。但本轮没有测真实配对的命中率。已知 reduced product、条件支持和差分激活机制仍是数学来源；编译回 HZ 不否定后续定义贡献，但 HZ 加完全相同事实具有相同逻辑强度，不能将本组件换名为创新已完成的 Neural-HZ。

## 真实结构接入的下一项具体工作

执行后只读检查 D025 的 census.py 和 D038 的 archive_worker.py，确认既有 CIFAR100-large 第一直接消费者 bank 已保存 64x576 系数、64 个 bias、五个窗口的原源界及固定相邻差界。这是足以准备下一项研究的数据，不是本轮新 census；不能外推 medium、Tiny 或后续残差块。

两个接入问题必须先解决：

1. 旧 receiver 权重和 bias 是 BN 后的认证区间，而当前组件接受精确 Fraction。下一候选须给出健全的区间系数或显式误差传递，保留真实参数身份；不能用区间中点替代真实模型。相同区间端点也不能作为系数相同的证据。
2. 1440 是五窗中固定 canonical pair 的 occurrence 数，含 padding，不是任意可合并的同锚集合。每个 seed 对应自己的原 bit；固定一个锚时，其余实际贡献只能使用同源的无条件范围或另有证书的条件范围。不得把不同原相位改名成同一个锚。

下一项应在完成上述数学契约后，预先登记真实输出通道的统一配对及完整人口，保留 padding、稳定源和失败前提，比较与原同预算界的关系。当前 archive 的 original_phase 仍是 tensor 坐标身份，不是经过验证的终端 HZ 相位列；latent、bit 编码及 decoder 的实际映射也必须补齐。不能在这些缺项下晋级为 native 或更新分数。

GPU 方面，本轮并行完成的 [D043 诊断](../d043_cuda_error_log_20260930/RESULTS.md) 在完整 3759 测试通过后，记录到 cuInit 返回 CUDA_ERROR_OUT_OF_MEMORY。它定位了错误位置，但未确定资源根因。没有 GPU 数值或性能资格，也没有改变上限或重跑失败版本；CPU 数学测试不是 GPU 目标的替代品。

## 记账与目标状态

mathematical_component_gate_passed=true 及 all_stages_passed=true 只覆盖本次冻结的完整数学组件与兼容性流程。native_HZ_admitted、actual_phase_column_binding_verified、source_census_qualified、gpu_computation_completed、complete_physical_qualification 均为 false，formal_gain=0。

正式基线仍为 1870/2413，即 1063 CERT 与 807 validated ADV；保留全部 13 家族与每个旧解的要求不变。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与正式分数相加。没有新正式 CERT/ADV、shadow、逐家族或全量回放，因此不更新任何默认路径或成绩表。

本轮属于 progress：形成新增非支配严格正控，完成默认关闭实现，并获得完整一次测试证据；这不是单纯重复计划或仅等候。整体 Goal 仍 active，完整定义贡献、真实结构适用率、GPU 与物理成本、新增解及全量保旧均未完成。

日期 2026-09-30；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。旧九个 tracked 文件仍为 3806 insertions、57 deletions。只新增 D043/D044 隔离源和结果文件，没有旧模型、旧证据、生产默认、commit 或 push 的修改。依照 write-page 技能把纸面证明、实测资格、未证范围与正式收益分别记录。

## 关键证据摘要

```text
44a4c4aba18ba81d92e27aa2d9d6a25d1cba3f71270bca5be3ee90373ae4152d  freeze.json
3275ce7f6debecd3328748b217757c414c3b9a58d3349529048c9e974ddb42d5  RUN exit.json
e7ade555ca971a3226657d5079519057b7b64bfea5327143b2ed182007ab8b85  RUN preregistered.json
6eafb2c5588e9637dbdb991b8a5ccc6e36f56d7375698662f8b20423fdc92a41  RUN inventory.json
d047d67a168948317666f9d6d9a066dd0c152afab15634456e8b8a75400b6a39  RUN tests.xml
```
