# 残差来源组件的一次性资格测试

本版本只申请残差后继提取的小型合成图组件资格，不申请定义创新、真实模型来源资格、native绑定或GPU资格。语义和静审修复见 CONTRACT.md。旧 D079 草稿逐字保留，新的 source_block.py 显式默认关闭。

## 完整测试人口和执行

继承已经成功的 D072 全部3813个有序 node IDs、180个原文件，追加一个文件中的四项固定无参数测试，总3817项、181文件。名称依次为 test_disabled_binding_and_source_caps、test_ordered_residual_parameters_and_boundaries、test_all_consumers_shared_cache_and_frontier_identity、test_unsupported_edges_shapes_and_cycles_fail_closed。只在测试体内构造小型内存 ONNX 图，不加载真实网络或运行数值推理。

四项覆盖默认关闭与batch绑定；完整Conv/BN/shortcut/Add/后继Conv/ReLU顺序；全部消费者、共享缓存和普通预激活旁路的既有门分类；未知算子、外来依赖、广播/形状错误、cycles及opset范围拒绝。旧3813项不筛选、不删除、不改名、不调整顺序。

唯一 RUN 为 experiments/neural_hz_20260831/results/d081_residual_source_qualification_20261001_v1。--enabled 首次创建即消费。冻结 CONTRACT.md、PREREG.md、source_block.py、test_successor_blocks.py、run_math.py、collection_contract.py 六文件后，才允许AST、导入、收集或执行。冻结前无候选数值预跑。任何失败保留，不修后重跑本版本。

使用一次 pytest 进程：收集结束时检查完整有序人口和实际路径，独占写inventory后才进入测试体；退出后核对全部JUnit，零skip/error/failure。启动、导入、收集、人口检查、执行和JUnit收尾合计60秒，CPU1、库单线程、CUDA不可见、AS16GiB。监督器RSS增量加65536 reserve，以及tracemalloc峰值加metadata加reserve分别不超过1GiB，保持D072范围；不冒充对子进程完整物理峰值的认证。

继承D072全部source、9个input身份、4417 GPU依赖身份和1011 decoder依赖身份，认证其manifest、exit、inventory与artifacts。旧成功/失败记录作为经哈希认证的历史合同保留，不重新执行历史main/worker，不把D070/D047/D057失败改写成功。运行前后检查source/input/production零漂移；缓存、临时文件和JUnit只写新RUN。

## 明确不执行和不授予的事项

不运行真实三源worker、来源数值普查、LP/MILP、GPU、shadow或2413/400回放。不放宽原256M whole、200M branch、40M evidence、64M retained、512位及240秒worker门；本次没有worker，这些尚未认证的完整费用保持未认证。

数学/组件通过仅说明该冻结源码及完整测试人口在此执行配置通过，不等于真实模型或生产默认可以启用。正式1870/2413与独立CIFAR100 25、TinyImageNet36不变，formal_gain=0。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。依赖为新freeze及继承manifest；仅新增隔离文件和唯一RUN。write-page技能用于分开语义证明范围、组件测试与尚未执行的资格；不发布外部页面。
