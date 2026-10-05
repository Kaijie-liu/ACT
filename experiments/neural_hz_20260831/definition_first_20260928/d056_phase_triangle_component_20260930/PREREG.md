# 共享原相位三角组件的数学资格实验

本次把已完成的三门证明实现为默认关闭组件，检验是否能用最多四条静态行补足独立双门关系缺失的联合信息。经典布尔三角关系不算新域定义；数学通过也不等于真实网络、GPU 或正式收益。原 HZ、全部连续与二元因子、EQ 和 LE、共享身份及具体输入 decoder 不变。

## 固定行为和测试人口

输入是同一原 source context 上的三个真实门的可靠区间仿射形式、原输出及原相位。按原 phase ordinal 排序，每条边统一用低 ordinal 到高 ordinal 的方向生成 D053 证书，不根据符号或收益改方向。严格稳定门只省略新增三角行，触零门保留；零边或偶负边不生成本种增强。其余两正一负和三负三角全部按 THEORY.md 编译，保留每条必要仿射平面。没有实例菜单、LP 查询、分裂、attack 或相位翻转。

完整继承 D053 的 3789 项、174 文件、精确 node IDs 和只读依赖。新增 test_phase_triangle.py 四个普通顶层测试，无参数化、无 skip：test_d054_physical_control、test_signed_triangle_projection、test_nonpoint_and_zero_semantics、test_identity_limits_and_default_off。合计必须 3793 项、175 文件。

新测试覆盖物理读出的严格增强、六个输入顺序不变、全部边符号排列和不等幅投影、普通三负门、非点偏置、全部零点合法相位、严格稳定及触零区别、原身份/循环来源拒绝和继承数值支持上限。非点权重支持继续由完整继承的 D053 测试覆盖，不宣称新测试穷尽所有参数。有限数学 fixture 不是运行时采样或相位枚举。

## 执行前冻结与唯一运行

freeze.json 使用 d056_frozen_v1，绑定 PREREG.md、THEORY.md、CONTROL.md、phase_triangle.py、test_phase_triangle.py、run_math.py 六文件和四个测试名称。冻结前只静态检查，不导入、collect 或试跑候选。唯一新目录为 experiments/neural_hz_20260831/results/d056_phase_triangle_component_20260930_v1，首次 --enabled 独占创建并消耗版本，失败也不重跑。

解释器 /data1/Kane/miniconda3/bin/python；assertions 开启，禁止 bytecode；CPU1、AS16GiB、库单线程、CUDA_VISIBLE_DEVICES 为空。collection 加 execution 合计仍限 60 秒，无新增 worker，不挪用 240 秒 worker 门。监督器两项内存观察仍为 RSS high-water growth+65536 和 tracemalloc peak+metadata+65536 各不超过 1GiB；不冒充完整聚合物理资格。

所有 cache/temp/日志/JUnit/工件只写本次新 RUN，源文件只写本次新目录。D053 已通过的数学门、D047 原三模型 census 失败状态均原样保留。旧 runner 不重跑，不调用旧 main/worker/writer，不修改旧 globals。新 runner 认证 D053 authority、D054 和 D055 文档证据，并继承全部 source/input/provenance、解释器、decoder 和 GPU dependencies；前后检查漂移。完整三模型研究尚未注册执行，不能把本轮数学测试当作补完 Tiny。

## 保留的后续门

每个 Fraction 输入与算术结果仍限 512 位，组合前预检实际输入出现次数和返回行支持总量不超过 65536。原 whole-work 256M、per-model 200M、evidence 40M、retained entries 64M、240 秒 worker 等门不变；本轮未执行这些完整物理候选门，不能凭组件数组少宣称已经过门。

本轮始终 formal_gain=0；source_census_completed、source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、complete_physical_qualification 均为 false。数学之后仍须完整真实同结构研究、shadow、逐家族与同候选全部 2413 回放，保每个旧解并有新解；E0 仍须独立同路径全部 400 回放，保全部 61 和两家族零回退。能力四并发不回退与纯速度门继续解耦。

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式 baseline 1870/2413=1063 CERT+807 validated ADV；独立 E0 CIFAR100 25、TinyImageNet 36，共61/400，不能相加。旧冻结源码、历史模型/日志/结果和 /data1/Kane/HyZor 只读；不改生产默认、现有 dirty changes，不 commit 或 push。

使用 write-page 将定义贡献、已知组件、证明、执行结果与未获资格分别记录。本轮属于实现和检验步骤，不缩小整体定义创新及 GPU 能力目标。
