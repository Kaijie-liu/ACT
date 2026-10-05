# 共同负项相位接口的数学资格预注册

上一轮为progress：D069证明统一共同负项支持逐槽支配D066，并给出跨层LP包含、普通解析控制及稀疏共享支持日程。本轮实现该组合的默认关闭数学参考，不另跑纯标量能力菜单，不宣称已经形成新域、GPU或真实模型资格。

新源码目录为definition_first_20260928/d070_joint_negative_component_20261001。先保存本预注册及THEORY.md，再实现hinge_support.py、negative_interface.py、test_joint_support.py和run_math.py。全部六文件静态审查后写freeze.json；冻结前禁止候选导入、py_compile、collection、单测预跑和数值调试。仅允许源码阅读、AST结构检查、哈希及纸面推导。既有源码和证据只读，不修改旧module globals或调用旧main/writer。

## 固定接口和证明范围

hinge_support提供prepare、query和decode，全部默认关闭，disabled不得读取任务输入。prepare接受带原ordinal的完整盒声明、两个点系数affine A/H，构造按收益率排序的持久AVL前缀树。query只接受原坐标上的稀疏增量，处理分母变号、变零、固定坐标与偏置；每次从原base派生，不复制全维系数或共享可变树。decode明确支付完整源向量费用，输出仅为盒支持见证，不是原模型ADV。

negative_interface复用D066已认证的source/phase类型、dyadic24 majorant及实际相位误差，在其prepare上建立上下两套支持base。condition按给定的两个不同原锚、相同四个完整盒查询生成新Table；原相位、连续源、EQ/LE、frame和decoder不变。Affine/ReLU/compile_rows沿用原接口，相位重命名、独立delta替换和按LP状态选规则均禁止。形式token仍不证明实际网络绑定。

正确性依据为D068原始填充证明和D069共同负项/omega包含定理。原HZ加同样关系逻辑等价；实现是待评估Neural-HZ关系片段，不把排序树或经典ReLU次可加性作为新域创新。完整代价与未覆盖项见THEORY.md。

## 完整固定测试人口

继承D066全部3809项、179文件及精确node IDs，再增加一个文件中的四个plain顶层测试，无decorator、无参数、无skip，依次为：

1. test_primal_support_and_sparse_updates
2. test_shared_negative_forward_control
3. test_interval_signed_wide_dominance
4. test_identity_default_off_and_limits

总计3813项、180文件。固定覆盖单hinge支持的独立全盒几何参考、非对称/固定坐标、阈值并列、分母变号/变零、偏置更新、base不变、完整盒见证及AVL聚合；D069三门严格控制及同相位后继/共同delta编译；宽混号/零权/区间参数和中点异相位、零点两原标签、与D066逐槽双侧包含；默认关闭、身份错误、ordinal/形状错误、512位和65536出现数拒绝。有限有理fixtures不是模型采样或健全性证明。

## 唯一执行和不变门

唯一RUN为experiments/neural_hz_20260831/results/d070_joint_negative_component_20261001_v1。第一次显式--enabled创建该目录即消费版本；成功、失败或超时都保存，不修改冻结源后重跑。解释器/data1/Kane/miniconda3/bin/python，-B、断言开启、CPU1、单库线程、CUDA不可见。collection和execution合计60秒，AS16GiB；监督器RSS增长加65536 reserve及tracemalloc峰值加metadata加reserve分别不超过1GiB。所有日志、缓存、临时文件、JUnit仅写新RUN。

继承并前后核对全部源、输入、生产provenance、decoder和GPU依赖，不刷新旧expected digests。原D064 native数学fixture、D047/D057失败普查分别保留原范围。必须精确匹配完整3813个node IDs和JUnit、零failure/error/skip、预算内完成、零漂移；否则本组合数学门失败，不追加缩小测试或提高预算的重试。

输入及中间Fraction仍限512位，显式比较乘积也检查，稀疏输入/合并出现数限65536。这里不声称已有whole256M/branch200M/evidence40M/retained64M的模型执行资格；这些真实后续门和240秒worker门不变。本版本没有worker、归档普查、模型/LP/GPU运行，不放宽仍待用户决定的GPU环境限制。

## 记录和资格

通过仅记mathematical_component_gate_passed和joint_negative_fixture_exercised。actual_model_binding、native_HZ_admitted、source_census_qualified、gpu_computation_completed、complete_physical_qualification仍false，formal_gain=0。结果自动保存来源、配置、完整人口、日志、JUnit、exit及哈希；无遗留任务也不伪造后台运行。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413和独立CIFAR10025、TinyImageNet36不变。不修改生产、旧档案或远端，不commit/push。Goal保持完整且active。
