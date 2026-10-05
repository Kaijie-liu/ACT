# 宽层共同相位接口的数学资格预注册

上一Goal轮为progress：跨领域审计定位经典共同混合机制，证明独立 residual 上表在明确强比较下无增益，并保存宽源四平面推导。本轮实现 THEORY.md 的关系接口与前向变换，不仅返回标量界。它是定义研究的可执行片段，不是已完成的新域或正式能力收益。

新隔离目录为 definition_first_20260928/d066_wide_phase_interface_20261001。先保存本预注册及数学定义，再实现 phase_interface.py、test_phase_interface.py、run_math.py。全部五文件审查后写入 freeze.json，冻结前禁止导入候选、编译、collection、单测预跑或数值调试；只允许静态阅读、AST检查和纸面证明。默认关闭/显式enabled=True，关闭时不得读取其他输入。

## 固定组件与测试范围

候选复用 D049/D053 的原值、原相位、共同源身份与 interval affine 类型，以及 D063 的已证共同盒支持原语；旧模块只读。提供 prepare(v,gates,receiver)、condition(prepared,i,j)、affine(tables,weights,bias,receiver)、relu(table,bias,receiver,own_phase)、compile_rows(tables) 的显式 opt-in 接口。prepare保留全部原项，condition不按LP状态选择，affine只能组合共同锚与共同context的表，relu保留新原门身份，compile_rows对同原锚对仅声明一个delta。

完整继承 D064 的3805项/178文件及精确node IDs，新增四个无装饰器、无参数、无skip顶层测试，依此顺序：

1. test_shared_phase_forward_control
2. test_wide_signed_cached_support
3. test_interval_actual_phase_and_quantization
4. test_identity_default_off_and_limits

总计3809项/179文件。覆盖D065普通强正控及共同/独立表差别、混号宽项和直接全展开对照、双侧前向与原相位不变、dyadic可靠secant、真实与中点异相位、原零点两标签，以及身份、位宽、聚合出现数拒绝。纸面证明负责全体整数赋值；有理数fixtures是实现检查，不是模型采样或攻击。

## 一次执行与不变预算

唯一RUN为 experiments/neural_hz_20260831/results/d066_wide_phase_interface_20261001_v1。第一次显式--enabled创建目录即消费该版本，成功或失败自动保存；不覆盖、不事后修改冻结源后重跑。解释器/data1/Kane/miniconda3/bin/python，断言开启、-B、CPU1、库线程1、CUDA不可见。collection与execution合计60秒，AS16GiB；监督器RSS高水位增长加65536 reserve与tracemalloc峰值加metadata加reserve分别不超过1GiB。所有cache/tmp/log/JUnit只写新RUN。

完整继承并前后核对旧source/input/provenance、decoder和GPU依赖；保留D047/D057失败状态。数学资格必须精确匹配全部3809 node IDs与JUnit、零failure/error/skip、时间内存达标且零漂移。D064的已通过native数学fixture仍保留，但不据此声称新条件表已安装到真实模型。

组件输入/合并支持聚合出现数65536，有理数及中间结果512位；24位斜率策略固定，不因测试调整。真实后续whole256M、branch200M、evidence40M、retained64M、worker240s均不变。本版本仅数学门，不注册worker、真实归档普查、模型/LP/GPU执行。下一次真实适配必须先固定完整人口、实际相位映射、费用与证据，不借本组件资格跳过证明。

## 资格与目标

通过仅记 mathematical_component_gate_passed 与 relational_interface_fixture_exercised。actual_model_binding、native_HZ_admitted、source_census_qualified、gpu_computation_completed、complete_physical_qualification仍false，formal_gain=0。不得以此替代实际结构非冗余、GPU或正式回放证据。

正式1870/2413（1063 CERT + 807 validated ADV）、独立E0 CIFAR10025与TinyImageNet36保持不变。全部原bits、连续量、EQ/LE、frame与decoder保留；不加入攻击、split、BaB、backward/dual rescue，不修改生产、旧档案或已有dirty changes，不commit/push。

日期2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。使用write-page分开定义、执行和未证收益。Goal保持完整且active，数学门只是通向真实结构与完整回放的里程碑。
