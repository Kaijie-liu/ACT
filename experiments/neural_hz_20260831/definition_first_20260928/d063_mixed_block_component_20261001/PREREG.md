# 四平面混权组件的数学资格预注册

日期2026-10-01，分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。上一轮D062属于progress：完成固定生成规则、全部两门共同源凸包交的严格分离、区间误差与完整构造成本，并关闭纯差值sector重复线性化路线。本轮只把该规则实现成默认关闭的数学组件；不以实现资格替代域创新、真实能力或GPU资格。

## 固定范围

数学规格为只读[D062 THEORY](../d062_mixed_block_support_20261001/THEORY.md)。固定正项相邻分组，负项与仿射残余均分；双项组使用四平面，单项组两平面，无正项使用残余上界。上下界同一规则；不优化匹配、斜率或乘子，不按实例、公开标签、历史结论、margin或求解状态选规则。

组件使用D049的原frame/value/phase身份及D053区间仿射形式。全部连续源、原门、原bits及零相位保留；声明既有接收读出并绑定上下界，不创建替代网络或原相位。真实接收读出等式、网络拓扑、原参数包络、native列、active方向与decoder仍是调用者和后续集成的认证责任。本轮形式token不能证明这些事实。

## 唯一新运行与冻结

唯一运行目录为 experiments/neural_hz_20260831/results/d063_mixed_block_component_20261001_v1。第一次显式 --enabled 创建该目录即消费此版本；无重试、覆盖或事后改源码重跑。运行前仅静态读取或AST检查，禁止候选导入、collection、单测试跑及数值调试。

freeze.json 必须固定当前目录的五个源文件：PREREG.md、IMPLEMENTATION.md、mixed_block.py、test_mixed_block.py、run_math.py；并固定四个新测试名。源码与freeze身份在执行前后检查，最终结果追加到新目录而不是修改冻结文件。

继承D057保存的全部3797项数学测试及176个文件，精确node IDs、source/input SHA、生产provenance、三原模型/spec、decoder和GPU依赖均保留。D057数学通过而整体source census失败；新运行显式保留该失败与其中更早D047失败，绝不重启worker或改写资格。

新增恰好四个无参数、无装饰器、无skip的顶层测试，顺序为：

1. test_complete_pair_control_and_forward：D062全部三对精确共同源凸组合、新上界11/10、假点63/40和后继3/5；保留原身份及bits。
2. test_cached_support_and_signed_groups：独立直接平面展开对照缓存支持，多组、正负、奇数尾、无正项、零权重、固定源及非对称盒。
3. test_interval_parameters_and_zero_phases：独立误差公式、固定有理参数端点与内部fixture、符号跨零权重和原门全部合法零相位；不运行模型采样或攻击。
4. test_identity_limits_and_default_off：错误frame/context、phase归属、重复身份/ordinal、自循环、接收身份、严格opt-in及资源拒绝。

总人口固定3801 tests/177 files。不能缩减旧人口、修改node IDs、用跳过代替失败，或用只跑新增四项冒充资格。

## 资源和执行条件

解释器/data1/Kane/miniconda3/bin/python，断言开启，-B禁止字节码。CPU亲和为一个现有可用CPU，数学库线程均1；CUDA_VISIBLE_DEVICES为空。collection加execution合计60秒，继承窗口不扩大，不从未运行worker的240秒额度转移时间。

进程AS上限16GiB；监督器RSS高水位增长加65536字节reserve不超过1GiB，tracemalloc峰值加metadata加同reserve也不超过1GiB。此测量不声称pytest所有子进程的完整物理存储资格。所有缓存、临时文件、日志、JUnit、inventory和exit只写本次新运行目录。

每个有理输入与中间结果最多512位。context源数、门数、仿射源出现数先聚合检查，总额65536，不能先合并再用unique数量绕过。共享支持缓存仍计所有真实输入、底式、组证据与行；不声称此组件已满足真实网络全工作或全物理预算。

原未来真实运行边界保持：whole_work256M、单分支200M、证据40M、retained64M及worker240秒不变。本轮没有worker、source census、native HZ、GPU、shadow或基准执行权限。

## 通过和失败的记账

必须完整collection精确匹配，3801测试全部通过且无skip/error/failure，合计时间和监督器内存过门，执行前后全部来源与生产provenance零漂移。任何失败保留完整日志，fail closed，不自动修改版本或重跑。

成功仅表示数学组件资格。source_census_qualified、native_HZ_admitted、actual_phase_column_binding_verified、gpu_computation_completed和complete_physical_qualification仍为false，formal_gain=0。正式1870/2413与独立E0 61/400均不变。之后仍需另行预注册真实同结构、shadow、逐家族和同候选完整回放。

所有新材料只写本隔离目录及唯一新结果目录，历史档案、生产代码和其他dirty changes不动，不commit/push。使用pages:write-page区分数学规格、执行事实与未认证收益。
