# 原生谓词绑定组件的资格预注册

上一Goal轮属于progress：D063完成3801项、177文件数学资格，收集与执行57.471842秒，源码与输入零漂移；并定位实际读出、原相位及后继界接入缺口。本轮实现[THEORY](THEORY.md)中的默认关闭native谓词桥，保持全部旧整数语义。它不是原模型接入完成、新域创新认证或正式提分。

## 冻结范围

新目录为definition_first_20260928/d064_native_predicate_binding_20261001。冻结五个文件：PREREG.md、THEORY.md、native_binding.py、test_native_binding.py、run_math.py。运行前只允许静态读取、手算、AST检查；禁止候选导入、编译、collection、单测试预跑或数值调试。freeze.json固定五文件SHA及四个新增测试名。

唯一RUN为experiments/neural_hz_20260831/results/d064_native_predicate_binding_20261001_v1；第一次显式--enabled创建目录即消费版本，不能重试、覆盖、改冻结源码后重跑或提高预算。结果与失败由监督器自动保存。

完整继承D063的3801项、177文件及精确node IDs，新增四个无参数、无装饰器、无skip顶层测试：

1. test_native_extended_multigate_binding
2. test_native_compact_residual_and_outward_rows
3. test_native_identity_and_predicate_rejection
4. test_native_default_off_and_original_state_preservation

总人口3805项、178文件。定向fixtures使用实际SparseHZono和已有原生ReLU算子，检查实际EQ/LE提取、shared latent残差、signed相位方向、原谓词与因子保留、安全RHS舍入以及不符合前提时拒绝。它们不是模型实例、攻击或采样。数学证明覆盖全体原整数赋值；fixture只检验实现，不替代证明。

## 不变的资源与边界

解释器/data1/Kane/miniconda3/bin/python，断言开启、-B；一个原可用CPU，库线程1，CUDA不可见。collection与execution共用60秒，不缩减旧人口，也不从未运行的worker借时间。AS16GiB；监督器RSS高水位增长加65536字节reserve以及tracemalloc峰值加metadata加reserve分别不超过1GiB。此范围不是pytest、模型、HZ或GPU全物理资格。

组件有理数及中间运算512位，合并前聚合出现数65536。真实后续whole work256M、单分支200M、证据40M、retained64M、worker240秒不变；本轮无worker。所有cache/tmp/log/JUnit只写本次新RUN。全部原来源、输入、decoder、GPU依赖与生产provenance绑定并前后核对。

D063数学成功但未取得source/native/GPU资格；其中保存的D057 source失败及嵌套D047失败必须原样保留，不重启或改记为成功。

## 成功与失败

完整collection和JUnit精确匹配，3805项全部通过，无failure/error/skip，时间与监督器内存过门，全部来源、输入和生产provenance零漂移，才算数学组件资格。任何失败fail closed并封存，不在原版本事后修复重试。

成功可以记native_predicate_fixture_exercised=true，但actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、source_census_qualified、gpu_computation_completed、complete_physical_qualification均仍false。fixture中的原列检查不能推广成模型原端口认证。

原模型、13家族和E0未运行，正式1870/2413及独立61/400不变，formal_gain=0。候选默认关闭，不修改生产代码、旧档案、原模型和任何已有dirty changes，不commit/push。使用pages:write-page分开记录数学命题、实际执行和未证收益。

日期2026-10-01，分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。成功之后仍需真实同结构、shadow、逐家族及同候选完整回放，不能将此数学资格作为缩小目标的终点。
