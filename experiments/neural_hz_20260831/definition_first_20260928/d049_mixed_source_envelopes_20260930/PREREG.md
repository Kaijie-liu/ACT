# 混合共同源包络的数学组件预注册

本版本验证 THEORY.md 中同时保留原连续源及原相位的上下包络、差值与伴随的前向规则。CONTROL.md 给出含非零偏置的普通混权正控。它是默认关闭的数学组件，不是完整新域、真实模型收益或 GPU 资格；固定盒上面和符号差分已有先例，贡献尚需真实适用性、完整成本与新颖性核验。

## 统一语义与算术范围

原 HZ 的连续因子、全部二元相位、EQ/LE、共同 latent/frame 和 decoder 不变。x_j 为有证原 source 值而不是独立副本，b_i 为原 bit 的有证 active 视图。混合形式 L(x,b)<=E<=U(x,b) 使用同一赋值；不同 context、类型、原身份不能因数值相等而合并。

先合并同一身份，再按连续源序号、原相位序号的固定类型顺序规范化。连续 literal 为有证盒坐标的正向或反向缩放；binary literal 是原 bit 或其补。上包络用一次固定 prefix；下包络用连续部分中点选择的有效支撑面，加 binary singleton 边际。无连续源时恢复 D046，不把 binary 下界外推为 fractional cube 上的逐点下界。

四条成对 ReLU 观察保留真实差值与伴随。Affine/Conv/Add/Concat 仍依赖真实共同源和原算子认证；默认关闭，只有 enabled=True 才检查和构造输入。任何身份、类型、上下界、位长、支持或未证前提拒绝都不宣告 SAFE、不删除任何原因子。零预激活处两种原合法相位保留。

实现采用精确 Fraction，输入与每步结果不超过 512 位。单形式与公开组合的稀疏输入出现次数保留 65536 上限，构造前检查；typed context 与原 formal frame 的存储同样有界。源上下界和原模型绑定由调用方认证；此数学组件只检查有限形式条件，不能伪称 native 列已绑定。

上、下包络必须在完整 source box 与 independent bit cube 上次序相容。这是保守有限子类，不是对依赖额外原谓词才相容的所有数学观察的总构造器。不得去掉该检查以取得正例。常量读出也须被其外部包络包含。

## 正控与精确测试人口

固定输入盒 [-1,1]^2，原门为

```text
q=ReLU(x), p=ReLU(3*x/4+6*y/25+1/200).
```

共同源差式为 `x/4-6*y/25-1/200`，范围为 [-99/200,97/200]。固定 x 后 y 的上包络给出

```text
q-p <= 97/400+x/400-6*y/25.
```

CONTROL 的旧点 `(x,y,q,alpha,p,eta)=(1/4,0,9/20,11/20,77/400,1)` 属于两个各自完整 source labelled hull，并满足旧 D020 的四条相位差分行，却违反新行 23/1600。物理读出 `Z=q-p-x/400+6*y/25` 的新上界为 97/400，真实输入 (1,-1) 取等。原所有 bits 保留，纸面旧点不是 ADV，新上界不是 benchmark CERT。

完整继承 D047 已通过数学部分的 3781 tests、172 files 和精确 node IDs。D047 的三源阶段失败仍保持失败；只继承其单独已证的数学资格，不将 all_stages_passed=false 改为成功。新增 test_mixed_source.py 四个普通顶层无参数化测试：

1. test_mixed_majorant_minorant
2. test_phase_only_and_typed_identity
3. test_source_relational_control_and_forward_composition
4. test_disabled_and_rejected_premises_and_rows

总数必须为 3785 tests、173 files。覆盖正负连续系数、非对称及固定 source 盒、正负原 bits、有限连续有理点和完整小维 binary cube、D046 纯 bit 一致性、typed identity、上述完整正控、后继混权组合、终端行和错误前提。单元测试的有限数学真值核对不是候选运行时相位搜索；不增加 attack、原模型 forward 或求解器实验。

## 单次执行与完整旧门

唯一 RUN 为 experiments/neural_hz_20260831/results/d049_mixed_source_envelopes_20260930_v1。首次 --enabled 独占创建并消耗版本；失败不重跑、不改人口、不放宽预算。执行前只允许静态阅读或 AST 解析，不预先导入、collect 或单跑候选以挑选结果。

冻结 PREREG.md、THEORY.md、CONTROL.md、mixed_source.py、test_mixed_source.py、run_math.py 六文件及四个测试名，schema=d049_frozen_v1。collection 与 execution 合计仍限 60 秒；任何失败、错误、skip、人口漂移或超时均不通过。固定解释器 /data1/Kane/miniconda3/bin/python、assertions 开启、CPU1、AS16GiB、库单线程、禁止 bytecode、CUDA_VISIBLE_DEVICES 为空。缓存、日志、JUnit 及终态仅写新 RUN。

认证 D047 的 preregistered、inventory、exit、freeze 和监督器，同时继承全部旧 source/input、原三模型与性质、decoder、解释器和 GPU dependency 身份。可读取已认证 D038/D015/D017 的 metadata、memory、绑定与 drift/provenance 帮助函数，不调用旧 main、worker、writer，不修改旧 globals。执行前后检查所有来源和 production provenance。

监督器保留 1GiB RSS high-water growth 和 tracemalloc peak+metadata+65536 两门。pytest 子进程受 CPU、AS、时间限制；不因此声称完整聚合物理峰值合格。异常仍保存终态、日志、已得工件和身份核验。

本轮不注册数值 worker，不能把新增测试移入 240 秒窗口。原 whole_work_cap=256000000、branch_work_cap=200000000、evidence_prepaid_work=40000000、retained_entry_cap=64000000 保留给后续完整物理候选；本轮没有支付或获得那一级资格。原三源人口未被缩成两个；未来真实阶段仍需新预注册的完整原人口，当前不补跑 D047。

无 worker 的 all_stages_passed 只表示本版本全部已注册数学阶段通过。始终 native_HZ_admitted=false、actual_phase_column_binding_verified=false、source_census_qualified=false、gpu_computation_completed=false、complete_physical_qualification=false、formal_gain=0。

## 后续范围与保管

通过本数学组件后仍须真实同结构、shadow、逐家族以及同候选全部 2413 回放；单独 E0 400 回放也不省略。每个旧 CERT/validated ADV、13 家族零回退、invalid ADV=0、默认关闭及能力四并发不回退门均不变。1.5x/2.0x/1.8x 仍只用于纯速度声明。

传播所触及的所有 source/bits 支持、排序、连续尺度恢复、数据移动、原 HZ、终端转换/A/A^T、证据及 decoder 必须完整计费。混合形式可能扩成全 root 宽度，不以行数小掩盖实际 nnz 或源展开。GPU prefix 与可靠舍入尚未实现，不能把本 CPU 数学通过当 GPU 进展。

日期 2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式基线 1870/2413=1063 CERT+807 validated ADV；独立 CIFAR100 25、TinyImageNet 36，共61/400，不相加。本版本仅写新隔离目录及唯一 RUN，全部旧源码/证据、历史模型与 /data1/Kane/HyZor 只读。不 commit/push、不修改生产默认。

依照 write-page 技能分开记录定义、经典机制、严格比较、有限数学资格和正式成绩。整体 Goal 不因局部组件通过而完成；无 attack/PGD、BaB、input/phase split、backward/dual rescue 或 LP 状态/实例身份菜单。
