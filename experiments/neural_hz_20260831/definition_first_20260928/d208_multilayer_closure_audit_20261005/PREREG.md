# 跨层闭包审计的单次执行预注册

本轮只审计冻结 D207 实现的组合精度，不修改其域、测试或任何旧结果。默认不执行；显式 --enabled 才运行。预期发现是：同域能够证明第一层差分，但下一层关系生成和查询没有保住该证明。审计测试通过不等于候选新增验证能力。

## 冻结与人口

在任何本轮 import、AST、compile、collection 或候选数值执行前，冻结 CONTRACT.md、PREREG.md、test_closure.py、run_math.py、collection_contract.py。schema 为 d208_multilayer_closure_audit_v1。唯一结果目录为 experiments/neural_hz_20260831/results/d208_multilayer_closure_audit_20261005_v1，创建即消耗该版本，失败也不编辑或重跑。

完整继承 D207 成功 receipt 的 4080 tests/214 files、7377 source identities、14 inputs、解释器、project import closure、decoder/GPU 依赖人口、CPU 和 production provenance。保留全部原 test paths 与有序 nodeids，并在末尾追加 test_closure.py 中四个普通无参数测试，总 4084 tests/215 files：

1. test_two_layer_counterexample_retains_full_interface
2. test_same_domain_proves_the_lost_preactivation_relation
3. test_fixed_queries_lose_the_relation_at_next_relu
4. test_direction_family_and_phase_preservation

新增身份和历史 receipt 加入同一清单，7377 不是新总数上限。新审计不继承新的算法资格。只复用认证的 provenance 函数，不运行任何旧 runner 的 main。

## 固定结构与断言

全部表达式使用 CONTRACT.md 的三个源、两个异质双门 bank 和最后一个 ReLU。保留完整十五维接口、五个原 bits、原输入 decoder、全部旧谓词和因子身份。主要断言为 J 上界 0、g1-g2/2 上界 -1/4、第二层盒差分 caps=(985/1024,3657/2048)、尺度 s2=l2=1、g1 已存上界 3/8、三个最终 p 证书均 >=1/4，且后继未被证明稳定为零。

全域 z=0 的依据是解析证明，不从有限取点推断。固定点为 (-1,1,-1)、(-1,1,1)、(-1,-1,-1)、(-1,-1,1)、(0,0,0)、(1/3,-1/2,0)、(1/2,-1/4,-1/4)。固定正缩放 lambda 为 1、2、3；保留符号相位并在最后一个点检查真实零 preactivation 的两个合法标签及非法分数标签拒绝。没有随机点、搜索、攻击或相位枚举。

同次测试写出 closure_counterexample.json 和 direction_family.json，记录实际精确证书、完整逻辑账及 false 的模型/GPU/正式增益标志。任何实际值不同、异常或资源失败均保存失败证据，不事后修改本版本断言。

## 仅证据写入重定向

沿用既有 D112 四个测试的 RUN 重定向到本结果目录下 inherited_d112_controls。另明确新增 D207 的三份证据重定向：source_phase_positive_control.json、exact_alias_control.json、complete_200_gate_control.json 写到 inherited_d207_controls，而非其冻结历史目录。

collection plugin 必须认证 D207 原测试模块和 _record 函数身份，只替换这个 JSON 写入函数的目的地；固定三个文件名、独占 x 写入、JSON finite 校验和目录检查保持。不得替换任何测试体、断言、域函数、Path 或 os。清单与 inventory 记录重定向及原证据 hashes，完整人口仍在同一个 pytest 进程执行。此例外只改变存档去向，不改变待测数学。

## 资源与完整性

使用 /data1/Kane/miniconda3/bin/python、认证 CPU [0]、数值线程一、CUDA 隐藏、RLIMIT_AS 16 GiB。完整单次 pytest 子进程从启动到 JUnit 保持 60 秒门；禁止选择性预跑。零失败、错误、skip、缺失或替代才通过审计执行门。插件自动加载关闭，importlib 模式，同进程有序 collection 验证，禁止 bytecode 写入旧目录。

监督器完整 pre/post 哈希时间另报；其 RSS high-water 增量加 65536 bytes 及 trace peak 加 metadata 加 65536 bytes 均不得超过 1 GiB。这不是 child RSS、native 完整物理内存或 GPU 资格。D207 本身的 whole256M、branch200M、entries64M 和 512-bit 有理数限制原样保留。不能以本审计支付真实网络构造费用。

自动保留 preregistered.json、inventory.json、tests.log、tests.xml、全部证据 JSON、pre/post source/input/provenance 检查、telemetry、artifact hashes 和 exit.json，包括失败。已有固定 LP 数学对照仍随完整人口执行；本轮不新增 LP 或 solver rescue。

## 不晋级

全部 actual model、native phase、GPU、source census、完整 physical、shadow、逐家族和正式回放资格保持 false，formal_gain=0、new_benchmark_solves=0。全部测试通过只说明审计按合同完成及反例被当前实现复现，不能写成新增 Neural HZ 能力或新颖性证据。

2026-10-05 Australia/Sydney；branch redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；预期 tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。所有新文件仅在本新目录与唯一新 RUN；历史实验、生产和 /data1/Kane/HyZor 只读，无 commit/push。正式 baseline 1870/2413 及独立 E0 61/400 均不变。
