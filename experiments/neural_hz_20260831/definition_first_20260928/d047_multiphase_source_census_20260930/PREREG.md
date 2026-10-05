# 多相位关系的完整三源适用性预注册

本轮继 D046 数学门之后，按 THEORY.md 认证原卷积来源的区间参数种子并测量多相位包络。不是正式验证器、native HZ 接入、GPU 回退路径或得分实验。默认关闭且只允许一次冻结执行，任何失败保留日志，不重跑版本或改门。

## 完整人口和统一规则

继承 D046 manifest 的三个原模型和各自固定性质：CIFAR100 large、CIFAR100 medium、TinyImageNet medium。全部原输入 SHA、source/dependency、decoder、GPU dependency 和生产 provenance 仍核对。使用已认证 D015 batch-one 原模型 extractor 和 source arithmetic helpers；不调用旧 main、worker 或结果 writer，不改旧模块 globals。

每个由统一 extractor 得到且 target_relu 非空的 direct Conv 分支，处理原四角与中心（只按原 anchors 去重）、全部输出通道和全部 canonical fan-in，含稳定、零系数、padding。记录全部未遍历 side consumers；不遍历动态 Add/Sub，也不声称覆盖全网或所有第一层空间坐标。固定相邻输出通道配对，奇数尾自配对；禁止按界、历史标签、margin、求解状态或事后收益选通道/源。

原 large 已有 320 接收行，medium 旧数值摘要为 640；Tiny 尚无完成原源证据，实际完整人口由同一 extractor 和原几何公式计算并核验。三模型任何一项未完成则整个 source census 不通过，不能缩成 large 单模型晋级。旧 D025 原失败记录不改，本版本不复活 D045 未执行草稿。

每个接收行实际计算普通界；每个固定 pair 先对共同源系数形成可靠区间差，再生成差值及伴随值的多相位仿射界和 D046 四条正部包络。不计算本规则不使用的逐源 pair 差界；这不删旧组件测试或任何原模型、接收行、canonical slot。稳定源只数值化其贡献，不删除或固定 backbone 原 bit。

## 数学门和单次冻结

完整继承 D046 的 3777 tests、171 files 及精确 node IDs，新增恰好四个 plain 顶层测试：test_interval_seed_soundness、test_stable_padding_and_identity、test_multiphase_transfer_and_metrics、test_disabled_and_rejected_premises。合计 3781 tests、172 files。有限有理 fixture 检查不是候选运行时相位搜索。

首次导入/执行候选前冻结本目录 PREREG.md、THEORY.md、seed_relation.py、test_seed_relation.py、census.py、run_census.py 六文件和四个函数名；schema=d047_frozen_v1。独占 RUN 为 experiments/neural_hz_20260831/results/d047_multiphase_source_census_20260930_v1。首次 --enabled 即消耗版本；执行前只做静态阅读/AST检查，不先单跑新测试或试选数据。

仅完整测试门通过才启动新 worker。collection+execution 合计 60 秒，worker 240 秒；CPU1、AS16GiB、单线程、assertions 开启、bytecode 关闭、CUDA_VISIBLE_DEVICES 为空，caches/tmp 只写新 RUN。任何 failure/error/skip/人口漂移/超时均不通过。

## 原资源门与证据

whole work 256M、每模型 branch work 200M、全局 evidence 40M、retained numeric entries 64M、Fraction 512 位、summary reserve 65536 均不变。40M evidence 和 summary 在数值前预付；全部实际来源读取/认证/解析、循环、算术、D046内部构造和排序须在操作前计费。输入 manifest/readers 的 pre-import 成本计入 whole，认证失败也记录已消费工作，不把失败前工作记零。

未修改的原 extractor/helper 沿用 D025 的解析分摊合同 `4096+model_bytes+8*spec_bytes`，不把它重新替换成按解析器全部最大容量的固定预扣，也不宣称这是逐 CPU 指令计数。本轮新增读取认证、源/接收循环、seed/clamp、输出和证据另行计费。D046 调用按实际支持 s 预付：form 为 `128+160s+16s*bit_length(s)`、positive-part 为 `256+256s+16s*bit_length(s)`，token 声明为 `64+192k`；wrapper 为 `256+80n`，有理算术另计。数值临时条目保守 reserve 为 `192*max_slots+8192`，原输入与全部保留对象仍另做 ledger。

监督器与 worker 分别检查 RSS high-water growth+reserve<=1GiB、tracemalloc peak+metadata+reserve<=1GiB；不声称聚合物理资格。worker 的 retained ledger 包含 raw model/spec、packet、完整 box、源 bounds 和身份缓存、接收系数、窗口及全部关系记录、认证 roots、最后数值临时量和 budget/meter 状态。另预留按最大 canonical slot 的临时条目上界。序列化和完整 roots ledger 共用一个 40M meter，不按模型重置。

源仿射形式在实际求界之后释放；保存原源坐标、可靠界与冻结的重构器/原 bytes 身份，不把释放说成无需构造。序列化证据引用原模型及冻结 extractor 来恢复完整 packet 参数，不重复抄写相同原参数；真实内存 ledger 仍计 packet、raw 和接收系数共存。每模型完整 evidence 写入 exclusive .partial，全部写完才 rename；保存完才释放该模型 roots。失败保留已完成文件及部分证据，不补作未计费恢复遍历。

每个 pair 保存原窗口、原接收端口/通道引用、普通接收界、四个源种子和四个输出相位包络的完整稀疏系数、全人口统计和来源规则。所有空支持/无改善 pair 保留。输出事实仅包括条件包络、支持规模、相位 hinge 是否跨零及限定单锚比较的充分指标；不使用潜在数目宣称非冗余、正式 SAFE 或 ADV。

## 资格及 provenance

source_census_completed/qualified 仅可在三个原源全完成且全部门通过时为真；actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、complete_physical_qualification 始终 false，formal_gain=0。没有模型 forward、LP/MILP、shadow、逐家族或全量回放，不改默认路径。GPU 原失败版本不重跑。

日期 2026-09-30；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式基线 1870/2413（1063 CERT、807 validated ADV）以及独立 E0 61/400 保持原口径。旧存档和 /data1/Kane/HyZor 只读，只写本新目录与新 RUN，不 commit/push。按 write-page 技能将数学、成本、实际结果和未完成资格分开记录。
