# 原生结构发现的冻结测试与真实归档诊断

候选默认关闭。先通过完整数学测试，再单独启动一次真实archive probe；两阶段不混称资格，旧失败及历史结果只读。

## 完整数学人口

完整继承D086的3821项、182文件，追加test_native_discovery.py四项固定测试：test_guard_catalog_and_uniform_groups、test_full_bank_transfer_and_projection、test_missing_ambiguous_and_compact_scope、test_default_off_and_budget_rejection。总3825项、183文件，有序人口不变，零skip/error/failure要求不变。

一次pytest进程在测试体前验证有序人口；CPU1、单线程、CUDA不可见、AS16GiB，启动至JUnit合计60秒。监督器主机1GiB观察、source/input/production前后漂移与全部历史身份链同D086。source_component_qualified=False不应被误判为D086数学失败，也不把数学通过升级为真实模型资格。

唯一RUN为 `experiments/neural_hz_20260831/results/d087_native_structure_discovery_20261001_v1`。先冻结CONTRACT.md、PREREG.md、native_discovery.py、test_native_discovery.py、archive_probe.py、run_math.py、collection_contract.py七文件，再允许任何新候选AST/导入/收集/数值执行。首次创建RUN即消费数学版本；不做单测预跑，不修改冻结版本后重跑。

## 真实归档阶段

仅数学门成功后，显式调用archive_probe.py --enabled，独占创建RUN/archive_probe；首次创建即消费这一阶段。240秒包含worker启动、导入、认证、解码、完整root计数、全结构发现、若预算允许的全部组转换，以及证据收尾。超时由外部监督记录；worker内部错误在finally保存完整摘要。无自动重跑。

唯一archive路径、大小、hash及来源资格见CONTRACT.md。与此接近命名但不同hash的归档不得混用。原snapshot层9全部HZ缓存及net/参数/bounds/precomputed要同时纳入root账。分析全部原谓词和全部认证候选，不能挑选几条成功关系代替完整人口。

两次archive读取/流hash预付2*68718806，全归branch/whole；解码owner、header及numeric root、完整发现和转换另外预付。256M whole、200M branch、64M retained、40M预留证据及65536摘要reserve不变。主机RSS绝对/增量及tracemalloc+metadata+reserve均核1GiB，AS16GiB不代替物理门。尚未实现或未取得完整physical资格的部分如实标false，不以字段名替代证明。

只要任一预算或前提失败，记录已完成阶段与失败原因，不换归档、不缩门/消费者人口、不升预算。若无可应用组，报告真实空人口而不是能力成功；若有组但全转换未通过，记录transformed=false。即使应用成功，也不输出CERT/ADV，不运行终端求解器，不宣称实际网络或decoder全资格。

## 记账和保存

正式1870/2413、13家族保旧、外部61/400独立口径及默认关闭要求保持。无GPU或速度资格，formal_gain=0。所有源码、结果与停止原因只在本新目录和唯一RUN保存，原corrected-prefix超时档不变，生产文件不改。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。write-page技能用于区分静态证明、数学测试、真实归档阶段与未获资格，仅本地存档。
