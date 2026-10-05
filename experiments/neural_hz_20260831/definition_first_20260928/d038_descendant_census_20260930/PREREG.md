# 跨层幅值规则的完整存档适用性预注册

这是一项单次、默认关闭的 CPU 参考审查，研究已知跨层幅值规则在完整已保存卷积块上的冗余情况，不是 Neural-HZ 新域、GPU 回退路径或新的验证器。数学依据见 THEORY.md 和冻结 D036。成功或失败都保留结果，不重跑当前版本。

## 人口和来源

只读取 D025 已完整认证并保存的 CIFAR100-large complete_0.json：SHA256 为 fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0，大小 12062002 bytes。原模型 SHA256 为 5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16，原性质 SHA256 为 caca9ef2245019dd70883661bd5c102544f64152ddc0ea8f967eb8ed0882e996。原 first ReLU 为端口 127，frame 为原 modelInput；完整输入盒 3072 维、1600 个已存档源仿射形式。

人口是该存档完整的一条接收分支、五个原位置 (0,0)、(0,31)、(16,16)、(31,0)、(31,31)，每处全部 64 接收通道。共 320 接收行，每行 576 个 canonical 槽位：合计 184320，其中 102400 真实边、81920 padding。稳定源、零系数、padding、所有接收行都处理，不按 crossing、标签、margin、旧结论或 LP 状态筛选。

这不是把 D025 原三模型人口缩小后重做资格。D025 完整研究仍失败，medium 保存未完成、Tiny 未进入；这次只分类已存在的 complete_0，不认证新 source census。supervisor 保留全部三个原始模型/性质身份、完整旧依赖和 provenance 验证；没有原模型 loader、forward 或求解器调用，也不声称覆盖另两模型或整个网络。

## 草稿保留与首跑冻结

前序 D037 三个草稿未经执行。本轮静态核对发现 worker 顶层报告漏了 runner 必需的 row_unresolved_counts；未启动有该问题的版本，也不覆盖它。本新目录从该草稿复制，机械更新本版本身份，并补齐报告字段；数学规则和门不变。D037 的原三文件保持原状态，新增数学测试文件也仅作为静态草稿保存。

根代理在任何本候选导入、测试或执行前，写入 freeze.json，固定本目录六文件的绝对路径与 SHA256：PREREG.md、THEORY.md、amplitude.py、test_amplitude.py、archive_worker.py、run_reference.py。freeze 还固定六个 plain 顶层新测试的名字与 3759 tests、169 files 的总人口。supervisor 在 candidate import 前核查它，随后将其 hash 及全部旧依赖记录到独占新结果目录的 preregistered.json。

结果目录固定为 experiments/neural_hz_20260831/results/d038_descendant_census_20260930_v1。第一条 --enabled 执行即消耗版本，成功、失败或超时均不得修改后重跑。静态文件读取、AST 检查和冻结前审查不执行候选。不得调用任何旧 main、worker 或结果写入器。

## 测试与不变的资源门

完整继承 D025 已冻结且通过的 3753 tests / 168 files，并加六个无参数、无 decorator 的普通测试，合计 3759 / 169。冻结精确 node IDs；collection 与 execution 合计仍是 60 秒，禁止 skip/error/failure。测试覆盖 D036 强局部 hull 正控、一般 interval 贡献/残差、稳定与 padding、零点两相位、宽 fan-in 全冗余、正负边、固定物理行有效性、默认关闭与 type/shape/work/bit fail-closed。固定数学 fixture 不是网络采样或运行时相位搜索。

使用 /data1/Kane/miniconda3/bin/python，CPU1、AS16GiB、单 runtime 线程、assertions 开启、禁 bytecode，全部 caches/tmp 放新 RUN，CUDA_VISIBLE_DEVICES 为空。只有完整测试门通过才运行 archive worker；worker 总墙钟 240 秒、whole work 256M、branch work 200M、64M retained numeric entries、512 位 Fraction、65536 summary reserve。两项既有内存门分别为 RSS high-water growth 加 reserve 不超过 1GiB，tracemalloc peak 加 tracer metadata 加 reserve 不超过 1GiB。supervisor 与 worker 分别测量，不声称 aggregate 物理资格。

证据 ledger 和 streaming 编码使用同一个全局 40M meter，并在数值处理前预付；summary 另预付 65536。pre-import 标准库认证的工作计入 branch/whole，按文件字节与解析访问计费；原 archive 原始 bytes、解析树、decoded 盒/源/相位/所有窗口及权重、全部行 masks、最后一份完整 kernel 返回值、认证 roots 都在 ledger 中。另保留每槽 64 数值条目加 4096 的临时 reserve。不得用只计 tensor 的口径替代完整 roots 和进程峰值。

supervisor 在执行前后核对完整 inherited source/input/provenance，包括旧 GPU 依赖身份。worker 在实际项目 import 前独立核对 anchored D025 manifest、本版本 freeze 六文件、全部实际 import 及存在的 package initializer；不会再次读取未 import 的 GPU 二进制作为 CPU 数值工作的重复认证。全部原身份仍由 supervisor 验证，没有删减旧检查人口。

## 输出和失败语义

每个接收行保存全部 576 个四位 row_masks 和计数，padding mask 为零。Q、L/H 和原系数通过 authenticated archive 加 frozen kernel 可无损重算；不省略实际算术。重算 ordinary_bounds 必须逐行与存档完全一致。证据写入 complete.json.partial，仅写完才 rename；完整输出必须有全部 320 唯一接收行且各层统计相符。

分类只有“上述充分条件证明冗余”与“UNRESOLVED”。potential_rows、potential_edges、potential_nnz 不表示严格非冗余、正确属性解或速度收益。即使所有 masks 为零也属于有用的研究负结果，不修改条件来追求非零。

任意资源、数值、认证、完整性或测试失败停止整个版本，保留日志和已完成证据，部分结果不晋级。archive_qualified 仅表示此次完整存档分类及其资源门通过；source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、complete_physical_qualification 始终 false，binding_mathematical_only=true，formal_gain=0。接收门的真实相位列绑定、完整 native 编译、GPU、属性求解、shadow 和全量保旧均不在本次完成范围。

## 研究去向和 provenance

先用完整存档结果判断廉价逐边证书是否值得保留。共同源跨相位组合的定义研究可并行，但任何新关系都须另外给出证明和完整成本，不能拿本组件的资格作整体晋级。此处不要求全网络 ideal，也不因单项失败放松预算或复活原草稿。

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。以 D025 preregistered/inventory/exit、原全部 dependencies 与 production provenance，以及 D036 THEORY/CHECKPOINT/SHA256SUMS 为只读 anchors。原九个 tracked 修改为 3806 insertions、57 deletions。旧 /data1/Kane/HyZor、源文件、模型、日志和结果只读，不 commit/push。

正式 1870/2413（1063 CERT、807 validated ADV）及 13 家族逐例保旧要求不变。独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不相加。全部原 bits、连续源、EQ/LE、共享身份和 decoder 保留；无 attack、BaB、split、backward/dual rescue 或身份/状态菜单。按 pages:write-page 将数学前提、人口、执行门和晋级限制分开存档；不发布外部 Page。
