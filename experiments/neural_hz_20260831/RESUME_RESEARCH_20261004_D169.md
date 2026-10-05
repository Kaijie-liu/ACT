# Neural-HZ phase-support研究恢复入口

用户重点仍是从数学定义提出强大的非凸 Neural-HZ，不是 helper、旧图改名或存储优化。完整 Goal 保持 active。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。

本轮[理论](definition_first_20260928/d169_phase_support_gluing_20261004/THEORY.md)与[记录](definition_first_20260928/d169_phase_support_gluing_20261004/RESEARCH_RECORD.md)均为纸面，未运行新候选。

- 正向结果：由共同 proxy 原生关系推出固定方向的相位支撑界；固定列序精确 RREF 的全部行可统一生成四条 LE/方向。它排除四门整数相位假点，补足已有有限表达的一项共同见证信息。
- 明确限制：在 D157 native 上不改变具体化；小例从54行增至至多70行，旧完整HZ已有对应结论。只能归档为支撑编译规则，不能算新域或正式新增解。
- 出生时共享两层 proxy 对整个父域健全，但含真实跨层相位乘积；若仍使用旧四行编译，且每个旧LP点都允许 eta=0，则新投影包含旧LP。共享名称本身不带来精度提升。这个限定结论不否定更强的联合像编译。
- 完整 separator 的关系拼接早已有证明；同名源/相位或均值不是完整共同赋值。二维Conv局部性不自动提供廉价跨层闭包。不要以拼接算法或恢复全部旧图绕开定义问题。

不启动上述两个未过门方案的实现。下一定义应展示完整共同像的可消费关系，在普通混权及下一激活后保住非冗余信息，给出直接查询和完整代价，并与强旧同信息路径比较。新定义仍要保留所有原二元相位、源/谓词/decoder及fail-closed语义；不准加入attack、BaB、split、backward/dual或实例菜单。

最后实际执行资格仍为D158的4032 tests/212 files。正式1870/2413=1063 CERT+807 validated ADV；独立E0 CIFAR100 25、TinyImageNet 36，共61/400；均零新增，不能相加。本轮没有模型/GPU/shadow/全量回放，因此不能宣称实测保全或升级。所有default-off、逐结构、全人口、shadow、2413/400及四并发不回退门不变。

所有新写入仅本轮新隔离文档，旧模型/结果/冻结实验及现存生产修改保留。tracked diff SHA256为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。未commit/push、未启动后台任务。新目录中ANCHOR_SOURCE.sha256和ARCHIVE.sha256从仓库根目录校验。后续任何候选执行前仍需新预注册及源码冻结；本轮研究记录不是执行许可。
