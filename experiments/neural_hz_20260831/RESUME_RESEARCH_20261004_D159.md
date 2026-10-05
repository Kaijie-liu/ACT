# Neural HZ 完整接口与定义研究续接

目标仍是定义优先的强大非凸 Neural-HZ，不是存储优化或 helper 的集合。用户再次强调该重点。上一轮只确认方向，归类 no progress；本轮完成了[定义边界](definition_first_20260928/d159_full_interface_definition_boundary_20261004/THEORY.md)和[研究记录](definition_first_20260928/d159_full_interface_definition_boundary_20261004/RESEARCH_RECORD.md)，证据改变了下一步选择，归类 progress。

冻结 D157 的 C=[B;ones] 在完整真实首块上不压缩：large/medium/Tiny 的新幅度分别131073/16385/50177，原门65536/14400/46656。large 的原 q identity skip 使原生域退回精确图，显式 I 行的四个 caps 也正好是旧四行 ReLU；不要再把这个情形叫作图替换创新。

medium/Tiny 不可由“矩阵高”误判注入。shortcut 是 W_s S，S选stride2空间全部64通道。载体 [B_main;S;ones] 完整重构全部消费者，只需12289/37633幅度，比原门少2111/9023。C1=F C0 保证投影后的新裸纤维包含于旧裸纤维；若行空间相同则等价。有限 caps、dominance、energy 方向不自动继承，行数/nnz/查询和物理费用仍需独立证明。该因子化仅为可复用支撑，不是新域，也不是已跑模型结果。

精确源能量 E(d)=||d||² 不是新候选：D156 已有其反例，D158 contraction 是其子集。新归档补全 P=C†C、N=I−P 的核球表示、椭球、固定源支持及单点条件；它仍丢核方向，同相位完整凸包仍丢 d=0 的原生修复。不实施此变体，不扩展 support helper 测试，不重跑旧版本。

接下来必须继续研究完整普通 Conv—激活—残差上的源—原相位联合定义和跨层变换。减少幅度不是唯一创新方式，也不是新硬门。不能仅把上述完整接口因子化做成工程后宣称完成用户目标；也不要重复做相同形状审计、点值球、比例别名、已有 phase series 或 Householder 正控。已有所有正负支撑保持可复用，新的实质定义仍须说明相较它们改变了什么，以及普通混权和活 skip 的全部费用。

本轮 paper_only，没有候选源码、数值执行、模型/GPU作业、shadow、replay 或生产集成，也没有待后台保存的实验作业。最近通过的人口仍为 D158 的4032项/212文件；实施前必须新预注册、冻结并继承原门。所有旧文件只读，仅新增 D159 文档与本 resume。

正式1870/2413=1063 CERT+807 validated ADV，独立E0为CIFAR25/Tiny36，共61/400；新增均0，13家族和全部旧解保全要求不变。GPU、smooth/Transformer、新家族与满分目标保持完整，目标 active，未完成、未阻塞。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。来源哈希与最终文档哈希分别见本轮 ANCHOR_SHA256SUMS 和 ARCHIVE_SHA256SUMS。
