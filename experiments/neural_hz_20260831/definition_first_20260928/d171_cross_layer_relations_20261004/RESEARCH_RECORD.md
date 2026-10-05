# Neural-HZ 跨层定义研究记录

本轮围绕强非凸 Neural-HZ 的定义继续研究，没有增加 helper、运行外部 rescue 或转向存储优化。得到 [残差纤维候选](RESIDUAL_FIBER.md) 的 whole-parent 健全性、原相位与 decoder 保留、整块预算传播及条件性 m→n 幅值替换；同时记录 [跨层与动态 Attention 边界](STRUCTURAL_BOUNDARIES.md)。这是纸面候选与比较范围的推进，不是已资格化的新域或正式能力提升。

上一轮 D170 的共同盒能量严格收紧已保留，但其强旧参考同样能证明控制，未启动实现。本轮不重复换球/换盒：研究对象改为跨层共同 q 与新 z 的关系，以及它是否能在保留活旁路后消除隐藏幅值。已知数学机制被明确标注，未把经典非扩张或低秩分解包装成新颖性。

候选的实际限制很具体：m=n 不带来幅值节省，m>n 也仍须支付原 m 个 bits、guards、全输出 caps、两项非线性范数和终端 lowering。没有证明训练网络的 U+λVᵀ 缺陷小，也没有决定可推广的固定结构参数规则。下一轮应先核完整普通块上的可付查询与适用性，而不是继续调该二维控制。已核两个 CIFAR 图的 Add 后有 Conv/BN，不能直接套未经混权的 post-Add ReLU 差分；Tiny 同等完整绑定仍未完成。

动态 query 作为尚未完整覆盖的真实重复结构得到范围核对，但没有新的前向算子。d=16、N=5/17 的自由 QK rank-only 约束是空的；这明确要求后续用真实同源系数与谓词，而不是仅添加低秩标签。它不解释全部 110 个未解 ViT 的原因，也不把研究优先级从普通 CNN 自动转走。

主代理只读旧归档、manifest、保存的图与生产 provenance，手工复核扇区证明、正控制和自由 QK 构造；三路独立审查分别负责相位兼容交叉能量、原生残差纤维、实际 Attention 覆盖与已有研究排重。文献只用原论文页面/PDF确认先例，没有引入论文的训练或优化流程。文档审查使用 write-page 技能按既有本地 Markdown 格式区分结论与资格；没有创建外部 Page，未验证网页渲染。

本轮无候选代码、freeze、import/AST/compile/collection、数值执行、模型解码/前向、LP/MILP、GPU、shadow 或全量回放。最后执行的组件资格仍为 D158 的 4032 tests/212 files，不把这组旧测试重新计为本轮通过。新文档的 checksum 是事后归档，不冒充实验前预注册。

2026-10-04 Australia/Sydney；分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新文件全部在 experiments/neural_hz_20260831 的新隔离路径，旧源码、历史模型/日志/表格、冻结候选和生产默认未改。没有 commit/push 或新后台实验。

正式成绩仍为 1870/2413，即 1063 CERT + 807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400，不相加；两边本轮新增均为零。旧档不变并非候选已实测保住旧解。Goal 保持 active，GPU、完整家族零回归和正式新解等整体目标尚未完成。所有后续执行仍需新预注册、冻结及原有完整门，不能因 paper-only 轮数多而降低验证边界。
