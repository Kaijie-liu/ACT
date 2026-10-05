# Neural-HZ 跨层纤维研究恢复入口

继续从 HZ 定义研发强非凸 Neural-HZ，不以 helper、旧图改名、存储优化或小例替代。完整 Goal 不变且 active；redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。

本轮新档为 [残差纤维定义](definition_first_20260928/d171_cross_layer_relations_20261004/RESIDUAL_FIBER.md)、[适用边界](definition_first_20260928/d171_cross_layer_relations_20261004/STRUCTURAL_BOUNDARIES.md)、[研究记录](definition_first_20260928/d171_cross_layer_relations_20261004/RESEARCH_RECORD.md)。不是实现预注册。

关键正向：对于 z=q+U ReLU(Vq+c)，利用 U=−λVᵀ+E 的显式 defect，可以从出生保存 q、全部原 γ、n 维 z、共同源相关误差关系及整块 norm。条件性地不保 m 维隐藏幅值 r，但只能在完整消费者满足条件且 m>n 时谈局部幅值节省。全 off caps、两类 norm 的 joint membership、旧 decoder 均已写出。整块预算 LR 可以跨层继承，前提是把它装进具体化而非只证明真实图。二维非零 defect 控制有效，不是新网络 CERT。

尚未过门：完整查询/LP lowering、确定性结构参数规则、真实 defect 和总费用。旧图可用同一证书不自动否决表示价值，但 m=n 或补回全图的成本必须如实计。不要只为新的名字实现已知球加 bits。下一步优先检验普通完整残差块的可付接口；仍未选定执行候选，不重跑旧版本。

重要边界：任意 mask 对任意向量保持固定 G 的命题会强制对角，但 sign-compatible ReLU 可保 PSD 且非正 off-diagonal 的 G。图 Laplacian只是子类。q≥0 时 ||ReLU(q+f)−q||≤||f|| 是已知差分关系，并不自动节省新幅值。真实 CIFAR Add 后还有 Conv/BN，不能直接套 small-f 情形。

Attention：动态 patch query 与两条 live residual 尚未完整覆盖；本轮没有新算子。自由 QK 在 d≥N−1 可生成任意正 row-stochastic P；真实两个布局 d=16,N=5/17，仅靠低秩没有新约束。该结论不适用于实际固定同源仿射 Q/K，后续必须研究这些绑定关系。110 个未解 ViT 不等于已知由动态 QK 导致。

没有新代码/数值/模型/GPU/shadow/回放。最后组件 D158 为 4032 tests/212 files。正式 1870/2413、独立 61/400 两边零新增，未宣称实测保旧。旧档及生产改动均未碰，无 commit/push、新后台实验。tracked diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新目录 ANCHOR_SOURCE.sha256 与 ARCHIVE.sha256 从仓库根目录校验。
