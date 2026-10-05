# 续接 Neural-HZ 定义研究并排除固定低秩幅度捷径

最新 [D152 研究记录](definition_first_20260928/d152_live_amplitude_rank_20261004/RESEARCH_RECORD.md) 已归档。用户强调重点是提出强大的非凸 Neural-HZ，而不是 helper、存储、构造优化或验证框架。正式突破尚未取得，不能用本轮负结论或元数据检查替代能力成果。

[数学限制](definition_first_20260928/d152_live_amplitude_rank_20261004/THEORY.md)：固定 q=Aq_K+Bz+C beta+d+E eta 在独立可达折面上要求 rank(E)>=未锚门数；即使 eta 任意源相关/非凸也一样。完整活读出 Vq 则为 rank(E)>=rank(V_U)。这是 D004/D131 推导延伸，不是新外部定理，不涵盖相位依赖加载或非线性 decoder。不能把实际可达折面前提视为已认证。

唯一冻结三模型 metadata 审计已 exit 0，最终归档计时 5.616436876356602 秒，完整7285/14身份前后通过。两个 CIFAR 原 ONNX 终端均100→100，Tiny为200→200；各终端ReLU恰一Gemm消费者，无输出幅度skip，三个图无pool。只证明没有形状自带的低维瓶颈，不证明实际满秩。30个ReLU全报告。RUN `results/d152_live_amplitude_metadata_20261004_v1` 已消费，不得删除重跑或修改冻结稿。原浮点系数、实际rank、forward、solver均未执行。

D151关于D150五行逐门范数被同坐标四行精确参照支配的决定保持。不要继续移植该版本；D149/D150组件原样只读可复用。D152固定低秩anchors与简单聚合也不推进候选，聚合是已知健全外包而非精确改进。

下一交付要直接回到域元素、具体化和可组合非凸算子：改变固定加载假设必须给共同源/phase/幅度绑定与完整终端费用。diag(beta)g不是新定义；共同nonlinear decoder若又在终端恢复m个幅度也没有支付优势。不要再以重复形状审计、小测试或适配层代替这个数学交付。尚无已选定且通过强参照的新候选。

正式1870/2413、独立61/400、formal_gain=0不变。完整数学人口4000/210是D150旧资格，本轮不替代。GPU/smooth/Transformer/全部家族能力仍是未完成目标。分支redu-hz、HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac、tracked diff29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5不变。无后台实验，goal active。本轮是有证据的路线收束，不是wait或blocked。
