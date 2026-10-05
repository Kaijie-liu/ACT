# Neural-HZ 残差块定义续接

D160 与本轮 D161 均归类 progress，但无新实测能力。本轮[理论](definition_first_20260928/d161_residual_block_projection_20261004/THEORY.md)与[研究记录](definition_first_20260928/d161_residual_block_projection_20261004/RESEARCH_RECORD.md)均为 paper_only；重点仍是定义优先的强大非凸 Neural-HZ，不是 helper 或存储优化。

真实图确认可以把 Relu2、主 Conv3/BN4/Relu5/Conv6/BN7 和 shortcut 合在一个完整块，止于 large Add8、medium/Tiny Add10；首个 q 不必再是独立外端口。这个边界想法 D126 已有，不能重复报创新。Add 输出仍分别进入 Conv9/BN10/Relu11 与 Add14，或 Conv11/BN12/Relu13 与 Add16；不是 Add 后立即 ReLU，也不能丢后续活 skip。

对 z=Kq+U ReLU(Vq+c)+d，若保留内部 Vq 与 z，固定原 gamma 的接口行空间 [V;K+UD_gamma V] 等于 [V;K]；K=I 时能完整反解 q。要真正简化须联合投影内部 q 与预激活，不只是改消费者名单。

普通有偏置混权2×2残差控制的四个严格开相位区表明源方向导数为 alpha_1(1-gamma_1)，phase-affine source generator 无法无额外幅度地精确表达它。共同块 proxy 的 H=(K+UD_gamma V)D_alpha 带二次相位系数，欧氏方向 Gram 最高三次，非对角 source Pi 时最高四次。这不是所有抽象域不可能性，也不要求未来必须精确或 phase-affine；只是不能免费省掉交互。

可复用窄定理：当待投影 t 的完整关系仅为 ||t-c(theta)||²<=E(theta) 加原 ReLU bits/guards，输出 z 的精确投影为 z>=0、inactive z=0、sum_active(z-c)²+sum_inactive(c+)²<=E；共同 witness 是 active t=z、inactive t=min(c,0)。原零点双标签保留，但输出零不意味着两标签必都可行。c=(-1,-1),E=9/4 的物理输出非凸。该公式不能遗漏额外 t 谓词/消费者，也不是原网络块的自动精确投影。

Young 后果有 gamma*c 及 inactive(c+)²，source-dependent 中心仍需产品/内部幅度或有损包络。固定中心又有旧 D146/D147 丢源风险；不为此单独启动组件。Ergen/Pilanci已有 rectified ellipsoid 几何，其一般 spike-free intersection是内包，不可当 verifier 的安全外包。D127图能量收缩亦已查重，不重启独立metric/helper路线。

下一动作仍需实质解决完整普通残差块的源条件联合像、下一非线性与全部查询成本；移动边界和新的球名称已经不足。允许换定义，不强迫继续 D160/161，不把强参照设成“必须在集合精度上超过精确HZ”的不可能门。

本轮无候选执行、模型/GPU作业、shadow、replay或生产集成，无本轮后台作业。完整数学人口仍为D158的4032项/212文件，任何新实施须新预注册/源码冻结/once-only与原完整门。正式1870/2413与独立E0 CIFAR25/Tiny36=61/400均零新增，13家族旧解、GPU、smooth/Transformer、新家族、满分目标不变，goal active。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。旧文件只读，所有来源与档案哈希见本轮 ANCHOR_SHA256SUMS 和 ARCHIVE_SHA256SUMS。
