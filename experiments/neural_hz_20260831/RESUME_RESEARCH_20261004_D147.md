# 保源仿射候选与不能靠半径修复的定义缺陷

Goal active，目标仍是强大的非凸 Neural-HZ，不是存储优化、helper 或完整图换名。上一回合 D146 是 progress；本轮从其候选继续得到一个实际改变下一动作的反例和修正，而不是再次归档相同状态。

[D147 主文](definition_first_20260928/d147_source_affine_norm_fiber_20261004/THEORY.md)证明：y=c+G beta+r 的 phase-only isotropic 球，可能在原整数相位下丢掉源与幅值关系；精确半径也不能救。64 维普通偏置控制里，旧 D009 已证明 J<50，新球允许原整数相位假点 J=63629/1024。该控制虽有严格 all-pair-hull 分离，但 D009 更强，不能晋级。

原生修正是明确的多块范数载体：y=c+C xi+G beta+sum E_k e_k，||e_k||2<=a0_k+a_k^T beta，保留原 P、bits、guards/masks、共同输入和 decoder。原 HZ 在 K=0 时精确嵌入。对新 ReLU，保留 q=(1/2)g+(1/2)|gbar|+e_new 的源仿射部分，只用共同球替换 e_new=(|g|-|gbar|)/2 的精确关系；不是保完整新门图再添 cut。

共同球半径可由源盒项、旧误差块 operator norms、原 phase drift 给出相位仿射公式；Affine/Conv/Add/Concat 共用块身份，真实算子像的包含性可重复证明。在本轮控制中半径 4 排除范数 8 的假误差，恢复旧 D009 界，尚无超越旧强参照的证据。

有证中点斜率的同一构造可容纳平滑激活；softmax 的零和余项球是已有 D108/D113 sector 的后果。没有 sigmoid/tanh/GELU、动态 QK/pV、LayerNorm 或 Transformer 实现资格。新 gamma 不会自动进入球中心或半径：初始 G=0、常半径时持续如此，这是尚未形成有效非凸相位耦合的重要缺口。

终端只发布固定实际消费者方向的有证线性外包，不调用 SOCP 等新优化器。数学递归必须保留球语义，不能拿有限平面的更大多面体继续冒用球半径。所有历史误差块、矩阵填充、source/guards、可靠范数和参考值、terminal 及 decoder 成本尚未过门，没有固定宽度证明。

下一项有意义的研究是普通非零中心 mixed readout、共享 skip 及后继激活上的跨层精度/成本，与 D009 能量及已有低成本混权关系比较；不能继续仅调 phase-only 球半径，也不能因修复这个控制就实施大回放。[其他假设的停止依据](definition_first_20260928/d147_source_affine_norm_fiber_20261004/OTHER_CHECKS.md)保留了共同能量、quotient 与低秩 decoder 的同轮审查。

只有纸面研究及一手文献核对，无候选 import/AST/compile、模型、测试、solver、GPU、shadow/replay 或后台启动。完整原测试人口与资源条件不变。正式 1870/2413，外部 CIFAR100 25+TinyImageNet 36=61/400，新收益均为零。生产与历史档案未改。

分支 redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本轮文档技能用于本地隔离归档及结论边界，不写外部 Page。
