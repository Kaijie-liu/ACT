# 端点归一化关系的先行研究与生产对照

本页仅记录本轮实际核对的直接来源，区分原作者结论、项目推导和实现事实；不是系统性综述或新颖性认证。

## 三个直接来源

[Huang 等的 Exact Finite Attention Responses From RoPE Derivatives](https://arxiv.org/pdf/2609.14127)，2026 年预印本，§2.2 Eq.5、Theorem 1 Eq.6，给出任意 score 增量的精确重加权及任意 value 的输出变化，不要求 V=K。§3.2 Eq.9 把非线性余项表示为共享非负权重对中心化 values 的加权和，总质量为 KL。它不是安全验证器证明；不迁移文中的 backward attribution、干预选择或实验分数。

[Maas 的 Gradient flows of the entropy for finite Markov chains](https://arxiv.org/pdf/1102.5238)，§4 Corollary 4.3、Assumption 4.4、Proposition 4.5，明确使用对数均值实现离散熵链式法则。因此 logmean 本身及这条链式法则都是已知。本文没有直接给出本项目端点矩阵 M_lambda 的 HZ 合同；这不构成我们新颖性的证据。

[Wei 等的 Convex Bounds on the Softmax Function](https://proceedings.mlr.press/v206/wei23c/wei23c.pdf)，AISTATS 2023，研究 score 差、simplex、指数／倒数和 LSE 分解的 Softmax 外包。这类逐行界属于强参考，不应删掉以制造割线模块优势。

本轮项目推导是：用概率质量守恒消去共同 logsumexp 差，将熵割线整理为 N 个共享 lambda 与一个 c，保持同一组系数经过任意共同 V 读出；并明确其线性外包、成本和不能当通用 Jacobian 的反例。对数均值、二次方差、McCormick 和共享噪声本身均不记为新发明。

## 已检查生产接点

`act/back_end/hybridz_tf/tf_transformer.py` 的 `_sparse_simplex` 与 `_softmax_ratio_inequalities` 逐 query 生成概率质量和同组 ratio。score difference 的 key_cache 复用 K 差计算，返回的仍为各 query 内的 key 差界。未在这些构造点发现两个真实 query 的端点割线模块。

`act/back_end/solver/solver_hz.py` 的 `softmax_ratio_weighted_extreme` 按单行固定 weights、概率盒和 score 盒求界；当前函数体没有实际使用签名中的 groups／score_differences，不能凭参数名认定它已经利用跨 query 关系。它已被 PV 输出求界使用，这不是未部署的草案。

`_softmax_taylor_coefficients` 保留方向导数、Hessian／跨度余项及 log 概率割线／切线约束。`_softmax_value_cross_radius` 的 cross 指相对单行参考的 delta_p×delta_V，包含正负概率质量平衡；它不是两个真实 queries 的差。大组和小组现有实现及其求界方法原样保留，本轮不迁移、改写或重新认证。

`_sparse_hz_softmax_value_fused` 保留共享 Q/K/V 仿射核心与谓词，再增加逐输出误差。其 constraint_parts 没有保留完整独立概率 HZ 或全部 PV 产品。因此它不是完全独立盒，但也不能免费充当 D110 需要的所有显式端点坐标。新旧比较必须保留这份已有源相关性。

本结论仅覆盖所读构造点，不是证明全系统没有任何间接跨 query 关系。D106 的单组源乘积、D108 的固定参考中心化行、D109 的循环关系均已阅读；D110 不将这些旧成果重复记为创新。

## 未完成的强对照

尚未找到通过指定强 Taylor／ratio／已知增量约束之后，被 D110 线性外包严格排除且影响真实后继的普通控制。完整精确 Softmax 图及共同 PV 乘积当然已经蕴含精确端点关系；不能把它们也宣称为被超越的对手。

论文恒等式可以启发新域，但必须进一步证明有限关系语言、统一前向变换、下游有效性和完整成本的实质贡献。目前不具备声称 PLDI 新域、GPU 加速或真实家族提升的证据。
