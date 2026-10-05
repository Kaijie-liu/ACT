# 共同误差与跨层能量的文献对照

本轮只用原始论文和官方数学参考核查具体机制，不采用其训练、对偶优化、分裂或替代求解器。下面的先例支持机制辨认，不证明本项目的新颖性或真实网络能力。访问日期为 2026-10-02；本地历史文档保持只读。

## 强凸误差球

[Ndiaye 等人 Gap Safe Screening Rules for Sparsity Enforcing Penalties](https://www.jmlr.org/papers/volume18/16-577/16-577.pdf)，JMLR 2017，第 3.2.2 节定理 6，PDF 第 6 至 7 页，利用 gap 和强凸性认证包含最优点的球。它支持“gap 到球是已有机制”的判断，不提供本项目的三角松弛无增益定理；后者在 THEORY.md 中直接证明。

[Parikh 与 Boyd Proximal Algorithms](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)，2014，第 2.3 节、PDF 第 13 页，给出 proximal 算子的 firm nonexpansiveness。D131 的充分统计证明与本轮可行投影球使用同类凸分析。四门反例及其整数相位范围是本项目的纸面检查，不是论文中的实验。

[Gu、Askari 与 El Ghaoui Fenchel Lifted Networks A Lagrange Relaxation of Neural Network Training](https://proceedings.mlr.press/v108/gu20a/gu20a.pdf)，AISTATS 2020，第 4.1 节式 4 和 5、PDF 第 3 至 4 页，将激活等式写成非负 Fenchel 关系的零集。本项目不采用其训练或优化流程，也不把 ReLU gap 重新命名为新抽象域。

## 固定谱多项式

[NIST DLMF 第 18.5 节](https://dlmf.nist.gov/18.5)，式 18.5.1 给第一类 Chebyshev 的三角公式，式 18.5.4 给第四类的公式。因此本轮 p_K 正是 W_K(1-2lambda/L)/(2K+1)。该识别排除了“新多项式族”的主张；误差及噪声常数在本项目直接推导，不声称最优。

[Wolfgang Erb Accelerated Landweber methods based on co-dilated orthogonal polynomials](https://arxiv.org/pdf/1206.1950)，所读版本日期 2012-12-20，第 1 节式 3 至 7、PDF 第 2 至 3 页，明确采用残差多项式及固定递推；第 5 节讨论非对称残差多项式。这里只据此辨认谱加速先例，没有宣称本项目的原相位符号接口已获廉价实现。

## 跨层共同能量

[Aleksei Kuvshinov 与 Stephan Günnemann Robustness verification of ReLU networks via quadratic programming](https://link.springer.com/article/10.1007/s10994-022-06132-9)，2022-03-16 发表，Machine Learning 111，2407–2433。第 3.2 节式 5 就是加权跨层 propagation gap；定理 1 处理加入输入平方距离后的 QP 目标的层权凸化，不是 gap 单独关于源的凸性。该直接先例改变本轮决定：不把全网络共同能量或选权凸化本身作为 Neural-HZ 新颖性。DAG 的静态 Young 证书和惯性推论另在 FORWARD_ENERGY.md 给完整纸面证明；论文的 QP/对偶求解不在采用范围内。

## 本地强对照和结果身份

- [D009 支撑证明](../d009_bounded_phase_energy_20260928/D009_PROOFS_AND_LIMITS.md)：已有共同源 Gram、非对角能量跨 ReLU 的反例及 SDP 对照，不能重复记作新发现。
- [D011 相位幅值审查](../d011_phase_value_audit_20260928/D011_RESULTS.md)：绝对幅值锚和标量互补关系的旧边界。
- [D114 二次关系边界](../d114_quadratic_relation_boundary_20261002/THEORY.md)：普通单层二次恒等式的有限范围分类，不否定跨层不等式，也不支持通用新颖性。
- [D126 有限误差消元](../d126_joint_phase_closure_20261002/THEORY.md)：已有多目标条件误差证书仍不保证共同实现，未知 bits 的系数费用不能用点值计算代替。
- [D131 共同 decoder](../d131_nonlinear_common_decoder_20261002/THEORY.md)：本轮数学起点，不是已经完成的域或来源资格。
- [D120 实际 CNN 结果](../d120_mixed_consumer_source_20261002/RESULTS.md)与[D130 实际 Attention 结果](../d130_import_isolation_20261002/RESULTS.md)：只是历史实验，本轮不改写、不重跑、不增加资格。
