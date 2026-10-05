# 共同解码研究的原始文献对照

本轮只采用作者或正式论文来源。查阅日期为 2026-10-02。下面区分既有原理与本项目推导，不把未检索到同一句定理当作新颖性证明；没有下载覆盖历史论文，没有引入这些工作的训练、分支、对偶救援或求解器作为执行路径。

## Proximal 算子的共同关系

[Neal Parikh and Stephen Boyd, Proximal Algorithms, 2014](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)，第 2.3 节、PDF 第 13 页，给出 proximal 映射的 firm nonexpansiveness 不等式。ReLU 是非负正交锥上的投影，故本轮逐坐标平方不等式属于该既有机制。我们的充分统计和直接 QP 差值证明写在 THEORY.md，不把这个经典性质命名为新 Neural-HZ 定理。

## Fenchel 激活图已经存在

[Fangda Gu, Armin Askari and Laurent El Ghaoui, Fenchel Lifted Networks A Lagrange Relaxation of Neural Network Training, AISTATS 2020](https://proceedings.mlr.press/v108/gu20a/gu20a.pdf)，第 4.1 节、式 4 和 5，PDF 第 3 至 4 页，使用 Fenchel–Young 关系表达激活图，并明确给出 ReLU 的对应形式。这是训练论文，不直接证明本项目的保原相位、全消费者解码、终端收益或任何验证得分。它足以否定仅凭 Fenchel 图表示主张新颖性的做法。

## 梯度统计量的势形式已有直接先例

[Laurent Meunier and coauthors, A Dynamical System Perspective for Lipschitz Neural Networks](https://arxiv.org/pdf/2110.12690)，第 4.1 节、PDF 第 7 页，明确使用 W^T sigma(Wx+b) 作为非降 Lipschitz 激活所产生的凸势梯度，第 4.2 节进一步构造 Convex Potential Layer。

此处按实际 PDF 题名记录，不采用搜索片段中混入的其他题名。已查段落支持势与平滑常数的先例；没有据此把“任意 Cq 可由同一 y 无损解码”说成该文的已引用定理，也没有据此宣称本项目的具体组合首创。原网络任意混权消费者通常不是这个势梯度，本轮的负控明确区分二者。

## 互补表示并不自动减少幅值

[Alp Aydinoglu and coauthors, Stability Analysis of Complementarity Systems with Neural Network Controllers](https://arxiv.org/pdf/2011.07626)，Lemma 1 和 2、PDF 第 5 至 7 页，给出单个 ReLU 和多层网络的互补表示。第 7 页说明该构造的互补变量维数等于神经元数量。因此把图写成一个隐式系统并不自动得到小幅值接口；本轮 QP 的 KKT 降级同样须支付重新出现的变量。这是结构成本对照，不采用其稳定性求解流程，也不外推其性能。

## 本地强对照仍然保留

[D004](../d004_observable_interface_20260928/D004_RESULTS.md)和[D051](../d051_interface_sufficiency_20260930/PROOFS.md)限制线性摘要，不能用于否决非线性 decoder。[D126](../d126_joint_phase_closure_20261002/JOINT_IMAGE_BOUNDARY.md)说明独立精确边际和两两关系缺少共同实现；本轮精确 decoder 回答语义存在性，但没有提供便宜的完整集合查询。[D114](../d114_quadratic_relation_boundary_20261002/THEORY.md)已研究原 bits 下的互补与二次恒等式，不能把 KKT 的重写称为发现新的二次关系。

比较完整旧 HZ 时，双方必须获得相同源信息和已认证关系。新坐标可表达同一集合不是失败理由，也不是能力证据；仍需证明有用的结构推理、完整成本与实际同路径回放收益。

配置为 paper_derivations、primary_literature、read_only_local_evidence；没有候选或数值执行。branch redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；正式与独立新增均为 0。
