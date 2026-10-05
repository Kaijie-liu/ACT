# 定义边界的一手来源与本地证据

本轮于 2026-10-03 核对一手文献，范围有限，不声称完成全面新颖性检索。以下论文的算法没有被引入实验路径。

- [Singh 等，Fast and Effective Robustness Certification，NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf)：已读取 Theorem 3.1 的 ReLU 仿射噪声公式及 Section 3.2 的 smooth activation 讨论。用于确认廉价对齐包络并非新公式；不据其结果推断当前候选性能。ETH 镜像打开超时后改用会议原文。
- [Ortiz 等，Hybrid Zonotopes Exactly Represent ReLU Neural Networks，2023](https://arxiv.org/abs/2304.02755)：本轮读取原作者摘要，确认 exact HZ ReLU 表达已有先例。这里的具体坐标对应、行数及残差反例由本项目直接推导，不归因于未阅读的全文定理。
- [Hashemi、Ruths、Fazlyab，Certifying Incremental Quadratic Constraints for Neural Networks via Convex Optimization，2021](https://proceedings.mlr.press/v144/hashemi21a.html)：本轮读取会议摘要，确认增量二次关系是既有验证方向。未据该摘要宣称其包含本记录的度量充要条件，也未采用其凸优化／SDP 流程。

本地比较对象为原地只读的 `../../nhz_v2_20261002/THEORY.md`，特别是 Definition 1、ReLU transformer、LP 与 shadow 交换及成本主张。相关实现和 helper 撤回依据见 [独立审计](../../claude_review_20261003/README.md)。本轮没有修改这些文件。

研究连续性由 [D132](../d132_shared_error_boundary_20261002/RESEARCH_DECISION.md) 给出。普通 shared-source 差分与成本边界分别参照 [D016](../d016_shared_secant_20260930/D016_DEFINITION_AND_PROOFS.md)、[D020](../d020_phase_difference_20260930/D020_PHASE_DIFFERENCE.md)、[D042](../d042_bounded_relational_transfer_20260930/THEORY.md)、[D100](../d100_envelope_factorization_20261001/THEORY.md)。没有把这些旧结果重新计作本轮新实验。

独立纸面审查：`aligned_domain_equivalence` 核查坐标对应、合法标签、输入投影有效界、一般 HZ 嵌入与完整成本；`forward_relation_candidate` 核查增量旧工作、固定度量充要条件及低维源范围；`d133_final_redteam` 复核两份推导并指出低维源矩阵必要条件还需局部源自由度，本轮已在新稿中补齐该前提。数学尚未机器认证，文件哈希只能证明保管身份。
