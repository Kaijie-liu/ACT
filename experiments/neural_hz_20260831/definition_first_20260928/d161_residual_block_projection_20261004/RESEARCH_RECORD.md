# 残差块定义研究记录

2026-10-04 Australia/Sydney；branch redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

前一 goal 回合 D160 归类 progress：完成并封存源平移纤维、canonical 递归及强参照负证。本轮核验其四项档案哈希全部通过，没有重新执行旧候选。本轮进一步研究完整残差块边界，[THEORY.md](THEORY.md)保存实际证据、定理与限制。

## 做了什么

1. current_helper_audit 只读核对 D059/D091/D094/D126/D152 与旧真实图。确认首个 q 的 identity skip 可被完整首次残差块吸收，且 Add 输出在下一块仍是活 skip；该边界思想已有 D126 先例，不重新计作创新。
2. fiber_kernel_review 推导 [V;K+UD_gamma V] 与 [V;K] 的固定相位行空间等价，给出普通四个开相位区域及 source generator 必须保留真实二阶交互的控制，并分析共同 proxy 的三次/四次 Gram 费用。
3. aligned_domain_equivalence 推导共同球过整组 ReLU 的精确投影、规范见证、物理非凸性与 Young 后果；强调 source-dependent 中心、额外内部谓词及活消费者的限制。
4. 主代理复核公式、坐标变换、真实节点与强比较，并核读 Ergen/Pilanci 原论文第2.2节。记录其 spike-free 集合在一般情况下的内包含方向，避免把训练 relaxation 当安全外包。
5. 静态查重发现图能量收缩已在 D127 CNN_RELATIONS，任意名义中心不成立。只保存这个查重结果，不再扩充已有 M-matrix 或能量 helper。

本轮得到改变后续研究动作的纸面证据，归类 progress；不是新颖性认证、组件通过、真实能力或速度提升。

## 来源与读取范围

真实图来自 D120 的 complete_0/1/2.json 的 original_source_relu、branches、side_consumers，以及 D152 的 model_0/1/2.json 的 nodes。主代理仅使用 jq 投影旧 JSON 中的节点与形状字段，没有计算新参数、秩、神经界或网络输出。最初一次 rg 命中单行大 JSON 导致输出截断，随后改用 jq；截断文本没有当完整证据。

项目内相关来源包括 D059 THEORY、D126 THEORY/JOINT_IMAGE_BOUNDARY/SOURCE_AND_PRIOR_ART、D139 THEORY、D140 THEORY、D152 THEORY、D155 THEORY、D159 THEORY 和 D160 已封存档案。引用的旧结果仅按其原声明范围使用，不转移任何组件资格。

外部仅采信[Ergen/Pilanci 一手论文](https://proceedings.mlr.press/v108/ergen20a/ergen20a.pdf)中 rectified ellipsoid 和 spike-free 的相关定义。未核读整篇所有训练定理，也未采用论文优化算法。没有对本轮投影公式作外部新颖性认定。

## 执行与存档

配置为 paper_only，candidate_execution=false、model_execution=false、gpu_execution=false、shadow=false、replay=false。所有算例是纸面推导和独立人工式审查；没有候选 import、AST、编译、pytest collection、数值测试、求解器或 source worker。没有需要继续等待的本轮后台作业，也不伪造 run manifest。

最近通过的完整数学人口仍属 D158：4032项/212文件。不授予新纸面方案，不重跑旧版本。将来实施仍需新预注册、源码冻结、once-only 新版本、全部继承测试与原资源门；本轮没有减弱任何边界。

文档归档技能用于在既有本地研究树分开保存定理、来源与未达成结论；没有外部 Page 写入。所有新文件仅在本轮新目录及新的根级 resume，历史模型、日志、源码、结果和生产 dirty worktree 均未改动。未 commit、未 push。输入与最终档案分别用 ANCHOR_SHA256SUMS、ARCHIVE_SHA256SUMS 记录，只证明文件完整性，不证明数学或实验资格。

## 记账与未完成事项

正式 baseline 1870/2413=1063 CERT+807 validated ADV；13家族逐家族及每个旧解必须保全。独立E0 CIFAR100 25、TinyImageNet 36，共61/400，不与1870相加。本轮正式新增0、外部新增0，没有新 ADV 或重放结论。

尚缺一个在普通完整网络结构上同时给出有用精度、完整费用与实际可消费查询的 Neural-HZ 定义。当前不为已知块消元或球像单独启动实现。保持 GPU、smooth/Transformer、新家族和完整同路径回放目标，不因纸面困难缩小目标或标记 blocked；goal 仍 active。
