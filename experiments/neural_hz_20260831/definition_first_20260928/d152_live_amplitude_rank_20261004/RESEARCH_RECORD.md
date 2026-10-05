# 共享幅度路线的数学限制与真实终端形状

本轮把重点保持在 Neural-HZ 的域定义上：检查固定 anchors 加共享误差能否实质替换逐门非线性幅度，并用固定三个原模型的结构元数据检验终端低维读出的适用性。结果不支持将这两种直接压缩设想推进为候选；没有新的 Neural-HZ 能力突破，也没有新正式解。

## 数学结论及已有工作的边界

[THEORY.md](THEORY.md) 证明：对于具有独立双侧可达折面的未锚门，固定仿射误差加载必须含相应坐标方向；全部满足前提时，k 个原门 anchors 加 r 个误差方向须满足 k+r>=m。eta 即使源相关、相位相关且非凸，也不能改变投影到 ker(E^T) 后的恒等式。完整活读出 Vq 的版本仅要求 r>=rank(V_U)，原输入旁路与幅度 identity skip 必须区分。

这是 D004/D131 已有折面跳跃推导的直接推广，不是新外部定理。不排除相位依赖加载、非线性共同 decoder 或不同抽象语义；也没有认证真实模型的全部可达折面前提。正权聚合四行是健全但不精确的已知外包，普通三门反例与完整 nnz 计数已记录，不能包装为新域。

本轮查阅 [LiNNA 原论文](https://arxiv.org/abs/2307.10891) 及其作者全文，用于对照“保留一组神经元作为线性表示基础”的已知路线。该工作不替本项目证明约束非凸域的健全性、实际能力或成本。当前自己的固定加载障碍来自上述明确局部假设，不从外部论文外推。

## 唯一元数据执行

按 [PREREG.md](PREREG.md) 冻结两文件后，仅执行一次 `audit_metadata.py --enabled`。外部工具会话 37238 完成，shell exit 0；内部最终归档计时 5.616436876356602 秒，低于 60 秒冻结上限。没有 rehearsal、AST、compile、候选导入或重跑。

完整 7,285 source 与 14 input 身份前后均通过；三份模型由认证原字节各解析一次，未读取浮点系数、计算新界、矩阵秩、模型 forward 或求解性质。全部 30 个 ReLU 都报告匹配或不匹配原因，不只保存命中的三个。

| 固定原模型 | 全图节点 | ReLU | 唯一终端匹配 | B 形状 | 整数维度上界 |
| --- | ---: | ---: | --- | --- | ---: |
| CIFAR100 large | 61 | 10 | Relu59 → Gemm60 | 100×100 | 100 |
| CIFAR100 medium | 59 | 10 | Relu57 → Gemm58 | 100×100 | 100 |
| TinyImageNet medium | 59 | 10 | Relu57 → Gemm58 | 200×200 | 200 |

三个终端 ReLU 均只有该 Gemm 一个消费者，没有幅度旁路；三个图都没有 AveragePool、GlobalAveragePool 或 ReduceMean。B 名字为 linear2.weight，transB=1，transA 缺省为 0。显式 alpha/beta 均存在，但按注册未读浮点值；不能据此宣布实际数值映射、A/batch shape 或 bias 广播已经认证。

结论仅为：此处没有由输入输出维数直接保证的严格压缩。方阵不是满秩证明，更不是任何压缩都不可能的证明。不扩大为整网结论，也不以接近低秩的数值近似绕开精确身份要求。这与 D004 的旧 descriptor 方阵记录一致；新增的是固定三份原 ONNX 身份与完整消费者证据，不是第一次发现该形状。

原始证据保存在 [result.json](../../results/d152_live_amplitude_metadata_20261004_v1/result.json)、[model_0.json](../../results/d152_live_amplitude_metadata_20261004_v1/model_0.json)、[model_1.json](../../results/d152_live_amplitude_metadata_20261004_v1/model_1.json)、[model_2.json](../../results/d152_live_amplitude_metadata_20261004_v1/model_2.json)。exit 和 artifact hashes 自动保留。

RSS high-water 为 184,045,568 字节，增量 164,630,528 字节；tracemalloc peak 为 107,449,840 字节，其 metadata 为 21,392,896 字节。分别加 65,536 字节 reserve 后均低于 1 GiB。它们只度量本次元数据审计，不能冒充 Neural-HZ 的完整物理内存或速度资格。

## 决定与下一交付

关闭本轮“固定低秩误差即可普遍压缩”和“此处终端输出维数更少”两项直接实施假设，不继续写对应候选或适配层。D149/D150 数学组件仍只读保留；D151 强参照负结论不变。

下一交付必须重新提出有实质差别的域元素、具体化及可组合算子，而不是再审计同一形状。重点仍是共同源、原相位与共同幅度关系如何在普通混权、残差及下一非线性中保留；相位依赖加载或共同非线性 decoder 只有在摆脱 D001 精确图改名和 D131 终端恢复全幅度的成本后才值得实施。当前没有已经解决该问题的新候选，不将研究方向写成既成突破。

全部原禁令和验证门不变。最近完整数学人口仍为 D150 的 4,000 tests / 210 files；本轮非数学候选检查，没有减少、替代或重跑该人口。生产源码与既有 dirty worktree 不变；无 helper、attack、PGD、BaB、split、backward/dual rescue、solver、GPU 或网络前向。本轮独立静审覆盖证明、图消费者、转置维度、身份认证与失败部分保留，不授予新域资格。

正式 baseline 1,870/2,413，独立 E0 CIFAR100 25 + TinyImageNet 36 =61/400 均不变；formal_gain=0，invalid ADV 没有新增提交。未作 shadow、逐家族或全量回放，未启用默认候选。GPU、smooth、Transformer 与各家族大幅能力提升仍未达成。

2026-10-04 Australia/Sydney，branch redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；`git diff --binary HEAD --` SHA256 仍为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新工作仅写隔离实验目录。文档技能只用于将证明、已知研究、真实证据和未获资格分开归档，没有创建外部 Page 或修改权限。所有本轮执行已退出，无后台实验；整体 goal 继续 active。
