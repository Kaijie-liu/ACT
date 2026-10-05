# 强参照审查与真实残差来源核查

本轮由上一轮已完成的 D150 数学组件继续推进。重新核对当前分支、工作树、权威目标与 D150 归档后，得到 [同坐标精确参照支配及共同幅度反例](THEORY.md)，因此改变原本准备进行的完整真实结构移植：先修改域定义，不为被该参照压过的版本继续增加实现框架或小测试。这是新的纸面证据改变下一行动，属于研究 progress，不是网络验证收益。

## 真实输入范围

主代理与独立只读审查共同核查 D025 large 完整工件、D015 medium 部分工件的 JSON 索引 [1]（数组第 2 项），以及既有 D120 对应引用。原始模型均存在，stat 为普通文件：

- /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_large.onnx，15243961 字节。
- /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx，10156168 字节。

本轮没有重算模型字节哈希、解码 ONNX 或运行模型。文件大小与已存 metadata 相同不能替代未来的源身份认证。

large 包已存 Conv0/BN1 和 Conv3/BN4，首两次 ReLU 为 1×64×32×32，并保留 Relu2 的 identity skip。要跨过 Add8 后下一真实 ReLU11，仍缺 Conv6/BN7 与 Conv9/BN10 数值参数。Add8 输出 133 除进入 Conv9，还进入 Add14，必须继续存活。

medium 包已存 Conv0/BN1、Conv3/BN4，以及 Conv8/BN9 shortcut。首 ReLU 为 1×64×15×15，Relu5 与投影 skip 为 1×128×8×8。跨过 Add10 后下一真实 ReLU13，还缺 Conv6/BN7 与 Conv11/BN12。Add10 输出 129 还被 Add16 使用，不能只绑定直接后继。

packet.graph.nodes 只含节点类型、索引、名称和输入输出端口，不含缺失节点的参数数组、卷积属性或认证 shape。model_raw 也只有 byte_count 和 SHA，不是原始字节副本。D120 引用明确不重复参数。需要新的预注册 decode/bind 才能补齐这段范围；不是已经做了完整残差实验。本结论限于本轮指定工件，不声称遍查历史所有文件。Tiny 同等完整来源仍未核齐。

## 范围与证据保存

数学审查分工为同坐标强参照、普通共同幅度反例、原始参数边界三路；主代理逐式推导四行包含、nnz、RHS、普通例子的全盒上界及新层误差延拓，并核对实际图端口。边规则只保留为已知有效关系的比较，不新增候选代码。

没有 candidate import、AST、compile、collection、pytest、LP/MILP、source worker、模型前向或 GPU 调用。未消耗新数值版本，不需要也不授权重跑冻结 D150。没有改测试人口、资源上限或任何权限边界。D150 的 4000 项通过仅作为此前完成件，不在本轮重新计一次成果。

查阅的一手外部资料包括 [神经网络二次约束框架](https://arxiv.org/abs/1903.01287) 与 [Ellipsotopes](https://arxiv.org/abs/2108.01750) 的作者/论文页面。前者用于确认已有关系抽象思想，后者用于辨认范数生成元载体已有先例；均不作为新颖性或本项目速度证据。一次已打开 PDF 的后续段落请求失败；不把未读取段落作为公式依据。核心证明由本地公式和纸面推导提供。

2026-10-04 Australia/Sydney，branch redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。tracked diff SHA256 仍为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。所有新文件仅在 experiments/neural_hz_20260831 新隔离目录；生产改动、历史文件与 /data1/Kane/HyZor 不变。文档归档将定理、负结论、真实来源和执行资格分开，避免后续把数学组件误报成创新成功。

formal_gain=0，正式 1870/2413 与独立 61/400 均不变。GPU 尚未获得成功计算资格，smooth/Transformer 和完整网络收益均未达成。没有本轮后台任务，整体 goal 继续 active。
