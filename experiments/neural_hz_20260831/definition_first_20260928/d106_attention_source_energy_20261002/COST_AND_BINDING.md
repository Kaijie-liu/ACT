# Attention 候选的完整费用与真实接入边界

当前候选只有纸面规则。现有元数据能证明旧未解 ViT 存在 Attention 结构，但还不能证明当前模型中的实际层绑定、乘积人口或完整成本。不能据小控制宣称真实可落地。

## 源码已经具备的能力

[融合实现](../../../../act/back_end/solver/solver_hz.py)的 _sparse_hz_softmax_value_fused（1476 起）保留 score/value 的共同仿射核心，可使用 Q/K 上下文，已有 Taylor、log-ratio、概率质量交叉余项界。它不是独立输出盒。

该函数在 1545 附近将 error_radius 交给 _sparse_add_error_generators；后者在 973 起逐输出写入各自 slot。[slot 分配](../../../../act/back_end/hybridz_tf/hybridz_tf.py)489 起使用不同连续槽。新增谓词由 score/value 或 Q/K/value 合并，并未出生共同 T 产品及列守恒。

[tf_mlp 调用方](../../../../act/back_end/hybridz_tf/tf_mlp.py)1711 起还交逐坐标输出界，后继也复用这些误差身份，所以不能称其在全网永远独立或无约束。此次静态检查发现的缺口限于该出生路径。

重要接入差异：fused 的 constraint_parts 不包含 probabilities HZ 本体，目前主要消费概率界。候选若显式使用 p，必须保留其真实坐标、simplex/ratio 谓词、Softmax 源和全部共享身份，不能把 p 当作免费已有的状态。新增行保存真实网络的规范共同见证，而非旧 exact=False 近似中每个独立误差赋值。

## 单组费用账本

设一组有 n 个 token、d 个输出通道，M 个实际源因子，其中原 bits 不删除。S_j 为源 j 在 score 或任一 value 通道出现的 token 集，K=Σ_j|S_j|，K_s 为 score 的产品出现数。Q 为各 value 仿射式的非零系数总数。所有数量都由实际源图确定，不能从模型名称猜测。

采用不做自动消元的显式实现时：

- 除原 HZ 和概率接口外，保留 K 个连续产品坐标，全部原 bits 仍在。一个 T_ij 服务多通道，不按输出再复制。
- 每个产品至多四条 McCormick 行、每行至多三个变量非零项，即至多 4K 行、12K 个变量 nnz；常量 RHS、变量界和可靠系数证据另计。此数字不包括原概率谓词或更强 RLT。
- 全 token 列守恒一条 EQ、|S_j|+1 个 nnz；稀疏列用两条遗漏质量界，每条至多 1+2|S_j| 个 nnz。它不复制遗漏产品，也不删除源因子。
- 单调性线性行至多 n+K_s+M_s 个变量 nnz，M_s 为 score 源支撑数；实际相同项合并及常数、舍入、证据都要支付。
- 输出读出保存 Q 个产品系数出现及至多 n*d 个中心概率系数；若物化 Y，还有 d 条链接 EQ 和 d 个输出系数。后继 Affine/Conv 仍按实际读出 nnz 支付。
- 复制每条源 EQ/LE 的完整 p 加权 RLT 是另一层增强，可能需 n 倍原谓词 nnz，不能算在上述小数目里或宣称已全部包含。

最坏 K=n*M，每个 query/head 的概率身份不同，不能跨 query 免费复用 T。对 q 个 query 为 Σ_r K_r 而非单一 K。多层额外状态会累积；旧 HZ、p 接口、乘积状态、终端矩阵、临时合并、证据副本和 decoder 同时存在的峰值都须计入。最终一条单调性行便宜，不代表取得它所需的整套产品便宜。

源成员、列身份、概率轴与区间前提检查也在同一费用内。没有完整资源实测、峰值、速度或四并发证据；不更改原 pool、存储、测试人口及 GPU 边界。

## GPU 与见证边界

产品索引 gather、同源系数归约、McCormick 行生成和输出矩阵操作具有 GPU 并行结构，但只是可能的实现方向。本轮没有设备代码或执行，不能将理论并行性报告成 GPU 加速。可靠舍入、原二元列身份、共享缓冲、设备与主机副本、证据及终端传输都要纳入同一个完整资格；此前 GPU 初始化资源失败不因此被豁免。

终端仍是原 LP/MILP。外包可行点不是 ADV；需由原输入 decoder 恢复输入，再经原具体网络及性质验证。真实见证重构中，每组至少支付 Softmax 和 K 次源产品、实际读出与网络运算；LP 给出的伪 T 不可直接当作真实乘积使用。未认证指数、数值失败或资源失败均不能推断 SAFE。

## 尚缺的真实来源与下一步

[正式未解清单](../../manifests/formal_unsolved_structure_manifest_v1.json)没有 tokens、heads、feature shape、Softmax 轴、逐 query 概率分组、Attention 前源因子数量或支持集。[生成器](../../generate_formal_unsolved_structure_manifest_v1.py)331 起只记录算子计数等。不能从 pgd_2_3_16/ibp_3_3_8 文件名推维度，也不能拿末端整网 n_g/n_b 作为 Attention 输入人口。

清单还提供了一个可行动的范围：ibp_3_3_8.onnx 为 303176 bytes，SHA256 9cc53b9edb35d40d70de1d816008a434c7e60b0404cf3cb38b9aefc189692883；pgd_2_3_16.onnx 为 325647 bytes，SHA256 246326387574617a274b6f73f7f52771df1195e03c99065ce4ad18c46b8e0984。二者都远小于 D015 特定来源工具的 64 MiB 原字节上限，但其 Conv/BN/ReLU 语法不能直接解码本候选的 Attention 语义。本轮仅从清单提取这些身份，没有加载或重新哈希模型。

下一步是在既有边界内预注册一次真实局部来源检查：先对这两个唯一未解 ViT 模型只读提取 Softmax→MatMul 连线、axis、静态 shape 和 initializer 身份，不执行网络、shape inference 或候选域。新的读取工具在任何 AST/import/执行前冻结来源、配置和自动保存路径，不修改 D015。图级检查不能自动证明 native HZ 的 s/V 共享 frame、真实非零支撑和原 bits；这些后续还需各自认证。只有取得这些实际人口，才能形成 K/nnz/依赖账本，并查清哪些原谓词需要保留。不直接启动全模型或复活 D099 已否决构造路径。

同时要以精确共享多项式乘积和已知单调/IQC 关系为强比较，不能只和更弱的逐产品 McCormick 比较来宣称创新。若真实费用不可承受或关系已被等价强基线覆盖，记录该范围的负结论，保留数学结果而不默认启用。ReLU、CIFAR100、TinyImageNet、其他家族及 GPU 的完整目标不缩减。
