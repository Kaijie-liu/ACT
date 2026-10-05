# 真实 ViT 图检查结果与适用边界

D107 的一次性只读图元数据诊断已完成，预注册的两个唯一未解 ViT 模型全部保存，没有遗漏失败模型。它确认了 Attention 的真实连线，也发现 D106 的仿射 score 假设不能直接套用。没有网络前向、求解器、GPU、shape inference 或候选数学测试；formal gain 为 0。

## 原始证据和执行状态

唯一执行目录是 [原始 RUN](../../results/d107_vit_graph_inventory_20261002_v1/exit.json)。[退出回执](../../results/d107_vit_graph_inventory_20261002_v1/exit.json)记录 complete=true、worker_returncode=0；worker 用时 0.4139445312321186 秒，supervisor 用时 0.533805513754487 秒。这只是解码诊断时间，不是验证器提速结果。

[工作进程回执](../../results/d107_vit_graph_inventory_20261002_v1/worker_receipt.json)记录图提取完成、candidate_qualified=false。worker peak RSS 为 81825792 bytes，traced peak 为 82074954 bytes，tracer metadata 为 6734592 bytes；监督器对应为 375513088、4637395、27888 bytes。父子分别通过预注册的观察内存门，不等于 Neural-HZ 完整物理资格。

两个模型身份、源文件大小及全部节点分别保存于 [IBP 图](../../results/d107_vit_graph_inventory_20261002_v1/ibp_3_3_8_graph.json)和 [PGD 图](../../results/d107_vit_graph_inventory_20261002_v1/pgd_2_3_16_graph.json)。均为 IR4、opset9，原始中间 value_info 为空。原注释仅给输入 [B,3,32,32] 和输出 [B,10]，B 是符号 batch_size。FLOAT 权重和缩放常数未解码求值。

## 从图结构手推的形状

下表的 token、head 和中间 shape 是依据保存的整数布局常量、Conv、Concat、Transpose 和 Reshape 语义作出的纸面推论，不是原注释或执行 shape inference 的输出。

| 原模型 | 原节点数 | patch 数加 cls | heads 和每头维度 | 全部 Softmax 到 value MatMul |
| --- | --- | --- | --- | --- |
| ibp_3_3_8 | 189 | 16＋1，即 N=17 | 3 个 head，各 16 维 | 54→55、110→111、166→167 |
| pgd_2_3_16 | 133 | 4＋1，即 N=5 | 3 个 head，各 16 维 | 54→55、110→111 |

Conv 的 kernel=stride 分别为 8 和 16，无 padding。所有 Q/K/V reshape 目标均为 [B,-1,3,16]，所以 Q/V 为 [B,3,N,16]、K 转置为 [B,3,16,N]，score 和概率 P 为 [B,3,N,N]。全部 Softmax 显式 axis=3；按 [ONNX Softmax 的旧版规范](https://onnx.ai/onnx/operators/onnx__Softmax.html)，opset9 的二维展平在这里恰为 [3BN,N]，没有跨 head 混组。

每个 Softmax 的唯一直接消费者是对应 MatMul 的左输入，V 为右输入。没有中间 reshape、mask 或 Dropout。完整 token 轴收缩满足 Y[b,h,q,r]=Σ_i P[b,h,q,i]V[b,h,i,r]，符合 [ONNX MatMul 规范](https://onnx.ai/onnx/operators/onnx__MatMul.html)。每层有 3BN 个概率组、3BN² 个概率坐标、48BN 个输出坐标。IBP 三层共 153B 个组和 2601B 个概率坐标；PGD 两层共 30B 个组和 150B 个概率坐标。它们不是 native HZ 因子数、稀疏支撑数或完整成本 K。

## 对定义研究的实际约束

两个模型首块 Q/K/V 的节点 24/26/28 共同读取节点 20。其前面节点 19 是参数均为 initializer 的五输入、单输出 BatchNormalization，不是 LayerNorm。固定统计 BN 的数值合法性、可靠折叠和 ACT 转换后等价仍未认证。首个原始 ReLU 为节点 70，在首 Attention 和残差之后；不能宣称这里已有前序 ReLU 相位证据。

节点 51 是两个动态输入的 Q×K，节点 53 缩放，54 Softmax，55 为 P×V。若首块 BN 可合法视作固定仿射变换，Q 和 K 对原输入仿射，但两者乘积通常是二次关系。仅有共同图来源不能证明 D106 所需的已认证仿射 score `s=c+Az`。下一候选必须保留真实 QK 关系，或证明 sound score-HZ 的同一扩展见证如何支持新增关系；近似误差因子不能冒充精确原输入因子。

score 缩放常数 κ 只保存了标量 tensor 的类型、尺寸和哈希，未认证其值或符号，不能猜成 1/4。生产前端会做版本转换与其他处理；本次原图检查也不认证转换后 fused Attention 的匹配、native latent/frame、原 bits、概率误差或 decoder 绑定。

## 已讨论但尚未立项的下一比较

保留一个待检验思路以免丢失：固定 query/head 时，若真实 score_i=κ qᵀk_i，则 Σ_i(p_i−1/N)score_i=κ qᵀ(Σ_i p_i k_i−mean_i k_i)。若 K/V 均为同一 h 的仿射映射，可利用共同的 Σ_i p_i h_i。这里只是已知代数交换，不是新颖性或节省证明。

不能只数最后的 16 个乘积：IBP 的 N=17 只比 16 多 1，PGD 的 N=5 反而更小；weighted-key 或 shared-hidden 的产品、共同 p 约束、原 HZ、终端和重构成本都须支付。不同 K/V 权重也不保证能从现有 Y 恢复 weighted-key。此思路未预注册为新候选、未实现、未执行。

## 不变的资格与成绩

最新已执行数学组件仍为 [D098](../d098_native_relation_bank_20261001/RESULTS.md) 的 3845 测试、188 文件。本诊断不缩减或替代该人口。native model binding、新域资格、GPU 资格、完整物理资格和真实新解均未取得。正式 1870/2413 与独立 CIFAR100 25＋TinyImageNet 36 共 61/400 不变，两个基线不能相加。Goal 仍 active。

配置、模型哈希、依赖和执行边界见 [预注册](PREREG.md)、[冻结清单](freeze.json)与 [解码依赖清单](DECODER_SHA256SUMS)。本次只补归档解释，不改已消费源码、预注册、freeze 或 RUN，不重跑诊断，不改生产和历史结果，不 commit/push。
