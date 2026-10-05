# 真实残差结构与 GPU 算子范围

本轮只读取已保存的拓扑和数学文本，没有重新解码模型、跑网络或初始化 GPU。保存拓扑是判断研究应覆盖哪类普通结构的证据，不是候选有效性或速度证据。

## 已保存的目标拓扑

CIFAR100 large 的 [完整原始图 packet](../../results/d025_interval_capacity_20260930_v1/complete_0.json) 中 `packet.graph.op_counts` 为 20 Conv、20 BatchNormalization、10 Relu、8 Add、1 Flatten、2 Gemm，无 Concat。八个 Add 包括五个 identity shortcut 和三个卷积 shortcut。

首块为 `Relu_2 -> Conv_3 -> BN_4 -> Relu_5 -> Conv_6 -> BN_7 -> Add_8`，Add 的另一输入是 Relu_2 的同一输出 port 127。Add 输出 port 133 直接进入 Conv_9、BN_10、Relu_11。后续 Add 输出也不是立即进入 ReLU，最后进入 Flatten。

CIFAR100 medium 的 [保存 packet](../../results/d015_source_shielding_20260928_v2/partial_source_evidence.json) 元素 `[1].packet.graph` 为 19 Conv、19 BN、10 Relu、8 Add、1 Flatten、2 Gemm，无 Concat；六个 identity shortcut 和两个卷积 shortcut。首块主支经两次 Conv/BN，中间有 Relu_5；shortcut 经 Conv_8/BN_9，在 Add_10 合流，随后进入 Conv_11。该旧数值 census 未完成，不能把保存的完整图等同数值资格。

Tiny 的[旧 81 层转换拓扑](../../results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json)记录 19 Conv、10 ReLU、8 Add；它有对应模型 hash 的旧图审计，但存在已记录的 BN lineage 缺陷。这里只用作残差结构线索，不作为本轮原图或数值认证，亦不据此宣布已认证不存在 Concat。D025 没有进入 Tiny，不能把它理解为 Tiny 没有残差。

## 必须覆盖的实际算子

常见合流坐标为 z=c+sum_j w_j*ReLU(g_j)+S*x，而非直接的单门 q+v。Add 及其后续卷积输入可以有符号，不能给 Add 虚造相位 bit；identity/projection shortcut、BN 的带符号 scale、stride/padding 对齐和共享祖先必须保留。

数学记录中的加权规则允许任意固定 w_k，但 v_k=z-w_k*q_k 仍包含其他门和 shortcut。支持界 B0、B1 的生成是核心未解决成本，不能省略，也不能按实例或待证性质挑选门。D025 的 THEORY.md 明确没有解释穿过动态残差合流，所以不能继承它的首层证书当作本轮跨残差资格。

## 仅纸面的矩阵自由算子

对固定 a,b>0，以及共同源上已认证的 M_ij>=max(|(a*f_i+b*f_j)/2|,|(a*f_i-b*f_j)/2|)，pair 行写成

```text
a*r_i+b*r_j-(a*f_i+b*f_j)/2 <= M_ij.
```

若四个物理坐标已有表示，一次作用是 gather 与加权和，转置是对应 scatter。相应乘子的贡献为 r_i 上 a、r_j 上 b、f_i 上 -a/2、f_j 上 -b/2。将 f 替换为源仿射式时，其矩阵转置和 bias 补偿仍需付账。

常数端点的残差行可以写成 z-delta*alpha_k<=B0，其中 delta=B1-B0。若 z 已是原系统坐标，作用与转置各需一个 z 和一个 bit 的访问；若 z 只是仿射 readout，须应用真实消费者矩阵及其转置，或新增有成本的定义等式。不能同时假定 z 免费存在、源支持也无需展开。

令 P 为 pair 行数、R 为残差行数、S 为共同源支持合并总量、Q 为全部 B/M 证书的实际工作、C_V 为残余/消费者算子作用成本。行生成有 O(S+Q+P+R) 的纸面记账；一次矩阵作用和转置为 O(P+R+C_V)，另计数值认证、scatter/reduction、临时量、完整原 HZ、solver 状态和证据费用。支持 DAG 如被使用，另加其全部节点、边、bounds 和乘子，不可隐藏在上述 P/R 中。

这些不是实际运行时间、峰值内存或 all-GPU 资格。512-bit 有理数参考操作不等于 GPU 单次 FLOP；任何未来浮点实现必须另证向外舍入及累计误差。未重试先前失败的 GPU 初始化或 tracing 版本，未放宽系统权限和资源门。

## 本轮核验的保存输入

```text
fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0  results/d025_interval_capacity_20260930_v1/complete_0.json
bbf0d17e48439edc11b352ad4d38ea8fc3ad0e9e0bf99108802b51fd7faac456  results/d015_source_shielding_20260928_v2/partial_source_evidence.json
f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274  results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json
```

主代理核对了保存文件 hash、CIFAR 原始图计数、首块节点及全部 Add 列表；独立只读审查补充 shortcut 分类和 Tiny 的历史限制。没有更改任一输入文件。
