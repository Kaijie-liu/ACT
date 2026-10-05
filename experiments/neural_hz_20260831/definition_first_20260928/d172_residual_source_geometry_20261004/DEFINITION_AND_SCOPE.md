# 残差纤维的真实系数诊断依据

本次不是新域实现或网络验证，而是检查 [上一轮残差纤维定义](../d171_cross_layer_relations_20261004/RESIDUAL_FIBER.md) 的普通结构条件和训练权重几何是否值得继续。数学健全性、可用查询、真实参数范围与完整基准收益是四件不同的事；本诊断最多补充第三项证据，不获得其他资格。

## 保留末端偏置和完整块

两份已认证原图中的全部五个 MLP 块均为同源 residual 加两次 MatMul、一次 ReLU 和推理 BN。按列向量约定写为

```text
BN(q)=diag(s)q+t,
V=W1ᵀdiag(s), c=W1ᵀt+b1,
U=W2ᵀ, d=b2,
z=q+U ReLU(Vq+c)+d.
```

原图 MatMul 权重为 W1∈R^(48×96)、W2∈R^(96×48)。s_i=scale_i/sqrt(var_i+epsilon)，t_i=BNbias_i−s_i mean_i。BN 的正分母、常量身份、transpose 排列、单输出推理模式和所有消费者必须核对，不能只按节点编号猜公式。末端 d 不可吞进 ReLU 前的 c。将上一轮 z 替换为 z−d，读出常数加 wᵀd，参考改为 zbar=qbar+U ReLU(Vqbar+c)+d，aggregate caps 约束 z−q−d，即保持同一定义的健全性。

解释依据为 [ONNX BatchNormalization 9](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9) 的单输出推理模式及 [MatMul 9](https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul-9) 的乘法方向。这里只解释 FLOAT 存储常量的实数网络；并非框架浮点前向的完整误差认证。

## 不搜索参数的统一规范

对矩阵 A 的可靠逐元素区间，令 a_ij=max(|lower_ij|,|upper_ij|)，定义

```text
Q(A)=min(Σ_ij a_ij², [max_j Σ_i a_ij][max_i Σ_j a_ij]).
```

Frobenius 与 1/∞ 范数不等式给 ||A||₂²≤Q(A)。采用一个固定公式，不按哪块成功来选参数：

```text
K=Q(V), λ=1/(1+K), α=λK/2,
E=U+λVᵀ,
ν=sqrt_upper(K), ε=sqrt_upper(Q(E)), u=sqrt_upper(Q(U)),
ρ=α+εν, L_sector=1+εν, L_triangle=1+uν.
```

λK<1，满足上一轮证明所需 λK≤2。全部量由可靠区间代数得到；λ 是由整个真实矩阵的共同 K 决定的确切有理数，不是取区间中心假装真实系数。Q 对实际 U/V/E 都是上界，不是谱范数测量值。比较 L_sector 与 L_triangle 只比较这两个统一证书的保守常数，不等于真实 Lipschitz 常数之比，更不等于端到端验证提升。

BN 平方根用整数平方根生成 2^-64 网格上的包含区间，端点有理运算每步进行 512 位检查并向外舍入。不能借舍入为负半径、忽略二次残余或让独立消费者各用一份系数。代码按每块保存同一组 V/U/c/d 的规范区间证据；未来复用仍须新组合资格。

## 为什么仍须完整成本比较

结构上每 token m=96、n=48，隐藏幅值只被 U 消费，因而上一轮 n 个幅值替代 m 个幅值的条件有实际候选。原 m 个 bits 与 guards 必须保留，完整 token 人口与共享来源也不能因按 token 复用权重而省略。

若用两项范数各自的正负坐标轴作有限线性 lowering，除了 2m guards 与 2n aggregate caps，还需 4n 行，总计 2m+6n=480 行每 token；旧逐门四行是 4m=384。新幅值为 48 而旧为 96，并不自动证明总成本更低。编译其他完整消费者方向还要支付相应行、系数和可靠舍入。范数原生 membership、有限 outer LP 与后继传播也不能混为一谈。

已有中点斜率恒等式提供强比较：z=zbar+(I+UV/2)(q−qbar)+(U/2)(|Vq+c|−|Vqbar+c|)。它不能普遍支配上一轮 sector 证书，例如精确负转置结构可给 L_sector=1，而某些中点方向的三角证书严格大于1；严格差在非零扰动邻域仍可保留。此事实只说明本诊断有辨别价值，不说明实际训练权重必然有利。本次不额外展开 UV 或执行 midpoint 查询，避免把固定轻量系数人口改成新候选实验。

本研究仍优先 CIFAR/Tiny 普通结构，不因 ViT 存在宽块就换掉完整目标。已知同宽 identity 残差没有上述幅值优势；投影 shortcut 也不能直接当成 q+分支。不同结构需其自己的定义证明，不能按模型身份选择救援菜单。
