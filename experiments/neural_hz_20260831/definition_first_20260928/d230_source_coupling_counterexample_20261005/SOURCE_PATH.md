# 真实 medium 前缀的可复用来源和接入边界

本页保存2026年10月5日只读源码检查的结果，避免下一轮重新寻找模型或把FX命名当成必须先解决的研究问题。没有执行原模型、解码新权重或建立完整native前沿。D230当前先检验来源相关性反例；本页不授予模型资格。

## 可复用的真实材料

D015 partial_source_evidence.json的第1项保存medium的完整3072维原输入盒及首双bank参数：Conv0/BN1的64通道、Conv3/BN4的128通道、Conv8/BN9的128通道shortcut。完整参数并不局限五个空间窗口，但该历史失败试验没有完整前沿结果，不能继承其资格。

路径 experiments/neural_hz_20260831/results/d015_source_shielding_20260928_v2/partial_source_evidence.json  
SHA256 bbf0d17e48439edc11b352ad4d38ea8fc3ad0e9e0bf99108802b51fd7faac456

完整图/形状/消费者由D179 model_1.json保存：  
experiments/neural_hz_20260831/results/d179_preterminal_domain_20261004_v1/model_1.json  
SHA256 f93c16ae170e62d688a76a4f630dfadfb8dcaf75a394b0e59eaf87390405aadb

D210 admission.json重核完整3072→14400→8192及全部首bank消费者：  
experiments/neural_hz_20260831/results/d210_full_prefix_admission_20261005_v1/admission.json  
SHA256 36be38c278b1327b1ec9d8b0797d4c7e78f59012b97261b1d7c809ebf39a06cb

这些哈希本轮已重新读取核对。它们没有提供完整14400/8192可靠L/U、实际native相位列或解码器证书；D211未冻结草稿也不能填补缺口。

## 真实消费链

原模型为 /data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx。完整路径：

- 3072输入，经Conv0/BN1到Relu2，形状64×15×15。
- port121同时送Conv3和Conv8，不得丢shortcut。
- Conv3/BN4到Relu5，形状128×8×8；port124再经Conv6/BN7。
- Conv8/BN9的port128与主支在Add10合流。
- Add10之后经Conv11/BN12，才到下一Relu13。Add10之后没有即时ReLU。

D015已有Conv3的73728权重及Conv8的8192权重，但未导出Conv6/BN7、Conv11/BN12数值。可从同一冻结原ONNX字节复用source_packet_v1._Reader.conv/.bn续读，原port/shape/全部消费者一并认证；不需要另造通用loader。source_binding_v2.extract_model提供显式batch1，census_worker_v2提供input_box/conv_shape/receptive/post_affine等原工具。不得直接用旧census()冒充完整前缀：它仅遍历五窗口并跳过shortcut。

若只在Conv6读出查询，须报告“真实后继仿射界”，不能声称已经收紧Relu13。

## 精确相位出生与原模型包络

D229只验证存储的原生图，并不自动证明它等于原ONNX。原extended出生含浮点c-Q，提取时解释为两个存储有理数相加，可能不同于存储的c。原BN折叠还包含sqrt及舍入；graph_error=0不能替代原源认证。这是实际接入的普通可靠算术要求，不应发展为极端case专项。

一个纸面可行但未实现/执行的桥是：先有完整证书

```text
f_true in c_hat + sum g_hat_k theta_k + [-E,E]。
```

在原同一H追加连续epsilon,zeta，并存字面EQ

```text
M*zeta - sum g_hat_k theta_k - E_hat*epsilon = c_hat，
```

E_hat向外覆盖E，M可靠覆盖真实|f|，实际theta含全部同源连续及原二元坐标。让零中心读出M*zeta进入已有extended ReLU，RHS=-Q与提取+Q可精确抵消。真实点选epsilon为证书残差归一化、zeta=f_true/M即有共同扩展。E=0时残差必须恰零。完整系数误差补偿及全部源绑定不可省略，不把中点网络替换成原模型。

这只是安全构造假说，不是新域创新、完整运行时认证或已实现代码。它还可能扩大局部cube、增加门和连续因子；须实际核精度及费用。D096已有精确carrier出生，D097有carrier等式商化；D229当前没有其证书入口，不能把固定carrier直接当自由盒后宣称等精度。

## 完整代价

medium首Conv结构项388800；Conv3未删值的有效结构项为128·64·22²=3964928，不是单个7,7窗口。实际原H nnz仍取决于完整传播，不能用此上限直接冒充实测失败。

D229源扫描对21个原数组收费近似Phi=8S+B（S为标量总数，B为总字节），每个公共操作前后两遍。一次bind加一次receiver至少4Phi，另有全部算术、Bank、完整前缀和证据费用。int32 CSR时每nnz约贡献28到Phi，故仅这四遍的256M粗必要上限约228万nnz；这不是新的门，也不是测得的整网费用。不能因为预计超账就缩到局部好窗口，不能套用D209特有9dm下界到所有实现。

在真实前缀投入前，D230反例要求先说明源关系如何在联合图生成和消费中存活。复制更少、载入更快或源映射完整均不能替代这一数学问题。
