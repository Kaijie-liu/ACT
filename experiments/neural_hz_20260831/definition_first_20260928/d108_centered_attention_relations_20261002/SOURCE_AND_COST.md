# 真实结构接点与完整代价边界

D107 显示真实 QK 有动态二次关系；D108 进一步确认首层存在恒定 query 的普通结构子类。这个发现缩小了首个可认证的来源范围，但消除恒定 query 的乘积误差已经是现有代码的能力，不得重新记为 Neural-HZ 新收益。

## 原图中可定位的子类

两个已保存的 [IBP 图](../../results/d107_vit_graph_inventory_20261002_v1/ibp_3_3_8_graph.json)和 [PGD 图](../../results/d107_vit_graph_inventory_20261002_v1/pgd_2_3_16_graph.json)均有如下链：14 ConstantOfShape 只读 batch 形状，15 加固定 cls_token，16 放为 token0，17 加固定 positions，18 转轴，19 固定参数 BN，20 转回，24/25 为逐 token query 仿射投影。

BN9 只有一个输出，按 [ONNX 对该版本的规范](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9)为测试模式，不从当前 token 重新计算统计量。因此在参数有限、BN 分母合法的前提下，首层 token0 query 不随像素值变化，各 head 的整行 score 对原像素仿射。这里是原图的结构推导，尚未解码浮点参数或认证 native 系数。

每模型首层有 3B 个这样的 query 组，分别为 N=17 和 N=5 个 token。统一条件应为“认证 query 行的原数值依赖为空”，不是 CLS 名称、token 下标、模型身份或 margin。后层 CLS 已经过 Attention，不再有该常量保证；训练态 BN、跨 token 归一化和输入相关 token 会破坏这条推导。

两个模型末端分别在 IBP186/PGD130 执行 ReduceMean(axis=1)，平均全部 tokens，然后 BN、Gemm。不是 CLS-only；不能只保 CLS 而丢掉其他 query。首 CLS 输出可经输出投影、残差、FFN ReLU 和后续 Attention 到达分类头，但图可达不等于非零敏感性或性质收益。

首块 patch 卷积没有重叠，局部原输入支撑上界分别为192、768个像素；覆盖全部 patch 时仍是3072个图像坐标。这些是结构上界，不是实际 Gc 的 nnz、输入扰动人口或已认证费用。

## 现有实现已经支付并利用的内容

[solver_hz.py](../../../../act/back_end/solver/solver_hz.py) 中 `_attention_score_affine`（1334 起）检查 Q/K 同 frame 与 term rows；1422 起用 Q 的 generator radius 收紧 bounds。1434–1440 的误差与 query 区间宽度相乘，常量行使其数学项为零，代码 nextafter 仍保留极小向外量。fused 在1512调用这一直接 score 接口。因此“首 CLS 的 QK 没有二次误差”本身是旧能力。

普通 `sparse_hz_matmul_relaxation` 的1037附近会将 Gc 每行截到256项，再把 omitted radius 付入误差；所以原图 score 仿射不等于中间 score-HZ 就是精确原输入仿射式。`sparse_hz_is_point` 的1868起使用1e-12容差，也不能替代新来源资格所需的严格零依赖证明。

[tf_transformer.py](../../../../act/back_end/hybridz_tf/tf_transformer.py) 136起发现 Q/K context，253起构造 score differences，305附近保存 bounds/rows/scale；转换后是否实际命中这些条件尚未验证。BN 转换见 [torch2act.py](../../../../act/pipeline/verification/torch2act.py)1568起。原始 Mul 转换、参数合法性、截断误差、frame 和全部源列都需要明确绑定，不能由原图名字推断。

现有 fused 出生合并 Q/K/value 约束，1546附近加入逐输出误差槽，不保留 probabilities HZ 本体作为 constraint_parts。故一般含 p 的新行不能把它当作免费存在；D108 研究的概率消去有实际费用动机。

## 代价按最终表达式计账

令 n 为每组 token 数，d 为输出维度，A 为预先按源结构确定的有限方向集合。对一个方向，已有可见变量上的中心化行最多需要 d 个 Y 系数、nd 个 V 系数和 n 个概率系数；若 wᵀΔp 被有证仿射下界消去，则换成 score 读出系数，而不是免费消失。

若直接把行展开到已有 HZ 因子，真实 nnz 由这些线性组合的支撑并集决定。要遍历各源表达、合并共同列、处理系数增长、可靠舍入、原 bits、RHS 和证据。不能把可见层的一行称成最终只有 O(n+d) 存储。通道、query 和层间所有行的生命周期与共享存储还要累积。

与 D106 的 K 个新增产品、最多4K条 McCormick行及质量约束相比，D108 为这条关系可以不新增产品列；这是符号层成本差别，不是相同完整关系精度的等价压缩。缺陷支持 R 与固定权重 Softmax 下界可能损失精度。旧 output HZ、其独立误差、全部原约束仍在；两者求交保持真实见证，不可悄悄替换为一条弱界。

若 s/V 是认证的同源仿射式，ρ 的界可通过残余系数的带证绝对值归约取得，临时和证据成本取决于实际 nnz；原二元列保留并参与支持上界，不连续松弛为新域。动态 QK 需要另计二次残余；本轮没有把它们算作零。

一般概率消去需要可靠的固定权重 Softmax 下界。现有 `_softmax_taylor_coefficients`（1151起）确实输出 affine、intercept、error_lower/upper，但其使用 score差值参考、概率和ratio界、Hessian和定向误差；若移植必须按原语义重建完整证书，不能将函数名当证明。它也含按 n² 布置的数组，完整 GPU/CPU峰值必须计入，不只数最终LE。

若其经认证的完整下界为 `wᵀp ≥ alphaᵀs+beta+e_lower`，则实际行应为 `aᵀY−Σ_i p0_i aᵀV_i−alphaᵀs ≥ beta+e_lower−wᵀp0−R`。不能漏掉参考项 wᵀp0，也不能取上侧误差。helper自己的展开中心不必等于D108的s0，但两者间常量换算和完整同组来源必须认证；nextafter或固定1e−10保护量本身不是这张新证书。

方向 a 的统一生成规则尚未冻结。可研究从源映射做固定前向线性代数投影，任何实际有限 a 都需重新认证 R；拟合最优性不是健全性前提。不得用终端性质、margin、标签或求解状态挑方向，也不能通过不断调配置宣称 smooth 创新。

## GPU 及下一项可证伪工作

源残余矩阵乘、共同系数归约、支持计算和行生成适合批处理，是待实现的 GPU 域操作；尚无设备代码或实测。原内存、证据、终端传输、四并发和完整物理门都不变。输入恢复仍用原 decoder；外包可行点须经具体网络验证，伪误差不能作为 ADV。

下一步先把统一的方向规则、参考和误差下界定义完整，并以CONTROL已给出的更强方向曲率关系为必需比较对象；不能再次只用逐坐标Taylor。只有证明新关系语言或统一消费的实质差异后，才预注册最小真实来源证书检查，读取两个模型的全部相关第一层恒定-query组及源参数，不重跑 D107。必须同时计 source、probability消去、新增行和旧输出的完整费用。若只有相同行换包装或实际残余过大，保留负结论，不进入生产或扩大数值人口。

GELU/SiLU 另需真实模型变体、gate坐标及同结构强比较，不能拿两个无GELU的ViT代表其通过。以上顺序不缩减GPU、CIFAR100/TinyImageNet、13家族和全2413的目标。
