# Smooth 激活与 Transformer 的定义研究存档

本页保存 Goal 新增范围及本轮静态研究，尚未选定新实现或预注册数值实验。目标是使非凸 Neural-HZ 适应 smooth activations 和 Transformer／ViT；“快速验证所有相关数据集”是目标，不是已实现能力或可保证结论。ReLU、CIFAR100、TinyImageNet 主线没有取消。

## 必须从定义处理的表达边界

有限二元因子、有限线性 EQ/LE 及仿射读出的 HZ，固定 bits 后为多面体的仿射像，整体是有限多面体并。它不能精确表示非平凡区间上真正非分段仿射 smooth 函数的输入输出图：图中的多面体段只能是仿射段，有限并只能得到有限分段仿射图。这是数学推论，不能误解成 smooth 函数的一维输出范围不能是区间。

因此新研究须明确选择更丰富的函数／非线性关系语义，或给出带证误差的健全近似。只新增有限线性行，不足以宣称精确 smooth 图。保留非凸原 HZ 并不等于禁止查询时使用有证外包；外包中的可行点也不等于具体 ADV。

## 已有能力和真实缺口

[正式基线](../BASELINE_LOCK.md)的 ViT 为 90/200：90 CERT、57 UNKNOWN、53 TIMEOUT。既有未解结构清单位于 [manifest](../manifests/formal_unsolved_structure_manifest_v1.json)。

| 已检查的未解模型 | 归档算子证据 | 解释边界 |
| --- | --- | --- |
| ViT pgd_2_3_16，95 例 | Softmax 2，MatMul 16，其中动态 4；ReLU 2，BN 5 | 没有 GELU／LayerNorm／Div 记录 |
| ViT ibp_3_3_8，15 例 | Softmax 3，MatMul 24，其中动态 6；ReLU 3，BN 7 | 旧 ViT 分数不能证明标准 GELU／LN Transformer 覆盖 |
| cgan small_transformer，2 例 | Softmax 2，MatMul 68，其中动态 4；Div 44，Sigmoid，Tanh | 没检查 Div 是否被常量折叠，不据此断言运行失败 |

本轮未加载 ONNX。上表仅限既有未解人口，不推断其他未检查模型；safenlp 的 FC–ReLU 基线也不因名称而归入 Transformer。

静态源码核对：

- GELU 在 act/back_end/hybridz_tf/tf_transformer.py:532 使用区间结果后 lift；不是保留精确输入输出曲线。interval_tf/tf_transformer.py:82 固定 tanh 近似公式，而 schema 允许 approximate 选项。后续必须先核对原模型是 erf-GELU 还是 tanh-GELU；没有做 parity，不能称已发现或修复具体运行错误。
- LayerNorm 在同文件 :410、:524 使用中心化、方差界和输出盒，特定条件下加零和关系；不是完整共享分母关系。
- Softmax 在 :545 已有差值界、概率盒、simplex、比例不等式，不能误称只有 [0,1]。比例行走 CPU／NumPy／SciPy。
- solver_hz.py:342 的 lift_bounds 保留旧谓词并引入输出区间因子，但不自动建立原输入到输出的精确非线性关系。
- tf_mlp.py:1655 及 solver_hz.py:985 的动态 MatMul 有共同 frame 下的 affine 核心加误差；tf_mlp.py:1697 及 solver_hz.py:1476 附近已有共享 score/value 的 fused Softmax×V。不能把 fused attention 或共享 Taylor 符号当成此次首次创新。
- dense Div 仅在非零 point 分母等条件下保留 affine HZ；稀疏 dispatcher 与 canonical exporter 的接入受限。本轮不扩充权限或修改算子。

这些是静态路径观察，不是运行资格、健全性全审或 GPU 域执行证明。

## 文献启发与不可移植部分

[Kochdumper 等，2023](https://arxiv.org/pdf/2207.02715)用共享多项式依赖及带误差的激活近似取得非凸外包。多项式映射可精确，不代表 sigmoid／tanh 原图精确。可借鉴依赖与余项管理，不整体替换原 HZ。

[Wei 等，2023](https://proceedings.mlr.press/v206/wei23c/wei23c.pdf)研究 Softmax 的指数倒数及 log-sum-exp 凸上下界。共同归一化和联合 logits 关系值得参考；这些界本身不是新的非凸域。

[ZonoGPT，2026 预印本](https://arxiv.org/pdf/2609.34457) §4.3 将 LayerNorm 作为整体展开，并保留原符号；附录 A.2 明确处理 tanh-GELU。该工作使用凸 zonotope，不能整体替代本项目 HZ。文中声称的证明不等于本项目已审计其实现。

[Certified Mechanistic Interpretability，2026 预印本](https://arxiv.org/pdf/2609.26112)是进一步线索：可关注 CPZ、共同多项式身份和归一化不变量。但其多层／端到端路径有采样余项乘安全系数及 MC／PGD 校准；这些不能作为本项目认证余项的依据。代理核对附录 A.3 对 epsilon 的处理，根代理已复核采样校准相关段落；不可仅据主文简写宣称整篇忽略 epsilon。该论文的完整证明和实现未审计。

## 待证的定义方向

研究假设是保留原 HZ 嵌入及全部原相位，再引入可被多个消费者共同使用的归一化／函数余项关系块，而非每坐标各换独立误差盒。

LayerNorm 可从以下精确实数关系开始：

~~~text
z=x-mean(x)*1
rho>0
rho^2*(epsilon+||z||^2/d)=1
y=gamma*(rho*z)+beta.
~~~

若 v=rho*z，则 sum v=0，||v||^2=d*var/(var+epsilon)<=d；epsilon>0 时不能写成恒等于 d。

Softmax 可从共同 shift a 和共同分母 D 开始：

~~~text
e_i=exp(x_i-a), D=sum e_i>0
D*p_i=e_i, sum p_i=1
attention_output=sum p_i*v_i.
~~~

LayerNorm 需要非线性代数关系，Softmax 需要超越关系。仅记录这些公式仍可能只是计算图包装，不构成创新。需要证明：共同关系在 Attention、残差和下一非线性中持续保留，与继承相同事实的强对照相比有严格收益，并且完整成本受控。旧 LP/MILP 终端边界不变时，须有带证 lowering；不得偷偷加入新非线性求解、backward／dual rescue、split 或采样认证。

后续先做结构定理与公平对照，再考虑默认关闭候选。GPU 不是给模型调用 cuda：须覆盖域操作及其误差、身份、缓冲和终端费用；旧 GPU 资源失败仍保留，不因此放宽上限。

## 证据状态

本页由根代理整合静态读取、两个文献／代码审查代理的回报及手工推理；没有新代码、模型运行、GPU、shadow 或全量回放。formal_gain=0。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；源代码已有 dirty 差异未改动，其哈希见本目录审计。完整目标以 [Goal 快照](GOAL_SNAPSHOT.json)保存，不覆盖旧目标文档的历史状态。
