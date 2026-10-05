# 真实归一化适用性与现有 Attention 参考

本次仅检查已有 ACT 文本、算子清单和实现；没有加载、重新哈希或执行模型。在已检索归档中，尚未建立一个标准 LayerNorm 的真实来源包。不能用模型名称、Div 数量或架构目录替代这份证据。

## 旧模型清单能证明什么

来源为 [正式未解结构清单](../../manifests/formal_unsolved_structure_manifest_v1.json)，其 [生成器](../../generate_formal_unsolved_structure_manifest_v1.py) 的 _operator_signature 保存算子计数、动态 Add/Concat/MatMul、IR/opset 及身份，不保存节点连线、归约轴、epsilon、gamma/beta 或中间输入域。清单绑定 benchmark commit 8b7b811b78ce6a329dc96f04ae6652da3c245948，默认模型根 /data1/Kane/data/vnncomp2025_benchmarks/benchmarks。下列身份来自清单，不是本次模型字节复验。

| 模型 | 未解人口 | 相关算子 |
| --- | --- | --- |
| vit 的 pgd_2_3_16.onnx | 95，其中 48 UNKNOWN、47 TIMEOUT | BN5，ReduceMean1，Softmax2，MatMul16 其中动态4，Relu2 |
| vit 的 ibp_3_3_8.onnx | 15，其中 9 UNKNOWN、6 TIMEOUT | BN7，ReduceMean1，Softmax3，MatMul24 其中动态6，Relu3 |
| cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx | cgan:19、cgan:20，均 UNKNOWN | Div44、BN6、Softmax2、动态 MatMul4，但无标准 LN 必要结构的完整证据 |

两个 ViT 签名没有 Div、Sqrt、Pow、GELU、LayerNorm 或 LayerNormalization。一个 ReduceMean 不足以判断用途。cGAN 模型记录大小 272784587 bytes，SHA256 为 10be6af09db7f6cd116a8b820bb93121b80a6b845df77c0347db4c443b354e35，IR4、默认 opset9、319 节点。其完整算子计数是：

```text
Add13 AveragePool10 BatchNormalization6 Cast24 Concat8 Constant20
Conv30 Div44 Gather6 Gemm2 MatMul68 MaxPool4 Mul8 Pad10 Relu13
Reshape10 Shape6 Sigmoid1 Softmax2 Squeeze1 Tanh1 Transpose4
Unsqueeze18 Upsample10
```

动态合流为 Add13、Concat8、MatMul4；没有 ReduceMean、Sqrt、Pow、Sub、LayerNorm 或 LayerNormalization。Div 分母是否静态不能由计数确定。

当前 [BatchNorm 转换实现](../../../../act/pipeline/verification/torch2act.py) 的 _convert_batchnorm 使用固定 running_var、running_mean：scale=gamma/sqrt(running_var+eps)，bias=beta−scale*running_mean，再构造 SCALE 与 BIAS。固定统计量推理 BN 是仿射，不是输入相关的径向 LN。这不构成所有归档 BN 的节点绑定和数值一致性认证。

## 仍然缺少的真实来源

[D015 source packet](../d015_source_shielding_20260928/source_packet_v1.py) 的 Conv/BN/ReLU 语法不支持 LN，且该特定包的 64 MiB 原模型上限低于上述 cGAN 大小。这不是放宽其他预算的理由。

[verifier 中 TinyRegressionBertLayerNorm](../../../../act/back_end/verifier.py) 明确使用 eps=1e−5，但属于合成回归 fixture；[Torchvision 映射](../../../../act/front_end/torchvision_loader/data_model_mapping.py)列有 ViT、Swin、ConvNeXt，也只是架构目录。二者均不是已冻结的真实模型、性质与 LN 输入域。

下一真实 LN 比较至少需要模型与性质身份、节点与标准公式绑定、axis/width/epsilon/gamma/beta、实际非零中心源域以及同源残差消费者。当前缺口不代表所有外部资产均无 LN；smooth/Transformer 的研究目标仍保留。

## 不得弱化现有 Attention 参考

静态阅读 [tf_mlp 的 MATMUL 路径](../../../../act/back_end/hybridz_tf/tf_mlp.py)及 [solver_hz 的融合实现](../../../../act/back_end/solver/solver_hz.py)发现：现有路径已识别 Softmax 来源、要求共同 frame，并把原 score_context、分数差及 value 源送入 sparse_hz_softmax_value_relaxation。

_sparse_hz_softmax_value_fused 已合并 score/value 的仿射核心及原因子，保留 Gb 和 EQ/LE；它还包含 Taylor 界、交叉余项界与 Q/K 上下文。_softmax_value_cross_radius 已有概率质量约束下的容量界。因此“共享 score/value”“利用 simplex 质量守恒”均不能重新计为创新。

值得进一步证明的结构问题是：该共享仿射核心之后，error_radius 经 _sparse_add_error_generators 形成逐输出余项，是否存在可以保留到混权或残差消费者的、更强且支付得起的同源联合余项关系？这是下一步假设，不是对现有实现不健全的指控，也不是已证增益。应先和上述完整融合参考比较，再决定是否预注册实现；不得由实例身份触发。
