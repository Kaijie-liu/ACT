# CIFAR 与 Tiny 的预终端完整仿射接口

本轮只读检查已经封存的 D152 全图元数据，发现应与“最后 ReLU 到最终方阵”的旧结论区分的实际结构：最后一个卷积 ReLU（全图倒数第二个 ReLU）后，到下一次 ReLU 前的整段路径均为候选仿射操作。它可能承载本轮共同幅值域；不是已经认证的新模型变换、宽度压缩或性能收益。

## 三个存档模型的直接证据

来源为 results/d152_live_amplitude_metadata_20261004_v1/model_0.json、model_1.json、model_2.json 中的 nodes、initializers、graph_outputs、source。三者 metadata_complete=true，哈希与封存 artifact_hashes.json 一致。本轮没有重新读取模型权重或运行模型。

| 模型 | 新 q 的出生 ReLU | 到下个 ReLU 的路径 | 原父 skip | linear1.weight 的保存形状 |
| --- | --- | --- | --- | --- |
| CIFAR100 large | Relu53，输出178 | Conv54 → BN55 → Add56 → Flatten57 → Gemm58 → Relu59 | Add56 输入175 | [100,4096] |
| CIFAR100 medium | Relu51，输出170 | Conv52 → BN53 → Add54 → Flatten55 → Gemm56 → Relu57 | Add54 输入167 | [100,2048] |
| TinyImageNet medium | Relu51，输出170 | Conv52 → BN53 → Add54 → Flatten55 → Gemm56 → Relu57 | Add54 输入167 | [200,6272] |

全 nodes 的端口使用检查中，large 的 178、179、180、181、182 分别仅进入链上的 Conv54、BN55、Add56、Flatten57、Gemm58；medium/Tiny 的 170–174 同理。这些中间量不是 graph_outputs。原父 skip 不是新 q 的 identity 消费；它必须保留，但不强迫把新 q 逐坐标暴露到下一非线性接口。

原模型身份：large 5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16；medium aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4；Tiny 234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776。这些是旧审计保存的身份，不声称本轮重新认证了原字节。

## 这改变了哪个研究决定

D152 确认最终 ReLU 后的 linear2 是 100×100 或 200×200，不能据维数声称严格压缩；该结论仍然有效。但不能把它或“没有 pooling”扩大成全网络没有较窄完整消费者接口。上述更早的 q 经整段仿射链后才到下一 ReLU，是另一个结构问题。

若属性、系数和 shape binding 后续全部认证，令 K 为该 Conv 的线性作用，T 为该 BN 的真实仿射线性部分，F 为 Flatten，W 为 Gemm 的实际线性部分，则下一预激活可写为

```text
h = B q + S p + c,
B = W F T K,  S = W F,
```

其中 p 是同一原父 skip，c 含全部实际偏置。转置、alpha/beta、广播、BN 参数误差、Conv bias 和轴语义必须按实际节点解释后并入，不可直接用 nominal 权重替代。C 可按统一规则为完整 B 加 mass，而不是只选有利性质方向。矩阵合成只是结构绑定，不是域创新本身；创新检验对象仍是新域对该 q 的源相位联合语义与后继消费。

保存的权重形状提示下一接口的预期输出宽为 100、100、200。当前记录没有解码该中间 Gemm 的浮点/整数属性、完整中间形状或系数，不能把 4096/2048/6272 当作已认证的出生 q 维数，也不能声称 B 的数值秩、稀疏性或总费用已知。

## 下一项受限资格检查

下一阶段只为这一普通结构预注册完整语义绑定与成本检查：覆盖全部声明的同类链，保原父 skip 和全部新原相位，记录合成系数、界证书、填充、存储、guards、energy、终端和 decoder 成本。规则由拓扑及数学条件统一触发，不按模型身份、公开结果或 margin 挑选。

新模型读取或数值活动仍须新隔离预注册冻结；本页本身不是运行授权或已执行计划。数学比较仍须同信息旧图和相同资源，不允许用此图匹配替代强参照、真实验证收益或完整回放。
