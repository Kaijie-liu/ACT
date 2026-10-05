# 联合约束的先行研究与本轮核对范围

本轮控制支持一个具体的信息缺口，不支持“首次共享概率产品”或“首次发现 Softmax 能量关系”。以下按来源实际核对程度分开，避免将文献线索升格为完成比较。

## 已核对的单调性主源

[Gao 与 Pavel，On the Properties of the Softmax Function](https://arxiv.org/pdf/1704.00805) Proposition 3 给出 Softmax 单调性；Corollary 2 给出与 inverse temperature 相关的共强制性。根代理读取了对应原文段落。

[Nair，Softmax is 1/2-Lipschitz](https://arxiv.org/pdf/2510.23012) Section 4.1 Corollary 1 明确给出更强的 2/lambda 共强制性。标准 Softmax、参考源为零时，直接得到本轮使用的 E≥2*||p−uniform||²≥0。根代理核对了该式，不依赖仅凭摘要推测。这个不等式及其常数均不算本项目新发现。

## 必须保留的 Transformer 比较对象

[DeepT，Fast and Precise Certification of Transformers，PLDI 2021](https://files.sri.inf.ethz.ch/website/papers/pldi21-transformers.pdf)已有 Fast/Precise 两类点积抽象和 Softmax 和为1的精炼。代理核到 Section 5.3；根代理直接 PDF 打开超时，随后读到作者原 PDF 的公开索引正文，确认两类点积实现、概率精炼和 GPU 实验描述。本轮没有完整核对 Section 5.1 全公式，也不称 Precise 为精确乘法。其整体凸域不能替换本项目 HZ，但其共享符号与专用点积不能被当作没有。

[Towards Formally Verifying LLMs: Taming the Nonlinearity of the Transformer](https://openreview.net/pdf?id=evDSvZBFRP)是进一步强比较线索。文献代理从原 PDF 索引正文定位到 Section 4.2 式11的多项式 zonotope 乘法和 Appendix B 式19的相关 Cartesian product；根代理当前访问遇到 OpenReview browser challenge，未能直接全文复核。应按原始投稿线索保存，不能据此宣称论文接收状态、实现正确性、Softmax 精确性或本候选已经胜过它。

本轮检索未核到一个 Transformer 验证器明确采用同一源能量行耦合概率与 value；这不是“不存在先例”的证据。完整 IQC、相关多项式产品和联合余项域都是需要检查的更强参考。

## 本项目已有的相同原语

[D018](../d018_order_relations_20260930/COMMON_SOURCE_COMPARISON.md)已经记录共同源 perspective 和完整提升成本；[D035](../d035_cross_phase_source_20260930/THEORY.md)已有具名乘积具体化及原 HZ 嵌入；[D049](../d049_mixed_source_envelopes_20260930/ALTERNATIVES.md)已有单源产品凸包审查。源乘积、simplex 乘源得到的列守恒及 McCormick 均不是这轮的新理论。

本轮真正新留下的项目证据是 [同一强参考下的严格控制](CONTROL.md)、该信息如何到达不稳定 ReLU，以及 [当前生产出生路径与新状态的费用差别](COST_AND_BINDING.md)。不能据此完成新颖性或正式能力晋级。
