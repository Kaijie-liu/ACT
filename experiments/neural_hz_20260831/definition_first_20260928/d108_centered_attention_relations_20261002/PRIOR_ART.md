# 已知数学机制与比较对象的边界

本轮不把单调性、中心化乘积恒等式或门控激活重新命名为新发现。主要引用均为原始论文或官方规范；没有用搜索摘要认证新颖性。

[Ramsauer 等，Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)，式63与附录式187–192，已有固定 keys 下加权 key 映射的 Jacobian／协方差形式；Lemma A24式479给出Softmax Jacobian谱范数β/2界，式478也给出无穷范数界。本轮根代理通过 [CMU 保存的论文](https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf)只读网络流提取对应原文，PDF解析器提示xref重建但成功输出公式段落。故D106引用Nair2025作为常数来源，不能理解为其首次发现；不改写D106已封存文档。

[Kim、Papamakarios、Mnih，The Lipschitz Constant of Self-Attention](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf)，§3.1讨论共同输入影响keys的额外Jacobian项，并证明标准点积self-attention在无界输入域不是全局Lipschitz。不能将固定keys的凸势单调性冒充整个self-attention的单调性；本轮引理只对实际score的Softmax使用单调性，动态QK变化全部留在ρ的界中。该无界结果不是否定本项目有界域认证。

[DeepT](https://files.sri.inf.ethz.ch/website/papers/pldi21-transformers.pdf)的节号本轮由文献审查复核：§4.8为点积，§4.9为乘法，§5.1为noise reduction，§5.2为Softmax重写，§5.3为sum=1精炼。D106“尚未核对§5.1点积全公式”的定位不正确，后续应使用§4.8。根代理直接网页读取该PDF本轮仍超时，未完成其全公式同事实比较；不能宣称D108胜过DeepT或共享PZ。

[Hendrycks、Gimpel，Gaussian Error Linear Units](https://arxiv.org/html/1606.08415v5#S2)，§2已经定义xΦ(x)并给出xσ(x)的SiLU及tanh近似区别。原文§4明确指出GELU本身非单调。D108使用的是单调gate，而非错误地假设整个GELU单调；这个门控视角本身亦为已有研究。

强比较还必须包含同方向有符号Taylor余项、完整联合余项/IQC及精确共享乘积。D108的控制通过最紧逐坐标余项，不意味着通过它们全部；新增关系若已由相同方向的强参考表达，就不能算精度创新。

用户本轮新增“在smooth function有突破性创新比configure selective要好”。当前仓库针对该短语的检索未找到具体对象，已发非阻塞澄清问题。这里保留原词和比较要求，不自行映射为另一篇论文，也不声称已超过它；未识别比较对象不阻止当前定义研究。
