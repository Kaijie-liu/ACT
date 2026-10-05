# 多门关系研究的证据范围与已有性

上一目标轮属于progress：D060完成了共同排序消费者投影和两门消元的成本审查，关闭了两个直接编码版本，而非仅重述计划。本轮先核对其八项校验全部一致，分支仍redu-hz、HEAD仍f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；没有重启或等待已结束的D057。

本轮新增[固定多门构造及强对照](THEORY.md)。它没有新颖性认证、实现或数值成绩，但给出了不同于已核固定pair/cycle集合的具体物理关系，以及穿过下一原门的解析安全性质。日期2026-10-01，配置paper_only、read_saved_json_only、primary_literature_only。

## 两条未采纳的定义捷径

支持函数差v=h_P-h_Q及ReLU(v)=h_conv(P union Q)-h_Q是已有热带神经代数。[Zhang、Naitzat、Lim 2018](https://proceedings.mlr.press/v80/zhang18i/zhang18i.pdf)的Propositions 5.1和5.6给出相关递推和实权重范围；本项目[D011 §3](../d011_phase_value_audit_20260928/D011_RESULTS.md)已经证明共同Minkowski因子消去与ReLU及混权Affine交换。重新读到这条规律不记成新发现，不实现整个热带规范化流程。

系数空间包含证书V=WT+R、每行绝对和不超过1，也没有新逻辑资格。[Sadraddini与Tedrake 2019](https://groups.csail.mit.edu/robotics-center/public_papers/Sadraddini19a.pdf)的Corollary 4已有线性映射型zonotope包含充分条件。这里只是比较系数几何，不曾将原HZ状态换成zonotope；论文的优化或枚举流程未移植。

更直接地，[D007 §6](../d007_conservation_bundles_20260928/D007_DOMAIN_AND_RESULTS.md)对相同T和逐项残余界，已经给出更强的三角关系，求和即可推出所提包含行。若共同残余保留仿射依赖后更紧，收益应归于共同残余证书，不能归于containment的新名称。这一审查促成了本轮固定equal-share规则先合并共同源secant、再求界的选择。

[TropNNC](https://arxiv.org/pdf/2409.03945)研究热带几何与网络压缩，使用zonotope Hausdorff距离联系函数误差。它是相关先行工作，不提供对本项目原网络的精确替换证书。本轮只核其相关表述，没有采用压缩网络、训练或采样结果。

## 新近多神经元先例的边界

另核Shmuel与Katz的[Neural Network Verification using Partial Multi-Neuron Relaxation](https://arxiv.org/pdf/2605.30155)，2026预印本，§3 Theorem 1和§4.2。其框架把多神经元关系反馈给后续界收紧，说明“加关系再前向使用”本身已有直接先例。具体BHSO生成器采用splitting与optimization，本文不移植；也不以其论文成绩推断本项目收益。

这不是完整新颖性检索。当前只能把可能贡献定位到固定、可认证、无搜索的神经结构算子及其完整成本，而不能声称第一个多门关系域。D049的一般残余式仍是最直接的项目内已有性对照。

## 真实结构可复用到哪里

根代理只读D025的[complete_0.json](../../results/d025_interval_capacity_20260930_v1/complete_0.json)，SHA256为fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0。其实际source为CIFAR100_resnet_large.onnx，模型hash为5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16；frame_identity为同一hash与modelInput。

已保存result.source_forms共1600项，result.windows为五个固定窗口，result.receivers只有原分支0；该分支有完整64通道、每通道576槽的BN后区间系数与偏置。只读审查者核对固定receiver0前四项严格正、slot13严格负；后者和共同源结构也见[D054记录](../d054_shared_phase_cycles_20260930/RECORD.md)。这些是多门混权确实存在的证据，不是选择仅这些门的实验授权，不证明新规则有用或原门均crossing。

完整可用片段是原共同输入到first ReLU bank，再到Conv3、BN4和Relu5。first bank的同源identity Add8侧消费者仍记账；Add另一臂Conv6/BN7和合流后的Conv9/BN10尚无本包完整参数。不能把纸面两层传递说成已运行真实残差块。

Medium的旧D015包保存相应128乘576主分支以及shortcut参数，但整体失败资格不变；Tiny完整新源证据未齐。native原相位列、真实frame生命周期及decoder认证也不是这些保存字段能够代替的。没有重新解码模型、重新求界或运行新的source census。

## 独立审查与执行状态

三项只读子任务分别完成固定联合关系及强对照、支持函数与包含证书的已有性检查、真实多门参数范围核对。根代理独立推导equal-share公式、区间误差版本和成本，复核纸面算式及保存JSON身份。

审查纠正了一个重要比较错误：早期解析点通过D020与D052，但不通过D049的一条固定差式，因此未用作最终严格增强证据。最终点在THEORY中给出完整单门凸组合、全部六对D020、两种源顺序D049、两向D052以及下一原门的检查。纸面修改不是运行后的调参；本轮没有数值执行或冻结后重跑。

所有新内容仅写当前隔离目录。旧生产dirty changes、模型、历史日志、所有冻结源码和结果均保留，不commit/push。没有新代码、候选导入、pytest、LP/MILP、GPU、shadow、家族或全量回放，也没有新的依赖安装。

formal_gain=0；正式基线1870/2413=1063 CERT+807 validated ADV，独立E0为CIFAR100 25、TinyImageNet36，共61/400。二者不相加，未声称本候选已保旧或新增解。Goal保持active；创新域、GPU、真实能力和满分目标均未完成。

使用pages:write-page将已知数学、候选变换、解析控制、真实参数范围及未获资格分开存档，读回本地文本，不发布外部Page。校验清单只绑定本轮文件与实际使用的旧来源，不为旧失败运行重授资格。
