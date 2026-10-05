# 共享端点割线研究续接

D110 新增了任意共同 value 下的 Softmax 端点割线定义、精确传递证明、普通终端线性外包和完整成本缺口。不是已完成的 Neural-HZ 新颖性、GPU 或能力突破。完整目标快照和旧成果导航仍见 [D109 恢复入口](RESUME_RESEARCH_20261002_D109.md)；不修改旧档。

## 本轮可复用的证据

- [定义及反例](definition_first_20260928/d110_endpoint_secant_20261002/DEFINITION.md)：p−q=(diag(lambda)−lambda lambdaᵀ/S)(s−t)，同一组 lambda 可以传到任意共同 V，但不能当成任意方向的路径平均 Jacobian。
- [线性外包和成本](definition_first_20260928/d110_endpoint_secant_20261002/LOWERING_AND_COST.md)：共同 c 的范围、四 McCormick 行、NC 个动态 value 产品的账单；纸面日程在两个真实布局上分别需要 39168B 和 1920B 个 value 产品，未计其余完整成本。
- [原始文献与源码对照](definition_first_20260928/d110_endpoint_secant_20261002/PRIOR_ART_AND_SOURCE.md)：已有熵链式法则及任意 V 的有限 attention 响应；旧生产的 cross-radius 是单 query 概率与 value 交叉误差，不是跨真实 query，但共享仿射核心不能被忽略。

这些证据明确了此前 D109 一般 PV 不能替代加权 key 梯度之后，可以怎样精确建立新的连接，也否定了把共享端点矩阵当通用 Jacobian 的用法。尚无足够强的正控制；不启动高成本全乘积实现，也不以增加数学测试数量代替真实效用。

## 继续研究的边界

下一步检验共享端点关系是否在完整有限 Taylor／ratio／增量参考后仍提供真实 PV 和非稳定后继可用的新信息；同时核算全部源、乘积、终端和见证成本。若只能恢复已知相同行，应归类为表述强化而非新域。此具体假设可以被否定，不要求为坚持它寻找极端控制。

主目标保持定义优先、非凸 HZ、GPU、smooth／Transformer 及全家族净提升。原连续／二元因子、EQ/LE、共同输入、decoder、fail-closed 和禁用搜索边界不变。能力和纯速度晋级门不变；默认关闭、逐层资格及完整同路径回放要求不降低。

正式仍 1870/2413，独立 CIFAR100 25＋TinyImageNet 36 共 61/400；formal_gain=0。最新实际数学资格仍 D098 的 3845 项／188 文件，最新图诊断仍 D107。本轮无候选代码、freeze、RUN、测试、模型、solver 或 GPU 执行，无后台实验。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；原 tracked binary diff SHA256 仍为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。仅新增隔离文档，用 SHA256SUMS 绑定；不改生产／旧结果，不 commit／push，仍为本机归档。文档整理技能用于分开定义证明、已有原语、未实现合同和未完成能力。Goal active，整体未完成。
