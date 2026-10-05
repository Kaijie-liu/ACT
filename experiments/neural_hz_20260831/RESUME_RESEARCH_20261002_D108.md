# 中心化 Neural-HZ 关系研究续接

最新工作是D108：从同源smooth门控语义推出可直接约束已有输出的中心化联合行。[定义与lowering](definition_first_20260928/d108_centered_attention_relations_20261002/DEFINITION.md)和[严格控制族](definition_first_20260928/d108_centered_attention_relations_20261002/CONTROL.md)已保存；不是存储优化，也尚未成为已认证的新抽象域。

## 当前目标和不变边界

完整更新后的Goal见[本轮直接保存的快照](definition_first_20260928/d108_centered_attention_relations_20261002/GOAL_SNAPSHOT.json)。本轮用户新增smooth突破及超过“configure selective”的要求；该比较对象尚待名称澄清，不影响继续研究，也不能虚报已超过。旧[定义优先章程](GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md)保持只读；新快照补充最新范围，不取消原八条硬限制。

必须从HZ数学定义研发非凸Neural-HZ，保留连续因子、原bits、EQ/LE、shared latent/frame及decoder；禁用退化Z/CZ、实例或LP状态菜单、attack/PGD、BaB、split、backward/dual rescue。GPU、smooth/Transformer、CIFAR100/TinyImageNet及全13家族目标均继续。没有完整同路径回放、逐旧解和逐家族保全、新增有效解及原能力门，不更新成绩或默认启用。

正式1870/2413、独立CIFAR10025＋TinyImageNet36共61/400不变；本轮formal_gain=0。Goal active，整体未完成。历史模型和冻结档案不动，新工作只在本实验根下新隔离路径。

## 本轮实际新增证据

- 中心化公式把匹配缺陷乘概率变化，认证余项维持O(δ²)，避免未中心化粗界的O(δ)损失。特定读出可不新建p或p×源坐标；一般情况必须支付固定权重Softmax下界或完整概率接口。
- 一个非均匀概率中心、非零匹配残余的三token双通道族，通过完整所列McCormick、列质量、精确概率图、tight坐标Taylor余项和真实输出像，却被新行排除；还能传给真正跨零的原ReLU。但已知同方向曲率关系明确给出更强的F≤0和后继r≤η，因此本控制不是胜过强参考的新颖性证据。
- 两个真实ViT首层确有每模型3B个恒定query组，可作为仿射score子类。但现有fused已经利用恒定Q减少乘积误差，不能把这点算新能力。末端平均全部tokens，不是CLS-only。
- 单调gate的乘幅度引理适用于erf-GELU、SiLU及保留原bit的ReLU；尚无GELU/SiLU真实资格或新颖性结论，不能将tanh近似与erf版本混同。

完整源码接点、费用及下一步见[来源和成本](definition_first_20260928/d108_centered_attention_relations_20261002/SOURCE_AND_COST.md)；先行研究及D106节号/常数来源补充见[文献记录](definition_first_20260928/d108_centered_attention_relations_20261002/PRIOR_ART.md)。这些修正只写新档，不改旧研究当时记录。

## 下一步及不能恢复的旧计划

先确定一条统一前向方向生成与概率消去规则，比较同方向Taylor强参考并完整核算残余。之后才预注册读取真实源参数的最小资格检查；不能仅靠图级“CLS常量”直接授予native来源资格，也不能以1e-12判点容差代替严格来源证明。不恢复D099费用否决路径，不重跑已消费D107，不通过多配置挑选结果。

原档案导航见[D107总入口](RESUME_RESEARCH_20261002_D107.md)，包含旧构造支撑库、文献综合、D103–D106的证明与负结论。最新已执行数学组件仍是D098的3845项/188文件；D107仅两模型只读图诊断。D108没有候选代码、freeze/RUN、测试、模型/solver/GPU执行或后台实验。

本轮使用文档整理技能区分新推导、已知原语、真实图事实和未认证事项。分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新增文件用SHA256SUMS绑定，不改生产，不commit/push；仍为本机归档，非异机备份。
