# 私有容量消元研究记录

本轮取得限定的正向结构定理，但尚未完成强 Neural-HZ：在可靠私有阈值覆盖条件下，可把完整联合神经查询中的私有条件幅值消掉，保留稀疏共同源与相位关系；给定合适的相位 bag 树时，还可避免物化全局相位表。详见 [定义与证明](THEORY.md)。

与上一轮不同，这次不是一阶纤维的重复排重；它显式保留多相位共同见证，并证明何时神经守卫可以由私有容量同步满足。[物理正控](CONTROL_AND_SCOPE.md)将旧 D175 的全部两门 hull 分离延拓到该条件类，且后继 ReLU 仍能消费。这个正控不是新不等式或新验证成绩。

另给有回路的周期局部带：在 COVER 下可用三元相位 bags 构成线性规模的一层联合 hull，且满秩、非零偏置的四环控制严格区分 singleton hull。主进程将协作者的边界见证改为严格内部源与 private 见证并手核。这避免把本来 singleton 已理想的树当成新精度正控；但同信息绝对值预算也能解决这个单项性质，不能报告算法胜利。

[公平费用](COST_AND_NEXT.md)在同 COVER 条件的四门 scalar-private 投影上给出448到160个辅助量、896到320条 LE，EQ仍32；有明确 nnz 账和更廉价局部旧参照边界。这是纸面扩展系统下降，不是全网络速度或存储实测。原像素条件关系、深层闭包、实际适用率、全 GPU、smooth/Transformer 和完整回放均未取得资格。

## 研究来源和进展分类

上一轮 D198 为 progress：其一般源和完整系数体包含证明排除了单纯一阶聚合继续增加查询精度的路线；本轮先重新读取其记录并验证归档哈希。首次校验误在子目录运行，因清单路径以仓库为基准而未找到文件；改在仓库根重验全部成功，未改任何历史文件。

当前 worktree、分支、HEAD 和 tracked binary diff 与既有记录一致。祖先及当前树未找到适用的 AGENTS.md；广泛 rg 只作定位，未将截断输出当作完整阅读。主进程读取目标权威正文，以及 D007、D035、D068、D175及完整pair见证、D180合同、D191相关边界、D194至D198相关定理与记录。一次 D176 THEORY 路径不存在，随后仅定位其实际文件，未声称读完 D176。

只读协作者提出并推导 COVER 下四容量胶合；主进程独立核对必要性、充分性、整数回落、零容量、原像限制，构造 D175 私有扩展正控及没有 COVER 的物理反控，推导精确局部 nnz。另外两名协作者复核公式、同信息比较和计数。另行提出的 abs/conservation、ray contraction 与简单双门能量方向已有旧直接先例，不另起候选或重记成果。

外部主进程核读 Wainwright与Jordan 2.5.2 Proposition1及相关树边缘段、Anderson 2.2至2.3节、Tsay摘要及3.2节相关命题；只作为已知胶合和单门分组参照。部分全文入口失败，Tsay改用可读PDF；tropical PDF超时未纳入新结论。宽泛检索不是系统综述，没有依据搜索摘要宣称新颖性或采用外部算法。

文档归档技能用于把定理、正反控、完整费用及资格分开写入本地新目录；没有外部 Page、渲染或形式化机器证明。文本读回、独立纸面复核与哈希封存不等于数值过门。

## Provenance 和执行状态

日期2026-10-05 Australia/Sydney；本轮 UTC 读数包括2026-10-04 18:00:40、18:06:57、18:10:44。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置 paper_only、definition_first、joint_private_capacity、default_off、no_candidate_execution。来源锚见 ANCHORS.sha256。

本轮没有候选 import、AST、compile、collection、数值脚本、LP/MILP、模型、GPU、shadow或replay；没有新 results、后台实验或旧任务重跑，也未检查旧后台任务。最近成功数学人口仍为 D180 的4056项、213文件；没有缩减、重跑或扩大原门。未来执行须另立新隔离冻结版本。

仅新增本目录文档与哈希，历史/冻结源码/模型/结果、生产默认及既有脏改动均不变，无commit/push。起始 tracked binary diff SHA256为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

正式仍为1870/2413，即1063 CERT+807 validated ADV；独立E0仍为CIFAR100 25、TinyImageNet36，共61/400；新增均0，不相加。保全部连续源、原bits及双零标签、EQ/LE、共同身份、所有消费者、decoder和fail-closed，不以表格证明实施phase split，不使用attack/PGD/BaB/backward/dual rescue或身份菜单。普通终端与原网络具体见证验证边界不变。

下一动作转为认证这一数学条件的真实结构适用性及完整原像费用，而不是继续被D198否定的一阶系数纤维或实现另一个helper。目标仍active、未完成；本轮为改变下一动作的研究进展，不是能力晋级或PLDI创新资格。
