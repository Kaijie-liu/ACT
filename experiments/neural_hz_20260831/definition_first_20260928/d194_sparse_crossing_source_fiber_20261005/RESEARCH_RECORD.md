# Neural-HZ 稀疏跨零共同源的研究记录

上一目标轮为 progress：D193 的同源半径反例及限定投影费用证明改变了下一动作。本轮再次核对当前 worktree、目标正文和 D193 档案，继续定义研究，没有转向存储优化、loader 或 helper 工程。

本轮最重要的增量是一个真实且范围明确的研究对象：[稀疏跨零共同源块](REAL_TARGET.md)。已有 medium 工件中的四个相邻门共享原输入并有活 shortcut；几何分解为 27 个共享坐标加四个精确私有仿射读出。原全局 latent 不消失，完整消费者不得遗漏。

[条件纤维定理](CONDITIONAL_FIBER.md)证明了固定同一共同源时、局部私有标量投影上的理想带相位凸包及乘积胶合。它不是全模型胶合定理；其廉价线性化恰为已知六行分区关系，不能拿来宣称新域。因此下一研究动作是针对这个普通小 crossing 接口寻找可组合的联合查询和精度收益，不再要求先压缩宽幅值接口，也不把局部别名组织当成定义突破。

[补充核查](SIDE_CHECKS.md)记录 anchor 实际幅值对误差界的作用，以及“仅复制源”的文献构造反例。前者补充源绑定所需信息，尚未解决共同乘积费用；后者限定一个不能借用的证明步骤。FNE、加权互补能量及旧 D173/D174 候选的排重未发现可重新晋级的新路线，原负结论保留。

## 来源与本轮实际操作

日期 2026-10-05 Australia/Sydney；记录前 clock 为 2026-10-04 16:29:33 UTC。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置 paper_only、definition_first、existing_artifact_readonly、default_off、no_candidate_execution。依赖由本目录 ANCHORS.sha256 锚定，外部原文链接就近保存。

只读 shell/JQ 检查了旧 JSON 字段、已有有理端点的符号及结构人口；没有除法、重算神经界、解码新权重、运行模型或产生新诊断结果。一次直接 sed 读取单行大型 JSON 被输出截断，后改用定向 JQ 字段读取；不将截断内容称为完整阅读。一次猜测 D150 路径失败，随后通过 rg 找到实际 REAL_NEXT.md 并读取。三位协作者分别检查候选关系、真实来源/几何、条件凸包/廉价参照；没有协作者写文件或执行候选。

无 import、AST、compile、pytest collection、数学数值测试、LP/MILP、GPU、shadow、replay 或后台模型作业启动。未检查已有后台进程。最近成功数学人口仍为 D180 的 4056 项、213 文件，未减少或重跑。任何后续数值执行仍需新隔离候选、先冻结、完整继承人口及原资源门，不能把本轮数学选题当作已冻结的执行合同。

新写入仅本目录的文档和哈希。历史数据与冻结代码只读；tracked binary diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，生产默认未改，无 commit/push。文档技能用于区分真实来源、数学推导、已知机制及待证事项；文本读回与哈希不证明渲染或数学机器认证。

## 资格和目标

没有新颖性认证、真实网络精度/速度、GPU、smooth/Transformer、完整物理资格或新增 CERT/validated ADV。正式仍 1870/2413=1063 CERT+807 validated ADV；独立 E0 仍 CIFAR10025、TinyImageNet36，共61/400，两边新增0且不相加。

保持全部原二元非凸相位、连续源、EQ/LE、共享身份、decoder 和 fail-closed；不使用 attack、PGD、BaB、split、backward/dual rescue、状态修复或身份菜单。13家族逐例保全、同路径完整2413回放、独立400回放、四并发不回退及更广家族目标未缩减。Goal active，未完成，也没有需要用户改变外部条件才能继续的阻塞。
