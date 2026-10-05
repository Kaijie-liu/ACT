# Neural-HZ 同源误差修复的筛选记录

上一目标轮为 progress：D192 完整相位误差像的证明与三门源绑定反例改变了下一研究动作。本轮继续从具体化关系出发，检查精确同源半径、完整源区间投影和共同图流见证。没有转为 loader、存储或外部 helper 工程。

[SOURCE_BINDING.md](SOURCE_BINDING.md)给出严格全父盒后继反例：即使误差半径就是精确的同源仿射距离，候选仍允许伪输出，已有一条 sector 行已能排除。[PROJECTION_BOUNDARY.md](PROJECTION_BOUNDARY.md)进一步证明，低源维数和两个完整输出不能保证旧四行 LP 的精确幅值消去便宜，并给出允许少量辅助量时的限定费用权衡。

这些证据使下一动作发生变化：不实施“把 D192 的 R 换成 r(theta)”或“直接把所有源约束投影进少数输出”作为通用强域。下一候选需要利用整个父关系对相位与源的共同约束，或明确接受并控制一种新的损失，而非仅收紧距离或隐藏展开成本。这不是新增硬限制，不要求所有新域保全某个旧 LP，也不放弃原定义创新与实际覆盖率目标。

## 共同图像路线的范围

另一只读探索要求完整消费者有认证分解 C=H*B*diag(w)，B 为固定图 incidence、w 为正权。在同一 q 上先合并源、相位和误差条件，得到 ell_i(theta)<=q_i<=u_i(theta)，再保留节点平衡 z=B*diag(w)*q、实际输出 Y=H*z，能保证各消费者使用同一个见证。不能先独立投影两个集合再相交，也不能在 H 非单射时未经投影就把 z 的全部割行当作 Y 的行。

四环的普通局部例：0<=q_i<=1，z_i=q_(i-1)-q_i。独立坐标界及 sum z_i=0 允许 z=(3/4,3/4,-3/4,-3/4)，共同见证却要求 z_1+z_2=q_4-q_2<=1。这只证明联合关系比这些边际摘要强，不是神经网络或强旧 HZ 分离。

连通图 m 条边、n 个节点有 m-n+1 个循环自由度；保留它们及 n-1 个独立节点平衡仍总计 m 个幅值。树的关系简单但 m=n-1，没有此维数节省；平行边可合并，但 source-dependent min/max 容量仍需支付逐边费用。任意 C 的星图表示也不自动压缩。没有真实 CIFAR/Tiny 的小图完整分解或强成本优势证据，因此不实现 flow 求解器或新的组件。

这些机制已有 D011 tropical 同源电路、D025 容量电路、D089 运输投影、D115 共同见证先例。这里只保留正确的“先共用幅值、再投影”边界，不重新计作新颖性。

## 文献定位

本轮核读 [Anderson 等的原文](https://optimization-online.org/wp-content/uploads/2018/11/6911.pdf)第 5.2 节 Proposition 13–14：其单 ReLU 的非扩展理想 formulation 可以有指数多个 facets，并给动态分离方法。本文的两个输出投影例并非该文定理的原样应用，但“少变量可能换来许多行”不是新发现。没有导入其动态 cut 回调或据其成绩申报本项目收益。

这里的源仿射支撑、min/max 投影和正依赖计数都是已知工具上的项目内推导；未确立 PLDI 级新颖性。纸面证明与原始论文阅读不等于候选执行或真实能力资格。

## provenance 与执行边界

2026-10-05 Australia/Sydney；记录前观测 UTC 为 2026-10-04 16:12:59。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置 paper_only、definition_first、source_binding_projection、default_off、no_candidate_execution。依赖为 ANCHORS.sha256 内的只读档案及上述原文，没有新模型或数据依赖。

三位协作者分别复核精确 LP 投影与 facet 证明、同源半径反例与扩展式计数、图像路线和旧先例；主代理提出二维族并核对来源、公式及归档。这是独立纸面复核，不称机器证明。

本轮仅新增本隔离目录的文档与哈希。无候选 import、AST、compile、收集、数值测试、模型 forward、LP/MILP、GPU、shadow、replay、新 results 或后台模型运行。未检查既有后台进程。生产 tracked binary diff 保持 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；历史模型、源和结果未改，无 commit/push。文档技能用于分开证明、反例、已有原理、费用与未完成资格；文本读回及哈希不证明渲染或数值正确性。

最近成功数学人口仍 D180 的 4056 项、213 文件，未缩减或重跑。正式仍 1870/2413=1063 CERT+807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400；本轮两边新增均 0，不相加。真实网络能力、GPU、smooth/Transformer、逐家族保全和全路径回放没有新资格。完整 Goal 保持 active，未完成，当前数学问题不构成需用户改变外部条件的阻塞。
