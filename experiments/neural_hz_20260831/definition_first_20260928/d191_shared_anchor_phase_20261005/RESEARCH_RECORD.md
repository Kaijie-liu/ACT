# Neural-HZ 共享相位候选的强对照筛选

用户再次明确重点是提出强大的 Neural-HZ。本轮直接修改候选域的相位—源—幅值具体化关系，未扩展存储或构造工程，也未用 helper、attack、split、backward/dual rescue 冒充域收益。

[定义与证明](DEFINITION.md)给出共享一个原门相位、全部原相位身份和完整消费者的非负误差预算。[同结构对照](CONTROLS_AND_COST.md)在满行秩、非平行、六十门的普通有偏置包上证明：候选强于简单逐门 LP，但仍允许严格内部混合相位伪输出；已有有序 ReLU 聚合关系排除这个伪点，并以更便宜的有限系统证明相关后继性质。

因此本轮结论不是“强大 Neural-HZ 已完成”，而是：一个新的具体化候选通过出生健全性推导，却未通过强对照的研究价值门，不应进入大规模实现。正例、负例和真实网络成绩严格分开。候选没有使 HZ 退化为凸域，但保留 bits 也不自动保证有用的相位幅值关系。

下一定义研究仍须处理原相位与实际共享源幅值的可查询关联，并证明它在完整消费者和后续非线性中，提供已知聚合/sector 行无法廉价替代的能力。不能通过增加 helper、忽略完整消费者、恢复整张逐门图却不计成本，或把构造提速称为定义创新来回避此问题。侧向精确惩罚与因子图检查见 [SIDE_CHECKS.md](SIDE_CHECKS.md)。

## provenance 与执行边界

日期 2026-10-05 Australia/Sydney；本轮记录前观测 UTC 为 2026-10-04 15:45:08。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置 paper_only、definition_first、shared_original_anchor、default_off、no_candidate_execution。依赖为 ANCHORS.sha256 中只读研究文件和明确列出的外部摘要，无新增模型或数据依赖。

三位协作者分别核查共享相位健全性/helper 边界、同结构强对照与费用、真实卷积因子图条件；主代理核对公式、来源与归档。这是纸面推导及静态读取，不是机器证明或数值运行。

本轮仅新增此隔离目录的文档与哈希清单。无候选 import、AST、compile、collection、测试、模型 forward、LP/MILP、GPU 执行或新 results 目录；无 commit/push；未修改默认路径、生产代码、旧研究或历史结果。tracked binary diff SHA256 保持 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。未检查既有后台进程，不对其运行状态作声明。

最近成功的数学资格仍为 D180 的 4056 项、213 文件；没有缩减或新增执行人口。新实现仍须先冻结并经过原门。文档归档技能用于区分推导、对照、取舍和未完成资格，只保存本地文档；读回与哈希证明文件字节保存，不代表数学、渲染或能力认证。

正式 baseline 仍为 1870/2413（1063 CERT+807 validated ADV）；独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400。两套成绩不相加，本轮新增均为 0。逐家族保全、实际新能力、全路径 2413/400 回放、GPU 和 smooth/Transformer 目标均未取得新的资格。Goal 保持 active，未完成；本次没有外部权限阻塞，也没有降低任何验证或晋级门槛。
