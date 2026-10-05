# 混权块研究的证据与执行范围

上一 Goal 轮属于 progress：新增跨领域复核及 QC 负对照，改变了下一步的比较范围。本轮先核对五项已存校验全部一致，分支 redu-hz、HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。没有等待或重启任何已结束实验。

本轮完成两项可复核数学结果：[增量接口的精确凸包与限制](SECTOR_LIMIT.md)，以及[固定四平面支持和完整两门凸包交的严格分离](THEORY.md)。三个只读子任务分别审查 sector 凸包、共同源联合对照、已有实现接口；根代理提出新三门控制、固定支持构造与共享底式成本，独立核算并合并审查意见。

已有 D061 四门控制被实际共同源的已知缩放 pair 行排除，所以不能作为超过完整 pair hull 的证据。新的三门控制明确给出全部三对的共同源带标签凸组合，原门预激活非零且源点严格内部；真实上界11/10，旧假点63/40。下一原 ReLU 真上界3/5，旧假点43/40。这些都是纸面解析数，不是运行结果。

新规则仍属于已知 max-affine、ReLU下支撑及条件观察的构造性组合；没有声称新域已经成立或完成 PLDI 新颖性审查。它形成了比上一控制更强的数学比较，并明确了一个无需免费 oracle 的统一前向生成器。没有要求达到全局 ideal，也没有修改既定晋级门。

只读实现审查确认 D049 的 owned frame/value/phase、form、readout、observe 和 compile_rows，以及 D053 的 interval affine，可支撑未来默认关闭组件。不能把旧 positive_majorant 当成任意新公式的等价替代。D057 保存的数学人口为3797 tests/176 files、数学通过，但整体 source census 失败；不得继承其未获得的真实资格或重跑该版本。

本轮没有执行候选导入、pytest、模型解析、LP/MILP、GPU、shadow、家族或完整回放，没有新依赖。未创建数值预注册或运行目录；后续实现须独立预注册并遵守原人口、512位、稀疏支持及完整资源规则。没有调大旧预算或缩小目标集合。

四平面规则可对应 GPU gather/reduction，但可靠舍入、区间参数、native 原相位/读出绑定、实际全物理存储及端到端速度均未认证。理论算术量不作 GPU 加速证据。

日期2026-10-01，配置 paper_only、primary_literature、read_archived_documents。只新建当前隔离目录，全部旧源码、档案、模型与既有 dirty changes 保持原样；不commit/push，不修改目标正文。

formal_gain=0。正式1870/2413=1063 CERT+807 validated ADV，独立E0为CIFAR100 25、TinyImageNet36，共61/400；两者不相加。没有证据宣称本候选完成保旧或新增解，Goal保持active。

使用pages:write-page分开记录定义接口、已知机制、项目推导、负结论和未验证收益；读回本地Markdown，不发布外部Page。新校验清单只固定本轮文件与直接依据，不重写失败记录或为旧候选补授资格。
