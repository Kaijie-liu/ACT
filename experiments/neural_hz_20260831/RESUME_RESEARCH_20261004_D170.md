# Neural-HZ 共同盒能量见证研究恢复入口

继续完整定义优先 Goal，不以 helper、存储优化或小例取代强 Neural-HZ。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。上一轮D169是progress，本轮也得到改变下一行动的定义证据；Goal保持active。

本轮[共同纤维定义与证明](definition_first_20260928/d170_capped_joint_fiber_20261004/CAPPED_COMMON_FIBER.md)、[替代定义边界](definition_first_20260928/d170_capped_joint_fiber_20261004/ALTERNATIVE_BOUNDARIES.md)和[记录](definition_first_20260928/d170_capped_joint_fiber_20261004/RESEARCH_RECORD.md)均已保存。

- 正向：盒约束与能量约束绑定同一个proxy，严格强于分别投影再求交。三门控制中旧见证最小能量24/25；共同盒内最小值57/50超过预算101/100。固定capped-square行可证明整个输入盒上的后继ReLU恒零，而旧native允许假幅值。
- 限制：相同信息下的强旧参照给出更紧证书149/140，甚至利用该源的特殊恒等式可到17/20。因此不能把正控记成新验证能力；暂不以它启动组件实现。
- 可复用定理：rank1有精确断点扫描成员检查；一般维数不是免费Gram消元。固定每轴六行可统一替换旧energy RHS，行数不增，但不能当作完整native查询。
- 已排除的直接路线：精确盒投影仍丢ker(C)里的源绑定；每层是凸势梯度不保证复合仍是势梯度；普通混权残差可不属于任何固定SPD度量下的势/凸prox；任意正权各层激活平方加光滑源正则也可能不凸。

下一研究不能只重复换球、换盒、增加势函数名称或恢复全部门图。需要保留真正跨层共同见证与非保守混权依赖，给出完整普通消费者上的非冗余能力或已支付的查询/表示优势。此处不要求不可能的“有损域集合比精确真图更紧”，比较的是实际表示与查询在同信息下的能力和完整成本。当前未选出满足该项的新实现候选。

无新代码/候选执行/模型/GPU/shadow/全量回放。最后组件资格仍D158的4032 tests/212 files。正式1870/2413=1063 CERT+807 validated ADV；独立E0 CIFAR100 25、TinyImageNet 36，共61/400；均零新增，不相加。本轮不能宣称实测保旧或升级；全部默认关闭、预注册冻结、完整人口、逐结构、shadow、2413/400及四并发不回退要求不变。

旧档、历史模型和结果、现存生产修改均未动；只新增隔离文档。tracked diff SHA256为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。未commit/push、未启动新后台任务。新目录ANCHOR_SOURCE.sha256和ARCHIVE.sha256从仓库根目录校验。所有后续执行仍需新的完整预注册与源码冻结。
