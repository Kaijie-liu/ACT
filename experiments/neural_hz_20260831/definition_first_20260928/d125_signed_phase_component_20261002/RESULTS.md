# 原相位条件幅值外包通过完整数学组件测试

默认关闭的非凸相位条件外包已从纸面定理推进到实现与数学组件资格。唯一运行完整通过 3881 项测试、196 个文件，没有失败、错误或跳过。这不是完整 Neural HZ 新域、新颖性、真实网络能力或 GPU 资格；正式及独立外部新增解仍为零。

## 执行证据

[freeze.json](freeze.json) 在候选导入、AST、收集或执行前固定八个新文件。唯一运行目录为 [本轮证据](../../results/d125_signed_phase_component_20261002_v1)，命令为该目录对应源码的 `run_math.py --enabled`。执行会话 23670 已终态退出 0，不重跑或修改冻结版本。

[退出回执](../../results/d125_signed_phase_component_20261002_v1/exit.json) 记录 component_tests_passed 和 mathematical_component_gate_passed 为 true。完整继承此前 3869 项并添加 12 项，全部在单一 pytest 进程完成；启动、收集、执行及 JUnit 合计 49.79337399452925 秒，未超过原 60 秒门。pytest 自报 48.53 秒，监督器总计 69.79470884427428 秒，包含前后只读身份认证。CPU1、单线程、CUDA 空、AS16GiB 均保持原合同；这不是性能对比试验。

JUnit 为 3881 tests、0 errors、0 failures、0 skipped。13 条 warning 来自继承的 TypedStorage 与 record_property 控制，没有把它们改成新算法失败或新收益。运行前 inventory 认证全量有序人口，结束后 source/input drift 均为空、provenance_drift 为 false。退出回执所列 11 个工件再次逐一通过 SHA256 校验。

## 实际验证的数学内容

实现从完整 target.g 提取父幅值权重与剩余仿射项，使用同源证书形成原相位条件误差预算，真正删除发出系统的父连续幅值列；全部原 signed bits、未替换 EQ/LE、source/frame 身份、可见读出和必要变量界通过重映射或源约束保存。这是语义层面的 sound 抽象变换，而不是仅压缩矩阵。

普通有偏置混权正控通过了以下精确有理断言：E_plus=7/60、E_minus=11/120；旧 LP 存在 J=16633/13440 的明确可行点，新系统有 J=29/24 的达到点。前向仿射界不调用新 LP，得到 J<=2899/2390，与下一 ReLU 的阈值 49/40 相差 23/1912，足以证明该数学小例下一门恒零。旧 LP 点与新整数伪点仅用于诊断，不是网络 ADV。

该正控的发出系统连续列从 5 变为 3，原二元列始终为 3；谓词行从 12 变为 8，nnz 从 34 变为 23。Projection.original 仍完整保留旧 System 供 decoder 检查，receipt 分开记录新旧条目与合计；不能由这些局部数字宣称净内存、完整物理成本或运行速度收益。

零误差控制保留三个零门的全部八种原相位标签。新增的作用域反例确认：即使 rho=0，旧目标门下界若额外限制源，目标关系等价也不自动等于全系统投影等价；实现只在另外证明重构目标范围时标注后者。非零误差外包允许整数伪点，decoder 会检查全部旧关系并拒绝它。数学 decoder 通过本身仍不是具体网络验证。

共同乘子组件通过普通内部最优 kappa=2、无穷远未达到下确界、有限常值段见证、零正权以及非对称源箱测试。它只优化共同乘子的 E_plus 子族，不保证 E_minus、最终输出或总成本最优，也不把无穷远当作有限证书。

## 仍缺少的研究结果

PWA 候选仍可由原 HZ 式 System 承载，不能凭实现完成就宣布新域定义或 PLDI 级新颖性。旧 HZ 加相同有效证书是更强或同等精度的必须比较对象。完整域定义、关系查询算法与可复现实际收益仍是主目标；Attention 非线性生成元方向保留在此前研究中，未被本轮测试覆盖或取消。

本轮没有新 source worker、实际模型/相位列绑定、native HZ 接入、GPU 计算、shadow 或完整 2413/400 回放。旧三模型来源资格只读继承，不转授给新候选。已核查真实残差消费者使单目标局部删列不能直接安装到这些网络；下一步必须处理所有受影响消费者，而不只是重跑局部窗口。详见 [真实结构接入边界](NEXT_REAL_STRUCTURE.md)。

正式成绩保持 1870/2413（1063 CERT、807 validated ADV）；独立 E0 保持 CIFAR100 25、TinyImageNet 36，共 61/400，两边新增均为 0。旧账本未变不等于本候选已经保住全部旧解。默认关闭，Goal active，完整普通 PWA/CNN、smooth/Transformer、GPU、13 家族和外部家族目标未缩减。

## 归档与身份

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

freeze SHA256 为 c83704e08e59b7a315572862ec3797073ed4a903d085ee6a229150424f2f67f4；exit SHA256 为 7d4ada01bd9902db9488f15327d84908fc9d503e4452fe5365cefbbe5b84efe3。历史源码、模型、结果和生产默认未改，无 commit/push。上一用户回合仅状态核查为 no progress；本回合完成实现冻结和实际完整数学测试，为 progress，但远非 Goal 完成。

pages:write-page 用于分开数学证据、局部成本、真实接入缺口和正式成绩并保存到既定本地目录；没有创建外部 Page。伴随归档清单在文档读回后生成，不冒充测试前冻结。
