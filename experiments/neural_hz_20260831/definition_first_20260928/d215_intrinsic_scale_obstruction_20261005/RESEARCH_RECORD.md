# 内禀尺度研究的证据和归档边界

2026-10-05 Australia/Sydney，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。当前用户再次强调定义优先，本轮据此研究关系的跨层组合，而没有开始先前考虑的完整权重适用率普查、共享身份优化或新一轮小控制实现。

上一轮用户交互属于方向和状态核对，没有新执行结果，按 Goal 记账属于 no progress。本轮完成一个具体候选定义的纸面审查：最小内禀尺度的有效性、固定原相位下的线性表示障碍、cap 收紧的逐点反例、单尺度消去边界，以及普通后继对 max-affine 闭包的反例。它改变下一行动：不实现这个缺少闭包保证的候选。完整推导见 [数学记录](THEORY.md)。

## 执行和独立复核范围

配置为 paper_only_exploratory。本文是已完成推导的真实记录，不倒填为预注册实验。没有候选 source、freeze、RUN、候选导入、AST、compile、collection、数值脚本、模型/GPU/终端执行或后台工作。

独立只读复核分别检查了语义尺度与 cap 反例、原 bits 编码边界，以及一般 feedback 与跨层闭包。关于 D214 方向比较，最终明确的乘数 (1,(1-r)/2) 对应第一 feedback 与第二 amplitude 两行；若改用两条 feedback，乘数应为 ((4-r)/3,2*(1-r)/3)。数学记录使用前者，没有把两个不同组合混同。

数学资格仍是 D214 的 4116 项、217 文件；本轮没有执行这些检查，也没有把纸面复核加入其测试计数。该旧资格不自动转授给一个新域。当前无新 source binding、真实 CNN、GPU、smooth/Transformer 或完整物理成本资格。

## 来源和不变的成绩

正式基线仍为 1870/2413，包含 1063 CERT 和 807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400。新增正式解为 0。基线权威及独立账本出处见 [基线锁](../../BASELINE_LOCK.md)，本轮不修改其中的历史阶段描述，也不把更早 59/400 口径恢复为当前基线。

所有规则仍须保留原非凸相位、共享 latent、EQ/LE 和输入重构；没有 attack、PGD、BaB、split、外部 helper、LP 对偶修复或失败后救援。讨论附加 max 的表示代价不等于授权启用它。普通终端和具体见证验证的边界不变。

只有本目录的新文档被写入。历史源码、冻结证据、生产脏工作树和 /data1/Kane/HyZor 均未编辑；未 commit、push 或创建外部 Page。前后生产 tracked diff 的 SHA256 均应为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，来源校验见本目录 provenance.json 与 ARCHIVE.sha256。

归档技能用于区分纸面数学、既有组件资格和正式能力，不改变研究标准。没有网页渲染或远程发布声明。完整 Neural-HZ Goal 保持 active，远未达到完成条件。
