# 相位中心 Neural-HZ 研究续接

主线仍是从 HZ 定义研发强大的非凸 Neural-HZ，不是算子存储优化、旧完整图外挂 helper，或新增求解算法。Goal active、未完成。最新[理论](definition_first_20260928/d148_phase_centered_norm_fiber_20261004/THEORY.md)与[控制及限制](definition_first_20260928/d148_phase_centered_norm_fiber_20261004/CONTROL.md)属于纸面进展，没有正式新解。

## 当前值得保留的定义

域读出 y=c+C xi+G beta+sum E_k e_k，保留同一源、原 bits、EQ/LE、全部共同范数块与 decoder。g=Wy+d、固定 tau、gbar=W(c+G tau)+d，新门采用

```text
q=(1/2)g+(diag(gamma)-(1/2)I)gbar+e_new
||e_new||2 <= (1/2)B(beta)
B=K0+sum Kk Rk(beta)+sum d_j |beta_j-tau_j|.
```

真实余项为 (1/2)diag(2gamma-1)(g-gbar)，故预算健全。新 G_gamma=diag(gbar)，旧 C/G/E 全部左乘 W/2；不出现相位乘积，但块数与列数增长。新门精确 active-value 关系被替换；若又保留完整门图，则仍只是 helper。

同一父状态、同一预算和相同原约束下，新整数球在父坐标、输出、原 bits 上包含于上一版。不能将两版新 e 视为相同坐标，也不能扩张为分数 LP 或独立整网递推支配。理论里已保存可靠同源二维分数反例。

## 已区分的普通结构

16 维 A=I-11^T/8、b=-1/2、源盒，可靠 L=-13/4、U=9/4。新球和原 mask 给 Q+(9/22)sum x<=72/11。带一个负输出权和同源 skip 的 J=sum前15 q-q16/4+(9/22)sum x 满足 J<33/5，下一 ReLU 恒零。

旧 fractional 点 x0、gamma1/2、q9/20 给 J531/80>33/5；全部至多三门完整 source hull、指定原 bank 的全部 D009 weighted Gram 和未重中心化 weighted perspective 族都允许，任何 gamma 延拓却都不能进入新域。另有整数点 x=-3/4、gamma1、q4/5 在上一版球内、新球外，说明不是仅查询器区别。

强比较边界必须同时保留：同信息 centered individual perspective-energy 已支配新球，精确 HZ 也不允许假幅值。这个正例不是超越它们的证明，不是实际 CNN/ViT 成绩，不是发现新反射原理。早期 8 维 Hadamard 点仅有标签分离，已排除出能力证据。

## 下一步和执行限制

下一步是冻结统一参考、可靠半径和有限终端查询规则，研究真实同结构层的界质量和完整代价；不用扩大解析控制或恢复全图再附加行替代定义价值。按原要求先预注册并冻结候选，再执行，不继承旧组件资格，也不缩减人口或抬高预算。

本轮无实现、import/AST/compile、测试、数值搜索、模型、solver、GPU、shadow、replay 或后台作业。最新实际组件执行仍是 D136：3965 tests / 208 files，startup+pytest+JUnit 50.532985638827085s，原60s墙钟门、CPU1/thread1、AS16GiB、CUDA隐藏；本轮不重跑或改这些门。

正式 1870/2413=1063 CERT+807 validated ADV，独立 E0 CIFAR10025+TinyImageNet36=61/400，新增均为0。任何默认启用或记分仍须完整2413及独立400的同路径保旧增益。历史和生产只读；只写本轮新的隔离归档，未 commit/push。

分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。[STATUS.json](definition_first_20260928/d148_phase_centered_norm_fiber_20261004/STATUS.json)及其目录内 ARCHIVE_SHA256SUMS 为本次续接记录；旧归档不覆盖。归档技能仅用于本地证明与能力边界的区分，未创建外部 Page。
