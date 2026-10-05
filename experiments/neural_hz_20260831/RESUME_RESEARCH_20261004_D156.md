# 继续研究相位分区的共同源商纤维

最新 [理论](definition_first_20260928/d156_two_branch_consumer_fiber_20261004/THEORY.md)
与 [研究记录](definition_first_20260928/d156_two_branch_consumer_fiber_20261004/RESEARCH_RECORD.md)
已保存。主线仍是强大的非凸 Neural-HZ 定义，不是 helper/存储优化。正式
1870/2413 和独立61/400不变，formal_gain=0，goal active。

本轮有一个新的明确候选，不再只停留在“考虑直接输出”。对完整consumer载体C，
d=g-gbar，s=Cd，a=y-h-Cdiag(beta)gbar，T=Cdiag(beta)C^T，U=Cdiag(1-beta)C^T，
用 a∈range(T)、s-a∈range(U) 和 a^T T†a+(s-a)^T U†(s-a)<=E 替换新q。
E必须覆盖整个当前父域的||d||²，原sources/bits/guards/decoder均保留。

整数语义恰允许共同代理dhat∈d+kerC，||dhat||²<=E，a=Cdiag(beta)dhat。
因此不是保u+v=d的individual perspective精确投影。误差方向在active/inactive
consumer子空间交内，交为零则输出精确；all-on/off原生精确。Q=CC^T正定时
center=TQ^-1s，K=T-TQ^-1T，能量完成平方，rankK=rankT+rankU-p。

固定lambda/mu直接给线性row，有限row不自动保range。另4p坐标phase-cap行
可保终端all-on/off精确。正文给加权Cauchy的相位仿射前向预算，支持整个当前
扩大父域的递归；会保留旧banks并填充，不是固定宽度/精度不降定理。

D148旧16门控制可由C=[ones;e16]两幅度保完整mixed J与下一ReLU；一条能量
行加caps证明J<=72/11<33/5，旧逐门LP假点J=531/80。有效能量原理是旧的。
本块源内联计数41rows/683nnz，对旧+同energy65/656；没有总nnz或速度阳性。
普通p1m2mixedphase即使点态exact E也允许假幅度，且通过phase-refined caps；
不要称无损或已保旧。相同源后的后继ReLU会产生假正值，正文数字已复核。

下一步先冻结真正统一的有限方向/载体/预算规则，再考虑新默认off组件，完整
数学人口仍4000/210，然后真实同结构、shadow和全量门；不得凭正控跳级。
注意primary matrix perspective/parallel sum已有先例，新颖性未成立。

本轮paper-only，无执行或后台任务。分支redu-hz，HEAD
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5未变。
生产和历史只读，无commit/push。GPU、smooth/Transformer及真实全家族收益
仍是完整目标未完成的部分，不能由这项定义候选宣布完成。
