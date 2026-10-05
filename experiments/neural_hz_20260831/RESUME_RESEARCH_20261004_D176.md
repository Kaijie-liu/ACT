# Neural-HZ 联合关系层级研究恢复入口

完整 Goal 继续 active，定义优先、保原非凸因子/共同源/decoder、GPU、13家族、CIFAR/Tiny、smooth/Transformer 和最终同路径全量目标不变。正式1870/2413和独立61/400均新增0。

本轮入口：[一般宽度定理](definition_first_20260928/d176_joint_relation_hierarchy_20261004/WIDTH_THEOREM.md)、[静态跨层消费与正控](definition_first_20260928/d176_joint_relation_hierarchy_20261004/FORWARD_CONSUMPTION.md)、[研究记录与文献边界](definition_first_20260928/d176_joint_relation_hierarchy_20261004/RESEARCH_RECORD.md)。D175及所有旧档保持冻结原状。

已知 Imm22 的 K_m 可用 prefix-halving 逆变换。任意同源 g，认证 ell<=K^-1*g<=u，t=max宽度，得到一条 sum(q)-lambda*g<=(Kell+t e1)beta-c ell；c_j=m-j，lambda=(1-2^(-(m-1)),1-2^(-(m-1)),...,1/2)。物理系数<=3m，无m² products，但源展开、2m支持、位长、全部组和terminal成本不免费。

结构定理：g=K_m z-e1、zbox时，Bell所有整数等号点满足真实 NN guards；其物理投影面维3m-1，真实中点的F=-1/2使NN hull满3m维。该行不能由全部proper-gate hull共同交推出，甚至可在各投影保所有原beta均值。一般m>=4证明允许原合法零标签，未证明普通扰动/训练适用率；若各proper witness还要共享指定高阶phase-slot质量，维数证明不自动覆盖。D175 m3已有明确非零偏置内部证据。

静态forward规则：aθ<=b，目标h=c+vθ，lambda取正同号比值v_j/a_j的最小值（空则0），U=c+lambda*b+BoxUpper(v-lambda*a)。保旧界，low对-h；不优化对偶/读LP状态，先按sharedframe合并。数学上是显式非负行证书，不能声称与对偶理论无关。

两层控制沿D175：F=Σq-2z1-z2<=.01(beta2+beta3)，h=F+(q1-z3)/20+.25，r=R(h)，J=r+z3/10。forward规则给h<=.37、J<=.47，故R(J-.5)=0。只留第一层Bell然后rebox的完整child source-boxhull却在真实父均值z=(.9,.9,.1),q=(.9,1.71,.01)允许r=.545,J=.555、后继.055；两个内部盒源见证已存。不是超过整个共同父域上的ideal child hull。

定义结论仍未完成：Bell/affine facet transport与模板传播均有先例，原HZ加这些行本身不能叫新域。下一工作必须面向非凸共同载体及其生成元/谓词组合规律，不能只扩facet/helper或重复组件资格。所有实现和数值诊断先另行预注册冻结，保持原数学、真实同结构、shadow、逐家族及全2413/独立400门。

本轮paper-only，无新后台实验；最后组件D1584032tests/212files，D172只读诊断不重试。分支redu-hz、HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac、tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。历史/生产均未改，无commit/push/default变化。
