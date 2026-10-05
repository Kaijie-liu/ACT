# Neural HZ 定向闭包研究续接

重点仍是从 HZ 数学定义出发提出强大的非凸 Neural-HZ，不是 helper 集合、恢复旧精确图、压存储或构造加速。本轮[定义与证明](definition_first_20260928/d163_directed_phase_budget_20261004/THEORY.md)给出可复用的定向预算修复，但没有取得新域、真实能力或晋级资格。

核心关系：D162 的 M 版距离域等价于 q=c+f+e，0<=f<=M(1-beta)，sum w|e|<=R，并保 q>=0、q<=U beta。active f=0；inactive 可取 f=(-c)+、e=-c+。所有消费者共用同一份 f。完整 K=[W;T] 的新中心为 K(c+f) 加原源仿射项，新半径为 kappa R，无 D162 inactive padding。保父 P、实际 g 的 sign guards、精确 live skip 和原身份；新 r 是有损代理，不保完整旧 r=ReLU(g)。可以额外加 r>=g。新中心 M 不能用实际 g 的下界代替。

非零预算正控：x,y∈[-1,1]，c=(x+1/4,y-1/4)，R=1/10，父 q 为距离域，U=(27/20,17/20)。g=q1-q2+1/4∈[-3/5,8/5]，z=q，kappa=2。新 active 时 J=r-g<=3/10，inactive 时 J<=3/5，所以 ReLU(J-3/4)=0。旧圆化在源零、父 q=(1/4,0) 仍允许 r=3/2，预算27/10、共同成本1、J=1。新域仍允许 r=7/10，用同一 f=(0,1/4) 花1/5，故严格有损。旧 HZ 单条上侧也证明 J<=3/5，因此这只是修复前版损失，不是能力突破。

账单：q/f/rho 共3m连续量、6m+1行，另付实际 guards、历史P、中心填充、所有旁路与decoder。消f回到2m、5m+1的M版，但再次对称圆化会丢方向。普通旧内联ReLU仅m输出量、4m行。无固定宽度、GPU、终端或端到端优势认证。

避免重复：相位中心diag(beta)c与下一相位的乘积属于D001，截断属于D121，共同替换合同D142；tropical共同符号抵消与闭包D011已有。相邻开cell证明固定C/S表示 y=Cxi+h(beta)+Seta 必须rank(S)>=rank(B_crossing)，不要求全部相位或m<=source维数，但D152已有主要结论。本轮只推广h任意相位函数，不计新下界；也不外推所有有损域。

下一实质问题是从普通共同输入盒产生有用的联合关系，经混权、非线性和真实旁路后，域自身仍能以可支付代价查询它。不能把预先给定的有利L1球当成真实CNN免费具备的输入。允许更换本候选；不要求有损域比精确图集合更小，只比较实际能力与完整费用。不要以独立helper或追加工程框架代替定义研究。

本轮为纸面数学与只读审查；没有候选导入、测试、模型/GPU、shadow或replay，无本轮后台作业。最近执行人口仍D158 4032项/212文件。未来实际候选仍需新的预注册/源码冻结/一次性版本与原完整门；不复跑消费版本。

正式1870/2413，独立CIFAR25+Tiny36=61/400，两边新增0。13家族逐例保旧、连续源/原bits/EQ/LE/共享身份、禁止helper补解、fail-closed、GPU与smooth/Transformer/新家族目标及全部回放条件不变，goal active。本轮属有限数学progress，不是正式能力progress。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新写入仅本轮隔离档案和本续接；旧数据不动。详情见[工作记录](definition_first_20260928/d163_directed_phase_budget_20261004/RESEARCH_RECORD.md)，文件身份见该目录的两个SHA256清单。
