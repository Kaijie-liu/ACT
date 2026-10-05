# 共同相位关系的纸面审计

本文件记录研究讨论中的代数与范围，不是已执行测试或数值资格。所有原 HZ 连续量、二元相位、guards、共享输入与 decoder 保留。下述 LP 凸包比较不把原非凸载体替换为凸域，不执行输入或相位分裂。

## 共同表与经典扩展表述的对应

对两个原相位 alpha、beta 的松弛及共同辅助量 delta，定义

```text
lambda00 = 1-alpha-beta+delta
lambda10 = alpha-delta
lambda01 = beta-delta
lambda11 = delta.
```

四条 McCormick 行等价于 lambda 非负，且四项和为一；alpha=lambda10+lambda11，beta=lambda01+lambda11。对每状态的非空有限输出盒 B_st=product_j[L_st,j,U_st,j]，同一 lambda 下

```text
sum_st lambda_st*L_st,j <= E_j <= sum_st lambda_st*U_st,j
```

恰描述加权 Minkowski 和 sum_st lambda_st B_st：盒的各坐标可独立取值，因此必要性与充分性都成立。再让 lambda 在单纯形内变化，得到带相位标签的四盒联合凸包。独立消费者各用自己的 lambda 会允许不相容的混合。

这是把经典 Cayley 扩展机制代入当前表语言的项目推导，不是新定理归属声明。它只对状态盒成立，不证明原网络源、guards 和全部消费者的联合凸包已经被精确表达。整数相位仍把 lambda 固定为对应状态；所有原相位包括零点的两种选择不变。

## 独立残差上表的无增益范围

设共同有界源 D 上 g1、g2 仿射，q1=ReLU(g1)、q2=ReLU(g2)，原相位为 alpha、beta。比较系统的父前缀属于包含源、值及原相位的完整双门联合凸包。于是存在一组真实父赋值的共同混合 mu，恢复比较点的源、q1、q2、alpha、beta。

对任意数量的消费者 F_j=a_j*q1+b_j*q2+R_j，仅使用各自独立 residual 范围 L_j<=R_j<=U_j。定义

```text
T_j,st = U_j + sup_{x in D}(a_j*s*g1(x)+b_j*t*g2(x)).
pi_st = Pr_mu(alpha=s,beta=t), delta=pi_11.
```

系数 a_j、b_j 可以任意符号。由原父关系 q1=alpha*g1、q2=beta*g2，

```text
F_j <= E_mu[U_j+a_j*alpha*g1+b_j*beta*g2]
    <= sum_st pi_st*T_j,st.
```

若后继满足精确图 r_j=ReLU(F_j-c_j)，单调性与 Jensen 给出

```text
r_j <= ReLU(sum_st pi_st*T_j,st-c_j)
    <= sum_st pi_st*ReLU(T_j,st-c_j).
```

同一个 delta 同时满足全部上表及其联合投影。因此这类上表不能排除该比较系统的任何点。证明只使用实际 residual 上界，并不假造一个共同全网分布；每个父前缀的共同分布已足够。结论也扩展到所述精确后继系统的联合凸包，但不能自动扩展到逐后继分别松弛的交集。

两个例外防止过度外推。其一，分数 big-M 后继不满足精确图：g1=x、g2=y，x,y in [-1,1]，父取真实 x=y=-1/2、q1=q2=alpha=beta=0；residual in [-1,0] 取 -1/5，F=q1+q2+residual。用旧界 [-1,2] 的 child big-M 允许 childbit=1/2、r=1/4，而表00的上界为零。这是强化较弱后继的例子，不推翻上述定理。

其二，下界截断不能反用 Jensen。令 F=q1-q2+1/4，其全盒下表为 (1/4,-3/4,-3/4,-7/4)，ReLU 截断后为 (1/4,0,0,0)。父共同分布

```text
1/2 * (-1/2,-1/2) + 1/4 * (1/5,-1/2) + 1/4 * (-1/2,9/10)
```

给 q1=1/20、q2=9/40、alpha=beta=1/4，精确 child r=F=3/40。下表却对任意合法 delta 要求 r >= (1/4)*(1-alpha-beta+delta) >= 1/8。这是已有条件下界机制可能更强的例子，不是上侧无增益定理之外的新颖性认证。

## 保留共同源的宽混合层候选

这段保存本轮尚未执行的构造，避免把讨论丢失或以后误认作已验证实现。令所有 g_k、v 仿射于同一源盒 D，原门 q_k=ReLU(g_k)，观察 F=v+sum_k w_k*q_k。对每门已知一个点态可靠的仿射上界 S_k>=ReLU(g_k)。写 w_k^+=max(w_k,0)、w_k^-=max(-w_k,0)，缓存

```text
B = v + sum_k w_k^+*S_k
H = sum_k w_k^-*g_k.
```

固定两个不同原锚 i,j，令 C=-w_i^+*S_i-w_j^+*S_j。四个条件上界平面为

```text
P00 = B+C
P10 = B+C+w_i*g_i
P01 = B+C+w_j*g_j
P11 = (B-H)+w_i^+*(g_i-S_i)+w_j^+*(g_j-S_j).
```

证明：非锚正项用 S_k 上包，非锚负项 N*=sum_{k!=i,j} w_k^-*ReLU(g_k) 同时满足 N*>=0 与 N*>=H*=sum_{k!=i,j}w_k^-*g_k。原锚状态 st 下其贡献精确为 w_i*s*g_i+w_j*t*g_j。前三态用 N*>=0，11态用 N*>=H*，利用 w_i+w_i^-=w_i^+ 化简即得。锚权重无需为正，零预激活的所有原标签均适用。

取 U_st=sup_D P_st，每个支持均在同一个完整盒上计算，不对四个相位子域求解。对 -F 用同一锚可生成下界。该式与独立 residual 上表不同之处在于先合并同一源的仿射系数，尤其保留 H*，而不是先把其他门变成相互独立的常数范围。

对盒中心 c、半径 r 的仿射式 b+a*x，支持是 b+a*c+sum_l |a_l|r_l。预存底式 B、B-H 及支持后，四平面仅在两个锚支持上发生变化。若固定锚对互不相交，单 receiver 全部锚的算术可按总系数出现数线性计费；它不是整网常数成本，也不是有理数位复杂度或运行时间证明。源身份规范化、上下两方向、majorant 构造、所有消费者、证据、终端矩阵与输入重构均另计。可靠 majorant 的数值策略尚未冻结；大宽度下精确系数位长、native 绑定和实际代价均未认证。

实际参数为可靠区间时，中点只可用于参考证书，不能用参考门相位替代原 bit。锚误差恒等式

```text
w_actual*ReLU(g_actual)-w_mid*alpha_actual*g_mid
 = (w_actual-w_mid)*ReLU(g_actual)
   + w_mid*alpha_actual*(g_actual-g_mid)
```

给误差界 rho_w*M_actual+abs(w_mid)*eps_g。非锚项由 ReLU 的 1-Lipschitz 性质得同形界；加上 v 的误差，可统一增加四个上端点并减小四个下端点。这是数学误差原则，不是已完成实际参数/谓词认证。

该候选的基本条件关系仍有已知先例。它尚无实际宽层非冗余证据、跨层共同源闭包、新域贡献或正式收益，不得据纸面线性算术复杂度宣称 GPU/性能通过。

## 静态来源与资格勘误

旧 D045 目录实际只有 THEORY.md、interval_relation.py、archive_worker.py，没有 PREREG、runner、freeze 或运行结果。D065 RECORD 所称其“未执行的预注册设计”应仅理解为设计草稿，不能当作已完成预注册或认证运行器；旧文件保持只读。

D025 已存 large 完整包覆盖五窗乘64消费者，每窗576输入槽、288固定源对。若每个源对跨全部64消费者共用关系，则是1440个源对组、92160张表。46080指固定双消费者配对，不等于已处理全部消费者的联合消元。这里是只读人口核对，不是新实验登记，更不能补记失败的 medium/Tiny 普查通过。

本轮没有运行以上构造。D064 的3805项成功检查仍只属于其原标量绑定资格，不能给这些新条件表授予数学、native、GPU或真实网络资格。证明由根代理与两个独立子任务纸面复核；formal_gain=0。
