# 共同源双端预算与原相位整数性的纸面研究

本记录补存 D103 的手工推导与独立纸面复核。结论是纯 epigraph 聚合的四面凸包成立，两条补充行属于已知 MIR 及其互补形式。它可以作为关系强化支撑，尚不是新的 Neural-HZ 定义、实现资格或网络收益。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；配置 paper_only，无实现、freeze、RUN、求解器、数值测试、模型或 GPU 执行。基线来源见 [BASELINE_LOCK](../../BASELINE_LOCK.md)，前序机制见 [D082](../d082_joint_gap_cardinality_20261001/THEORY.md)；完整目标见[本次 Goal 快照](../../archive_checkpoint_20261001_d103/GOAL_SNAPSHOT.json)。

## 从原门得到两端共同源信息

保留全部原连续因子、原二元相位及零点两种标签、EQ/LE、latent/frame 和 decoder。令 q_j=ReLU(h_j)，n_j=q_j-h_j，b_j 为认证的原 active bit 视图；可靠容量 A_j、U_j 给出 d_j=A_j+U_j>0。定义：

~~~text
a_j=A_j/d_j, C=sum a_j,
H=sum h_j/d_j in [c,d],
N=sum n_j/d_j, m=sum b_j,
T=N+sum a_j*b_j.
~~~

原容量蕴含 T<=C 和 T<=m-H。H 的上下界须在同一源合并后可靠取得；不能由不同消费者各自选择源见证。令 s=C-T，X=C+H，L=C+c，U=C+d。本节研究的是下面的纯 epigraph，明确假设 0<=L<=U<=p：

~~~text
S={(X,m,s): L<=X<=U, m in {0,...,p}, s>=0, s>=X-m}.
~~~

这里 s 没有上界；原门的其他容量、T 下界和源谓词尚未加入。

## 四面凸包及完整性

令 k=floor L，ell=L-k，r=floor U，u=U-r。conv(S) 正好由连续盒 L<=X<=U、0<=m<=p 及下式给出：

~~~text
s >= 0
s >= X-m
s >= ell*(k+1-m)
s >= X-r*(1-u)-u*m
~~~

下端行的有效性来自整数 m：m>=k+1 时右侧非正；m<=k 时，X-m 减去该右侧至少为 (1-ell)*(k-m)>=0。上端行由反射 (s',X',m')=(s-X+m,p-X,p-m) 的同一论证得到。

完整性可显式构造，不需要执行时枚举相位。整数 m 已在 S 中。非整数 m=j+theta，0<theta<1，定义：

~~~text
a0=(1-theta)*j, b0=X-theta*(j+1)
A=max((1-theta)*L, X-theta*U)
B=min((1-theta)*U, X-theta*L)
t=clip(a0,A,B)
Xj=t/(1-theta), Xj1=(X-t)/theta.
~~~

X 在 [L,U] 保证 A<=B；两点 Xj、Xj1 均在 [L,U]。以权重 1-theta、theta 混合整数层上的点 (Xj,j,[Xj-j]+) 和 (Xj1,j+1,[Xj1-j-1]+)。所得最小 s 为

~~~text
min_{t in [A,B]} [t-a0]+ + [b0-t]+
  = max(0,b0-a0,A-a0,b0-B).
~~~

当 k<r，按 m 所在区间化简为：

~~~text
m<=k:          X-m
[k,k+1]:       max(X-m,ell*(k+1-m))
[k+1,r]:       max(0,X-m)
[r,r+1]:       max(0,X-r*(1-u)-u*m)
m>=r+1:        0.
~~~

这些恰为四面最大值。当 k=r，中央带 m=k+theta 的式子为 max(ell*(1-theta),X-k-u*theta)，同样相符。更大的 s 可由两个整数见证同时增加非负竖直余量获得。整数端点导致相应新增行冗余，不需要新特例算法。

换回原聚合：

~~~text
T <= C-ell*(k+1)+ell*m
T+H <= r*(1-u)+u*m.
~~~

第二行也可视为对互补门 h'=-h、q'=q-h、b'=1-b、C'=p-C、H'=-H 使用 D082；这是原量的数学视图，不新增或删除实际相位。

## 不能外推的精确性

四面只对上述纯 epigraph 聚合精确。加入 T 下界、逐门容量或共同源谓词后，行仍有效，但“凸包与附加约束相交”等于“附加约束后再取凸包”并不一般成立。

例如下述三门的 C=1.35、H>=0.05，原整数系统不可能 m=0，因此其真正凸包满足 m>=1；但四面加 T>=0.45m 仍允许 m=0.5、H=0.05、T=0.225。不得把本定理称为完整神经聚合凸包或完整 Neural-HZ 凸包。

## 三门的可用推论

复用 D082 的普通三门：

~~~text
x,y in [-1,1]
h1=4x/5+y/5+1/10
h2=-3x/5+2y/5+1/10
h3=-x/5-4y/5+1/10.
~~~

A=9/10，U_j=11/10，d_j=2，C=27/20；H=3/20-y/10 在 [1/20,1/4]。因此 L=7/5、U=8/5、k=r=1、ell=2/5、u=3/5。两端预算是：

~~~text
sum n_j <= 11/10-m/10
sum q_j <= 4/5+3m/10.
~~~

第一行乘 3，第二行乘 1，再除以 4，并使用 sum n=sum q-3/10+y/5，得到：

~~~text
sum q_j + 3y/20 <= 5/4.
~~~

在 (x,y)=(1,1) 和 (-1,-1) 取等。这是已知 MIR 的项目推论；尚未证明它优于继承相同事实的全部现有神经关系或终端求解器。

## 尚未完成的对照线索

为避免遗漏，保存一个手工候选松弛点，但不赋予它强比较资格：

~~~text
x=y=0, b1=b2=b3=7/15, m=7/5
q1=q3=77/150, q2=7/30
delta12=2/15, delta13=0, delta23=4/15
P100=1/3, P010=1/15, P001=1/5,
P110=2/15, P101=0, P011=4/15, P000=P111=0.
~~~

该点满足旧负预算取等，违反新增正预算 1/25，违反物理读出界 1/100。根代理手工构造了单门源凸组合和 D052 等式线索，但完整 D066/D091 对照尚未结束，未独立复核，未数值执行。不能把它标成实际反例、validated ADV 或已超过强对照的结果。下一次若继续，先完成公平比较，不扩张测试包装。

## 文献归属和后续使用

[Mingling: Mixed-integer rounding with bounds](https://atamturk.ieor.berkeley.edu/pubs/mingle.pdf)，Atamtürk 与 Günlük，已检查作者稿的 MIR 背景。这里只借鉴有效行机制，不移植分支切割或违反项目限制的搜索策略。D082 已保存其他 MIR 来源。

两端生成最多增加两条聚合 LE，不需新 bits；但完整源绑定、容量证明、位复杂度、行展开、设备缓冲、终端和见证费用尚未认证。默认没有选定实现；保留为精确对照及候选支撑，不将其改名为定义级突破。最新实际数学资格仍为 D098 的 3845 测试和 188 文件；formal_gain=0，正式 1870/2413 与独立 E0 61/400 不变。
