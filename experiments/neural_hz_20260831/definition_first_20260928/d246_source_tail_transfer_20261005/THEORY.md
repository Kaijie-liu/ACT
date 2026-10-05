# 宽输入残差的两父限制与尾项闭包

本研究解释D245共同源恢复之后仍可能遇到的结构障碍，并保存一条健全尾项闭包及其红队反证。它不是新数值候选，也不授予实际模型资格。保留原非凸HZ的连续因子、全部原二元位、EQ/LE、共享来源与decoder；以下分数点只比较松弛，不是ADV。

## 未选不稳定输出的不可消去宽度

固定同一已封存H中的父ReLU层。先忠实折叠稳定门，合并同一来源列；仅将仍独立存在的同层crossing输出记为Q_k，可靠界为[0,U_k]。父预激活只依赖较早来源A及误差，不含同层Q。两子完整预激活为

~~~text
g_1 = sum_k C_1k Q_k + S_1(A,errors),
g_2 = sum_k C_2k Q_k + S_2(A,errors).
~~~

若选父i,j，归一化尺度s_i=max(-L_i,U_i)，tau=[s_i(C_1i-C_2i)-s_j(C_1j-C_2j)]/4>0。共同源Gram只能修改A等来源上的系数，不能消掉未选Q列。因此D245完整列盒求界的宽度必满足

~~~text
W_r >= (1/2) sum_{k not i,j} |C_1k+C_2k| U_k,
W_e >= (1/2) sum_{k not i,j} |C_1k-C_2k| U_k,
W_r+W_e >= sum_{k not i,j} max(|C_1k|,|C_2k|) U_k.
~~~

证明是盒宽等于各规范化列的系数绝对值乘范围宽度；其他列只增加非负宽度。最后一式使用(|a+b|+|a-b|)/2=max(|a|,|b|)。这是当前盒求界合同的下界，不是原非凸H精确支持函数的下界。

mixed守卫要求W_r<=2*tau。因此未选共同均值贡献大到使第一式超过2*tau时，无论Gram怎样恢复源相关性都不能通过。即便守卫通过，仍可能留下很宽的W_e；两defect的e支付总宽至少2W_e。不能把守卫通过数直接当作验证能力。

令sigma_k=|C_1k+C_2k|U_k，delta_k=s_k(C_1k-C_2k)，S=sum sigma_k。若

~~~text
S > max_{i != j} [(sigma_i+delta_i)+(sigma_j-delta_j)],
~~~

则固定两子门下全部eligible父对都失败。因为任一通过对必须有S-sigma_i-sigma_j<=delta_i-delta_j。最大值可按第二项的带身份最大值和次大值在线性扫描中求得；不需要LP、相位枚举或按结果选择规则。完整C/U、来源合并及范围的取得成本仍必须另付，不能将该O(m)扫描当作免费模型运行。

普通五父例：x_k,t在[-1,1]，q_k=ReLU(x_k)，g_plus/minus=plus/minus(q1-q2)+(4/5)(q3+q4+q5)+t/10。S=24/5，而上式右侧最大为4，全部父对失败。严格内部点x=(.9,-.9,.9,.9,.9)、t=0使z=54/25、实际子差=9/5；未经守卫的K=z+1=79/25，下侧要求34/25<=1/5而不成立。这不是零点约定或极小舍入误差问题，不能直接删除guard。

## 同层源前沿的精确共享

此结论限定于D245复用D243展开器时的child策略：在原ReLU或选中q边界停止，不在选中父f处停止。每个crossing ReLU本来就是停止边界，因此同一父层中改变选中哪两个crossing父，不会改变子门的完整来源前沿；选中的Q停止标记并未创造新的边界。不能将该结论直接套到还在选中父f处停止的原D243 X attach。

设归一化父源行为w_i，子源行为v_c，Gamma为可靠来源半径平方构成的对角矩阵。父w_i在全部同层Q列上为零，所以从子均值扣去两项选中Q不改变内积：

~~~text
G_ij = w_i Gamma w_j^T,
h_i = w_i Gamma (v_c+v_d)^T / 2.
~~~

可共享这些精确收缩，不必每个候选对重做全部源展开。但稳定门折叠后的前沿必须一致，Gram条目、缓存和算术都收费；r/e的带符号端点和仍需合并真实来源系数后求得。该结论没有把整个矩阵常驻内存算作免费，也没有解决完整Conv预算或终端消费。

## 保留非负共同尾项的健全闭包

把全部未选同层Q在共同均值中的系数按符号拆分为两个非负、同H的实际读出u_plus、u_minus，令

~~~text
z = z0 + u_plus - u_minus,  u_plus,u_minus >= 0,
d = tau(q1-q2)+e,          e in [Le,Ue].
~~~

z0仍由D245的选中父x/q和完整core残余表示。只对z0要求原mixed条件：在h=alpha1-alpha2非零时，|z0|<=tau。所有未选差项仍完整进入e，不能因拆出了共同尾就省略。

记D=ReLU(z+d)-ReLU(z-d)，K0=h(z0+tau)，以D245六产品表达K0。新增条件观察u_i=alpha_i*u_plus、l_i=alpha_i*u_minus并保留完整MC，可加入

~~~text
 D-K0-u_1-l_2 <= 2*tau*p2 + 2*max(Ue,0),
-D+K0-l_1-u_2 <= 2*tau*p1 - 2*min(Le,0).
~~~

证明：Phi(d,z)=ReLU(z+d)-ReLU(z-d)对d单调且2-Lipschitz；d-tau*h=tau(p2-p1)+e给出两侧p/e支付。在参考d=tau*h处，h=1时Phi对z非减且1-Lipschitz，尾增量在[-u_minus,u_plus]；h=-1时在[-u_plus,u_minus]；h=0时恒零，所写支付仍非负。core守卫给Phi(tau*h,z0)=K0。叠加即得两行。原位整数时产品MC精确，所有旧点有唯一扩展；旧H及decoder保留，零标签无遗漏。没有新增相位或求解子问题。

四个额外产品总成本为10连续观察、40MC+2defect+3容量=45LE。若core残余及两尾各为一个非退化物理列，关系132nnz；另加20条产品界后65LE/152nnz。一般内联支持s_r,s_plus,s_minus的关系上界为120+4(s_r+s_plus+s_minus) nnz。原H、实际读出EQ、分组/范围和终端完整费用另付。

## 红队排除了不必要的四产品实现

更弱但便宜的统一规则已经健全：

~~~text
 D-K0 <= 2*tau*p2 + u_plus+u_minus + 2*max(Ue,0),
-D+K0 <= 2*tau*p1 + u_plus+u_minus - 2*min(Le,0).
~~~

它直接来自Phi对z的全局1-Lipschitz性，也可由上述条件观察的MC上界推出；无需四个新观察。

我们纸面提出的正控是D245原混合core加u_plus=(4/5)(q3+q4+q5)、u_minus=0，并以实际读出Ftilde=F-(2/5)u_plus比较。固定原两子与这五父，D245选core两父时W_r>=2.4>2tau=2，选一个core一个尾时W_r>=1.6>2tau=1，选两尾时tau=0；因此当前固定规则全部拒绝。

新上行可以恢复Ftilde<=258/125。将D245强参照的各原子共同附加三个真实inactive点x=-1/2、q=alpha=0，尾为0；原物理假点2587/1250、严格差7/1250及全部对应源/相位分解保留。真点尾为0时仍取等。

但两个独立红队指出：Ftilde中的减项恰好抵消了廉价统一支付，后者也得到完全相同的证书！所以该例证明core规则在共同尾项下的适用性闭包，不证明条件tail产品带来了额外强度。内联三尾的廉价版仍为6观察、29LE/102nnz，含产品界为41LE/114nnz。当前不实现四个额外观察，不为此正控新建数值runner。

D244自由tail的负定理不能直接否定这个有core共同源容量的闭包，但也不能反过来据此宣称新颖性。D061等旧档已经记录共同残余及单调/Lipschitz组合原则；这里没有证明其具体固定生成器已追平整图，也没有证明超越这些组合原理。宽差分e仍是未解决问题。

## 真实网络范围和下一选择

只读D241的三个完整prefix确认必须保留全部后续消费者：large为port133到Add14及port136到Conv12；medium/Tiny为port129到Add16及port132到Conv14。其工件明确activation_bounds_propagated=false、actual_complete_child_affine_map_constructed=false、actual_native_phase_columns_bound=false。所以目前没有实际稳定/crossing人口或所需完整C/U，不能声称上述全pair证书已经拒绝三张真实模型。

将整个576输入窗口都当作独立crossing余项会错报：大部分稳定门可能折叠回共同A来源。本轮只得到可供完整真实来源消费的必要条件和闭包证明，不得换成选几个有利窗口或删除其他消费者。

研究选择是保留D245已认证组件，不追加无强度证据的条件tail产品。真实下一阶段仍必须取得完整同H范围、消费者、原相位与完整终端消费；GPU不是数学能力替代品。本文没有代码/import/AST/compile/pytest/模型前向/新求解调用，没有新成绩。
