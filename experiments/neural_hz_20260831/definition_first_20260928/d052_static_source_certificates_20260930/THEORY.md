# 两个原门的共同源证书与前向传递

本轮得到一个不读取求解状态的共同源证书生成规则：对两个原 ReLU，在共同源盒上固定匹配系数、支付未匹配余项，可直接生成至多两条跨门关系。非点参数也有健全版本，中点仅用于选择证书，不替换真实网络。另给出共同相位系数下穿过下一 ReLU 的前向规则。

这些规则补上现有混合源观察语言中的生成步骤，但没有超出已知 Anderson 单门关系、条件观察与 RLT 的机制。现有 D049 语言已经能表达输出行，因此不能将本轮包装为新的域定义。完整 Neural HZ 创新、真实网络收益和 GPU 资格均未完成。本轮只作数学研究，不导入或执行新候选。

## 保留的非凸语义

固定同一个原 HZ frame，保留连续因子、全部原二元相位、EQ/LE、共享 source 身份、物理读出和输入 decoder。研究两个原已有门

```text
q=ReLU(f(x)), 原 active bit alpha,
p=ReLU(g(x)), 原 active bit beta,
x in D=[l,u].
```

D 是实际共同源的可靠外包，不替换其原非凸谓词。alpha、beta 只是原相位的认证 0/1 视图，不能新建替代 bits。预激活为零时，两种原合法相位均保留。源 x 可以是已认证的内部物理 frontier；将内部值当成输入的仿射函数则另需证明。

加入下述有证行不改变原整数具体化。查询时放松 bits 可取得更紧外包，但不是将原域退化为凸域。空证书嵌入原 HZ，原 decoder 不变。相同名字、区间或数值不证明两个源是同一赋值。

## 一个不需要精确系数匹配的定理

选同一 D 上两个确切仿射函数 P、N，并实际构造以下证书：

```text
P(x)>=0, N(x)>=0,
f(x)-P(x)<=kappa0,
f(x)-N(x)<=kappa1,
g(x)-P(x)+N(x)<=U.
```

kappa0、kappa1、U 是可靠确切常数，不要求非负。原门语义给出

```text
q <= P + kappa0*alpha,
q <= N + kappa1*alpha.
```

alpha=0 时 q=0，用 P、N 非负；alpha=1 时 q=f，用两条源证书。该证明不求解相位子问题，也不要求 P、N 与真实 f 的系数同号。

记 F=q+p-P、Delta=kappa1-kappa0。对原 beta 的整数恒等式 p=beta*g，得到

```text
F <= kappa0*alpha + U*beta + Delta*alpha*beta.       (1)
```

beta=0 时使用第一条 q 界；beta=1 时由 q+g-P<=q-N+U 使用第二条。两门的零点原相位都包含在非严格证明中。证明分情况不是候选运行时切分。

## 单个消费者的完整两行投影

只为 (1) 引入假设辅助量 theta，并用 alpha*beta 的四条 McCormick 行约束它。在 0<=alpha,beta<=1 上，消去这个没有其他消费者的 theta 恰好给出：

```text
Delta>=0:
  F <= kappa1*alpha + U*beta,
  F <= kappa0*alpha + (U+Delta)*beta.

Delta<0:
  F <= kappa0*alpha + U*beta,
  F <= kappa1*alpha + (U+Delta)*beta - Delta.
```

Delta=0 时两行相同，只需一条。Delta 的符号是已认证数学常数的结构条件，不是 LP 状态或实例菜单。

必要性由 theta<=min(alpha,beta) 或 theta>=max(0,alpha+beta-1) 得到。充分性在 Delta>0 时取 theta=min(alpha,beta)，Delta<0 时取 theta=max(0,alpha+beta-1)，这些选择同时满足四条 McCormick 行。Delta=0 时任意合法 theta 都可扩展。

这是该单消费者线性提升的精确投影，不是整个双门 source hull，也不是原非线性乘积的连续精确表示。若其他消费者共用 theta，各自独立消去可能丢失相容性，不能据此删除已有共享 overlap。实际生成器可直接输出两行，不需要创建 theta；全部原 bits 不 pivot、不删除。

## 固定的系数匹配构造

先考虑确切仿射 f、g。在两者共同 source 支持的并集上，按 f 每项的符号将原盒坐标精确写成 y_i in [0,1]，使

```text
f=m+sum_i A_i*y_i, A_i>=0,
g=e+sum_i B_i*y_i.
```

负 f 系数使用反向坐标 y_i=(u_i-x_i)/(u_i-l_i)，非负系数使用正向坐标。f 系数为零时采用固定正向；只出现在 g 中的坐标不能省略。固定坐标先精确折叠。偏置、原点与尺度补偿必须完整计入 m、e。

统一定义

```text
t_i=min(A_i,abs(B_i)),
P=sum_{B_i>0} t_i*y_i,
N=sum_{B_i<0} t_i*y_i,
p0=sum_{B_i>0} t_i, n0=sum_{B_i<0} t_i,
M=m+sum_i A_i,
kappa0=M-p0, kappa1=M-n0,
U=e+sum_i max(B_i-sign(B_i)*t_i,0).
```

P、N 显然非负。因为 A_i-t_i>=0，前两条源支持由完整盒直接得到；最后一条是余项 g-P+N 的完整盒支持。没有忽略不匹配系数，也不要求两门权重相等或成比例。

这不是全局最优匹配声明。它是一条按数学系数统一执行的选择，保证所需前提可构造，不调用 LP、支持函数 oracle 或后向优化。两行投影的精确性只针对这个已固定的提升系统。

原单门 q<=P+kappa0*alpha 也可视为 Anderson 上侧关系的凸组合。真正的新增项目步骤是同时选择两个源证书并在另一个原门消费它们，非新的凸化原语。

## 非点参数的健全扩展

真实参数可能来自 BN 的可靠区间。一般定理并不要求 f、g 系数确切：只要 P、N 是固定的确切非负函数，三个上界同时覆盖原参数即可。

可用系数区间的中点按上一节规则选择方向和匹配幅值，构造确切 P、N。中点只决定要证明哪个函数不等式，不替代 f、g，不创建参考相位；即使真实系数区间跨零，P、N 的非负性仍成立。

对区间仿射形式 a0+sum_i a_i*x_i，定义直接可靠支持

```text
Upper(a,D) = upper(a0)
           + sum_i max(lower(a_i)*l_i, lower(a_i)*u_i,
                       upper(a_i)*l_i, upper(a_i)*u_i).
```

将确切 P、N 的系数及常数先从原区间形式中相减，再取

```text
kappa0=Upper(f-P,D),
kappa1=Upper(f-N,D),
U=Upper(g-P+N,D).
```

这完整保留真实参数的不确定性。参数间或参数与源之间的相关性若被忽略，只会放宽外包，不使它失效。不能先分别把 f、P 的值区间化后又宣称得到了相同相关支持；更不能漏掉 P、N 的原点和尺度常数。输入区间、BN enclosure 与原门绑定仍必须已有可靠来源。

精确参数时，该版本恢复上一节公式。浮点实现还需逐步向外舍入与溢出检查；这里是精确有理区间的证明，没有浮点、BN loader、native 或 GPU 实现资格。

## 共同相位系数下的前向传递

令 u、h 为同一物理 frontier 的真实仿射读出，psi(b) 是全部相关原 bits 上的一个固定仿射式。若已实际构造

```text
u-psi(b)<=c0,
u+h-psi(b)<=c1,
```

对下一原门 r=ReLU(h)、原 bit gamma，可得

```text
u+r-psi(b)<=c0+(c1-c0)*gamma.                       (2)
```

两条前提分别乘以 1-gamma、gamma 再相加，共同 psi 抵消，没有新 gamma*b。零点两种原相位均保留。(2) 是既有 D029 上侧条件观察规则的一个前向组织方式，不是新的不等式类别。

若两前提分别使用 psi0、psi1，直接插值会出现 gamma*(psi1-psi0)。当有关 bits 独立可行且差系数非零时，该项有非零混合离散差，不能等同一条只含原 bits 的仿射式。可以逐项取系数最大值（常数并入 c0、c1）作共同上包络，再使用 (2)，但这是健全近似，不是无损闭包。已有确切 gated 读出或相位恒等式可用于代换，必须另有证明。

仿射、Conv、Add、Concat 通过共同身份上的确切线性组合形成读出；两条前提的非负证书组合及其成本必须实际支付。这里没有普遍自动产生所有所需前提的算法，也不声称有限模板对任意后继精确闭合。CONTROL.md 给出不用 oracle 的普通混权两层实例。

## 完整成本和设备形式

若两门有 d 个共同规范化 source 支持，匹配和三次盒支持是 O(d) 标量工作；原稀疏支持合并、排序和索引成本另计。固定已排序支持可以线性合并，不按每门的单独 fan in 冒充共同支持。

两条最终行各至多 |support(P)|+4 个物理坐标 nnz，零新辅助量、零新 bit。若 P=p_const+sum p_i*x_i，则 q+p-P<=a*alpha+b*beta+c 必须编译为

```text
q+p-sum p_i*x_i-a*alpha-b*beta <= c+p_const.
```

如保留 P、N 的单门行作为可消费证书，还需支付这些行、支持和证明；不能声称它们已免费存在。负原 bit 编码、bias、同一读出抵消都要精确转换。

(2) 中若 u=u0+sum_{j in V} a_j*v_j，psi=psi0+sum_{i in B} d_i*b_i，则新行在物理坐标上至多 |V|+|B|+2 nnz。多层相位支持与旧关系会累积；小当前 frontier 不等于恒定总存储。

设备数据流可用固定 gather、abs/min、区间乘积和分段 reduction，再形成稀疏行及 A/A^T。此描述没有执行设备 kernel，也没有把普通终端转置称为 dual rescue。真实 HZ latent 展开、可能的别名列与等式、原矩阵、主机设备共存、缓存、可靠误差、终端求解、证据与具体见证检查都要付账。不能由 O(d) 的证书算术推出全验证器加速。

## 与已有研究的边界

[Anderson 等的五作者完整版本](https://arxiv.org/pdf/1811.01988)，§5.2 Proposition 12 给出单门 source box 的强表述；Proposition 13 的 LP 点分离不在本规则内。我们使用静态系数生成有效关系，不采用动态 cut loop、BaB 或 LP 状态触发。

与 [DeepPoly](https://files.sri.inf.ethz.ch/popl19-paper264.pdf) §4 的有限神经多面体约束一样，有限条数有计算价值但限制精度。这里保留原非凸 HZ，不整体替换为 DeepPoly；多放几个前向模板也不能据此宣称新颖性。

D029 已给条件观察投影，D035 已给共享 overlap，D049 已有混合 source/phase 语言。本轮输出能由这些已知机制表示。研究决定是保留这个可构造、参数健全的生成器作为支撑，不再为它另造一个域名称。真正的定义贡献仍需超出这组已知局部规则的结构不变量、适用性与完整成本证据。

## 执行和保管状态

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置仅为纸面推导、独立只读复核、原始论文相关段落核验；没有候选导入、测试、模型、LP、GPU、shadow 或完整回放。

上一轮 D049 的 3785 项数学组件通过保持原状态，不重跑、不当成本轮执行。所有旧失败结果与生产 dirty changes 保留。本轮 formal_gain=0；正式 1870/2413 与独立 E0 61/400 不变。具体 provenance 和研究决定另见 RECORD.md。依据 write-page 技能区分来源、推导与执行；未发布外部 Page。
