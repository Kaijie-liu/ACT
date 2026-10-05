# 真实共同源窗口中的相位关系图

本研究把已有默认关闭三门组件接到真实卷积窗口，检验共享关系是否实际存在，并完整计算生成这些关系的代价。表示仍保留原非凸 HZ、所有连续与二元因子、EQ/LE 和输入重构；窗口图只是有认证来源的关系层，不是以新名字整体替代原域。

经典布尔三角行、原 HZ 加同样行的逻辑强度与本层相同；局部缓存也不是新颖性。本轮不能授予 PLDI 级定义贡献。它回答的可反证问题是：在原固定真实三源中，可靠共同源证书能否产生奇负相位三角，以及整个固定人口能否在原资源边界内完成。

## 原数学身份与具体语义

一个窗口使用同一原 source context。每个 source ordinal 对应 first-bank ReLU 的真实 NCHW 坐标，身份还包括原模型hash和实际tensor port；其区间由原first Conv、输入性质与可靠BN得到预激活界，再取非负部分。Padding 为字面零，不伪造source或phase。不同原坐标不能因系数相同或区间相同而合并。

全部第二bank接收门使用实际原Conv/BN参数的区间enclosure。每个输出ordinal编码其原通道和空间位置，phase是该输出原active bit的形式引用。窗口source与目标output属于不同角色，不能按整数ordinal碰巧相等而合并。真实native列、active编码、全局frame生命周期和decoder尚未绑定，形式token不证明这些事实。

对已声明的同源原门 qi=ReLU(fi)，原bi保留零预激活的全部合法选择。固定方向边证书为

```text
qi+qj-Pij(source) <= k0ij*bi+Uij*bj+dij*bi*bj.
```

Pij、Nij及常数由冻结D053从真实区间形式生成。中点仅用于选择一个确切P/N函数，实际支持仍在原区间参数上求取；不把中点网络当作真实网络。D056的两正一负/三负合成给出最多四条静态物理LE。健全性来自上述原门前提与整数三角恒等后果，不来自终端状态、对偶或搜索。

当原HZ谓词确实包含这些原门与source前提时，新增行对其整数具体化是已证后果；可能加强的是连续松弛。若前提只覆盖真实网络而非整个既有抽象状态，只能声称包含真实网络像、不能未经证明声称与整个旧抽象集合相等。本次不安装新行到native HZ，也不声称已构成完整仿射/Conv/ReLU/Add/Concat抽象变换或精确商化简。

## 完整人口与共享证书的等价性

先计算所有n个门的可靠普通界，只有lo>0或hi<0的严格稳定性可用于省略本三角增强。触零保留；原bits始终不删除。令m为其余门，完整人口为C(n,3)，稳定省略为C(n,3)-C(m,3)，剩余全部C(m,3)均组合。m<3时不构造用不到的边，仍完整记录所有原通道界和总计。

当m>=3时，每一对剩余原门按原phase ordinal定向，仅调用一次D053.generate并缓存返回证书。每组三元组取同一context、同一原output/phase、同一affine形式的三条缓存证书；没有外部可注入的certificate缓存。D056内部对任一组三门调用的三个确定性D053生成结果因此逐字段相同。随后采用相同的EDGES顺序、奇负判定、tau、残项平面、P/source/readout和A/phase合并，故返回行应与完整D056重算逐行一致。

这种等价仍要求相同的三门支持/数值上限，不因单边可构造就绕过D056三门组合限制。每个原affine界核对与普通接收界相等；缓存不能跨context、source box、tensor或模型复用。新的有限数学测试对固定多门窗口的全部三元组逐行比较，而不是仅复用一个旧通过证书。

每个窗口所有plain pair证据只保存一次；每个三元组记录原通道、固定边索引、状态及全部新行。共享引用只减少重复证据，不省略零边、偶号或无增强三元组。原source/target身份、P/N、k0/k1/U/delta和pair投影仍可追溯。其他窗口采用各自的数学context；它们的全局native合并尚未发生。

receiver_coefficients_ref明确引用source_packet_ref所绑定的原模型字节与extractor，给出原分支index和可靠post_affine规则，以重构该窗口使用的完整接收参数；它不指向不存在的序列化receiver bank。实际解码bank仍被内存ledger计入。独立证据检查认证这些引用及人口/行结构，不重新解析原模型并重跑数值生成器。

## 可报告的代数差

对于奇负三角，令tau=min(abs(dij))。在三个连续bit均值都为1/2时，三条独立pair上界之和与合成三角上界之差恰为tau/2：唯一负边时两个正剩余项各减tau/2并由中心项加tau/2；三负时三角项为tau*(1-3/2)，其余负剩余项为零。

该量仅证明辅助phase-box上包络存在严格差，不证明有共同source/guard/physical readout实现那个旧最坏点，也不是新增CERT、ADV或具体网络非冗余见证。字段cube_midpoint_gap需显式保留这个限制。没有采样或LP去挑选使用哪些三角；所有数学合格三角统一记录。

## 全部传播与存储代价

每窗口必须先检查和传播n乘F个canonical参数位置，包括padding和稳定门。若m>=3，则E=C(m,2)条边各生成一次，T=C(m,3)个三角全部处理。以每门实际支持约S计，边构造约O(E*S log S)，三角标量符号/平面约O(T)，但每个三角实际物理行仍须合并和输出，最坏O(T*S log S)工作与O(4*T*S)nnz。这不是只计四个标量操作的常数时间算法。

typed缓存包含原context、source/value/phase registries、m个affine形式、E个完整D053证书及其P/N和旧pair行；这部分占O(m*S+E*S)支持并与plain证据共存。当前wrapper在调用前支付声明、原support计算、排序、检查和组合，再逐访问支付plain转换。具体预付公式如下，其中ell(s)=ceil(log2(s+2))：

```text
source/value/phase声明及context：4096+256*m+256*R+16*R*ell(R)
每门affine声明：1024+192*S+8*S*ell(S)
每门一次可靠界核对：64+64*S
每边D053生成，T为两门实际支持和：4096+768*T+32*T*ell(T)
每组三角合成，H为三P实际支持和：2048+384*H+24*H*ell(H)
```

R为真实非padding源数；affine声明中的S为传入的实际非padding出现数，之后edge使用规范化后的实际非零支持数。完整canonical位置的遍历、普通界、序列化及局部对象生命周期另付费，不被这些式子替代。这是有界标量与container操作模型，不是CPU指令数或速度声明。

typed临时numeric-entry上界为65536+512*F+48*m*F+192*E*(F+8)+256*n，E在m<3时为0；允许每个Fraction额外计其分子与分母，覆盖窗口context/registries、所有形式和边缓存及一组正在合成/序列化的暂存。原输入和全部plain证据仍另外经过ledger。临时frame的value/phase registries存在反向引用环，不能假定return即释放；新wrapper须以预付的finally清理仅由本调用创建的临时registries。所有plain原身份保留，不接收或清理外部HZ/native frame，因此这不删除原二元相位或修改旧模块全局状态。

worker同时保留的原raw模型、spec、decoder packet、box、接收参数、source bound cache、plain窗口输出、认证roots与报告全部经过原bounded ledger。typed缓存及组合temporaries使用独立保守numeric-entry上界再相加，不把旧ledger不支持的对象当零。最后窗口之前释放的原first-form和parser临时对象仍使用继承的额外reserve。实际RSS/tracemalloc门和原全部工作/证据/entries上限不变；不据此声称完整聚合主机设备物理资格。

原全网谓词、全部原二元相位、后继传播、终端矩阵展开、普通LP/MILP和反向输入重构都没有在本source census执行。完整端到端费用不能从本次局部结果省略。源模型解析成功也不证明后继第三Conv/残差参数已经解码；旧extractor该部分仍只保存拓扑。

## GPU与定义创新仍需的证据

边证书独立生成后统一组合，提供了一个可批处理的固定工作集合；其作用是为后续可靠GPU算术提供明确数据流，不是GPU实现。外向舍入、可靠归约、设备/主机共存、输出材料化及完整终端消费仍需单独实现和测量。

真实结构实验若没有奇负关系或无法完成，应如实否定当前版本在该范围的晋级资格，不将资源失败扩大成数学不可能；若发现奇负关系，也只晋级到后续真实使用研究。创新域定义、结构性精确简化、原HZ嵌入、各类算子覆盖、全部旧解保留和新解、GPU净收益分别需要证据，不能由本轮局部组件替代。
