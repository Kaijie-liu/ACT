# 原生 HZ 上的共享相位幅值变换

本候选将已有共享幅值公式编译到实际 SparseHZono 的 EQ/LE 和原二元列上，覆盖两个父门、多个混权后继及全部未选输入残余。它是定义研究的第一项可执行原生关系变换，不是已经完成的新抽象域。公式来自已有 RLT、析取提升和前轮 D085；不能将本次接入重新称为数学新颖性或 PLDI 级成果。

## 语义与原生绑定

输入 H 保留全部连续因子 xi in [-1,1]、原 signed bits b in {-1,1}、EQ/LE、输出读出、frame 和 exact 标志。选择两个具有实际原门谓词凭据的父门，以及一组具有同样凭据的 child；这些选择是函数的结构参数，不读取实例身份、公开结论、margin 或求解状态。本版尚未实现模型级统一 selector，因此不能称完整验证路径。调用期间原 HZ 不允许被另一线程并发修改；不把可复用的 frame 数字当作跨运行身份。本次返回的原数组采用独立副本，不冻结或修改调用方数组。

复用 D064 对 extended/compact 原门的逐行认证，而非按 ONNX 名称推测求解器列。原 q=Q*(1-eta)，Q>0，active=(1-b)/2。父输出归一化 u=(1-etaA)/2、v=(1-etaB)/2，故 0<=u<=alpha、0<=v<=beta。所有选定门必须 graph_error=0；非零误差 compact 门拒绝该候选，不宣布 UNSAT、不修原模型。stored-native preactivation 不等于已认证的原 ONNX/具体浮点网络，外部模型绑定仍未取得。

域元素的研究语义沿用 D078：原 HZ 加共同读出与联合关系，全部消费者必须由同一个 xi,b 解释。本次只实现这一语义的一次 upper 变换，不实现完整格、最佳抽象或跨任意深度精确闭包。输出保留全部原列和约束，追加连续辅助量，不增加、固定、删除或 pivot 原 bits。

## 一次共享变换

共享 delta=alpha*beta、zu=beta*u、zv=alpha*v，使用 D085 的12条 LE；三个新 unit-cube 因子分别表示 2*delta-1、2*zu-1、2*zv-1。定义 pi00=1-alpha-beta+delta、pi10=alpha-delta、pi01=beta-delta、pi11=delta；u10=u-zu、u11=zu，v01=v-zv、v11=zv，其余父幅值为零。

对每个原 child 提取完整 g,q，给定结构系数 a,b 后以精确有理运算定义 g=c+a*u+b*v+r。c 为差的完整常数，r 保留其余全部原连续及二元系数，不能丢弃非父项或假设它们独立。a,b 不匹配实际卷积权重不会制造虚假等式，差异全部进入 r；但可能使界失去效用。这不认证调用方结构来源。

令 M 为 r 的全 latent cube L1 范数。r=0 时不分配残余量；否则前三槽 r_s=M*xi_s，第四槽 r11=r-r00-r10-r01，添加八行 -M*pi_s<=r_s<=M*pi_s。不同 child 的残余槽独立分配且全部收费，不宣称保留它们在 fractional relaxation 中的全部联合源关系。

槽 s,t 的界为 L=c+min(a,0)*s+min(b,0)*t-M、U=c+max(a,0)*s+max(b,0)*t+M。U<=0 用 k=d=0；L>=0 用 k=1,d=0；其余用 k=U/(U-L)、d=-k*L。统一添加 q<=sum_s[k_s*(c*pi_s+a*u_s+b*v_s+r_s)+d_s*pi_s]。所有 child 共用同一 delta,zu,zv；本次不保存 child 的逐槽输出，因此不声称下一层已经消费这些条件输出。

## 投影与数值健全性

任意原整数可行赋值取上述真实乘积；pi 只有一个活动槽，r_s=pi_s*r。所有新因子在 [-1,1] 中，各槽 h_s=pi_s*g，原 q=ReLU(g)，所以全部新行成立，包括零预激活的两种合法原标签。故每个原状态都有同时扩展。反向因全部旧约束保留，新系统投影必在 H 中；投影回全部原连续、二元及输出坐标恰等于原 HZ。

编译任意精确行 A*z<=d 时，将每个系数转为有限 binary64 Ahat，精确累计 e=sum|Ahat-A|，RHS 取向上舍入的 d+e。全部 latent 在 [-1,1]，所以真实扩展仍满足存储行；输出舍入后辅助量不一定仍等于其理想乘积。不能在后续证明中把这些浮点辅助量无条件视为精确乘积。原坐标投影的双向包含仍成立，但不会提高原 exact=False 的资格。overflow、超过512位或支持限额等均拒绝，原对象保持不变。

decoder 仍读取原列前缀；不需要从新辅助量倒推输入，也不进行终端列投影。这个论证只说明原 decoder 的代数映射未变，不证明具体模型来源、输入形状或整个 decoder 实现已经通过本候选回放。

## 有限范围的严格控制

普通块为 x,y in [-1,1]，u=ReLU(x)、v=ReLU(y)，r=ReLU(u+v-3/4)、t=ReLU(u-v+1/4)。原父上下界为 [-1,1]，两 child 为 [-3/4,5/4]。原 signed phase LP relaxation 中，x=y=0、u=v=1/4、alpha=beta=1/2、r=5/16、t=5/8、child active=(1/4,1/2) 满足逐门原谓词。

本候选上行相加得到 r+t<=1/4+5*u/4+3*delta/8。结合 delta<=alpha，右侧为3/4，而旧点左侧15/16，排除间隙3/16，且不依赖如何选择新幅值。这是对原逐门连续松弛的定向严格控制，不运行 LP，不修改原整数域，也不是具体网络 ADV。已有联合约束可能同样排除它，不据此宣称优于先行工作。

正向扩展测试使用原生产 affine/ReLU 算子创建的小型真实 SparseHZono 对象，以 Fraction 核查完整原/新谓词；它们不是预训练网络或 CIFAR100/TinyImageNet 样例。宽残余控制加入原连续项和原上游 bit，检查全部残余被保留。浮点测试支付新系数的完整误差，不使用容差冒充精确。

## 完整成本与 GPU 边界

若 K 个 child 中 R 个残余非零，其残余总支撑为 S，则增加3+3R连续因子、12+8R+K条LE，EQ及bits数量不变。新增行的nnz上界为26+8K+31R+3S：共享父项26，每非零残余28+2*support，每child上行至多8或11+support。原行零填充和新CSR也要支付。返回 input/output buffer bytes、两者共存字节、新行缓冲字节以及 native nnz 是矩阵和向量计数；Fraction、Python对象、构造临时CSR、证据、终端转换、求解与decoder仍另计，尚未取得完整物理/工作量资格。

结构规则适合批量 gather、系数组合、四槽掩码与稀疏作用，但本实现是 CPU 可靠参考，不是 GPU 实现或加速成绩。GPU 迁移必须让同一语义、舍入证据和全部物理费用过门；不借本次 CPU 测试放宽未授权的 GPU AS 边界。本轮不执行 GPU、模型、LP/MILP、shadow或正式回放。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413与独立E0的61/400不变。上轮文献复核明确既有分组提升的先例，但没有可运行新候选；本轮用原生实现及一次冻结测试推进，而非再扩书单。依赖和baseline provenance由继承D081的完整manifest及本目录freeze认证。
