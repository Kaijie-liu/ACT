# Neural HZ 源相位闭包续接

本轮[定义与证明](definition_first_20260928/d162_source_phase_budget_audit_20261004/THEORY.md)给出了源相关中心、原相位和共同 L1 距离预算的 mixed affine、ReLU、live-skip 健全闭包；仍是纸面候选，未取得新域创新或能力资格。不要把它重启为孤立 helper 实现。

核心公式：对 rectified-L1 成本域，||y-c||_w<=R+sum_gate w_i M_i(1-beta_i)，M_i>=sup(-c_i)_+。完整消费者栈 K=[W;T] 的加权 L1 范数 kappa 给共同新半径；新中心保 source-affine，部分 ReLU 的 distance 与所有 identity skip 共用一份预算。原父 P、输入身份、全部 bits、实际 guards、同身份 skip 都保留。无需二次求解器，但历史宽度、中心填充和终端成本没有免费消失。

决定性反例：x,y∈[-1,1]，q=(ReLU(x+1/4),ReLU(y-1/4))，g=q1-q2+1/4 全局 crossing，紧界[-1/2,3/2]，live skip=z=q。x=y=0 时 q=(1/4,0)、g=1/2，闭包半径5/2却允许r=3/2，成本仅1。真实 J=ReLU(g)-g<=1/2，而假点J=1，使后继ReLU(J-3/4)错误允许1/4。旧一条active upper就排除；不是稳定门遗漏。损失来自把定向inactive位移改成无方向球。

两个容易混淆的投影：center-M形式只声明整数精确；原四行LP的完整消t形式须用rho>=c-q、rho>=q+L(1-beta)-c、rho>=0，见证t=clip(c,[q+L(1-beta),q])，它对分数beta也精确。若额外相交实际t box，inactive成本需要到[L,0]的距离。真实内部谓词/消费者不可遗漏。

LP包含定理有版本及量词限制：仅对L版旧四行LP的精确投影，新预算覆盖旧LP全部父状态时，放松原仿射t再消元不可能提升LP精度。仅覆盖native整数父域时不能套用；M版也不能套用。M版确有局部分离：三门共同父球sum|t_i-1/4|<=1，L=-3/4,U=5/4；旧LP允许t_i=1/4,beta_i=1/2,q_i=5/8，M=0的新预算却需总成本9/8>1。它是共同像/范数非扩张原理的正例，不是真实网络收益；不能被错误的L版定理排除，但也未修复上述crossing递归损失。

signed absolute-value transport 已被 D007 三角关系非负组合与 D061 覆盖，未实施。[支撑记录](definition_first_20260928/d162_source_phase_budget_audit_20261004/SMOOTH_ATTENTION_NOTES.md)另保动态query的sym(Q^T DeltaK)=0消项定理、GELU斜率平移势差公式，以及D130真实38/192部分Attention改善/零稳定门/超时的准确范围；这些都不是新能力或下一默认路线。

上一目标轮为no progress；本轮通过新闭包定理、强反例、编码/量词审查及封存归为progress。无候选/模型/GPU/测试/shadow/replay执行，无本轮后台作业。数学人口仍D158 4032项/212文件；未来实际候选须新预注册/源码冻结与原完整门，不复跑旧消费版本。

正式1870/2413、独立CIFAR25+Tiny36=61/400，两边新增0；13家族逐例保旧、非凸原bits、所有禁令、GPU、smooth/Transformer、新家族和满分目标全部不变，goal active。

下一实质动作是保留并使用相位位移方向的共同源表示，不再重复换范数或给消元加新框架。允许更换此候选；不要把负例扩大为所有Neural-HZ都不可能，也不要求有损域集合精度超越同信息的精确HZ。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新写入仅本隔离档案与续接文件；历史只读。来源身份见 ANCHOR_SHA256SUMS，档案身份见 ARCHIVE_SHA256SUMS。
