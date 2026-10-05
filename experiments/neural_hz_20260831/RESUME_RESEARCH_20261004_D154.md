# 继续定义优先 Neural-HZ：不能把幅度共享当成能力突破

用户重点仍是提出强大的非凸 Neural-HZ，不是 helper、存储或构造优化。当前
Goal 已包含此边界，无需改弱目标。正式1870/2413与独立61/400均无新增；
formal_gain=0，goal active。

最新 [D154理论](definition_first_20260928/d154_ordered_shared_phase_20261004/THEORY.md)
及 [研究记录](definition_first_20260928/d154_ordered_shared_phase_20261004/RESEARCH_RECORD.md)
保存了 ordered bank 的精确中位相位幅度共享、完整消费者列差秩付款条件及
更强分数反例。不要把它说成已获资格的新域或重跑旧反例当新成绩。

核心正面式：m=2r，f_L>=f_R且全组有序，rho=原beta_r，
t_i=max(0,-f_i,f_(r+i))，q_L=f_L+(1-rho)t，q_R=rho t。
弱顺序也保全零点所有原标签，不强加bit-prefix。
完整消费者D=C_R-C_L，精确D=L R秩d时z=rho R t仅需d产品，总幅度r+d。
d<r才能保留变量优势；全部原q的identity consumer使d=r，raw source skip不自动如此。
R内联产品只有两行重复系数，成本2nnz(R)，不是4nnz(R)。
mask minimax rank=ceil(m/2)；实际维数下界须独立源chart，不能套一维链。

六行max整数精确但LP可能弱；m4混权C=(2,-1,1,3)有3幅度/16结构行/40nnz，
旧4/16/32，额外旧relaxed sign guards另8行16nnz。全部源/界/readout另计。
非平行4门/5源控制给新Y53/8、旧同标签<=45/8，差1。

更关键：f=(x+3b,x+b,x-b,x-3b)，b=1/4+y/20，源[-1,1]^2。
A=(x1/5,y0,beta1100,t00)，B=(x-3/10,y0,beta1000,t=(0,1/20))，
半半平均属于完整同源全部标签max图的joint hull；所得rho1/2,t=(0,1/40),v1/10。
即使给tight条件界rho0:v[0,14/5]、rho1:v[-3/5,4/5]，独立product仍容z1/10，
Y51/40>旧同源同bits上界6/5，差3/40。不是global CERT/实际回归/ADV。
all-bit关系可排这个点，但未证明全修；不能改成helper补丁宣称新域。

决定：不实现当前独立max+product lowering。不重复元数据普查，不扩测试框架。
下一需要有联合源/相位/值的消去或跨层变换定理，同时解释原终端的强度和
完整费用；尚未选定新执行候选。不能仅把旧D014/D018/D137/D143重新命名。
GPU及smooth/Transformer目标仍在，尚无对应突破。

本轮只有纸面、只读检查、primary摘要及新隔离文档，无数值尝试/测试/模型执行，
无生产修改或后台实验。D152已消费run不改不重跑；D150数学人口4000/210不减。
分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5保持。
