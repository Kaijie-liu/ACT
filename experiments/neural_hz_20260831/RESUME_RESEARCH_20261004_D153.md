# 续接源相位联合生成的 Neural-HZ 定义研究

最新 [D153 理论](definition_first_20260928/d153_phase_source_loading_20261004/THEORY.md) 与 [研究记录](definition_first_20260928/d153_phase_source_loading_20261004/RESEARCH_RECORD.md) 已归档。没有新域能力突破，formal_gain=0。用户重点仍是从 HZ 定义研发强大的非凸 Neural-HZ，不是 helper、存储或审计框架。

本轮新必要条件：在同一父仿射 chart 内、anchors inactive、n个未锚预激活独立且全部相位可达时，q=B_beta*t+c_beta+E_beta*eta 若源加载变化的所有像空间包含于共同t0维K，则统一误差宽度r必须满足r+t0>=n。固定B即使E_beta任意相位相关，也需某相位r>=n。证明用有限差分与商映射后的多重仿射行列式，不需要eta可微、不执行相位枚举。t0不是max单相位rank，也不是乘法次数；真实模型前提未认证。

普通三门有偏置混权A、可逆后继W、raw source skip和下一ReLU的控制已逐式核实；源全模式见证||x||∞<=7/18，后继两端分别<=-1/90和>=7/1440。因此不是死门/重复门特例。

另一关键反例：g1=x-y/4+3/4、g2=x+y/4+3/4，真实参考3/4。源(-11/16,0)、原相位11、假q=(23/32,23/32)通过D150范数及每个新增参考能量。旧LP对J=q1-q2/4-4x/5+y/5证J<=1，而假点J=697/640，能接下一原ReLU(J-21/20)=5/128。不能为“加真实参考球就修好了D151”启动实现。有限非零参考点一般不恢复exact；零参考则回到已知互补性。一般共同球有曲边，plain LP无损终端缺口仍在。

grounded图能量即使M正定、一个原门精确仍允许另门膨胀，具体二门例子和primary QC冗余比较已保存。不要将QC文献的copositive/SDP强参照偷换成普通四行LP，也不引入求解helper。

下一交付必须改变源相位联合生成或共同非线性读出并提出新的可组合规则，证明普通混权/残差/下一非线性及完整原终端费用。单纯E_beta、phase中心、finite-anchor、grounding补丁不推进；D001精确图、D137径向clique、D143聚合product已经有边界，勿重新换名。当前没有已选定的新实现候选，不以重复图审计/文献复述代替构造。

本轮仅纸面与只读检查；无代码候选、数值执行或后台任务。上一回合与本回合均为progress，不是wait/block。D152冻结run不改、不重跑，D150数学人口4000/210保留。正式1870/2413与独立61/400均未新增；GPU、smooth/Transformer、全家族能力仍未完成。goal active。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5不变。
