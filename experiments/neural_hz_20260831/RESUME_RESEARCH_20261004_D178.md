# Neural-HZ 源观测候选恢复入口

完整 Goal 继续 active。正式 1870/2413、独立 E0 61/400 均新增 0；全部原非凸身份、保旧、资源、fail-closed、禁止 helper 和全量晋级边界不变。

本轮已有具体候选而非仅重述目标：[定义与证明](definition_first_20260928/d178_reference_observed_fiber_20261004/THEORY.md)。C 为完整消费者加 mass，tau 为既有结构参考，H=C D_tau；共同 proxy 满足 Ct=Cd、Ht=Hd、||t||²<=E。只保存 delta=C(D_beta-D_tau)t 这 p 个新幅值，不物化 masked-H 输出。读出为 C D_tau g+C(D_beta-D_tau)b+delta。

已纸面证明 whole-parent soundness、严格细化且仍有损，以及 ||M delta||²<=Emax sum_i||MC_i||²|beta_i-tau_i| 的一次相位预算。全部历史 bank 在各自参考相位时误差归零；从已失真的旧父域接入不能恢复旧损失。五门混源、mixed consumer、原源 skip 和下一 ReLU 控制已独立复核，但仅为相位切片正控，旧四行也能证明；没有强参照或正式能力胜利。

[真实结构落点](definition_first_20260928/d178_reference_observed_fiber_20261004/REAL_FRONTIER.md)来自保存的 D152 全图：large Relu53、medium/Tiny Relu51 经 Conv/BN/Add/Flatten/Gemm 才到下一 ReLU。linear1.weight 形状分别为 [100,4096]、[100,2048]、[200,6272]；原父 skip 不是新 q 的 identity 消费。这个接口与旧 final-ReLU 方阵不同，值得下一阶段定向资格检查。尚未解码属性/系数/中间形状，不能声称 q 宽度、B 秩或实际节省已经认证。

下一项工作应直接针对该普通完整仿射接口，先完成新隔离预注册下的语义与完整成本检查，并与同信息旧路径比较；不要再为插值/共轭/差分旁线扩建框架，也不要只做矩阵合成而忘掉域定义。新数值/代码活动仍须先冻结。候选尚未实现或默认启用，新颖性、GPU 和全部真实验证门仍待证。

本轮 paper-only 加只读旧 JSON 证据检查，无新后台实验。最后组件 D158、最后系数诊断 D172 不重跑。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。仅新增隔离文档，历史和生产原状。[研究记录](definition_first_20260928/d178_reference_observed_fiber_20261004/RESEARCH_RECORD.md)保存来源与未过门范围。
