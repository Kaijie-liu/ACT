# 强 Neural HZ 主线与共享池化研究续接

用户再次强调：重点是从 HZ 数学定义提出强大的非凸 Neural-HZ，不是只做算子、存储、构造优化或辅助算法。现有 goal 已明确这一点，不需要再次删除重建。保持 active，不把本轮支撑定理晋级成定义创新。

本轮[共同投影证明](definition_first_20260928/d165_shared_pool_projection_20261004/THEORY.md)处理 ReLU→MaxPool 的共享叶子。固定保留变量后 q_i 的共同可行区间为 [max(0,g_i,max_j(p_j-D_ji)), min(u_i beta_i,g_i-l_i(1-beta_i),min_j p_j)]。枚举所有上下界比较给出精确投影，包括不可省的跨窗口行 p_j-D_ji<=p_k。取左端点重构同一份共享 q，保留原 beta、原 one-hot delta、所有零标签和并列选择。定理覆盖分数 LP，故它保存而非增强旧 LP 强度；额外 q 旁路或谓词必须共同处理，不能直接删列。

费用：E=sum d_i，X=sum d_i(d_i-1)，S=sum nnz(g_i)，Sd=sum d_i nnz(g_i)。旧幅值 m+P、bits m+E、行 4m+2E+P、nnz 2S+6m+6E；新幅值 P、同 bits、行 2m+3E+X+2P、nnz 2S+2Sd+2m+P+8E+3X。重叠会增行，内联会增 nnz；单窗口普通两源例从 5 连续/4 bits/13 行/32 nnz 变成 3/4/12/37。没有实现净优势或速度证据。值恒等式 max ReLU=ReLU max 还会改变原并列标签，不能作为无条件替换。

已有 D123 唯一叶子投影及外部 max-of-affine 强公式不能被重新命名成创新。当前 MaxPool 代码确有清空 HZ 并退回区间的缺口，但冻结未解清单仅两个 cGAN 条目有 MaxPool，且没有邻接/窗口/消费者证据。不得因此启动偏离主线的完整池化工程，或声称 CIFAR/Tiny 受益。定理保留供未来真正的新域算子复用。

下一候选的研究对象应是域元素、具体化及跨非线性共同关系的可组合表达，要求相对明确强参照的能力或完整成本优势；不要反复重启范数换名、精确旧图包装或局部消元。普通终端 LP/MILP 和具体见证验证不是被禁 helper，但不得借其他求解算法计成域收益。

本轮仅纸面证明、只读源码/元数据/论文审查和新文档。无导入、数值、模型、GPU、shadow、replay 或后台作业；最后执行人口仍 D158 的 4032 项/212 文件，未来必须重新预注册并冻结新版本，保持完整原门。正式 1870/2413=1063 CERT+807 validated ADV，独立 E0=25 CIFAR+36 Tiny=61/400，两边新增 0；本轮未做保旧回放。

日期 2026-10-04 Australia/Sydney，分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只写本次隔离档案和续接文件；[研究记录](definition_first_20260928/d165_shared_pool_projection_20261004/RESEARCH_RECORD.md)及其哈希清单给出来源和边界。
