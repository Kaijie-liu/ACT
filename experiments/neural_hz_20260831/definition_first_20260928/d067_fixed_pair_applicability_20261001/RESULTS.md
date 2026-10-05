# 真实归档中没有双未定固定锚对

完整适用性审计通过，但得到的是一项局部负结论：这份CIFAR100 large归档的1440个固定配对中，没有两个原相位都未确定的真实配对。因此，在双方均加入已证明稳定相位事实的比较里，不能用这份归档展示共享delta相对独立delta的额外收益。原计划的D066宽层关系表运行器不据此继续搭建。

这不是整个D066、共同源支持或Neural HZ的否定，也不是原native LP已被证明冗余。四状态表本身相对旧标量界仍可能有收紧；本轮没有计算这些表，没有新增CERT/ADV。

## 实际完成的检查

唯一命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d067_fixed_pair_applicability_20261001/audit.py --enabled`。会话21941返回exit_code=0，运行6.736291266977787秒，无本次遗留后台进程。源码与文档在首次执行前冻结，运行后没有修改、重跑或增加预算。

重新计算全部3072输入盒坐标上的1600源形式，涉及34752个系数项。每个源的上下界均与归档完全相同；源状态为727严格正、863严格负、10严格跨零，触零与精确零均为0。这只验证归档形式与界的一致性，不代替从原模型重新提取形式或原网络forward验证。

全部五窗口、576原slot、64消费者和原global-even配对都已核对，包含旧pair_keys逐项相等、padding身份和全部源覆盖。

| 固定配对分类 | 对数 | 对应消费者映射数 |
| --- | ---: | ---: |
| 两个相位均已稳定 | 535 | 34240 |
| 只有一个相位未确定 | 9 | 576 |
| 两个相位均未确定 | 0 | 0 |
| 一侧padding | 512 | 32768 |
| 两侧padding | 384 | 24576 |
| 全部 | 1440 | 92160 |

544个双真实锚对全部满足至少一位稳定的充分条件。10个跨零源中有一个与padding配对，其余9个各与稳定源配对；没有过滤padding后重新配对。92160是完整消费者映射数量，不是已计算的关系表数量，kernel_tables_evaluated=0。

根代理在运行后从完整source_records与pair_records独立重聚合状态、五窗口分类、1600个不同源ID和92160映射总数，均与诊断一致。没有用旧summary代替此次逐源重算。

## 为什么改变下一行动

[THEORY](THEORY.md)证明：在允许未固定bit取[0,1]的表系统中，alpha=0使每个delta_j=0，alpha=1使每个delta_j=beta。因此任一锚稳定时，独立辅助量已经唯一相等，共享不会再改变投影可行集。原非凸域的整数bits仍保留，未被删除或连续替换。

本次所有真实固定对都落在这个范围内。继续为该归档支付完整D066 prepare、condition、clamp、编码及typed对象账本，不能产生原先要寻找的共享delta优势证据。因此关闭的是“在当前完整归档及当前固定配对上证明这一额外收益”的实验路线，而不是关闭项目、其他结构或原表的全部价值。

没有测量D066核表成本。静态审查提出的高费用上界也不是已证明的200M预算下界，不以它虚报候选超时或资源失败。此次停止该路线的根据是完整结构证据及上述等价定理。

## 资源与证据

branch work为59104571/200M；whole work为99170107/256M，其中包含预付40M证据额度与65536摘要reserve。实际证据账为17519053/40M。

RSS高水位341725184字节，增长321232896字节；tracemalloc峰值136083713字节、metadata69278992字节，按预注册加reserve后通过主机门。持有对象账本为65586361字节及505044个数值出现项；含保守parser瞬态reserve的条目上界24633144/64M，不等于实际native矩阵条目或全系统物理内存。

完整证据[complete.json](../../results/d067_fixed_pair_applicability_20261001_v1/complete.json)为856320字节，SHA256 f8e54eb8a0d18a8953b72b9a702fa1eefae61dfb775ecd216474e9334eacdcff。[diagnostic.json](../../results/d067_fixed_pair_applicability_20261001_v1/diagnostic.json)记录完整状态与费用，SHA256 f000fc5dc96ea48e575d484514b2af4d892fcce88667a796551c232ee670c4eb。两者已经根代理重新计算哈希核对。

三个新源码文件、五个直接证据身份及freeze在运行前后核对一致，source_drift、input_drift和identity_unchecked_paths均为空。此诊断没有候选、模型、GPU或求解器模块导入，不声称完成旧完整依赖树或新的候选兼容性门。

## 后续研究与不变的资格

下一研究应面向真实非线性后继所需的相位内部共同源关系，明确哪类观察在下一ReLU或残差合流中仍可复用。不能仅把此处两个未定门重新拼在一起、挑一个阳性坐标，或再增加同类fixture来声称定义突破。任何新的结构规则、适配器和真实人口都必须另行说明语义、全成本与预注册，保留完整旧人口及其负结论，不重启旧失败版。

已有D066的3809项/179文件数学资格保持原范围；本轮没有重跑这些测试，也没有授予拟议3813项adapter组合资格。D025原三模型普查、D047与D057的失败状态原样保留，TinyImageNet没有新增完整来源证据。

applicability_audit_qualified=true；actual_model_binding_qualified、actual_phase_column_binding_verified、native_nonredundancy_verified、source_census_qualified、gpu_computation_completed、complete_physical_qualification、candidate_admitted均false。

正式成绩仍为1870/2413（1063 CERT+807 validated ADV）；独立E0为CIFAR100 25、TinyImageNet 36，共61/400，formal_gain=0。没有执行shadow、逐家族或全量回放，不能以这次诊断宣称新候选已经保旧。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。仅新增隔离文档、脚本和唯一RUN；原九项tracked修改仍为3806 insertions、57 deletions，没有改历史数据、生产默认、Goal或远端。write-page技能用于分开结构结论、实际执行和未证收益，文档已读回，无外部Page发布。本轮属于progress：取得改变下一实验选择的完整真实结构负证据；整体Goal仍active。
