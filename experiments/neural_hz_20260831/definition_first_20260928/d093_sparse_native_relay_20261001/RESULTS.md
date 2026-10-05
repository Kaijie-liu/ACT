# 自动原生跨层绑定通过完整数学测试

D093已完成默认关闭的原始谓词自动绑定与规范稀疏观察接口，并一次通过3833项测试、185文件。该结果使D092关系可以由原生门结构统一生成，不再要求调用者手填观察权重；但没有执行预训练网络、GPU或全量回放，正式收益仍为0。

## 执行证据

七文件先冻结于freeze.json，冻结前仅文本和来源检查，没有候选AST、import、compile、collect或数值执行。唯一入口run_math.py --enabled，唯一RUN为results/d093_sparse_native_relay_20261001_v1，session22146已确认exit_code=0。没有本轮后台候选继续运行，消费版本不修改、不重跑。

完整继承D092的3829项184文件，加四项新控制，在同一pytest进程中收集并在测试前认证全部有序nodeids。3833项全部通过，无失败、错误或跳过，13条警告保留在日志。测试子进程启动、导入、收集、执行和JUnit共48.326484901830554秒；pytest自身报告46.94秒。监督器含前后身份核查及封存总计59.341201255097985秒，三种时间不混称。

退出回执的component_tests_passed、mathematical_component_gate_passed、inventory_validated_before_execution和all_stages_passed均为true，tests_exit和supervisor_exit均为0；source_drift=[]、input_drift=[]、provenance_drift=false。监督器的traced peak为14,212,040字节、tracer元数据4,709,648字节、RSS高水位增长0，仅属监督器观察，不能代替候选完整物理资格。

## 已通过的控制

稀疏接口与D092補零后的稠密接口在普通结构、混号观察与完整连续及二元余项、空W行、空C行上，理想行、舍入误差、安装RHS及完整HZ数组一致。原坐标、全部原相位、EQ与LE、输出、frame、exact=False及零ReLU标签保留。D092三门严格排除控制的相同行因此得到保留，不是只检查单个辅助赋值。

64个不同的原始后继门由完整D088目录自动发现。每个后继产生自己的观察与约束，没有缩成一个代表门：新增195个连续辅助量和590条LE，保留全部原始二元相位。稀疏参数为196个系数、192个索引，对应稠密接口的逻辑系数人口4228。这里比较的是接口人口，绝不是16倍端到端加速或完整内存节省。

自动绑定在普通子门物理尺度Q=5/8时正确恢复W=(1,1)，没有误用父门归一化系数公式；混号结构恢复W=(9/8,−1)，保留前级余项半径3/4和后级余项半径3/8。planner到内核的结果与独立手写稠密公式对照一致，原状态存在共同扩展。

默认关闭不读取毒输入。非规范索引、显式零、错误系数、坏门、局部资源超限、错误owner或seal、非有限谓词均拒绝。测试还在H0中构造了两个实际认证门共享eta但使用不同原相位的情形，planner拒绝整个计划；无后继关系返回空计划，不计收益。

## 资格限制和下一步

actual_model_binding_qualified、source_component_qualified、source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、complete_physical_qualification和gpu_computation_completed仍为false。candidate_physical_gate_evaluated=false，worker_launched=false。小型原生结构上的自动绑定不自动获得预训练网络、完整费用或在线分配器资格；全人口行列上界不是完整执行收费。

下一步是新的真实同结构预注册：使用全来源认证、完整存活根与构造收费的闭合H0，测量全部适用组及其传播成本，再谈shadow和正式回放。不可重新启动D090或消费过的D093；不可直接恢复缺失allocator状态的旧snapshot继续forward。更大的layer20归档的读取预算缺口见同日[静态来源补充](../../research_handoff_20261001_native_binding/README.md)，不能把换大归档或局部稀疏人口下降当作这个缺口已解决。

本候选复用既有共同实现和有效关系数学，未取得独立新颖性认证，不因稀疏接口或自动发现就宣称完成Neural HZ新域。GPU执行、真实覆盖率和完整端到端保旧增益仍需独立证据。Goal保持active。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413（1063 CERT、807 validated ADV）和独立CIFAR100 25、TinyImageNet 36共61/400不变。tracked差异仍为9文件、3806 insertions、57 deletions，未改生产或历史档案，未commit或push。write-page文档技能用于分开数学通过、局部人口变化、实际执行和未取得的资格。
