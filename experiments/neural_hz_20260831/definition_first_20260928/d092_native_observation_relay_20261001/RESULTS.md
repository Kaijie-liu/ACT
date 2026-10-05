# 跨层共同观察通过原生数学组件测试

本轮将D091共同规范实现和跨层条件观察落实到实际 SparseHZono；完整继承测试加四项新增控制，共3829项184文件，一次执行全部通过。它推进了可执行关系变换，不是仅优化存储；但尚未取得真实模型适用性、完整物理成本、GPU或正式能力资格，也不据此宣称完成新抽象域或新颖性认证。

## 一次执行的证据

唯一RUN为 results/d092_native_observation_relay_20261001_v1，监督入口 run_math.py --enabled。六文件先冻结，冻结前没有新候选AST、导入、编译、收集或数值执行。唯一执行session为9990，最终exit_code=0，监督器已保存exit.json并结束，无本轮后台测试继续运行。

完整有序人口在同一次pytest收集后、测试前认证；3829tests/184files，无失败、错误或跳过。测试子进程启动、导入、收集、执行和JUnit共49.46075773611665秒，满足原60秒门。监督器含前后身份核查和封存总计60.69334441609681秒，scope不同，不能将总时间混称测试时间。日志的48.16秒为pytest自身统计。13条pytest警告不改变通过结论，不隐去其日志。

source_drift=[]、input_drift=[]、provenance_drift=false。mathematical_component_gate_passed与component_tests_passed均为true，supervisor_exit与tests_exit均为0。监督器的host观察在原门内；此观察不是候选或整worker物理资格。完整继承D090原测试与来源人口、9inputs、4417GPU dependencies、1011decoder dependencies，D091理论额外身份绑定。旧D088和D090归档失败原样保留，不重跑。

## 本轮实际支持的结论

四项新控制分别核查：原H0共同扩展和零ReLU的全部原标签；相对D086加原第三门的严格排除；混号W/C及两级完整连续和原二元残余；默认关闭、坏绑定与资源拒绝、零尺度观察、数值buffer和nnz记录。

正控的第三门已在H0中，采用真实全局预激活界[-1/2,1]。旧分数点在完整H0及D086均可行。新观察前级行可精确存为binary64，其代数关系强制全部可行辅助扩展有Z00=1/8、Z10=Z01=0、Z11=5/8。下一门理想上界11/24排除旧w=1/2，精确间隙1/24；测试对所有单位盒辅助量扣除E+(installed_rhs-exact_rhs)，仍证明严格余量大于1/48。不是只挑选一个不利的辅助赋值，也不是新增具体网络ADV。

宽混权控制保留原连续和原binary残余，前级半径3/4、后级3/8，增加15连续列和47LE；全部原bits、EQ、LE前缀、输出、frame、exact=False及原输入列不变。原对象未修改。正控相对H0增加6连续列和23LE，其中相对D086新增3列、8观察行和1下一门行。这些是所测小块的实际组件结构，不表示真实宽CNN费用已过门。

## 未完成事项和下一步

结果中的actual_model_binding_qualified、source_component_qualified、source_census_qualified、native_HZ_admitted、complete_physical_qualification、gpu_computation_completed均为false。没有归档、预训练模型、终端LP/MILP、shadow、GPU或全量回放。candidate_physical_gate_evaluated=false，不以数组描述代替全部Python/Fraction/CSR临时、证据、host/device、终端和decoder账单。

下一步是为同一关系规则给出真实网络的统一消费者/后继绑定与完整成本预注册，然后执行真实同结构检查。须覆盖所有真实读出及残余，不能挑实例或删除难组；不先回到泛书单、任意提高冻结预算或重跑D090。GPU仍需独立可靠编译及原资源边界下的资格。本冻结版本已消费，不再改动或重跑；复用需要绑定本源码与本次完整证据。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413（1063CERT+807validatedADV）及独立CIFAR10025、TinyImageNet36共61/400不变，formal_gain=0，Goal active。生产tracked差异仍为9文件3806insertions57deletions，历史只读，未commit/push。文档技能用于区分数学通过、原生组件控制和尚未取得的真实模型及GPU资格。
