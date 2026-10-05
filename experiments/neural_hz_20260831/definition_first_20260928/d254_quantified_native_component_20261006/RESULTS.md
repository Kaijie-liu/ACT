# 定量守卫原生组件验证结果

统一定量守卫组件完成一次完整回放：4241项测试、226个文件，零失败、零错误、零跳过。新公式在旧guard外的构造样例中给出实际存储行证书，并由原普通终端消费；尚无真实模型、GPU或正式提分资格。

## 数学关系及执行结果

实现保留全部原连续因子、四个原二元位、EQ/LE、输出、skip、共享frame和五维输入decoder。guard内29条精确行、存储行和整份native状态与旧规则逐行一致。guard外不选另一算法，而是同一公式计算四个相位条件超额项。27个固定整数点、16个全零合法标签及非对称上下尾的延拓检查通过。这些点不是新增ADV。

固定guard外结构取a=7/8、b=1/16、epsilon=3/32、tau=1，完整异相位界为[-31/32,33/32]，eta_minus=0、eta_plus=1/32。实际存储行的非负组合及原EQ消元证明F<=265/128，即2.0703125；全部辅助列在证书中消去。两个明确的强参照允许完整带来源/标签的假点F=8509/4096，严格差29/4096。阈值1061/512下，新证书给负margin −1/512、参照假点给正margin 21/4096，因此可证明该构造的后继ReLU为零。

两次固定普通LP对照观测到旧上界2.75、新上界2.05859375（527/256）。新数值恰等于一个已核原整数点的值，但本轮精确上界证书仍为265/128，未证明该较强数值界的精确紧性；不把浮点求解输出升级为最优性证明。这个结构与D249的主fixture参数不同，不用相同数值冒充跨实验统一成绩。

完整单child前缀以及共同child对凸父hull的见证均带原输入和原标签。没有声称超过完整跨层RLT、完整四门原源hull或所有重调tau的hard-mixed规则。候选自身不调用solver、bound generator或第二验证器；测试的两次普通终端LP不是救援路径。

## 成本及保护边界

主fixture新增六个连续因子、29条LE、104个实际native非零项；整份谓词有4 EQ、37 LE、150 nnz，全部四个旧binary保留。双关系整批、高水位、旧谓词和decoder检查通过；失败不发布半批。非平行来源产生25/48的Gram系数，逐项系数误差及向外RHS补偿检查通过。未支持的门误差、秩、来源依赖、身份篡改及资源失败均拒绝，没有更换终端路径。

普通正控共用Budget最终work=2270044、entries=713216，未提高256M/200M/40M evidence/64M entries/512-bit边界。CPU0单线程，CUDA隐藏。完整pytest进程观测52.133650秒、日志51.02秒，低于原60秒；监督器总66.743476秒不在该pytest时限口径内。监督器traced peak为22115450字节、tracer metadata7751472字节、RSS高水位增量0，均不代表pytest完整物理峰值、GPU或全模型资源资格。

## 执行与历史隔离

唯一RUN为experiments/neural_hz_20260831/results/d254_quantified_native_component_20261006_v1，会话69002退出0。完整继承D249有序4225项，追加同一16项；执行后独立读回JUnit、inventory和manifest，确认全部4241个实际case、226个文件及旧有序人口不变。来源身份7841项、输入14项；33份运行工件哈希重新核对全部一致。执行后source_drift/input_drift为空、provenance_drift=false。

前序D253因D249 writer定义行误登记46而非44，在收集门被拒，零测试运行。该失败源码、日志及RESULTS/ARCHIVE完整保留；本次在新目录修正两处登记，数学核心逐字不变，16项测试仅更新import/RUN/schema身份。没有对已冻结版本原地修补重跑，没有降低人口、预算或绕过身份核查。

freeze时刻2026-10-05 16:26:50 UTC（悉尼2026-10-06 03:26:50）。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

freeze SHA256为7abf8af75b5538d581482366cabae9fc571ef18408d0e9f26c89ceeaf326782a；exit为b8ba7e0af59e3c69184b873a7b3b7de45382deb7b42b71c3497d0f086652d9de；summary为6de7513add996d2a858f3cc04e387440533f48faa3d23fe8a12279db6059a9cf。

## 尚未取得的资格

quantified_native_transport_passed=true，仅表示这次原生数学组件运输通过。new_domain_qualified、native_HZ_admitted、实际模型/生产在线/GPU/完整物理资格仍全false；domain_definition_changed=false。不转授D249旧native_mathematical_transport_passed标志。

固定三来源、同结构shadow、逐家族、全部2413、独立400及四并发不回退均尚未执行。正式1870/2413（1063 CERT+807 validated ADV）和独立CIFAR10025、TinyImageNet36，共61/400不变，三个gain计数均为0。

下一实质接入问题见NEXT_INTEGRATION.md：生产重基底可能使同一幅值关系在新坐标中被当前提取器遗漏。该源码反例及修复合同不是额外测试通过、新域创新或真实网络收益。
