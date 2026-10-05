# 混合源相位组件的完整数学回放结果

本轮把D244的统一mixed关系落实到同一个非凸H中，完成冻结后唯一一次完整数学回放：4209项测试、224个文件全部通过。定理的强参照分离现有可复查的实际存储行证书，而不只是纸面算式。这是组件研究进展，不是实际模型提分、GPU完成或新抽象域创新性已确立。

上一轮只回答中断状态，属于no progress；本轮新增实现、证书及唯一运行证据，属于progress。Goal保持active，完整目标未缩减。

## 定义关系与精度证据

实现保持D243已封存H、连续来源、全部原signed二元相位、EQ/LE、共享frame及输入decoder。沿同一来源展开父预激活和子预激活，以固定的半径加权二乘二Gram恢复共同源系数；偏置、未选输出、误差及全部余项仍保留于r/e。不同展开停止策略使用独立memo，原图检查排除选中父输出依赖。

唯一mixed守卫同时处理父预激活x与ReLU输出q，不在失败后选择pure-X/pure-Q或调整tau，也不读取实例身份、property方向、LP状态。六个相位条件连续产品及29条关系共用原相位；原相位整数时，产品有唯一真实扩展。D244的全体整数投影保持证明继续是理论依据，机器检查不是用有限点替代该证明。它不宣称增加HZ可表示集合的类别。

固定声明图的shortcut和真实F输出均读取原A来源，没有为父预激活插入直接连线。实际13条LE的非负组合加3条原EQ消元，得到F<=258/125=2.064，真实点取等。原松弛可行物理点F=2587/1250=2.0696被全部辅助扩展共同排除，严格差7/1250。对应阈值413/200的后继预激活上界为-1/1000；这只是控制图的界，不是VNNCOMP CERT。

同一反例分数点具有完整单子门前缀的源与相位标签分解，也具有凸化父域上的联合真实子图分解；精确边界加非负斜率证书还覆盖原D240 pure-X的全部tau>=51/50及pure-Q的全部tau>=44/25。没有搜索tau。比较不涵盖原网络完整凸包、全部RLT或额外跨残差一致性，不能推广成超过所有已有验证方法。

另外检查固定27个完整整数点的扩展和原输入重构、全零处16组原相位标签、非零完整r/e、非对称归一化、共享BN误差、binary来源、原谓词与状态保持、秩/依赖/守卫拒绝及sticky资源失败。新组件没有调用数值求解器或验证helper；继承的既有固定LP控制仍按原登记完整执行。

## 完整费用与资格边界

主控制旧H有21列、8 EQ、58 LE、109 nnz；新关系保留全部旧行，增加6连续列、29关系LE、96 nnz。终端显式产品界另有12 LE及12 nnz，因此完整结果为27列、8 EQ、99 LE、217 nnz，增量41 LE、108 nnz，原4个二元相位不变。这不是压缩收益。

普通正例共用原Budget，实测累计work=569777、entries=230829。负fixture中为检查拒绝后健康性而建立的独立状态不计入这项普通正例累计数。限制仍为256M总work、200M单公共操作、64M累计逻辑entries及512位，未放宽。该逻辑账不等于全模型物理或GPU峰值。

mathematical_component_gate_passed=true仅授予本声明组件数学阶段资格。actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、complete_physical_qualification、gpu_computation_completed、new_domain_qualified、new_capability_qualified均为false。相同frame下多个Relation辅助列的全网合并也尚未取得资格。

## 唯一执行与可复现来源

源目录为definition_first_20260928/d245_mixed_source_component_20261005；唯一RUN为results/d245_mixed_source_component_20261005_v1。2026-10-05 12:24:30 UTC冻结六源及测试人口，冻结前未import、AST、compile、collect或试跑候选。主代理与三代理完成静态审阅；最终测试文件仅将预算记录键精确为ordinary_positive_fixtures_share_budget，反变换SHA与此前所审全文完全一致。

命令保持PREREG.md原样，工具会话69664终态exit0，没有重启或二次执行。freeze.json SHA256为9ea7e0c6f642688127ee8d6e2abf3dc6c29c0beaf614dc8a5f8596b08682cc95。源码执行后不再编辑。

完整测试为4189项既有测试加20项新测试；原223个文件及有序nodeids前缀保持不变，总224个文件4209项。JUnit为0 failure、0 error、0 skipped。pytest自身报告50.22秒、13条既有告警；包含导入、collection和JUnit的测试进程墙钟为51.39150989986956秒，低于60秒门。监督器整体65.98810685798526秒另计，不混称测试墙钟。

来源清单7752项，输入14项；source_drift=[]、input_drift=[]、provenance_drift=false。D24320项完整重跑，其summary重定向到新RUN/inherited_d243_controls，与原summary的SHA完全一致；其他历史writer同样只换新证据目的，不改原测试或旧文件。

独立只读验收还逐项核对实际JUnit与登记nodeids的完整顺序、旧7709项来源映射、全部6份冻结源及31份exit登记工件的实际SHA。旧19份继承工件与D243对应结果一致，14份输入逐个核验通过；没有仅凭exit中的通过标记作结论。

监督器traced_peak=22100505字节、tracer_metadata=7768880字节、预留65536字节；RSS高水位增量观测为0，不意味着RSS为0，也不代表pytest或全模型完整物理占用。原AS16GiB、CPU0、单线程、CUDA隐藏的数学阶段配置不变。

本轮post-run复核D243历史ARCHIVE的42项及D244历史ARCHIVE的25项均一致。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。没有修改生产默认、历史模型/结果、冻结版本或/data1/Kane/HyZor，没有commit/push。

## 下一步的真实适用性缺口

本轮修复的是数学规则与普通residual结构的错位，不证明三张实际网络已经命中。Gram允许没有父f直边时恢复共享源关联，mixed判据允许同时消费x和q；但Gram不保证余项的L1界缩小。未选父输出、BN误差、非反对称子系数及全部skip差项仍完整进入r/e。det>0、系数非零、守卫通过、实际输出界改善必须分别记录，不能混称成功。

D244已证的D243单层逐tap构造至少237895680 work超出200M单操作门，未被新Gram解决。生产路径遇未消费lazy H会丢弃该状态，不能绕过终端后报证明。直接重跑已知超账的完整构造，不是新的能力实验；改成更松界或只保留有利子块，也不能作为保留精度的证据。

下一真实阶段必须保持原三模型完整人口及共同预算，建立以下同一路径证据：

- 原模型/property字节、输入盒、原位方向及原输入decoder的完整身份链。
- 三个完整前缀、共同见证、共享BN逐scalar误差及全部后续消费者；large保留主支、Add8及Conv9/BN10，medium/Tiny保留主支、Conv8/BN9 shortcut、Add10及Conv11/BN12。
- 可靠全H界及事前冻结的结构配对人口，逐项输出Gram、tau、完整r/e、mixed守卫和拒绝原因，不按后验精度或标签挑选。
- 新关系进入同一个实际终端H，旧约束、原相位、其他消费者不删除；全链构造、查询、存储和设备成本有完整记录。

这些是实施缺口，不是新增的成绩晋级门。仅取得真实守卫命中可报适用性；实际后继读出或下一ReLU的可复现界改善才是进一步能力证据。随后仍需同结构shadow、逐家族、全2413及独立400、原四并发不回退门。GPU、smooth、Transformer与其他家族仍属于完整目标，不能用本CPU声明图阶段替代。

## 成绩与归档

正式baseline仍1870/2413=1063 CERT+807经验证ADV；独立E0仍CIFAR10025、TinyImageNet36，共61/400，不相加。formal_gain=independent_e0_gain=new_benchmark_solves=0。未动生产不等于已证明新候选全量保旧。

文档归档技能用于分开定义/精度证据、资源观测、实际模型缺口和正式成绩；沿用项目本地隔离目录，没有创建外部Page。ARCHIVE.sha256保存本轮源码、冻结文件、文档、RUN全部工件及关键历史引用。文本和哈希保存检查不是GPU、完整模型或论文新颖性认证。
