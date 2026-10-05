# 真实残差块上的共同观察接入检查

状态：静态否决，未冻结、未执行。本文件保留当时拟议合同，不是当前运行授权。条件性最低收费266477776已超过256M，详见 [静态结果](RESULTS.md)；不得修补草稿后沿用本版本或启动其 runner。

本候选从原始 TinyImageNet 模型与性质重新生成当前 HZ，在首个包含三组 ReLU 和残差 Add 的完整块末尾，尝试 D098 的全部原生关系认证和共同观察安装。目的不是增加小型测试或再优化存储，而是检验已经证明的关系语言在普通真实网络中的适用人口、后继消费和完整接入成本。成功和失败均保存；任何中间层结果的 formal_gain 都为 0。

## 研究问题与不变语义

直接复用已冻结 D098 数学实现，不改变原 HZ 生产出生、共享 frame、变量分配器、deferred 物化或合流。实际关系为 g-q=L(s+b)，q=Q(1-eta)，L<0<Q，并带原两行相位守卫；在原整数状态上 q=ReLU(g)，两种合法零相位均保留。共同观察必须对应同一个原状态，而不是让不同消费者选择不同输入。

统一规则读取完整当前谓词，认证所有可识别的原门，固定列序选择父对，保留完整余项，并检查全部源组和所有后继。不得按实例、模型、家族、标签、margin 或求解状态选择有利组。预算失败不返回成功前缀；未知和歧义不是 SAFE，也不删除任何原二元因子。这里研究的依赖是实际 eta 关联，不预先宣称为已认证的 ONNX 时序边。

需要由新结果回答：实际有多少门可认证，多少源组真正被后继共同读取，全组安装是否在原门内可行。如果只获得部分前缀、未完成普查或资源拒绝，明确报告其停止阶段，不把合成样例收益外推为真实网络收益。即使安装成功，也不能仅凭它宣称新抽象域已完成或达到 PLDI 新颖性要求。

## 固定来源与完整结构

模型为 `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/onnx/TinyImageNet_resnet_medium.onnx`，SHA256 `234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776`。性质为实验树中 `vnnlib_v2_full_v1/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib`，SHA256 `d105a0c7ca711eb46b9f20ab772a0564444569d06f6206bca0e95d5a19990cca`。输入形状 1×3×56×56，原模型 float64；这些标识只固定实验来源，不进入关系选择规则。

从 INPUT 新鲜转换和正向传播，采用已归档的 corrected BatchNorm producer graph，核对其既有 graph-faithfulness 锚点。完整保留 ACT ReLU5、ReLU9、ReLU20 及 Add16 的两条来源：ReLU9 经 Conv10 的主支和 ReLU5 经 Conv13 的旁路，再经 Conv17 到 ReLU20。不得用旧 pickle、输入盒重构的中间层或不完整 deferred 状态代替当前对象；不得调用会隐式配置旧 hooks 或写大快照的旧 main。

本轮执行前查明一个重要来源差异：旧 `c5_corrected_prefix_20260905_v2/events.jsonl` 在2.279秒记录 layer17开始，之后达到240秒上限，没有完成该卷积；旧 integrated 前缀的约6秒传播依赖 C5 显式构造支撑。不能把后者误作未启用 C5 的性能证据，更不能重复同一已知超时路径。

因此在冻结前固定复用原 `c5_runtime_materializer_v2.installed(enabled=True)`，仅包围此次从 INPUT 到 ReLU20 的新鲜传播，不调用其旧 main。其结构选择只看完整两Conv及对角仿射岛，不读实例标签；已选中岛保持原跨请求256M产品池、逐分支200M、quarter gate和64M边界，失败不回落另一路径。先核对原 live evidence 的 SHA256 `5b2af877318515dd7daa06eb0c79181d13c3dd71a351c6cb4b2cbeddc91fde94`、通过字段及源身份；3845项继承测试已包含该 runtime v2、budgeted/ordered compiler 的相关测试。新的组合仍须自行检查，旧资格不直接移植。记录全部 C5 阶段和费用，退出 context 并验证原函数恢复后才进行 D098 检查。这是保留旧构造成果的显式复用，不是 Neural-HZ 新增收益。

上述旧 JSON 实际为194038279字节，包含大量历史存储路径；本轮只读核实其 `live_registered_gates_passed=true`、`input_roots_unchanged=true`，不把另两个 `relu_executed=false`、`cache_publication_executed=false` 隐去。D099 数学与 source 监督器对该精确 SHA 的全文件身份前后认证，worker 认证本次已封存 manifest/exit 后引用这个固定历史事实，不重复解码旧存储人口，也不把其当作本次数值输入。这符合旧预注册“live evidence and source hashes still verify”的边界；不是给任意外部哈希授予资格。监督器实际读取字节和耗时单列，不能用此分层宣称完整请求免费或全物理通过。当前 C5 代码、全部当前根及新 bank 仍按新调用实际检查。

本次在 ReLU20 闭合。新连续列尚未发布给生产 allocator，故不把安装后的对象塞回缓存继续出生；不运行终端求解或给出 CERT/ADV。在线生命周期、完整具体输入重构和后续全模型传播仍是独立未完成义务。

## 资源统计和资格边界

保留 CPU1、单线程、CUDA 不可见、AS16GiB、worker 240 秒及内部 235 秒收尾闹钟。主机观察保留原 RSS、增长和 tracemalloc 加元数据的 1GiB 边界。监督器身份读取及封存另列，不能把它们说成免费。

新诊断采用一个 whole256M 池和嵌套 branch200M，覆盖本次身份读取、元数据、当前完整 root 审计、D098 的完整目录、全部组和证据；不按组重置或拆分池。evidence40M、numeric 与已证临时合计64M、512bit 有理数以及原局部65536限制保持。费用须在相应操作前支付，不能在先建巨大 bank 后才拒绝。

旧 C5 的256M约束原本针对 selected affine island，不能追溯改成全模型生产算术门，也不与 D099 新 bank 池相互借款。模型解析、转换及带明确登记 C5 支撑的正向构造在此次 worker 时间、AS、RSS、trace 窗口内完整记录，但没有现成的全过程标量工作证明或全部构造临时人口证明。因此 `production_scalar_work_qualified` 和 `complete_physical_qualification` 不能因当前对象账本通过而置真。本次是实际来源诊断，不降低正式资源晋级门。

当前强可达根包括 model/wrapped、ACT net、tf 全字段、before/after、所有 HZ/expression/precomputed/allocator 缓存、输入与 decoder，以及 bank、精确行、回执、最终对象和仍存活的工作数组。数值 owner 去重不删除其所有者；模型不能仅以 state_dict 或一个 HZ buffer 代替。未知对象 schema 或无证明的分配界限 fail closed。生产构造阶段未知临时成本、具体输入解码资格和 GPU 均继续独立标为未完成。

## 晋级与归档

候选默认关闭，仅显式启用。先按新冻结版本完整运行继承的3845项数学测试、188文件，不新增无关测试；通过后才允许本次唯一真实 worker。完整性、数学成功、当前结构适用性、安装完成、物理资格、速度、新颖性和正式成绩分别记录。

正式1870/2413（1063 CERT、807 validated ADV）和外部61/400（CIFAR10025、TinyImageNet36）不变。不运行 attack/PGD、BaB、输入/相位 split、backward/dual rescue，不触碰普通终端决策或独立具体见证验证的原边界。新旧结果不混合。GPU 初始化的旧失败不重跑、不绕权限或提高 AS；当前 Fraction/SciPy 实现不能称为 GPU 域实现。

所有新文件仅位于本新隔离目录；原拟唯一 RUN 未创建。生产、历史资料和冻结代码只读，不 commit/push。分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。上一目标研究轮为 progress：D098 完成一次3845项数学执行并封存；随后的用户归档问答仅核验已有证据，没有新研究结论。本轮取得明确的静态不可执行原因，排除 D098 接入说明中尚未完成费用核验的原计划；不是模型执行结果。
