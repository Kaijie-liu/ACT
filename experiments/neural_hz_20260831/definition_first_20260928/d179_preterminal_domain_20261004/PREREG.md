# Neural-HZ 预终端完整结构诊断预注册

此合同先于新代码的 import、AST、compile、collection 或执行冻结。它只认证 D178 候选的真实结构适用前提，不能作为域数学测试、模型验证或性能实验。数学问题与后续否证条件见 DEFINITION_TEST.md。

## 固定来源与执行

仓库 /data1/Kane/FSE/ACT，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。执行前后检查，生产文件不写。

完整认证 D158 preregistered.json，其 SHA256 为64cd535e4ae10bfaeb8204d21841f991781aa981e2aeddb359a87ca8050103d9；其中7327 source和14 input的字节身份全部在首次第三方/新模块导入前及结束后检查。新输入人口固定为inputs.json的三个原模型及三份D152完整图记录。全部处理，不因结果筛选；属性值来自同一份认证后的原模型字节，逐模型只解析一次。

freeze.json 的schema为d179_preterminal_domain_v1，source_sha256精确覆盖六份新文件：PREREG.md、DEFINITION_TEST.md、inputs.json、bind_structure.py、run_structure.py、launch_structure.py。唯一输出目录为results/d179_preterminal_domain_20261004_v1，排他创建即消耗版本。仅 --enabled 执行；失败或超时保存部分证据，不修改冻结版或重跑。缓存、临时文件及外部回执只在新结果目录，不写旧bytecode。

解释器 /data1/Kane/miniconda3/bin/python 必须解析到继承认证的原解释器。使用已认证 ONNX protobuf loader，不运行 checker、shape inference、ORT、torch、原网络 forward、求解器、GPU 或候选。原模型每份<=64MiB；JSON每份<=8MiB；禁止外部tensor data、子图/函数和未支持结构。FLOAT权重数值不读取，只有运算属性的有限float解码并以hex留证。

## 统一结构人口

按标准opset12的实际端口、常量维度及属性解释Conv、BatchNormalization、Relu、Add、Flatten、Gemm，显式将原符号batch绑定为1。覆盖整图尺寸，不以Gemm权重宽度冒充出生q宽度。所有原ReLU各给完整链匹配或拒配理由，匹配规则见定义页。内部q依赖的端口必须唯一消费，且无图输出旁路；skip必须是出生前已存在的共同父端口。

全图与保存D152元数据的节点/端口/initializer维度必须一致。未知域、属性、广播、错误尺寸、训练BN或不完整图均fail closed，不能跳过后声明完整。三份结构诊断全部成功且前后身份无漂移才标diagnostic_complete。无匹配结构也是可记录的完整结构结论，不据此重选模型。匹配数不强制成预期值。

只报告统一规则的m/r/p、设计所需原相位槽位数、候选和旧逐门行数、C密集系数数量上界及未经数值认证的完整仿射链。没有读取原HZ状态或phase ID，不能声称已认证身份保全。禁止从这些数字推导实际秩、非零系数、nnz、精度、速度或新增CERT。BN variance/scale数值、有效B与source/energy未认证，必须保留相应false字段。

## 固定资源与保留证据

CPU0、数值线程1、CUDA隐藏，地址空间16GiB。由launch_structure.py --enabled一次启动唯一诊断子进程，外部timeout60秒，内部58秒；外部超时由subprocess终止并等待该唯一子进程，不重启，自动写EXTERNAL_RECEIPT.json。RSS高水位增量+65536字节与tracemalloc peak+tracer metadata+65536字节各<=1GiB，分别报告，不冒充完整物理候选资格。外部监督器不读取模型、不执行诊断，只记录子进程退出和有限长度日志。

保留whole_work<=256000000、branch_work<=200000000、evidence预付40000000、metadata retained-entry累计<=64000000。此为metadata-only计费，不进行D172的逐FLOAT精确区间算术或8倍数值载荷展开；认证文件按每64KiB缓冲16单位，读取原字节按1单位/byte，protobuf结构解析另按1单位/byte，JSON解码按4单位/byte，metadata节点/属性/边/尺寸按显式操作和保留字段计费，证据序列化预付遍历与字符上界。原字节缓冲不是已展开的数值因子，实际驻留仍由上述两个独立内存门检查。这些work单位不是跨实验性能指标，也不改变候选的完整物理存储门。

自动保存每个模型的完整或失败记录、总结果、前后来源检查状态、环境/版本/时间/内存/计费及artifact hashes。完整诊断必须前后全部认证；异常或超时可能来不及完成后认证，此时显式保留false与原因并判证据无资格，不从缺少漂移记录推断未漂移。外部timeout若早于内部回执，保存外部退出记录并判不完整；不能因缺回执重启。最后汇总保留64KiB预留。任何未证、异常、漂移、资源失败均不能给资格。

## 不变的全量门

本次不运行或取代D158的4032 tests/212 files，亦不向新代码转让其数学资格。未来候选仍先过完整数学人口，再真实同结构、shadow、逐家族和同路径全2413/独立400回放。所有candidate/math/native/model-verification/GPU/complete-physical资格为false，formal_gain=0。正式1870与独立61不变，不相加。

此前被撤回的Claude路径不恢复；原历史文件只读，不修改生产、不commit/push、不默认启用。完整Goal继续active。
