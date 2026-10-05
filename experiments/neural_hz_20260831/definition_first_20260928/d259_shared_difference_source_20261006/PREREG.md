# 共享来源差分界的执行预注册

本轮要检验的是一个可能增强相位关系的必要条件：在三个完整真实残差前缀上，共享来源先合并、选中父项再剥离的余项界，能否逃出已证的投影冗余充分条件。逃出不是严格收紧，更不是新增验证结果。本轮不安装QG关系、不生成原生相位列、不执行验证器终端查询。界计算是对完整非凸来源的可靠证书，不把来源替换为区间域。

日期2026-10-06 Australia/Sydney，分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。旧实验、生产、模型和历史结果均只读。研究依据为本目录CONTRACT.md及只读D258定理，不将来源诊断声明为新域创新。

## 冻结和两次独占执行

八个冻结源为CONTRACT.md、PREREG.md、source_bounds.py、source_observer.py、test_source_bounds.py、run_math.py、collection_contract.py、run_audit.py。freeze.json保存这些源的SHA256、既有基线与完整测试人口；写入并核验冻结之前只允许文本与静态数学审查，不允许导入、AST/compile、collection或数值执行。

数学RUN唯一为experiments/neural_hz_20260831/results/d259_shared_difference_source_20261006_v1，命令为：

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d259_shared_difference_source_20261006/run_math.py --enabled
~~~

仅在本版本完整数学收据、后置检查及工件身份全部成功后，才能执行独立来源RUN experiments/neural_hz_20260831/results/d259_shared_difference_source_20261006_sources_v1：

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d259_shared_difference_source_20261006/run_audit.py --enabled
~~~

每个RUN独占创建一次。任何阶段失败都保留原冻结源与失败证据，不在同版本补丁后重跑，不启动来源阶段绕过数学失败。观察超时只轮询已存在的实际会话。默认关闭；没有自动降级到第二种界算法。

## 数学人口与覆盖

完整继承最后成功D257的有序4269项、228文件，追加本版本8个plain函数，共4277项、229文件。原人口、原测试语义及原依赖全部保留；旧证据writer仅定向到新RUN内的隔离子目录。固定新函数顺序如下：

1. test_01_default_off_and_limits
2. test_02_outward_arithmetic
3. test_03_complete_channel_conv
4. test_04_bn_whole_carrier
5. test_05_additive_peeling_not_arbitrary_bounds
6. test_06_difference_payment_controls
7. test_07_complete_padding_classes
8. test_08_summary_boundary

覆盖默认关闭、共享sticky资源账、普通浮点的精确有理包络对照、完整小卷积与实际零padding、BN整个nominal加独立误差来源、可加账安全剥离、共享差分正控与剩余相关性丢失负控、全部边界mask人口和资格分账。浮点误差检查先作精确有理包含，再作普通小样例的非空泛精度检查；不以容差代替健全性。

新组件与新8项均没有LP或其他solver调用。继承字段fixed_component_lp_controls_registered仅指旧测试已有的固定普通终端对照，不代表本轮新登记LP。数学监督器中的worker_stage_registered=false指不启动候选验证器；source_audit_stage_registered=true指上述另一个来源诊断，不转授原生域或模型验证资格。

数学门仍是CPU0、单线程、CUDA隐藏、AS16GiB、完整一次pytest墙钟60秒、零失败/错误/skip及原双1GiB宿主观察。监督器总墙钟与pytest墙钟分开报告，不用监督器观测冒充子进程/GPU完整物理资格。完整旧测试并非新增已解样例数。

## 完整来源与费用

来源顺序和model/spec身份严格沿用D241 source_0/1/2及D179 metadata：CIFAR100 large、CIFAR100 medium、TinyImageNet medium。从完整输入盒至第三个原ReLU的全部节点、参数、通道、空间位置及后续消费者均保留，不抽窗口。三个前缀共70个initializer、875072个标量；其中869440个Conv权重。原模型没有为本实验裁剪或改写。

统一登记第二与第三ReLU相同位置的相邻通道对。三来源登记人数分别64512、8128、24892，共97532；空间几何共同计算映射为2853条pair/class记录，覆盖所有原位置，不是采样。不能因非crossing、零tau、失败或结果阴性事后缩小登记人口。条件只读本来源的数学结构，不读实例标签或terminal状态。

整个来源RUN共用256000000 work；每模型最多200000000；evidence只预付一次40000000；累计numeric/metadata entries及完整保留根观察不超过64000000；有理证据512-bit；完整生命周期240秒、CPU0单线程、CUDA隐藏、AS16GiB和双1GiB宿主观察均不变。解析、身份检查、源参数、界计算、分类、证据和后置检查都计入。内部融合操作必须减少实际遍历和中间数组，不能只取消实际发生的收费。

原权重packed FLOAT32精确提升到FLOAT64，不走旧逐权重Fraction解码。BN认证包络及模型完整语义不因此省略。负的名义tau统一交换父门顺序及对应尺度、系数、上界，子门次序不变；选择的tau只作差分恒等式中的正参数，系数不确定性全部保留。阈值比较对最终存储的二进制端点用Fraction精确计算，不以浮点等号决定冗余。

完整静态费用检查是冻结前条件。早期通用Interval逐步骤实现的全crossing主体下界已超出预算；其费用不能当作拟融合实现的收费，也不能声称实际模型已经执行失败。若最终冻结前无法给出可信的完整费用计划，保留未执行草稿及静态否定记录，不假装进入数值阶段。

冻结前独立静态审查确认，融合版已消除旧逐步骤实现仅h和offset主体就至少478214784 work的全crossing执行分支下界。下面是最终源码的费用计划，不是已经通过资源门的收据；各项不能相互替代或漏记。

| 项目 | 静态依据与尚未实测范围 |
| --- | --- |
| 原模型与spec解析 | 沿用原完整来源字节人口，约53.703M work |
| 证据预算 | 全阶段一次固定预付40M work，内部另守40M限额 |
| 监督与旧证据读取 | 约35M，包含三份D241 JSON约8.759M、数学manifest/inventory、JUnit及全身份前后检查；新收据大小仍待实测 |
| 完整packed参数 | 16乘875072加3倍TensorProto字节及小量元数据，约24.5M |
| 完整Conv前缀 | P=869440、M=111168、Co=1408；保守主项16P+135M+120Co=29087680，另计小量节点/几何固定项 |
| 差系数与offset | T=328896；h不超过54T、offset主体不超过104T，另有2853次标量证书/元数据开销，计划约56M |
| BN与端点转换 | 31104个输入端点和5632个BN端点；基于已存整数不超过39位十进制的保守位长计划约29.24M，另有输入解析/BN算术约1–2M。实际位长可能更小；不以64/input解析费冒充转换费 |
| 分类与最终证据 | 全2853条记录的外部标量系数、完整偏置/误差、类内合并、精确Fraction阈值与metadata约11.8–12.2M全活跃保守计划，最终观察另按实际计入 |

前六项约238.2M是含保守主项的计划小计，不是整个来源费用上界。完整总量还取决于端点位长、tiny编码实际触发、crossing人口及元数据；尚未静态证明总量低于256M，也未证明固定来源必然超限。累计core entries计划约35M，packed、所有临时/metadata及保留根仍另计。审查决定以不变硬上限作一次完整fail-closed测量，不能因为计划有余量或粗上界偏大就授予或否决实际资格。若实测越界，本版失败即封存，不修改收费、预算或人口重跑。

最终审查的全活跃、全tiny及最大已知整数位长组合给出约281–283M的粗保守费用计划，累计entries约55–61M；这些上界组合不等于固定来源的实测代价或必超下界。一次完整执行会同时解决实际代价和关系分类两个未决问题，超过256M/64M/240秒或任一观测门即不合格。此风险在冻结前明确登记，不在结果出来后改解释。

## 结果与晋级边界

每个登记成员区分ineligible、excluded及not_excluded。excluded只对应本轮形式及同一普通范围下D258的充分冗余条件；not_excluded仅表示尚未被该充分条件排除。没有严格收紧证书不得声称强于旧域，没有完整原生接入不得声称模型验证能力。

来源阶段保留完整逐class证据；report只保存小结与工件身份。来源文件不超过8MiB；终态report/exit共同使用65536字节预留。所有失败、资源消耗及资格必须如实记录。数学资格仅在完整数学门与后置检查后授予；原生接入、完整物理、GPU、新域、新能力、CERT/ADV和正式gain在本轮均不授予。

正式1870/2413及独立CIFAR100 25、TinyImageNet 36不变。后续仍需同源严格收紧、完整接入、同结构shadow、逐家族与全部2413/独立400回放，以及原四并发不回退门。完整Neural-HZ、GPU、smooth和Transformer目标不因本诊断结束。
