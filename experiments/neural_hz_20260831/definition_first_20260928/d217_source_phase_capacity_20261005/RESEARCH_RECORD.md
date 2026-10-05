# Neural HZ 共享源与物理能力研究记录

上一轮 D216 属于有进展：完成并冻结了局部关系定理及物理投影负结论。本轮先重新核对其14项归档哈希，全部一致，然后继续定义研究，没有重启旧候选或数值任务。

## 本轮改变下一行动的证据

[相位容量和投影判据](THEORY.md) 给出完整单门源关系下的最大可延拓相位。对额外相位单调关系，物理点是否真的被排除，可在这些最大相位上精确判断；不能再用任意分数相位违反点当作能力阳性。固定点计算可排序扫描，但不是整个验证问题的廉价解法，也没有实现成 helper。

同页将 D216 的负结论扩大到有不同偏置、不同主系数和横向权重的一类普通非对称双门。仅改变 kappa 或小系数仍可能没有输出收益。三源族有可检验必要条件，但本轮没有新的 physical 或 child 阳性。

[真实共享源证书](REAL_SOURCE_CERTIFICATE.md) 用已存原参数和实际正宽度源盒证明 CIFAR100 medium 四门候选的共同源映射秩4。两个子式分别有 >3/2000 和 <-41/5000 的余量，不依赖数值近零或极端特殊情况。由此排除用一个标量整体替代该块共同源的精确仿射读出接口（包括有限原相位选择）；不否定保留全源时额外加入一个尺度关系。

这些结果服务于强 Neural-HZ 的定义，但没有建立强新域、PLDI 新颖性或新增解。不为这些负结论新建 toy runner，也不转做 loader 或存储优化。

## 真实接入路径的只读审查

现有冻结 source_binding_v2.extract_model、source_packet_v1 及 census_worker_v2 的 input_box/receptive/conv_shape/post_affine，以及 D025 first_form 已能提供可复用来源。无需新通用 loader。但旧 census 只走五窗口，并跳过 target_relu=None 的 shortcut，不能原样当完整能力实验。

当前 D214 也不能直接接完整真实前缀：其 Form 只接受精确 int/Fraction，而现有冻结 BN 来源路径返回可靠区间系数。D205 有健全区间推广的纸面证明，尚无已过门的 owned-domain 组合实现；取中点不合法。

边界上，D214 的 support 查询会沿 birth 历史反向遍历，并组合非负证明权重，不能把它描述成没有 backward 计算。它的已有组件资格不自动证明新 Neural-HZ 的能力或免除 helper 审查；本轮没有调用它，也不以额外验证器、solver rescue 或外部算法的收益记账。

其次，当前 affine 物化稠密矩阵，首 bank 每门视图和12张方向证明复制完整父身份，累计 entries 下界为13*d*m。最小 medium 的 d=3072、m=14400，仅这项就是575078400，已超过64000000。该结论是当前实现范围，不是数学规则不可能实用；不能以减少人口、重置 lineage 或修改收费掩盖它。解决执行成本只是必要配套，不能单独记成定义创新。

完整真实 scope 仍为原三个 model/spec 的 Conv0→BN1→ReLU2→Conv3→BN4→ReLU5，并保留所有活 shortcut：

| 模型 | 原性质标识 | 输入数 | 首 bank | 第二 bank |
| --- | --- | ---: | ---: | ---: |
| CIFAR100 large | idx_1059_sidx_6596_eps_0.0039 | 3072 | 65536 | 65536 |
| CIFAR100 medium | idx_1190_sidx_8846_eps_0.0039 | 3072 | 14400 | 8192 |
| TinyImageNet medium | idx_1024_sidx_5112_eps_0.0039 | 9408 | 46656 | 25088 |

large 保留到 Add8 的 identity skip；medium/Tiny 保留 Conv8→BN9→Add10。model/spec 联合来源沿用 D120 的 selected_sources，不用仅绑定模型的元数据代替。四门证书只是研究锚，不能替代完整 population。

后续新组合仍传承 D214 的4116 tests/217 files、7473源身份/14输入及实际复用依赖；完整60秒 pytest 门、240秒 source worker 门、whole256M/branch200M/entries64M、512-bit、单CPU单线程、AS16GiB及原内存/证据门均不变。没有在本轮放宽任何门槛或启动尚未预注册的数值实验。

## 下一项研究问题

核心是多维共同源、原相位和幅值怎样构成可组合的非凸关系，并在真实混权消费者上保留有用的物理约束。应先用本轮相位容量判据排除可延拓的伪阳性，再研究不能被共同相位修复的源关系；真实四门秩证据禁止默认 rank-one 简化。

保持连续和二元因子、EQ/LE、同源身份、全消费者、具体输入 decoder 与 fail-closed。下一研究仍是前述数学问题，不把保存计算图、已知单门 hull、更多证书或外部 helper 直接改名为 Neural-HZ。

## 执行与记账

2026-10-05 Australia/Sydney，分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本轮是纸面推导、独立并行复核、定向读取旧 JSON 参数以及实际 model/spec 哈希核对；没有 import、AST、编译、collection、候选执行、模型解码/运行、LP/MILP、GPU 或回放，也没有后台运行。

新写入只在本隔离目录；生产、旧实验、模型和结果不改。tracked diff SHA256 前后均为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，它不代表全部未跟踪源码的冻结。新页和只读锚点另记哈希，无 commit/push 或默认启用。

正式1870/2413=1063 CERT+807 validated ADV不变；独立E0 CIFAR100 25、TinyImageNet 36，共61/400不变；本轮两边新增均为0。未改旧文件不是新候选已经保旧的证明。强 Neural-HZ、GPU、smooth/Transformer、其他家族和完整能力目标仍 active。

pages:write-page 仅用于本地归档，分开定义判据、真实结构证据、运行资格与成绩；没有外部 Page 发布。
