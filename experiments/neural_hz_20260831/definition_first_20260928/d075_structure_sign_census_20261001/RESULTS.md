# 共同负项符号普查因整数中间量超限未完成

本次完整归档诊断失败，未获得共同负项符号分布。唯一运行在有理数比较的交叉乘积检查处触发512位限制并fail closed，没有完整或部分端点文件，不能据此报告单侧比例、跨零比例或候选界改善。原D072数学组件资格不变，但不扩张为真实归档适用资格。

## 唯一执行及保存证据

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。唯一命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d075_structure_sign_census_20261001/audit.py --enabled`。会话26503已终止，实际exit_code=1；没有本次仍在运行的任务。

唯一RUN为 `results/d075_structure_sign_census_20261001_v1`，仅留下 `diagnostic.json` 和空的临时目录。没有 `complete.json` 或 `complete.json.partial`。冻结前只有全文静审、AST及身份核验，无候选导入、编译或数值预跑；三路静审发现的字典预付顺序和结果发布顺序在冻结前修正，数学规则与原预算未变。

| 记录 | 观测值 |
| --- | ---: |
| 实际失败 | ValueError，new comparison intermediate exceeds 512 bits |
| 完整运行时间 | 4.9827491119503975秒 |
| branch work | 66685090 / 200000000 |
| whole work及证据预留 | 106750626 / 256000000 |
| 实际证据meter | 0 / 40000000 |
| RSS高水位 | 306380800字节 |
| RSS增量 | 285073408字节 |
| tracemalloc峰值 | 105122018字节 |
| tracemalloc metadata | 87697776字节 |

失败前没有执行候选支持器、真实模型、LP/MILP或GPU。audit_completed、structure_sign_audit_qualified和全部实际能力资格均为false。虽然所观测资源数值低于相应上限，memory_gate_passed仍为false，因为完整诊断未完成；不能将局部资源观测解释为完整通过。

运行前后身份认证通过，source_drift、input_drift和identity_unchecked_paths均为空，分支及commit未变。根代理又独立核对全部3份冻结源码与11份输入哈希，确认失败诊断以及不存在complete文件。freeze.json SHA256为 `088690c37241f2fae6324e43be6feeb39d8b9e00dcc42dfb5fbe07b7d1ec9a49`；diagnostic.json SHA256为 `ebd8e8afa037339e2df91f90f4356d7cfdb4af7f2fc597ee33f8fd0c85dd0f76`。

## 可以和不可以推出什么

本次只证明这个冻结精确算术实现未能完成既定归档。比较器先形成分子与另一分母的乘积，再检查位长；异常没有保存具体操作数、消费者或锚对，所以尚不能定位是哪一条比较，也不能断言公因子消除就能使全量通过。它不证明原模型参数幅度极端，也不证明数学共同负项规则错误。

冻结D070支持器也采用检查交叉乘积的比较方式；这只是静态实现相似性，不是D070在同一记录上失败的运行证据。不能把D075诊断失败改写成D070数学测试失败，反过来也不能凭D072成功宣布真实网络数值资格。

不增加位宽或预算，不原地修改和重跑本版本，不为失败前的内存中间端点补写结果。后续若改变算术实现，必须另立版本并证明规则、可靠性、费用及全部原验证范围；目前不把数值实现修补作为定义研究的主线。下一项纸面检查区分固定相位纤维的已隐含支持、分数相位四槽重组合的严格强度，以及后继激活能否继续消费该关系。

正式1870/2413（1063 CERT与807 validated ADV）、独立CIFAR100 25加TinyImageNet 36即61/400均不变，formal_gain=0。所有旧档、冻结源及生产文件保持原状态，未commit或push。write-page技能用于明确失败证据及推断边界，只作本地存档；整体Goal保持active。
