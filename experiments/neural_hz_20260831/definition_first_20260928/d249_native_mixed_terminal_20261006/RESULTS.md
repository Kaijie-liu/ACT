# 原生混合关系运输的测试结果

完整组件回放通过：4225 tests、225 files，零失败、零错误、零跳过。
冻结版本只执行一次，完整 pytest 用时51.833274秒，日志报告测试用时50.73秒，
监督器总用时66.476613秒；60秒门作用于完整pytest，而不是监督器全流程。
13条既有警告不改变结果。正式成绩与独立外部成绩均未增加。

## 实际取得的进展

已有混合相位关系已能直接认证并追加到构造样例的实际 SparseHZono 矩阵，
不再为了这一步重建 D243 Source，不调用旧 bound generator、Bank 或求解
救援。四个原二元相位、全部旧 EQ/LE、原输出、skip 和五维输入 decoder
均保留。27个固定整数点和16个全零合法相位标签均可延拓并重构。

固定物理对照中，正常终端的 LP 松弛上界从2.75收紧到2.05859375，即
527/256，恰等于已证明的真实最大值。主证明是实际存储 LE 的非负组合加
原 EQ 消元，不依赖浮点LP的成功状态。完整单child源hull及共同child对
凸父hull的假点给4233/2048，与真界相差17/2048，不能延拓到新关系。
这说明关系在此样例中确实被终端消费，不只是保存了一份公式。

单关系新增六个连续因子、29条LE、104个实际 native 谓词非零项，无新相位。
产品界由 native [-1,1] box 编码，不重复增加12条界行。双关系整批及超过
局部分支宽度的完整旧高水位也通过；整批失败不返回半状态。这仍是隔离
terminal snapshot，不是生产在线 allocator 或完整网络接入资格。

普通非平行来源控制产生25/48的 Gram 系数，实际逐项舍入误差及 RHS 向外
补偿检查通过。非零 compact 门误差、秩不足、守卫不成立和依赖不合格均
拒绝，没有切换到其他算法。旧终端 signed 到0/1转换的不精确 .1/.2 EQ
被新入口准确拒绝；没有修改旧求解器或者偷偷增加第二条终端路径。

## 证据及成本口径

唯一 RUN 为 `experiments/neural_hz_20260831/results/d249_native_mixed_terminal_20261006_v1`。
会话83898正常退出0。完整继承 D245 的4209项有序人口、224文件及证据，
新增16项；源身份集合7792项、输入14项，前后来源/输入漂移为空，工作区
provenance未变。原测试证据写入本RUN隔离子目录，旧存档不改写。

普通正例共享逻辑 Budget 最终为 work1,781,186、entries559,792，未提高
256M/200M/64M/512位上限。监督器 traced peak 为21,665,481字节、tracer
metadata7,562,832字节、RSS高水位增量0。它们不是pytest进程总物理峰值，
也不是GPU或完整模型资源证书；complete_physical_qualification仍为false。

冻结时间为2026-10-05 14:03:51 UTC，即悉尼2026-10-06 01:03:51。
freeze SHA256：`75693fca242e5c059e539e9ecb4a1b976573b2c81b469c4d75ea4c5387706d32`。
exit SHA256：`59e8fde92dce75d191cad3aaa446cda1e5f61090d76cc0334c48b253b54d20df`。
summary SHA256：`7617e42fab07c14ce92b1624bfcdf1e68d057e3b93a728f4cd1f058c0603791c`。

## 尚未取得的结果

native_mathematical_transport_passed为true；domain_definition_changed为false。
这次只是运输已经证明的关系，不是另一项定义创新。真实模型绑定、生产
native安装、online lifecycle、GPU、完整物理、新域及新能力资格均为false。
没有执行三个完整实际模型、shadow、13家族、2413或独立400回放，也没有
四并发晋级、smooth/Transformer或新家族收益。下一步边界见
[接入清单](NEXT_INTEGRATION.md)，不能以终端组件通过缩减整体目标。

正式1870/2413（1063 CERT、807 validated ADV）不变；独立 CIFAR10025、
TinyImageNet36，共61/400不变。formal_gain、independent_e0_gain、
new_benchmark_solves全部为0，默认关闭，Goal继续active。
