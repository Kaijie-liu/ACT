# 零值谓词运输的执行预注册

本候选是默认关闭的D256语义接入实现，分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，日期2026-10-06 Australia/Sydney。tracked binary diff SHA256保持29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。此前所有源和证据只读。

唯一新RUN为experiments/neural_hz_20260831/results/d257_zero_predicate_materialization_20261006_v1。六个冻结源为CONTRACT.md、PREREG.md、predicate_transport.py、test_transport.py、run_math.py及collection_contract.py。freeze.json写入并核验前，只允许静态文本编辑及审查，禁止import、AST/compile、collection、数值试跑或模型执行。

唯一执行命令为

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d257_zero_predicate_materialization_20261006/run_math.py --enabled
~~~

RUN独占创建即消耗版本。失败保持完整证据及冻结源，不修改后重跑；观察超时只轮询同一实际会话。

## 人口及预算

继承最后成功D255的全部有序4257项、227文件，新增固定12个plain test函数和一个文件，共4269项、228文件。名称、顺序及完整依赖由runner和freeze固定。单pytest进程在任何测试执行前认证完整人口，执行后复核source、input、provenance以及人口；必须零失败、错误及skip。不做子集试跑，不删旧测试或延长冻结门。

CPU0、单线程、隐藏CUDA、AS16GiB、完整pytest墙钟60秒。保持原双1GiB host观察边界与65536 summary reserve；监督器观察不代表pytest/GPU完整物理资格。正控共同Budget；事先声明的非法输入或低预算负控可使用较小独立Budget检验sticky rejection。原真实来源256M/200M、一次40M evidence、64M entries、512-bit、240秒等要求不变，本次不运行真实模型worker。

## 固定覆盖

1. 默认关闭及私有函数环境，生产函数和类身份不被改写。
2. 可达局部source与全部旧高水位，完整批次返回及fresh辅助。
3. 另一source的额外谓词、非零skip、完整输出及bias保留。
4. 严格零项的长线性链、Add及内部checkpoint规范化。
5. 部分及全false的keep_rows仍保留谓词，按原物化语义保留bias。
6. 实际隐式卷积类型，以及未知算子、非有限值和坏形状拒绝。
7. 完整物化输出中的原D255关系与普通终端固定对照。
8. 固定合法整数相位赋值的canonical延拓、全部输出和原输入decoder。
9. 内部selective入口确实调用私有物化wrapper，稳定正支路与core的范围如实记录。
10. 来源内容漂移、错误frame或aux冲突的整批拒绝，旧对象不变。
11. 共享预算、资源失败sticky及无生产全局改动。
12. 汇总全部记录和资格分账。

主正控直接复用只读D255 fixture与attach构造helper，不执行旧测试函数或旧证据writer，也不增加另一套数学结构。新增第7项只对固定原目标调用两次普通终端LP，以对照原view与带关系的完整物化view；候选本身无solver。精确存储行/谓词保留、canonical赋值及终端降级审计是主要正确性证据，不把浮点LP或分数点称为CERT/ADV。所有原数学测试仍按冻结人口完整运行，其原有控制不改。

本次没有在线分配器发布、HybridzTF安装、完整真实前向或GPU执行资格。zero_predicate_materialization_passed只能在完整测试、后置身份和观察门都通过后置true；旧rebase、quantified等资格仅保留为历史收据，不转授到新组合。其他actual_model、online、GPU、complete_physical、new_domain及new_capability标志均false，所有gain为0。

每项结果、失败原因和接入范围均写入新RUN及新研究目录。局部通过不改变正式1870/2413或独立61/400，不结束完整Neural-HZ研究Goal。
