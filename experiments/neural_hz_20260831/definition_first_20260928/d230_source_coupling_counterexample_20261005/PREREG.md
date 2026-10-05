# 源均值接合反例的冻结验证

本轮是负向数学诊断，不是新抽象域候选，不跑实际模型。固定六源为CONTRACT.md、PREREG.md、diagnostic.py、test_diagnostic.py、run_math.py、collection_contract.py；在任何候选import、AST、compile、collection或数值执行前冻结SHA256。只允许一次执行，失败保留且不修改重跑。

## 固定内容

唯一新增plain test为test_source_mean_binding_counterexample。诊断入口默认关闭。固定CONTRACT中的四源、四ReLU与F；逐项验证完整原H、D229自然查询、两份实际单child源混合、原native LP点以及D228原(z,t)共同坐标控制。所有数值与参数在运行前固定，不搜索系数、不选择事后较好方向、不调用新的LP/MILP/helper。

必须得到：真四平面支持(0,0,1/8,1/8)，达到点F=1/8；当前D229上界5/8，完整R=0；假点F=3/20，严格缺口1/40；局部A/B为真实局部图但不分别满足原共同源关系；完整native EQ/LE可行点为分数LP视图而不是整数反例；原(z,t)坐标的D228上界1/8。全w不可能性、Bank lambda扩展及后继R(F-11/80)后果由合同中的精确证明解释，不冒充有限数值枚举、已求lambda或模型结果。

正控只恢复既有D062，无新强参照优势；其结果不触发另一路径或默认启用。新证据只有summary.json，由独立writer独占保存，并明确所有资格边界。

## 完整继承与隔离

继承D229完整4140 tests/219 files、有序nodeids、7544来源身份、14输入及完整依赖/证据闭包；新增1项后为4141/220，不降低旧人口、不跳过失败、不把先前receipt当新执行。实际新来源清单可能增加旧工件身份，按新manifest报告实数。

原D112/D207/D208/D209/D214 writer重定向保持。D228内联写入的精确Path.open facade沿用D229，不改旧函数、字节码、断言或数学。D229的独立_record_file（源首行48）只重定向到新inherited_d229_controls；原8项均执行。旧summary哈希与全来源前后核验，旧目录不得写入。

唯一RUN：
experiments/neural_hz_20260831/results/d230_source_coupling_counterexample_20261005_v1

唯一入口：
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d230_source_coupling_counterexample_20261005/run_math.py --enabled

RUN首次独占创建即消费版本，无后台重试。自动保留preregistered.json、inventory.json、tests.log、tests.xml、summary.json、exit.json和所有继承新证据，失败也保留已生成工件。

## 固定预算和资格

CPU0、单线程、CUDA隐藏、AS16GiB；完整pytest导入/collection/执行/JUnit阶段60秒，supervisor观测1GiB、reserve65536。现有Bank组件的256M累计work、200M单操作work、64M累计entries、512位、65536稀疏出现次数门不改；两个固定Bank对照都计入共享或明确记录的原预算，不作速度比较。诊断fixture与证据费用不冒充完整原网络物理账。

schema=d230_source_coupling_counterexample_v1；required_tests=4141，required_test_files=220。mathematical_stage_only、negative_audit_only、fixed_component_lp_controls_registered（仅继承既有测试对照）、new_component_solver_free为true。domain_definition_changed、new_set_class、solver_rescue_registered、worker_stage_registered以及实际模型/native/GPU/完整物理/newdomain/capability均false，formal_gain=independent_e0_gain=0。

分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；原tracked diff29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5应保持。全部新文件只写本新目录及新RUN。文档归档分开证明、实测和未获资格；不发布外部Page。目标仍是强非凸Neural-HZ，不以本轮诊断通过宣布完成。
