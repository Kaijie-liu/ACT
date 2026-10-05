# 原相位兼容出生的唯一数学验证

本预注册只覆盖默认关闭的跨层激活组件。固定六源为 CONTRACT.md、PREREG.md、birth.py、test_birth.py、run_math.py、collection_contract.py；在任何候选 import、AST、compile、collection 或数值执行前冻结 SHA256。冻结版本只执行一次，失败原样保留且不编辑重跑。

## 固定测试与解释

新增一个文件的12项普通测试，顺序如下；无参数化隐藏人口。

1. test_01_default_off_and_frame_identity
2. test_02_mixed_fifth_gate_strict_gap
3. test_03_all_vertices_remain_true_graph
4. test_04_cut_completeness_independent_geometry
5. test_05_integer_compatible_mixture_is_graph
6. test_06_zero_labels_and_signed_support
7. test_07_repeated_crossing_birth
8. test_08_common_source_residual_control
9. test_09_resource_failure_is_sticky
10. test_10_exact_arithmetic_and_invalid_inputs
11. test_11_compiled_rows_and_counterexample
12. test_12_original_bank_immutable_and_record

合同的混权第五门控制必须实测 sup F=63/32；完整旧联合图凸包接独立精确新门接受 F=2，严格差1/32。保留此控制全部旧物理坐标、原 signed bits、frame 和真图点。测试必须区分局部支持见证与原网络 ADV，不给分数点记分。

独立二维几何参考可在测试中穷尽固定有限原相位以核对完整交点及支持，不能进入候选运行时或被称为网络 phase split。重复出生、真实共同余项、所有生成点的逐点真图检查、原零标签、编译行和固定合法混合一起检查。支持/编译/下一出生都使用同一个共享终身 Budget；拒绝测试分别建立明确的新预算，不把资源拒绝当能力通过。

完整二维控制、再次出生的具体系数、余项坐标、查询及拒绝输入均由冻结测试源固定，无搜索、调参或事后改变。新组件不调用求解器、其他验证算法、helper 或具体模型；继承数学人口保留其原固定 LP 对照。新结果 summary.json 由独立 writer 独占保存12项记录，失败亦自动保存已完成的监督工件。

## 完整继承及旧写入隔离

继承 D230 的全部4141 tests/220 files、有序 nodeids、7579来源身份、14输入和完整依赖/证据闭包，新增后4153/221。实际新身份数按最终 manifest 报告。旧 receipt 是来源门，不代替重新执行全部继承测试；不得删测试、跳过失败或降低人口。

D112、D207、D208、D209、D214、D229 的独立 writer 重定向和 D228 内联 Path.open 窄 facade 原样继承，旧函数、字节码、常量及断言不修改。新增 D230 test_diagnostic.py 首行19的 _record_file 重定向到本 RUN/inherited_d230_controls，仅准写其旧 summary.json。旧 D230 是 negative_audit_only、domain_definition_changed=false，必须按它自身冻结 flags 验收，不用新候选语义重写旧成绩。

本候选的语义定义是本 CONTRACT，最后一个旧成功候选定义仍指向 D229；D230 的负向诊断不伪称新域候选。全部旧源码、历史结果及实际模型只读。

## 唯一运行及原预算

唯一 RUN 为 experiments/neural_hz_20260831/results/d231_phase_compatible_birth_20261005_v1，首次独占创建即消费版本。唯一命令为：

```text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d231_phase_compatible_birth_20261005/run_math.py --enabled
```

CPU0、单线程、CUDA隐藏、AS16GiB；完整 pytest 导入/collection/执行/JUnit 阶段仍60秒，supervisor观测1GiB、reserve65536。组件共享 Budget 仍累计work256M、单操作200M、entries64M、算术512位。接口和去重费用计入，不以逻辑账冒充完整物理存储、真实模型、GPU或并发性能。

监督器独占保存 preregistered.json、inventory.json、tests.log、tests.xml、summary.json、exit.json 以及继承新证据；故障时保留已生成工件，不后台重试。结果同时核对完整有序 collection、JUnit全部项零failure/error/skip、全部源/输入前后哈希、原工作区 provenance 及自动封存工件。

## 资格和账本

schema=d231_phase_compatible_birth_v1，required_tests=4153，required_test_files=221。mathematical_stage_only、fixed_component_lp_controls_registered、new_component_solver_free、domain_definition_changed 为 true；negative_audit_only、new_set_class、solver_rescue_registered、worker_stage_registered 为 false。domain_definition_changed 只表示本次新增非凸因子的前向算子规格，不表示新集合类或创新资格。

实际模型、native安装、完整物理、GPU、new_domain_qualified、new_capability_qualified 和任何正式提分资格全部 false。formal_gain=independent_e0_gain=new_benchmark_solves=0，正式1870/2413及独立E0 61/400不变。没有全量回放不声称新路径已保住全部旧解。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff应保持29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本轮只新增隔离研究，不改生产、不提交或push。数学组件通过之后也必须继续真实同结构、shadow、逐家族、2413和独立400晋级流程；整体 Goal 保持 active。
