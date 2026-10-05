# 原生混合关系组件的完整数学回放

冻结前只允许静态编辑和审查，禁止 import、AST、compile、collection 或数值试跑。
六源固定为 CONTRACT.md、PREREG.md、native_mixed.py、test_native_mixed.py、
run_math.py、collection_contract.py。审查后 SHA256 冻结，唯一执行一次；失败
原样保留，不修改冻结源或重跑。所有新写入只在本目录及唯一新 RUN 内。

## 完整固定人口

继承 D245 完整有序4209 tests/224 files及来源、依赖、14输入、全部历史证据，
新增一个文件16项 plain test，共4225/225；不 skip，不以历史 receipt 代执行。

1. test_01_default_off_and_snapshot
2. test_02_native_gate_extraction_and_scales
3. test_03_dyadic_stored_physical_certificate
4. test_04_complete_single_child_witnesses
5. test_05_joint_children_convexified_parents
6. test_06_integer_extensions_and_zero_labels
7. test_07_preserved_predicates_readouts_and_decoder
8. test_08_whole_batch_and_global_widths
9. test_09_exact_compact_and_graph_rejections
10. test_10_full_residual_and_shared_source
11. test_11_rank_guard_and_dependency_rejections
12. test_12_outward_coefficient_rounding
13. test_13_complete_cost_and_normal_terminal_lowering
14. test_14_snapshot_isolation_and_fail_closed
15. test_15_sticky_shared_budget
16. test_16_summary_and_qualification_boundary

普通正例共享一个 Budget；固定完整整数网格/零标签、强参照、原行保留、
源与余项、向外编译、整批失败、正常终端、sticky资源门均按源码冻结。
独立 malformed/resource 拒绝 fixture 允许事前声明的独立小 Budget。
测试侧仅在固定 F 方向对旧/新增行模型各执行一次普通 LP 松弛查询，辅助核对
正常终端消费；候选不调用 LP，不读取其状态或生成后续关系。实际存储行的
精确非负组合仍是主要上界证书，不把浮点求解器状态本身当作数学证明。
所有旧 writer 继续重定位到本 RUN；新增 D245 _record_file（首行56）重定位到
inherited_d245_controls/summary.json，不改变断言或数学函数字节码。
D241仅作历史来源，仍由 D243 source_reference 配合原 D240 manifest 读取。

## 唯一执行和证据

RUN 为 experiments/neural_hz_20260831/results/d249_native_mixed_terminal_20261006_v1。
首次独占创建即消费本版本。唯一命令如下。

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d249_native_mixed_terminal_20261006/run_math.py --enabled
~~~

CPU0、单线程、CUDA隐藏、AS16GiB；全部pytest含导入/collection/JUnit限60秒。
监督器1GiB及reserve65536、候选work256M/operation200M/entries64M/512位不变。
不改变任何门槛或扩大预算。独占保存 manifest、inventory、tests.log/xml、
summary.json、exit.json及继承证据；失败也保留。超时观察只轮询原进程。

schema=d249_native_mixed_terminal_v1。domain_definition_changed=false，
mathematical_stage_only、fixed_component_lp_controls_registered、
new_component_solver_free=true。native_mathematical_transport_passed 仅可在完整
数学门通过后置真；不等于 actual_model_binding/native_HZ_admitted/online/GPU/
完整物理/新域/能力资格。formal_gain=independent_e0_gain=new_benchmark_solves=0。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256
29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。
不改生产、默认开关、旧存档，不commit/push。上一轮用户中断询问仅核对状态，
就Goal推进计为no progress；本轮推进可被反证的原生运输，整体Goal保持active。
