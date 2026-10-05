# 普通两层结构复现了关系闭包缺口

唯一冻结运行通过全部 4084 tests/215 files，包括原 4080 项和四项新增审计。关键结论是负面的：当前 Neural HZ 组件能够在自己的父域证明一个关系，却不能让后继 ReLU 查询继续利用它。这是组合精度缺口，不是健全性错误，也不是新的正式解。

## 同次执行的证据

[反例记录](../../results/d208_multilayer_closure_audit_20261005_v1/closure_counterexample.json) 给出：第一层 J 的域内上界为 0；同一父状态对 g1-g2/2 的上界为 -1/4。解析证明据此得到 p=r1-r2/2-1/8<=-1/8、后继 z=ReLU(p)=0。

然而下一 bank 的尺度构造只看旧幅值的独立范围，得到 caps=(985/1024,3657/2048)、共同尺度 1。对 p，实际三个完整证书为：

| 固定域内证书 | 精确上界 |
| --- | --- |
| 单门 triangle | 16217/46094 |
| 常数 pair | 6423849967/9582552064 |
| 源相位 pair | 284806913/445700096 |

三项均大于 1/4。后继没有被证明稳定，最终抽象界为 [0,16217/46094]。此处没有外部 helper 或求解器补证；p<=-1/8 的全域结论来自 CONTRACT.md 的解析推导，不是用若干点代替证明。

[正倍数方向记录](../../results/d208_multilayer_closure_audit_20261005_v1/direction_family.json) 验证 lambda=1,2,3 时同一缺口保持。这只是同一方向的齐次族，不算三个不同结构突破。固定点 membership、原输入 decoder、完整十五维读出、五个原 bits 和真实零点的双标签也均通过。

完整小结构保留十三个因子，其中八个连续、五个二元；三 banks、两 pairs、四十条谓词、102 predicate nnz，计入非零常数和 RHS 为 146 个系数。该次控制全部反复查询和检查累计 8610 work、2011 logical entries、20 个 readouts。它不是实际网络单次请求成本或完整物理内存资格。

## 原因与研究决策

递归查询的健全性已经有证据，但不代表递归精度闭包。新层生成共同 deficit 时，原生谓词和先前查询已证明的关系没有进入 cap 认证；保留它们的存储并不等于消费它们。

下一研究先处理这一语义接口。固定 pair 的差分通过同域查询认证后，ReLU 单调性给出 q_i<=a*q_j+max(0,c)，可以在纸面上修复此例；但这正是已有 D138 规则的组合，未实现，不能重新计作新定理。更一般的方向是让共同非负谓词余量参与后续源相位关系生成，见 [下一定义假设](NEXT_DEFINITION_HYPOTHESIS.md)。该假设尚无实现、新颖性、性能或真实网络资格。

另一个并行结论是：简单删除原 active-value 行会改变整数具体化；有损替换又被同样十行的“精确原图加两条 h 反馈”严格支配，仅节省两个系数，当前不值得晋级。不能把多行包装为 min 运算就少记全部费用。详见本轮合同中的普通反例。

因此目前不把扩大重复 pair 样例、原生接入或 GPU 移植当作解决本体问题。GPU 仍是最终目标，但要加速的是具备有效组合能力的演算；本轮没有 GPU 工作或新性能声明。

## 执行与存档完整性

完整 pytest 子进程耗时 48.36630479618907 秒，低于原 60 秒门；pytest 自报 47.18 秒。包含独立来源检查的监督器总计 62.575013764202595 秒。没有选择性预跑、重试或修改已消耗版本。

本次前后认证 7399 个 source identities 和同一 14 个输入，source/input drift 均为空，production provenance 未变化。仍是 CPU [0]、单数值线程、CUDA 隐藏、16 GiB AS。监督器 RSS high-water 增量为 0，trace peak 为 19060633 bytes、metadata 为 6621072 bytes、reserve 为 65536 bytes；这些仅是监督器观察，不是 child 或整个 native/GPU 的物理门。

有 13 项原有 warnings，零失败、错误或 skip。D207 三份继承证据只写入本次 inherited_d207_controls，三份内容哈希与历史证据逐一相同；旧测试体、断言和域函数未改。D112 的既有输出例外同样仅改变目的地。

[退出凭据](../../results/d208_multilayer_closure_audit_20261005_v1/exit.json) 的 negative_audit_tests_passed 和 all_registered_stages_passed 为 true；new_domain_qualified、new_capability_qualified、capability_improvement_claimed 为 false。全部模型、native、GPU、完整物理和回放资格仍为 false。进程已正常退出，无本轮遗留后台任务。

正式 1870/2413=1063 CERT+807 validated ADV 不变；独立 CIFAR100 25、TinyImageNet 36，共 61/400 不变，新增均为零。旧文件未变化不等于新候选已取得全量零回归资格。完整 Goal 继续 active。

2026-10-05 Australia/Sydney；branch redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 保持 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只新增隔离研究文件和唯一新结果目录，未改生产、旧实验、历史数据或正式表格，无 commit/push。文档技能用于本地归档和区分证明、负结果及资格，未发布外部 Page。
