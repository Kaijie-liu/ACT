# 多原相位前向关系的数学组件验证

本轮检验稀疏原相位仿射包络能否在普通混权仿射、ReLU 和残差组合中共同传播，避免先分别按单个相位求界再丢弃其他相位的信息。THEORY.md 定义共同赋值上的载体、变换及健全性；CONTROL.md 给出严格强于所列局部组合与逐单锚比较系统的物理读出正控。固定顺序超模链及差分传递都有文献先例，候选不是新颖性已完成的 Neural-HZ。

## 语义和统一构造

原 HZ 的连续盒、全部原 bits、EQ/LE、共同 latent/frame 及 decoder 保留。观察的读出是原坐标仿射式，其上下界是同一赋值上的稀疏原 bit 仿射式。形式身份 token 不是原模型或实际 HZ 列证书；调用方仍须认证来源、门关系、范围与 bit 的 active 方向。

对相位仿射式，先合并同一原 bit 的系数，再将负系数视为原 bit 的补 literal。按原结构 ordinal 固定排序，不按 LP 点、margin、模型身份或事后收益排序。ReLU 正部上界用固定前缀的边际增量，下界用 singleton 边际。边际可写成对正系数上限的 clamp，不必将两个大前缀正部相减。差值下界使用负正部的上包络，不能误用同一方向；伴随幅值也继续传播。

上包络在完整连续 literal cube 上也成立；singleton 下界一般仅保证原 bits 的整数语义，不能伪称在连续 cube 上逐点低于 hinge。它们都可成为真实原图凸包的有效线性行，不能用连续放松替换域本身的非凸语义。一个原 bit 时退化为 D044 的条件端点 clamp；无 bit 时为常数。零点保留两个合法原相位。

原网络仍按已有精确 HZ 操作解释，新增观察变换只是健全的有限外包，不保证无损闭包、全网 ideal formulation 或所有读出的最佳界。缺失、错配或不支持的前提拒绝安装观察，不宣告 SAFE，不删除原 bit。候选函数默认关闭，须显式 enabled=True。

本次实现只接纳上下包络在整个独立原 bit cube 上次序相容的有限子类，不是 THEORY 所有有效观察的总构造器。例如仅由原谓词认证 alpha<=eta 时，lower=alpha、upper=eta 可以在可达赋值上合法，但本组件仍保守拒绝；不删除这一检查或把拒绝当反例。常量读出的外部输入包络也受全 cube 包含检查。第八项同时覆盖这一范围及位宽、稀疏输入数量拒绝。

## 严格正控和预定人口

CONTROL.md 的同一网络为 q=R(x)、p=R(y)、t=R(z+q/10-p/5)、r=R(z-2q/5-7p/10+1/4)，输入盒为 [-1,1]^3。原 bit 按 q、p 的结构顺序处理，生成 t-r<=alpha/4+eta/2。加入原负幅行后，物理读出 Z=t-r+(q-x)/4+(p-y)/2<=3/4。

预定旧点由四个真实前缀点凸组合，再接连续前缀上的最后门精确图产生；它满足全部六对 D020 的 24 行，以及明确登记的逐单锚界，却有 Z=63/80。真实输入 (-1,-1,-1/2) 达到 3/4，四门预激活非零。此前未通过负幅 D020 的点已弃用，不作为正例。纸面松弛点不是 ADV，数学测试通过不是正式 CERT。

本版本完整继承 D044 已通过的 3769 tests、170 files 和全部精确 node IDs；增加 test_multiphase.py 中八个普通顶层无参数化测试，总计 3777 tests、171 files。预先指定函数为：

1. test_default_off_and_identity
2. test_majorant_and_minorant_soundness
3. test_single_anchor_matches_d044
4. test_mixed_phase_affine_transfer
5. test_strong_hull_control
6. test_physical_separation_and_attainment
7. test_second_mixed_block
8. test_rows_and_rejected_premises

有限数学覆盖包含固定小维有理系数、完整二元顶点及少量有理 cube 点、一个原 bit 的旧规则对应、异号系数 literal 视图、混权合流、完整四点见证和六对旧行、真实取等、同一原相位继续经过下一混权块、终端 LE 的 RHS 符号及普通错误前提。单元测试中的有限真值核对不构成候选运行时相位搜索；不加入模型 forward、源 census、真实基准 LP、攻击或 GPU 实验。

## 成本和 GPU 边界

纯 Fraction 组件保留 512 位及 65536 个稀疏输入出现次数的有限规模界。对合并后 k 个原 bit，固定顺序扫描为 O(k)，未排序输入的身份规范化及排序另计。每条最终关系的系数、RHS、读出展开、原位身份、临时量、历史证据、终端正向与转置、求解及 decoder 全部不是免费。只记录算法计数上界，不声称本次已测完整物理成本。

GPU 可采用分段前缀、clamp 与系数恢复；但 scan 的认证舍入、系数区间、完整峰值共存及终端查询尚未获得执行证据。已有 GPU 初始化失败版本不重跑，不修改其预算，也不把 CPU 数学资格冒充 GPU 完成。

原 whole_work_cap=256000000、branch_work_cap=200000000、evidence_prepaid_work=40000000、retained_entry_cap=64000000 仍为后续完整物理候选的门。本次没有新增数值 worker，不把新测试移入 240 秒窗口，不减少既有验证人口。

## 单次冻结与执行

新 RUN 为 experiments/neural_hz_20260831/results/d046_multiphase_envelopes_20260930_v1，独占创建；首次 --enabled 消耗版本。首次执行前冻结 PREREG.md、THEORY.md、CONTROL.md、multiphase.py、test_multiphase.py、run_math.py 六文件的绝对路径与 SHA256，以及八个测试函数名。schema 为 d046_frozen_v1。执行前允许静态 AST 解析，但不先导入或单跑候选来选择结果。

collection 与 execution 合计仍为 60 秒。任何 failure、error、skip、人口漂移或超时均不通过，不事后改人口、放宽门或重跑失败版本。固定解释器为 /data1/Kane/miniconda3/bin/python；assertions 开启、CPU1、AS16GiB、库单线程、禁止 bytecode、CUDA_VISIBLE_DEVICES 为空，缓存与输出仅写新 RUN。

来源以 D044 已通过的 preregistered.json、inventory.json、exit.json、freeze.json 和 run_math.py 认证，并继承全部 source/input、原模型与性质、decoder、解释器及 GPU 依赖身份。允许使用摘要认证的 D038/D015/D017 只读 metadata、memory、绑定与 drift/provenance 帮助函数，不调用旧 main、worker、writer，不修改旧 globals。执行前后检查完整来源身份与生产 provenance。

监督器沿用 1GiB RSS high-water growth 及 tracemalloc peak+metadata+65536 门；pytest 子进程受原 CPU、AS 和时间门约束，但不因此声称完整聚合物理峰值合格。异常也保存终态、日志、已得 JUnit 和身份检查；无 worker 的全部阶段通过只表示数学组件资格。

始终 native_HZ_admitted=false、actual_phase_column_binding_verified=false、source_census_qualified=false、gpu_computation_completed=false、complete_physical_qualification=false、formal_gain=0。只有后续同候选完整保旧增益回放才可能更新正式成绩或默认路径。

## Provenance 和不变条件

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式 1870/2413（1063 CERT、807 validated ADV）、13 家族及每个旧解保全；独立 E0 CIFAR100 25、TinyImageNet 36，共 61/400，不相加。本轮不改变能力与速度解耦门。

所有新文件仅在当前隔离目录，结果只写新 RUN。D045 未认证草稿、旧冻结实验、历史模型及 /data1/Kane/HyZor 保持原状。不 commit/push、不改生产默认、不重定义整体 Goal。禁止 attack/PGD、BaB、input/phase split、backward/dual rescue、对偶或求解状态菜单；普通终端和独立具体见证边界不变。

依照 write-page 文档技能分开定义、文献先例、纸面证明、有限测试、未证范围及正式计账；本地存档，不发布外部 Page。
