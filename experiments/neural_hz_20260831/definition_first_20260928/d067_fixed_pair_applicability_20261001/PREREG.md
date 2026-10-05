# 真实宽层固定相位对的适用性审计

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。目标与基线继承GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md：正式1870/2413、独立E0 61/400不变。

本轮首先核查D066共享相位机制在现有完整真实归档上是否具有结构适用空间。它是只读历史证据的诊断，不是新的验证候选或D066模型集成。此前静态查看旧summary已知1600源门中727严格正、863严格负、10跨零；全部source形式34752项。尚未执行新固定pair分类或重算源界，不预设分类结果。

唯一输入为results/d025_interval_capacity_20260930_v1/complete_0.json，12062002字节，SHA256 fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0。冻结其原manifest、原source_box_bounds及census源码、D066接口及本轮源码；运行前后核对这些直接依赖。原失败D025全三模型普查不因复用其中完整large归档获得成功资格。

## 人口与计算

验证3072原输入盒坐标、1600具名源形式、五窗口(0,0)、(0,31)、(16,16)、(31,0)、(31,31)，各576原slot及64消费者。按原global-even索引固定(0,1)、(2,3)…(574,575)，共1440对，对应92160消费者记录。padding不造bit、不先过滤再配对；真实源身份及bounds须与window记录逐一相同，全部1600源须被覆盖。

独立使用精确Fraction重算全部1600源形式在完整3072输入盒上的区间和；每项乘法取四个端点积的最小及最大，逐项累加。必须与已保存bound完全一致。此计算只核归档形式的界，不重新加载网络，也不认证归档形式从原模型重新生成的完整流程。

源状态仅由可靠界分类：lower>0为strict_active，upper<0为strict_inactive，其余为unfixed，并另记跨零、触零和精确零。pair分类为padding_both、padding_one、both_fixed、one_unfixed、both_unfixed。所有pair保留原位置与两个归档源门语义ID，padding为null，不声称native列已绑定；完整计数，不跳过失败项。对每类同时报告pair数量及乘64的消费者映射数量，但kernel_tables_evaluated固定0。

若both_unfixed为0，只可依据THEORY说：在B1和B2均含上述可靠stable-bit事实的比较中，现有全部真实pair没有由“共享delta”单独产生的额外约束。不能据此说四表相对标量界无效、原native LP必然冗余、所有模型无收益或D066整体已否决。若非零，保留全部对应记录，继续原真实适配设计，不改变配对或人口。

## 运行边界

默认关闭的audit.py仅在显式--enabled下读取证据，仅用标准库，无候选或项目模块导入，无求解器、模型、GPU、相位枚举、条件子域或搜索。脚本与此页及THEORY先冻结到freeze.json，再首次运行。只使用一个新RUN results/d067_fixed_pair_applicability_20261001_v1；目录已存在即拒绝重跑。完成及失败均保存diagnostic.json，完整输出先.partial再原子发布，不覆盖旧数据。

沿用CPU单核、AS16GiB、240秒上限、512位有理数、256M whole及200M branch，另保留40M证据账和65536摘要reserve。预算在读取、解析、遍历及运算前支付；输出遍历及字节另记。主机ru_maxrss与tracemalloc含metadata均须<=1GiB。此诊断不授予完整native/HZ/GPU物理存储资格。

原3809项/179文件数学门保持其原资格，本轮不导入新候选、不重跑或削减该门，不以诊断成功声称新adapter或组合通过数学门。之前讨论的3813项adapter组合及D067宽层kernel运行器未实现、未冻结、未执行；此次不是它的缩人口重跑。若后续实施仍必须另行冻结完整数学与实际评估。

记录branch/commit、直接源码SHA、输入SHA、配置、运行时间、资源、所有状态及失败原因。正式gain恒0；actual_model_binding_qualified、native_nonredundancy_verified、gpu_computation_completed、complete_physical_qualification及candidate_admitted恒false。
