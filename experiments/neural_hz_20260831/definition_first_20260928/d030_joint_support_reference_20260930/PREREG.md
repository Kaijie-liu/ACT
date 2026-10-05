# 共享源支持与条件观察的参考实现预注册

本轮实现冻结 D029 THEORY.md 的数学组件：共享源正项配对、负项固定仿射上界、显式区间误差，以及六条条件提升行的四条精确投影。目标是检验这些组件能否可靠落地，并测量保存的真实共同源结构上的支持界。它不是已完成的新 Neural HZ 域，不是 GPU 替代路径，也不更新任何正式成绩。

## 单次执行与只读边界

新代码只在当前 d030_joint_support_reference_20260930 目录。唯一结果目录为 experiments/neural_hz_20260831/results/d030_joint_support_reference_20260930_v1，必须 exclusive 创建。冻结本文件、joint_support.py、test_joint_support.py、archive_worker.py、run_reference.py，以及 D029 三篇数学文档、其 hash 清单和全部继承依赖后，才能首次导入或执行候选。首次执行即消耗版本；失败不改源码、重试、缩人口或放宽预算。冻结前只允许静态审查。

命令为 /data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d030_joint_support_reference_20260930/run_reference.py --enabled。所有数值入口默认关闭，必须显式 opt-in；错误前提、类型、形状、共同源、位长或资源越界 fail closed。不要运行任何旧 main、writer、census 或诊断重试。

## 组件及固定测试人口

继承 D025 全部 3753 个测试、168 文件和精确 node IDs，新增九个无参数、无 decorator 的顶层测试，共 3762 个、169 文件。collection 与执行合计仍不超过 60 秒，无失败、错误或跳过；JUnit 和收集结果均须与冻结人口逐项一致。旧 D025 组件测试成功与其 source census 失败分别保留，不能把后者改写成成功。

九项新增测试覆盖：D027 的联合支持与公平单门对照；带符号、零项、无正项与常数；共享源身份与抵消；跨零区间权重的显式误差及固定端点算术检查；稀疏缓存与独立直接支持参考；分数相位下六行与四行投影存在量化等价；零 preactivation 的两种原 bit；默认关闭及类型、形状、源身份拒绝；工作预算、位长和无效账本拒绝。具体函数名在首次执行前由 AST 冻结，不通过参数化隐藏人口。

求界接口只返回观察上下界及对照，不创建新相位或改动原门。prepare_source 可在一个窗口的所有通道间复用名义源和误差，但其内部缓存必须深不可变，prepare 和所有消费都计入同一总工作账。完整缓存通过 evidence_roots 暴露给物理存储计数，capsule 本体另外计入。

正权重按输入 slot 次序相邻成组；未配对单门两侧使用相同支持，不给 dummy 人为分配残差制造增益。数学参考中恒等零的 affine form，在完整输入和误差计费后不参与正项分组；这是零表达式规范化，不删除原槽位、真实门或零点二元选择。若实际区间 form 并非零，其误差仍完整计入。稳定但非零的源不能借此过滤。

系数中点仅定义数学参考 F0，所有实际系数区间的误差 E 按 D029 公式进入上下界。名义相位不能绑定到原 bits。四行投影 API 的固定系数及界必须为经过调用方认证的确切 Fraction；该 API 不负责证明这些前提，也不把区间中点直接写入原 HZ。

## 真实存档的完整组件人口

唯一输入为 D025 已完整保存的 CIFAR100-large complete_0.json，固定 12062002 字节，SHA256 为 fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0。保留原模型和 spec 的 provenance，不重新解码或执行原模型。此输入选择定义一个新的存档组件研究，不是把失败的原三模型实验缩为一模型重跑通过。

处理该档全部 3072 个 box 坐标、1600 个源 forms、五个窗口 (0,0)、(0,31)、(16,16)、(31,0)、(31,31)，以及每窗口全部 64 个 receiver channels 和 576 个 canonical slots，共 320 个读出。保留 stable、crossing、zero、padding 和没有改善的行，不按旧结果或 margin 筛选。核对原 phase、frame、canonical Conv 几何、原 weight/bias 区间和每一条旧对照的引用。旧 288-pair 分组与本候选按正项形成的组数不同，不混算人口。

每行记录新上下界、共同误差 E、相同负项规则和误差下的公平单门对照、旧 ordinary interval、旧 D025 phase-capacity 行在所有 phase 上取范围后蕴含的标量区间，以及旧两者和新界的交集。该标量投影不等于完整 D025 条件 LP，因此不能据此声称超过整个旧验证器。共同误差下 paired 不弱于指定 single 对照；新界未必优于旧 ordinary 或旧联合事实，必须原样报告。

报告全部 320 行中的 paired 严格改善数、相对旧组合上下界的改善数、旧外包跨零数及新增界稳定数。跨零是外包状态；新稳定界不是网络性质 CERT。projection API 本轮通过数学测试检验，不伪称已在真实原 HZ 终端装入这四行。

medium 缺少完整逐行记录，Tiny 当时未进入；第一残差块第二 Conv/BN 也未存齐。本轮不声称覆盖这些缺口，不改变原三模型或正式 replay 人口。完整残差集成仍需新的受控来源提取和后续晋级。

## 完整资源与证据

worker 仍限 240 秒、256M 全局工作、200M 当前 archive 处理工作、64M retained entries、512-bit 有理端点、AS16GiB、CPU1，保留 RSS 高水位增长加 65536 字节及 tracemalloc peak 加 metadata 加 65536 字节均不超过 1GiB 的双门。整个过程禁用 CUDA，线程数为一，缓存和临时目录位于新的 RUN。CPU reference 不算 GPU 资格或生产 fallback。

开始数值工作前预留全局 40M 证据工作及 65536 summary reserve。archive 字节读取、JSON 解析、整数、重复键核对、Fraction 重建、共同源身份、算术、缓存和逐行记录均计费；所有算术和循环操作在执行前付账。D025 evidence.py 的有界 ledger 与 encoder 可只读复用，但不能运行旧 writer/main。共享源缓存的单观察复杂度不能当作整层免费成本。

完整物理 ledger 包括原 archive bytes、完整解析树、转换后的 source/receiver/box/window 数据、当前 prepared cache、全部输出和 manifest；还计 capsule header、64*576+4096 的临时 numeric-entry reserve，并测量真实进程峰值。上一个窗口缓存的释放属于预注册的正常生命周期，不丢弃任何输出或输入证据。

新 complete.json 保存每一行完整结果和输入引用，冻结输入可通过其确切路径、大小和 SHA 重建，不重复序列化 12MB 原档。但所有仍持有输入根都纳入物理账本，不能将引用当作免费存储。编码先写 exclusive complete.json.partial，完整成功后才 rename。失败保留 partial、日志和小 summary，不做超预算恢复序列化。

diagnostic.json 不超过 65536 字节，包含实际耗费和停止原因；supervisor 独立核对完整输出 hash、五窗口乘 64 通道的 exact 人口和资源字段，读取输出的内存也纳入 supervisor 自身双内存门。worker 完成与整个 source census 资格分开记录，source_census_qualified 永远为 false。

## 不变目标与报告

本轮新实现的数学先例和未解决范围见 D029 THEORY.md 与 LIMITS.md。没有 attack/PGD、BaB、输入或相位 split、backward/dual rescue、LP 状态修复或实例菜单。普通终端与具体见证边界没有改动。所有原连续/二元因子、EQ/LE、共享身份和 decoder 保持原样。

日期 2026-09-30，分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。保全当前 dirty production provenance，不覆盖历史档案。正式 1870/2413 和独立 E0 CIFAR100 25、TinyImageNet 36，共 61/400 不变。始终 report formal_gain=0、native_HZ_admitted=false、gpu_computation_completed=false、complete_physical_qualification=false。通过只代表本组件及此完整存档人口，不代表新域完成、13 家族保旧、实际 GPU 速度或全量晋级。

使用 pages:write-page 技能区分预注册事实、数学假设和资格边界；只写当前本地隔离目录，不发布外部 Page。运行后另存结果，冻结文件不再修改。
