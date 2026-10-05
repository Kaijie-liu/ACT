# 联合残余真实来源候选预注册

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。本轮承接 D263 理论和 D264 数学核，上一用户回合只核对状态，不作为研究增益。历史目录及生产文件只读，所有新输出隔离在本目录和新的 results/d265_joint_source_20261006_v1、results/d265_joint_source_20261006_sources_v1。

冻结前只允许源码/文档编写、文本审阅及身份读取。候选 import、AST、compile、收集、测试、来源数值计算均须等待完整 freeze.json。冻结清单包括 CONTRACT.md、PREREG.md、source_arithmetic.py、joint_pair.py、jp_certificate.py、source_observer.py、test_joint_source.py、run_math.py、collection_contract.py、run_audit.py。冻结后不得修补或重跑本版本。

数学人口为最后成功4285项/231文件原序，新增test_joint_source.py中的四个定向测试，合计4289项/232文件。测试覆盖共同组/原epsilon与bias、固定Gram及残余舍入、JP精确正负对照、默认关闭/资源拒绝与唯一summary。测试中的固定有理数或枚举只是独立oracle，不成为候选运行时helper。旧证据写入必须由新plugin完整重定向到新RUN，前后源码和14输入身份全部核验。

来源人口沿用 D261 全部三个原始模型/性质与D241来源身份，完整前缀和未来消费者不变；不改变模型、父对、位置或mask人口。数学成功和成本预检成功是来源执行的两个共同前提。冻结前源码审查已证明本实现的必做准备费用过门失败，因此freeze明确source_static_preflight_passed=false，本轮不启动来源worker。source_audit_stage_registered=true只表示入口已随源码冻结，不授予执行资格。所有费用、墙钟、物理观察及证据条件见CONTRACT；数学通过也不改变这个停止决定。

报告区分：数学组件通过、来源完成、联合付款下降、已有hull蕴含、尚未排除，以及尚未取得的原生/全网/GPU/正式资格。只要没有完整同路径网络回放，formal_gain、independent_e0_gain和new_benchmark_solves始终0。不得把not_excluded改名为严格强化。
