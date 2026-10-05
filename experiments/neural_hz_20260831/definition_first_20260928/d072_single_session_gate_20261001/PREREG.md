# 同一进程内执行完整数学人口的隔离验证契约

本版本只改监督执行，不修改D070的数学候选、四项新测试、旧3809项测试或其身份。D070 v1在完整收集后联合60秒门超时，永久保留FAILED。它的源码、freeze及RUN只读；本版本不能将其失败改成成功。

当前双调用先collect-only，再在第二个pytest进程重新收集并执行。新契约用一次pytest进程，在pytest_collection_finish中核对完整3813个node IDs和180文件后才允许执行，父监督器结束后仍逐项核对完整JUnit。省去的是重复启动和收集，不是人口、验证项目或计时阶段。两路独立静态审查已核对本机pytest时序：该hook读session.items，不读尚未更新的testscollected；失败直接抛非零异常，不只设置shouldstop/shouldfail。

## 固定执行与成功条件

唯一RUN为experiments/neural_hz_20260831/results/d072_single_session_gate_20261001_v1，显式--enabled首次创建即消费。冻结本PREREG、run_math.py、collection_contract.py三文件，并认证D070的全部六份原冻结文件、freeze、失败exit及前序成功人口。冻结前不导入候选、不compile、不collect、不做数值预跑。D070核函数和测试必须逐字不变。

插件在collection完成时逐项核验与原D070一致的有序人口、唯一性、原实际文件路径及无collection错误，并独占写出inventory；只有成功关闭证据文件后才允许进入测试循环。循环入口复核未发生后续人口漂移。参数测试仍逐node ID核验，新增四项仍作无装饰无参数的AST检查。禁止挑选、过滤、调序、skip或只运行新测试。

父进程从启动这个唯一pytest子进程前计时到其退出，含启动、导入、收集、hook验证及写盘、所有测试和JUnit收尾，总计仍限60秒。CPU1、库单线程、CUDA不可见、AS16GiB、监督器RSS增量加reserve及tracemalloc峰值加metadata加reserve各不超过1GiB，reserve65536，全部缓存/临时数据/JUnit只写新RUN。完整物理资格不因监督器通过而获得。

成功还要求pytest退出0、3813个JUnit逐项匹配、零failure/error/skipped、完整inventory身份一致、全部source/input/production在运行前后零漂移、artifact哈希完整，以及原4417个GPU依赖和1011个decoder依赖人口不变。没有独立额外collect命令或重试。任何缺失证据、插件未执行、错误或超时均fail closed并保留版本。

## 声明边界

这不是新域、精度改进或纯速度晋级，不改数学算法与resource gate，也不把测试启动开销减少算成验证器提速。通过仅授予本执行配置下的D070数学组件资格；旧D070 v1仍FAILED。原D064成功、D047/D057失败分别保留。无模型/source worker、LP/MILP、GPU、native HZ admission、shadow或全量回放。

原256M whole、200M branch、40M证据、64M retained、512位及240秒worker门保持原范围；本版本没有worker，不授予这些尚未执行的全系统资格。通过后下一步才是完整真实结构的付费预注册，不继续扩张相似测试。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413及独立E0 CIFAR10025、TinyImageNet36不变，formal_gain=0。仅新隔离文件及唯一RUN；旧档、生产和远端不改。Goal保持active。
