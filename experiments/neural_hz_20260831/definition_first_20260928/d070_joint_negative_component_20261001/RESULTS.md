# 共同负项组合完成实现但完整数学门超时

本版本完成默认关闭的共同负项支持器及相位接口，经过独立静态审查后冻结并执行一次完整数学门。结果为失败：收集与测试的联合60秒预算耗尽，mathematical_component_gate_passed=false。不能将源码完成、成功收集或未见断言失败当作数学资格。

## 已实现与审查范围

hinge_support.py 实现完整盒上的 A−ReLU(H) 支持，使用持久AVL前缀聚合及锚稀疏更新，处理分母变号、变零、固定和未使用坐标；完整decode核对源盒与原目标值，但它不是原HZ或ADV见证。negative_interface.py将上下八次支持接入D066原Table、实际原bit参数误差及共享delta前向接口，不调用旧condition选择救援路径。

根代理通读四份新源码，两个独立静态审查分别核对AVL/支持/解码，以及相位接口的符号、身份和误差。测试作者与根代理核对独立小维几何参考、混权区间参数、原零点标签和资源拒绝。冻结前只作源码、AST、哈希及纸面检查，未导入、compile、collect或预跑。冻结前修正了一个tie fixture的措辞并增加所选见证H=0核对，没有减少人口。

继承准备算术维持约分后Fraction的512位检查；新支持器还检查其显式整数中间运算及比较乘积。不把支持器的更强检查泛称为所有旧准备算术均采用同一内部实现。原源码、原连续因子、全部原bits、EQ/LE、frame和decoder的保留前提不变，当前仍只是形式接口而非真实模型绑定。

## 唯一运行及实际证据

唯一命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d070_joint_negative_component_20261001/run_math.py --enabled`。RUN为 `results/d070_joint_negative_component_20261001_v1`。会话73763已返回exit_code=1；监督器与pytest PID均已不存在，没有遗留运行任务。没有重跑或修改冻结源码。

- collection完整核对3813个node IDs、180个文件，与继承3809项及新增四项完全一致；日志报告12.62秒。
- 随后的pytest执行被联合时间门终止，failure.type=TimeoutExpired，test_wall_s=60.03481234051287，监督器总wall_s=74.08799614571035。总时间还包括依赖认证与收尾，不是纯候选运算时间。
- tests.log只留下进度，没有JUnit或完成总结。进度共3230个点，未见F/E；按收集顺序推断停止在旧C111测试附近，而四个新测试排3810–3813。它不是逐项完成证书，不报告“3230项认证通过”，也不宣称新测试已执行或通过。
- 6721个source身份、9个input身份、4417个GPU依赖和1011个decoder依赖按冻结契约核验。source_drift=[]、input_drift=[]、provenance_drift=false。
- 两项监督器主机观察门通过，tracemalloc峰值14342927字节、metadata4274064字节；它不等于完整pytest/模型/HZ/GPU物理资格。
- 根代理独立重读manifest和inventory并验证全部六项已保存artifact SHA一致。D064数学成功和D047/D057普查失败保持各自原状态。

本轮证明了执行管线尚未在原预算内完成所需验证，而不是证明支持定理错误或新算法慢。缺少逐测试时间和完成证据，不能进一步归因；没有为了获得诊断而启动额外候选运行。

## 后续行动与未取得资格

暂不启动真实结构、模型或GPU运行。数学、native、实际模型绑定、真实source census、完整物理及正式收益资格均未取得。原1870/2413与独立E0 CIFAR10025、TinyImageNet36不变，formal_gain=0。

当前监督器先用一个pytest进程收集，再用第二个pytest进程重新收集并执行，重复支付启动与收集费用。可另行审查新的隔离执行契约：在同一pytest进程中先核对完整人口，再执行并逐项核对JUnit，保持全部3813项、60秒和原资源限制。这是待审监督改进，不改变本次失败，不授权删检查、放宽预算或重跑本版本。

通过数学门之后，真实范围已明确：D025 large完整归档可覆盖固定五窗口、64消费者、全部92160条原slot映射，其中34816张双真实锚表。D047 medium记录缺少完整源与接收参数，Tiny无完整包；不能按阳性统计改选人口。D067的固定配对负结论仍有效，不重启共享delta额外收益实验。新种子比较必须保留完整fan-in、BN区间参数、稳定事实兼容槽和原next-ReLU，另行支付全预算并预注册。

2026-10-01，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。只新增本隔离目录代码、文档、freeze及唯一RUN，未修改生产、旧模型或历史结果，未commit/push。write-page技能用于明确失败、推断与未取得资格；记录读回核查，不发布外部Page。Goal保持active，整体目标未完成。
