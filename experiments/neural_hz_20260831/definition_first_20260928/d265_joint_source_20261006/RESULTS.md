# 共同来源程序数学通过但未获真实来源执行资格

完整数学回放4289项、232文件全部通过，零failure、error或skip。测试进程53.52946017868817秒，低于原60秒门；pytest日志52.36秒，监督器总68.24817442521453秒。新增结果是完整小结构上的来源程序与精确证书检查，不是真实网络新解。

## 新实现和实测覆盖

实现从原父卷积、BN、两层残差和原skip来源生成共同残余关系。原BN epsilon、Conv/BN bias、共享A坐标和选中Q的舍入残差均保留；负tau只交换父顺序，不交换两个子输出。使用一个固定名义模板Gram选择声明系数，每mask实际残余仍完整支付。没有攻击、PGD、BaB、输入/相位split、backward、dual rescue或新求解器调用。

新增四项测试的同一正例预算为672,240 work、361,780累计entries。identity skip和步长二projection skip两类完整小结构，各有18个pair/mask记录，全部eligible；每个结构用两份精确有理数完整前向赋值检查关系，原BN误差也取非零正负端点。这72次行检查不是穷举健全性证明，健全性还依赖已写定理及可靠算术，也不是原始benchmark实验。

共同分组的负BN尺度、bias以及64组区间端点oracle通过。固定Gram正控为32/65和−4/17，选取的binary64常数在其可靠包络内；完整残余未被舍弃。JP的100个真实图点检查通过；宽残差负控同时满足“联合付款严格下降”和“新关系冗余”，精确冗余余量1/16。

## 原人口与隔离

冻结时间2026-10-05 21:44:54 UTC。唯一数学RUN为results/d265_joint_source_20261006_v1，session69024完成且exit0。保留原4285项/231文件顺序，仅追加四项；13个警告均来自继承测试。没有冻结后修改、另行收集、补跑或来源数值执行。

执行认证8146个源码/依赖身份和14个输入身份，source_drift与input_drift均为空，生产provenance未变。旧writer保留原隔离并新增D264 summary重定位；历史源码/证据无覆盖。CPU0、单线程、AS16GiB和双宿主观察门保持。监督器traced peak为21,103,976字节、tracer metadata为7,235,184字节；这不是全网或GPU物理资格。

summary在监督器后置检查前写入，故local_joint_source_math_completed=true而joint_source_math_passed=false；最终exit完成检查后才将后者置true。exit的all_registered_stages_passed只表示该数学入口的阶段完成，不能解释为真实来源阶段已通过。

## 来源阶段明确未通过

冻结前静态审查已经证明本实现的三项必做准备下界68,890,624 work，超过可替换预算空间上界66,679,110，且尚未计Gram、共同支持等。详见COST_PREFLIGHT.md。本轮未创建sources RUN、未启动真实来源worker；freeze中source_static_preflight_passed=false和入口保护保持，不因为数学成功重写它们。

因此本版没有新的三来源数值普查、原生HZ安装、完整终端、GPU、shadow、2413或400回放。最后完成的真实来源普查仍是D261，而不是D265。

## 正式记账和研究决策

正式仍为1870/2413，即1063 CERT和807 validated ADV；独立E0为CIFAR10025、TinyImageNet36，共61/400。formal_gain、independent_e0_gain、new_benchmark_solves均为0。此次数学通过不能作为13家族正式零回退证据。

新增WIDE_RESIDUAL_THEOREM.md证明一整类普通宽残差结构中，联合付款虽严格改善，JP仍被旧普通约束蕴含。下一步不继续把本实现的预算打磨当作域创新；优先研究是否能在原非凸相位/共享来源语义中避免两个残余守卫各自取最坏值，并先给指定旧强参照下的严格分离证据。这个问题尚未解决，也尚未注册新的数值候选。
