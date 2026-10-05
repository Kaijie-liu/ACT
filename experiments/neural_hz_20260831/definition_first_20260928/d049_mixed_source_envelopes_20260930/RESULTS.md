# 混合源包络数学组件执行结果

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。冻结后完成唯一一次执行，数学组件资格通过；这不是完整 Neural HZ 创新、真实网络收益、native 绑定或 GPU 资格。

## 实际结果

- 完整测试人口为 3785 项、173 个文件，包括原 3781 项及本候选四项；没有删减、skip 或重跑。
- pytest 报告 3785 passed、13 warnings，执行 44.55 秒。收集加执行 57.76604776829481 秒，满足原 60 秒限制。
- 监督器总时间 66.34476432576776 秒，包括测试窗口外的认证、存档与身份核对；不能把两种时间混为一谈。
- 终态 supervisor_exit=0、mathematical_component_gate_passed=true。源码和输入 drift 均为空，production provenance 未漂移。
- 监督器 host 门通过；traced peak 为 14291380 字节，tracer metadata 为 4917456 字节，另有原 65536 字节 reserve。该遥测不是完整候选及子进程聚合物理峰值资格。

固定命令为：

```text
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d049_mixed_source_envelopes_20260930/run_math.py --enabled
```

执行入口首次独占创建 [结果目录](../../results/d049_mixed_source_envelopes_20260930_v1/exit.json)。进程句柄 2936 已返回 exit 0；随后限定进程检查未见该候选存活进程。不因观察等待或输出延迟重启。

## 通过证明了什么

四项新增数学测试核对混合 source 与原 bit 包络、纯 bit 与 D046 的一致性、身份保留、普通非零偏置强对照及后继混权传播、LE 行归并与 fail closed。详见冻结的 CONTROL.md、THEORY.md、PREREG.md 和 test_mixed_source.py。

有限测试补充纸面证明，不是全网健全性认证。原源坐标、原相位方向、网络与 decoder 的 native 绑定仍由未来集成承担。测试中的分数松弛见证不是具体 ADV；其严格间隔不是 benchmark gain。

## 未取得的资格

本版本没有 source census worker。source_census_completed、source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、complete_physical_qualification 全部为 false。all_stages_passed=true 仅指本版本已预注册的数学阶段。

原 D047 的数学通过与三源 census 失败均原样保留。其 Tiny 阶段 whole work 超限没有被补跑、洗白或缩成人口更小的成功；旧源码、日志和结果没有改写。

组件保持默认关闭。正式成绩仍为 1870/2413，1063 CERT 加 807 validated ADV；独立 E0 为 CIFAR100 25、TinyImageNet 36，共 61/400。本轮 formal_gain=0，没有 shadow、逐家族、2413 或独立 400 回放，也没有生产默认、commit 或 push 变化。

## 存档与研究衔接

冻结六文件在执行前完成主代理与独立静态核验。完整日志、JUnit、人口清单和 provenance 自动保存在新的结果目录。失败和拒绝分支仍 fail closed。

该数学组件可作为下一项共同源与原相位关系研究的支撑，但新增关系语言、结构定理、真实适用性及完整成本仍需分别证明。另见新的接口充分性研究记录；不通过继续增加测试数量来替代定义创新。

按 write-page 技能将数学资格、未完成的集成和正式成绩分开记录；文件已读回，无外部页面或渲染预览。
