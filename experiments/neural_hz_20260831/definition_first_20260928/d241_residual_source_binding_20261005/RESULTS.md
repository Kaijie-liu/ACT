# 残差前缀原参数审计结果与实际域接入缺口

三个既定原始模型的完整残差前缀来源审计已通过。唯一运行正常退出，用时15.355831727385521秒，原模型、property、依赖和生产快照前后身份一致。此次补齐了此前缺失的Conv6/BN7及合流后Conv/BN参数和全部消费链；没有执行D240关系、建立实际HZ或产生新增CERT/ADV。

这是一项真实来源前提的完成，不是新抽象域、GPU或验证能力资格。完整Goal继续active；正式1870/2413和独立E0的61/400均未变化。

## 唯一冻结运行

四个源文件在2026-10-05 10:34:37 UTC记录冻结时间，随后写入[freeze.json](freeze.json)。冻结前只做文本审阅和文件身份检查，没有新源码import、AST、编译、collection或试跑。最终静态修正包括历史provenance三字段比较，以及恢复D179已有的ONNX模块来源核验；均发生在冻结前。

唯一命令如下，终端会话25460正常返回exit 0，未重启或重跑：

```text
/data1/Kane/miniconda3/bin/python -B /data1/Kane/FSE/ACT/experiments/neural_hz_20260831/definition_first_20260928/d241_residual_source_binding_20261005/run_audit.py --enabled
```

输出保存于[本轮独占目录](../../results/d241_residual_source_binding_20261005_v1)。[退出记录](../../results/d241_residual_source_binding_20261005_v1/exit.json)与[完整报告](../../results/d241_residual_source_binding_20261005_v1/report.json)均确认diagnostic_complete=true、status=0。报告引用D240已经通过的4169项、222文件数学记录并检查有序JUnit人口；本来源阶段没有执行或减少那些测试，也不把其资格转移给新读取器。

冻结清单SHA256为5409ca5968df7f6f1c79120f4714a225efb79e0fe87d381972c1736d08274515。RUN登记含7691项来源和14项输入，完整前后身份检查均通过。五个RUN工件按exit清单重新读取SHA256，全部一致；exit自身SHA256为9cb62fb6f392a993fa61c24acffa448101ce5ce5434fb5f82864f9c4014ad1bd。

运行后的两项独立只读复核确认：全部原7655来源保持相同SHA、14输入及D240有序4169 nodeids/222路径不变；三份源记录的BN全通道、Conv参数引用、残差主/shortcut和截止后的消费者完整，资格flags未越界。复核只查已写JSON、源码及哈希，没有重算BN包络或重跑候选。

## 完整三模型人口

每个模型均从原始输入审计到第三个原ReLU，不挑窗口或通道，不因模型差异删减人口。模型和property顺序完整继承D240登记。

| 原模型 | 前缀节点 | Conv数 | BN通道总数 | initializer数 | 逐值审核参数数 | 三个ReLU的标量人口 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CIFAR100 large | 12 | 4 | 256 | 20 | 113344 | 65536 / 65536 / 65536 |
| CIFAR100 medium | 14 | 5 | 576 | 25 | 380864 | 14400 / 8192 / 8192 |
| TinyImageNet medium | 14 | 5 | 576 | 25 | 380864 | 46656 / 25088 / 25088 |

证据分别为[source_0.json](../../results/d241_residual_source_binding_20261005_v1/source_0.json)、[source_1.json](../../results/d241_residual_source_binding_20261005_v1/source_1.json)、[source_2.json](../../results/d241_residual_source_binding_20261005_v1/source_2.json)。三者参数总数875072，所有读取值通过精确Fraction、有限性、形状及位宽检查。BN保留原gamma/beta/mean/variance/epsilon以及全部通道的外向系数包络，没有替换为中点网络。

完整输入盒分别为3072、3072、9408个坐标，身份同时绑定原model/spec SHA、batch1、NCHW和原port。每个前缀均保留一个真实残差Add及两个通向截止范围之后的消费者槽。medium的Add10/port129仍记录后续Add16的shortcut消费者，第三个ReLU/port132仍记录Conv14；并未因局部范围结束而丢弃它们。消费者身份已与完整原图及D179逐项核对，不代表这些后续节点已经执行或验证。

全部权重实际解码并审核，结果文件通过原始模型SHA、唯一initializer名、dtype/shape、TensorProto及payload标识保存无损重构来源，不重复序列化百万级Fraction。原模型仍是必需的只读来源。小规模BN原参数、包络、完整输入盒与几何直接保留。记录中的source_binding_complete仅指raw-prefix来源与几何完整，不指native-HZ绑定。

## 资源与物理范围

全程仅一个WorkBudget、一个共享40M evidence账本；模型边界不重置全局支出。CPU0、单线程、CUDA隐藏、禁字节码、AS16GiB，以及原240秒、256M全局work、200M单模型、64M条目、512位和两个1GiB宿主门均未扩大。

| 项目 | 实际记录 |
| --- | ---: |
| 完整运行墙钟秒 | 15.355831727385521 |
| 全局work | 246412868 / 256000000 |
| 最大单模型work | 80075829 / 200000000 |
| 共享evidence work | 9632292 / 40000000 |
| 完整条目保守峰值 | 7428956 / 64000000 |
| tracemalloc峰值字节 | 84247023 |
| tracer metadata字节 | 13772320 |
| 小报告保留字节 | 65536 |

三模型各自work为35697755、68464802、80075829。临时数值条目上界分别1107264、3288256、3491008，均在任何参数标量解码前预付；之后对真正保留的根作有界账本遍历。下一模型开始前释放上一模型完整根。实际物理峰值持续观察，不因临时权重释放而重置。

RSS高水位初值和末值均为354562048字节，增长为0；这不表示审计零内存。tracemalloc峰值加metadata及reserve为98084879字节。哈希读取总字节数14305019924，沿D179固定块记账；该work不是哈希CPU指令复杂度声明。全部资源记录是本来源审计范围，不是完整候选、GPU或四并发性能资格。

全局work距离256M上限仅9587132。不能据此假设额外网络传播或D240附加关系也已支付；未来新组合必须预注册并计算其完整端到端代价，不能沿用本轮通过标志。

## 数学主线与下一项实际缺口

[THEORY.md](THEORY.md)给出同一个实际H上的关系代换义务：保留原相位和全部谓词，用不移动ReLU零点的正比例归一化，保留完整r/e，包括不匹配幅值项、skip、偏置与共同误差。Q关系新增四个连续产品与十八条LE，在原二元位整数时应投影回同一个旧H，原decoder不变。这份条件定理不证明其守卫在上述真实模型成立。

已明确不能把跨Add的两层卷积在边缘位置直接合为通用5x5核；中间padding和偏置的有效路径不同。完整源审计保留两层各自几何，供后续同源展开使用，不用一个方便的合核替代原网络。

下一研究步应直接建立完整前缀的共同来源证书并测试D240实际适用性，而不是再做一轮同类来源普查。当前仍缺：

1. 原property到实际H的包含证明和输入decoder。现有生产输入构造的浮点中心/半径和exact标记本身不是这个证明；本轮没有修改它。
2. 每个原门唯一、被所有消费者共享的预激活与误差载体。BN及实际系数误差必须完整保留，不能按支路或查询重新选择误差。
3. 完整原相位列、全部读出及可靠L/U，并在同源坐标上合并r/e后求界。本轮没有这些缓存资格。
4. 将D240四观察列及有限行精确代回这个H，保留旧EQ/LE、所有bits、消费者和decoder，再测判据及真实查询收益。

D064的原Graph鉴别、D096的精确出生模板/证书、D015的来源算术可作为支撑；复用不等于原模型已认证。不能接回D228 Bank、D229独立局部盒或其他辅助算法菜单。D096当前完整schema/readout扫描与全CSR复制有自己的费用，不能逐门免费调用，也不能把该实现的昂贵常数当成任何同H桥的数学下界。后续批量认证仍需逐项覆盖与真实计费。

GPU、smooth、Transformer、新家族、同结构shadow、逐家族及全2413和外部400回放仍未完成。没有真实guard命中数、native-HZ资格或新颖性资格。上述接入仅是检验定义是否实用的前提，不应替代定义创新主线。

## 记账与存档边界

formal_gain=independent_e0_gain=new_benchmark_solves=0。正式1870/2413与独立CIFAR100 25、TinyImageNet 36的61/400不相加。没有默认启用、生产集成、commit或push。

分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，生产候选字节SHA256仍为15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75。历史模型、已冻结源码和旧结果保持只读。

上一交互轮只核实中断状态，按Goal推进口径是no progress；本轮完成完整三模型缺失参数审计并产生冻结证据，属于progress。没有运行中实验需要下一轮重启。四冻结源、freeze、六个RUN文件及本文由本目录ARCHIVE.sha256连接；结果文档是运行后记录，不伪装为冻结前合同。

文档归档技能用于按项目现有本地格式区分数学、来源、资源和正式收益，没有发布外部Page。完整目标保持active，没有缩小成功定义或提前标记完成。
