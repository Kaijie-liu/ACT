# 共同相位表研究记录与真实来源边界

2026-10-01，分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。权威目标仍为GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md；Goal active，未改变目标、资源、保旧或晋级要求。

上一Goal轮分类为progress：已结束D064的唯一运行得到3805项数学通过并完成归档；文献审查明确多消费者关系缺口。本轮新增[数学证据](THEORY.md)改变下一步选择：关闭无增益的裸乘积lift，保留有严格对照的共享相位表构造，尚不授予实现或真实模型资格。

## 复核与修订过程

根代理推导接口、四槽前向规则、最终普通控制、物理输出性质和三组完整双门凸包见证。一个独立代理核对相位选择器障碍、六行乘积及no-gain延拓、投影和代价；另一个独立代理逐项复核正控算术、原门松弛、四个独立表、pair-hull见证、取等输入及两后继跨零事实。

纸面设计中曾考虑c=1/10、G=q1+q2-p-x-z的例子；其固定四平面G上界虽有效，但一个廉价标量证书G<=11/5就能排除假点，故不能支撑主要强对照。它没有被实现或运行。最终改为THEORY中的c=1/2、G=q1+q2-p/2-y；四个全局输出上界均由真实输入取等，修复了该比较缺口。这个修改发生在任何数值执行前，不是修改已冻结实验结果。

原文只补核Sharp Hybrid Zonotopes的RLT范围；D029、D035、D042、D059、D062以及此前文献记录均只读。没有穷尽式新颖性调查，不宣称论文级创新已通过。

## 可复用真实参数与不能外推的部分

只读完整旧包results/d025_interval_capacity_20260930_v1/complete_0.json，SHA256为fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0。它属于失败三源普查中已经完整保存的large单模型证据，原失败状态不变。

已保存Conv_0 [64,3,3,3]、BN_1、Relu_2端口127，以及下一Conv_3 [64,64,3,3]、BN_4、Relu_5端口130。两个Conv均group=1。完整已存人口为五个固定空间窗口乘全部64通道，共320行；每窗576个canonical槽，由64个接收通道共同使用；result.source_forms有1600个条目。不是整层所有空间点。

对应字段为packet.first_conv/first_post_ops/first_relu、packet.branches[0].conv/post_ops/target_relu、box、result.source_forms、result.receivers和result.windows。共同源、接收权重和可靠区间足以作为新的两层结构评估输入，但本轮没有计算候选新界或选择阳性实例。

后续Conv_6、BN_7、Add_8(132,127)到133及Conv_9、BN_10仅保存图拓扑，没有对应参数解码。model_raw只有字节数与hash，不是原模型字节。因此现包不能证明完整残差合流或第三Conv的候选收益。D045的46080个源对乘消费者对是未执行的预注册设计，不是已得到的结果；medium/Tiny缺失部分不借large填补。

下一次真实候选需要预先固定完整结构人口、端点生成方法、共同身份以及全套预算，记录无改善和不适用项。不能按本页正控或某个阳性坐标选菜单；也不能将新单模型分析补记成旧三源通过。当前资料定位不授权重跑消耗版本或放宽预算。

## 实际操作与资格

配置为paper_only、read_archived_json、primary_source_check。只读shell/jq定位来源和hash，未导入候选、编译、collection、单元测试、模型执行、LP/MILP、攻击、采样、GPU或benchmark回放；没有新后台进程。不存在运行通过次数或性能测量。

只新增本目录THEORY.md、RECORD.md和SHA256SUMS。原九项tracked dirty changes保持3806 insertions、57 deletions，未改生产代码或/data1/Kane/HyZor，不commit/push。原D064及此前冻结源码、日志、清单全部只读。

正式1870/2413（1063 CERT加807 validated ADV）、独立E0 CIFAR100 25和TinyImageNet 36均不变，formal_gain=0。尚未验证真实模型接入、实际非冗余、全端到端代价、GPU、同结构shadow或任何保旧回放。目标未完成，保持active。

使用write-page技能区分证明、已知机制、实际执行和未证收益；交付前读回本地文档并检查校验清单，不涉及外部Page发布或渲染预览。
