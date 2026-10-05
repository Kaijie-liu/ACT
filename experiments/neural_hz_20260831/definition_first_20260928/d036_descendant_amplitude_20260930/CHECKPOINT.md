# 跨层关系研究检查点

上一 Goal 回合分类为 progress：D035 形成共同两相位接口、真实源条件信息反例及已知 BQP 的准确定位，改变后续研究方向。本轮也为 progress：[数学记录](THEORY.md)证明自由观察路线无投影收益，给出强逐门凸化对照下的物理输出正控，并明确宽 fan-in 上的充分冗余条件。没有以状态汇报或未执行计划代替这些证据。

## 本轮决定

不实现没有实际消费者的 D035 条件观察包装。保存无需新变量的跨层幅值关系作为已知机制的前向证书组件，未称为新 Neural HZ 域或新不等式类别。正控是普通两层 ReLU 与混合线性输出；旧侧允许 63/100，新行证明 3/5，并有明确具体输入达到上界。各单门乃至以完整父凸包为源的子门 hull 都不能单独排除旧点，未缩放跨层相位差分行也通过。

新的区间系数版本允许在新增行中使用认证端点，不修改原网络系数。这绕开了该组件对 BN 中点替代的依赖，但不豁免 loader、原参数、frame、向外舍入和原谓词正确性。

宽卷积没有被宣称成功：若每条边的幅值都不超过独立盒正负总容量，廉价全边下界会全部冗余。必须先预注册真实结构适用性审查，再决定是否实现更强的共同源证书。不能凭已知 five-window 潜在人口上限假报实测非零边数或覆盖整个网络。

## 检查和执行边界

根代理与两个独立数学审查核对了无增益扩展、perspective 局部投影、正负幅值行、真实网络正控、完整父 hull 混合见证、原坐标输出及区间系数健全性。另一审查核对了旧相位容量覆盖范围、全边代价和宽 fan-in 冗余条件。

本轮仅执行文件读取、版本状态/哈希核查、一手文献阅读及本目录文本写入。没有测试执行、候选导入、模型 forward、LP/MILP、输入或相位搜索、GPU、shadow 或全量 replay。所有有限凸组合均为数学证明，不是验证算法的运行时分裂。没有运行中实验需要等待，没有创建数值预注册或首跑结果。

使用 pages:write-page 将已证系统、强比较对象、已知先例、规模条件与未验证资格分开记录；仅保存本地文档，不发布外部 Page。原文核对覆盖 Günlük、Linderoth 作者稿相关章节，不声称全面复现或创新认证。未使用模型性能、公开 sat 标签或终端状态作规则菜单。

## 版本和正式记账

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置为精确纸面推导、相关文献与只读独立复核；无新增执行依赖。原 tracked 九个修改保持 3806 insertions、57 deletions，未改生产代码、默认配置、旧模型或结果，未 commit/push。D033 未执行草稿保持原状。

本轮 get_goal 已返回 active，与新的继续目标消息一致；D035 记录的 paused 是其当时核查结果，旧档不改写。本轮不重设、暂停或完成 Goal。

正式 baseline 仍为 1870/2413，即 1063 CERT、807 validated ADV，全部旧解及 13 家族必须保住。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400，不与 1870 相加。本轮 formal gain=0。没有候选实现资格、GPU 加速声明、正式新增 CERT/ADV 或默认晋级。

连续源身份、原二元因子、EQ/LE、共享 latent/frame、输入重构和 fail closed 均保持。无 attack/PGD、BaB、input/phase split、backward/dual rescue、LP 状态修复或原 bit 删除。数学局部投影不会改变实际域的原 bits。完整 2413 与独立 400 的保旧、零无效 ADV 和性能门均未执行，不能声称目标完成。

只读输入 SHA256：

```text
0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c  GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md
5ffe411a62b9f0713c5c14a756b8f698ffd018904ae572b7369ceade6eff7d83  d035_cross_phase_source_20260930/THEORY.md
ac1e12326f11439f3ce300b1749b8c9251634ef324b0f9c6cd60a8ccb1992596  d035_cross_phase_source_20260930/CHECKPOINT.md
f573076e11dddc4de92eada9c9956ed91e179680b3e9f545174d67c4903a7e3e  d022_phase_coherence_20260930/PHASE_COHERENCE.md
65879b0d8d8a4eedd8e204f2d53a0fc8814e907677c0efb3bdfec7e4a2f01bda  d029_conditional_observations_20260930/THEORY.md
5538f41275a93bcf7e1f268085128431a99f0a1d82ad2c28bf7dc19571903b60  d025_interval_capacity_20260930/DEFINITION_AUDIT.md
```
