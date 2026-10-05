# 三门共同相位规则的只读适用性审计

本轮先判断上一版三门相位关系在已有真实来源记录中是否具备必要条件，不实现新的 cut 组件，不以测试数量代替真实结构。上一轮的数学正控和已封存文档属于 progress。本轮以下只读诊断先固定查询范围与解释口径，再执行聚合；不是新的模型实验、完整 source census 或正式资格回放。

## 固定证据和完整诊断范围

读取旧 D047 RUN 的 complete_0.json、complete_1.json 和 exit.json。前两份分别包含 CIFAR100 large 的 320 个、medium 的 640 个原接收门记录，各五个窗口，全部原 admitted 分支和通道均纳入。读取第三份 complete_2.json 是否存在的事实；不存在时 Tiny 保持 missing，不能从部分日志估算，不能把两模型诊断变成三模型通过。

不重读模型、不重算前向、不导入或执行 D047/D049/D053。不写旧目录，不改变旧失败状态、测试人口、资源门或正式结果。所有新文件仅在本新目录，由 apply_patch 保存。

固定诊断使用每个 pair 的 ordinary_h/ordinary_w 及其原 channel 身份，恢复每窗口全部原接收门。校验窗口数、pair 数、receiver 数、分支位置与各通道的完整唯一人口。尾部自配对若出现，只保留该原接收门一次并核对已认证 self-pair 标记。

仅按可靠预激活区间端点符号分类：lower>0 为 strictly_active；upper<0 为 strictly_inactive；lower<0<upper 为 strictly_crossing；其余合法触零情况为 zero_touch。zero_touch 不固定任何原 bit。分类只使用 JSON 整数的正负/零，不以浮点近似比较两个大有理数；原有理区间及原输入真实性依赖冻结来源，不用本诊断重新认证。

每窗口报告各类数量和 unresolved 的原通道位置，其中 unresolved 为 strictly_crossing 或 zero_touch；统计任取三个 unresolved 通道的组合数，以及原顺序三个连续通道都 unresolved 的个数。只计数，不枚举相位、不构造三角证书、不计算候选增强。还报告整个已记录模型范围上的任意三门组合上界，以免把同窗口结论误报成全模型结论。

这些数是必要条件的规模上界，不是实际相位图的负圈数，不是满足全部真实前提的候选数，更不是新解。坐标仅供审计，不授权据此选择未来实例或通道。未来新真实实验仍需原完整三源人口、预注册和全部原门槛。

## 使用的结构命题

令 a=d12>0、b=d23>0、c=-d13>0，tau=min(a,b,c)。旧三个独立双门投影相加的上界为

```text
O=a*min(b1,b2)+b*min(b2,b3)-c*max(0,b1+b3-1).
```

新三角上界 C 与 O 的差是 tau 倍

```text
D=b2-min(b1,b2)-min(b2,b3)+max(0,b1+b3-1).
```

当 b1 固定为 0/1 时，D 分别为 b2-min(b2,b3)、b3-min(b2,b3)；b2 固定为 0/1 时，分别为 max(0,b1+b3-1)、1-b1-b3+max(0,b1+b3-1)；b3 固定为 0/1 时，分别为 b2-min(b1,b2)、b1-min(b1,b2)。六者均非负，所以旧上界不大于新上界，新规则的全部仿射行均冗余。

这只说明已知共同 phase-cycle 机制在可靠稳定事实已安装时没有增益。原 bits、guard 和 decoder 全部保留，证明中的代入不授权删除或 pivot bit。零点合法相位不能随意固定。若 production LP 没有正确绑定这些稳定事实，尚不能称其实际已经蕴含；修复绑定本身也不是新域贡献。

如果至多剩两个不确定 bits，唯一非平凡 overlap 的全部上侧消费者系数同号，则同一 McCormick 极值同时扩展各独立投影，同类纯相位圈增强无增益。异号消费者或额外 source/guard 条件关系不在该结论内。

## 配置与 provenance

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。配置只读归档聚合，不运行数值 worker、GPU、LP 或测试。查询 audit.jq 的哈希在首次聚合前记录，输出以原样 JSON 另存，不修改查询以追逐结果。

```text
0743a8aef9496ab484466e727a3b6f379fd9da84143d129dbbd0e1db5a362f93  complete_0.json
7728a92c5fb800eb3ca71f4f49c2102c0f0e8f640a48b02531e095bf976c1e76  complete_1.json
b2e3f2682c830249199c40fc0ad322b10f21c478340aa3c62138184852371dc3  exit.json
47e2ecfc2ce163a327feac6cd3faf1cb7d2f9e0d3d03201e573812c5bdf5b41c  D047 seed_relation.py
```

正式 baseline 1870/2413 与独立 E0 61/400 不变。新颖性、数学证明、归档诊断、真实数值资格和正式成绩分别记录。按 write-page 技能保持这些范围清楚；本页不发布为外部 Page。
