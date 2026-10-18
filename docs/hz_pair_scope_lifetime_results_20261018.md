# 同 pair 生命周期 R1：失败留档及已完成对照

结论：**整批 FAIL，11/12 测试方法通过；不推广、不自动重跑。**
默认关闭的实现已提交，但初始 factor 映射污染控制尚未完成。本轮没有真实模型、
GPU、原生优化器、依赖变更或新证书收益；G1–G6 均未因这些控制而关闭。

## 身份与运行边界

- 冻结执行：`7477c2d2be47fa1083b5dd3c4d77576d0ed3f0b7`。
- [协议](hz_pair_scope_lifetime_controls_20261018.md)及
  [163 文件冻结清单](hz_pair_scope_lifetime_freeze_20261018.json)。
- 原始目录：`/data1/Kane/MOE/results/pair_scope_lifetime_20261018_r1`；保留所有来源、失败日志和终态。
- [紧凑归档及逐文件身份](hz_pair_scope_lifetime_results_20261018.json)。
- 单次批次终态约 36.388 秒，最终制品 1,751,708 字节，进程组采样 RSS 峰值
  41,218,048 字节；无超时、资源停止、迟到终态或报告解析错误。

启动前检查：可用内存约 55 GiB、所在磁盘剩余约 271 GiB；其他用户任务未受干预。
当时 ACT 目录约 52 GiB、独立 results 目录约 1.2 MiB。本批新增约 1.67 MiB，
没有删除冻结来源、失败或其他数据。8 GiB 地址空间、2 GiB 采样 RSS、原 Work／制品／
位数及 300 秒限额均不变。

六输出来源完整有理数编码的 18,081,728 字节需求在冻结前已被静态拒绝；原候选累计
限额 16,769,024 字节未改。它仅运行首 tile 的有理数轨迹控制，没有被计成全目录完成。

## 失败及覆盖缺口

`test_07_initial_canonical_faults` 在 `selected > 1` 断言处失败，实际为 1。
guarded 是一维纯仿射、无 ReLU 的固定来源，仅有一个输入 factor，不满足双 factor
置换的夹具条件。这一前置条件在执行前的人工／AI 静态检查中被遗漏。

| 同一测试方法中的子用例 | 实际覆盖 |
| --- | --- |
| nodes／missing | 已执行预期拒绝及 `traversal_ready=False` 检查 |
| nodes／permuted | 已执行预期拒绝及 `traversal_ready=False` 检查 |
| factors／missing | 在夹具断言停止；尚未污染或调用检查器 |
| factors／permuted | 未进入 |

这是测试覆盖缺口，不是“污染后的 factor 已被接受”的证据，也不是确认实现正确的理由。
其余 11 个方法通过，包括完整差分、真正跨 tile 的缺证据、别名／身份污染、旧 handle
失效、末 pair tile 截止／异常、部分报告解析、容量拒绝、发布异常和搬迁。
部分报告解析控制并非操作系统强杀的实测，不扩张其范围。

冻结审计器原样执行，正确拒绝 `complete execution`；没有修改它绕过失败。
后续只读[失败记录审查器](../scripts/audit_pair_scope_lifetime_failure.py)核对保存制品身份、
完整分母、角色 max／端点 min 及成本算术，输出 `FAILED_BATCH_RECORDED`，不是 PASS。
审查耗时约 0.049 秒，独立于本批在线计时，不能补成在线成功。

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B \
  scripts/audit_pair_scope_lifetime_failure.py \
  /data1/Kane/MOE/results/pair_scope_lifetime_20261018_r1
```

## 已完成的四臂记录：仅作描述

12 条完整四臂记录、3 条 oracle 记录（656 次角色比较）、2 条搬迁接收均已保存。
同来源／同候选的候选 hash、精确 receipts、结果和算术计数相同。离线复核验证了
这些记录的一致性，没有重新生成来源或重新执行 oracle。

| 来源／编码 | 原 global | 原 scoped | 保留 global | 保留 scoped |
| --- | ---: | ---: | ---: | ---: |
| guarded／0 | 9,522 | 9,819 | 9,780 | 10,065 |
| guarded／1 | 10,554 | 10,851 | 10,812 | 11,097 |
| eight_private_six_outputs／0 | 1,314,572 | 1,024,100 | 1,279,824 | 952,196 |

表中为完整消费路径计账的 Work，不是时间、内存或真实模型性能。

- 多 tile 来源：scope 构造／释放从 56 次降至 28 次，实际续租 28 次。global 路径生命周期
  净节省为 34,748（已扣除新增 lease／seal／plan）；retained global→retained scoped 节省 327,628。
  原 scoped 本身已有 290,472 的遍历节省，**新生命周期相对原 scoped 的增量为 71,904**，
  不能把全部 327,628 都归给本轮改动。
- 无跨 tile 机会的 guarded：保留 scoped 比保留 global 多 285 Work；比原 scoped 多
  246，比原 global 多 543。不是所有形状都受益，不能默认开启。
- `scope_work + global_seal_work + lease_work + plan_work + setup_work + selector_work
  - planned_iteration_saving` 的差量完全解释四臂 Work 差异。扣除后共同成本分别为
  9,252、10,284、1,238,412；没有默认生命周期费用抵消。
- 按表中三个来源／编码组合，四臂各自相同的 owned payload 峰值分别为
  125,700、158,724、492,826 字节。
  这不是完整 Python／原生堆或峰值 RSS 的不增长证明。

所有臂的数学结果也相同：六输出合成来源 140/140 性质为正；guarded 两编码分别为
0/12、2/12，两个完整请求都为 UNKNOWN。新增完整正结果为零。
记录中 `CAPTURED_REAL_SOURCE` 指捕获声明的实数语义，**不是训练所得真实模型**；
`checkpoint_correspondence_checked=False`、`native_float_proof=False` 保持可见。

完整运行四臂秒数在 JSON 中保留；单次组件计时受执行顺序、控制断言和系统负载影响，
不报告真实请求加速比。独立 oracle 在各臂计时之外、统一批次预算之内。

## 停止与最小后续决定

本批结束，保持 FAIL、默认关闭及原冻结文件。无需新增来源或继续刷性能控制。
若另行批准修复，只将两个 factor 污染子用例指向本批已经存在、含私有 ReLU factor 的
`eight_private_six_outputs`；保留断言及全部拒绝门。先冻结新执行身份和有限复测清单，
不能在旧目录补写、把两轮拼为全通过或重跑后覆盖本批。

即使覆盖修复通过，13,922,736>8M 的完整来源检查下界及 membership 限制仍未消失。
本批不能用于完整八专家训练拓扑准入、GPU 批量候选验证、G1 关闭或外部竞争性升级。
接下来不能继续增加包装层来替代这些未达成的研究目标。
