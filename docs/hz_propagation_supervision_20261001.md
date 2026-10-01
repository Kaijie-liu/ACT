# 实际 HybridZ 传播的独立预算监督

2026-10-01：冻结协议的 18 次合成 CPU 调用完成，12 项控制与 67 项回归通过，
独立归档审计通过。四个正常来源重新创建网络/HZ、执行实际 analyzer、序列化、
独立检查及接收；输出和支持界与上一轮一致。**没有新真实模型证书或 GPU 成绩。**

先行冻结提交 `fcd7df31561f3dfd268bdb6f602eef9d61b0f512`，
[协议](../configs/hz_propagation_supervision_20261001.json)、
[设计及范围](hz_propagation_supervision_design_20261001.md)、
[紧凑归档](hz_propagation_supervision_20261001_r3.json)。

## 这次接了什么

[独立监督入口](../scripts/hz_propagation_supervised.py)执行 produce → check → receive；
[worker](../scripts/hz_propagation_worker.py)在计时内创建原冻结合成对象，再调用
`propagate_checked_guarded`，没有从历史结果免费读入来源或下界。
生产入口和旧冻结执行器均未改动。

独立 checker 核对 scope、全部声明的 ReLU 层及实际使用的支持义务，对保存的候选
逐条精确重算。父进程在成功退出和清理之后锚定实际输出文件，receiver 再核对。
这不是独立重算整个 network→HZ，也不是完整 MoE 请求的路由覆盖证明。

一个调用的时钟包含创建、导入、构造、传播、支持提议/检查、序列化、独立检查、
接收、父端哈希、清理、终态写入及观察到的返回。阶段内部成本不重复加到总成本。
独立审计不重新提议或传播，不算入原请求；离线审计不能补成在线成功。

## 结果与失败分母

| 终态 | 次数 | 解释 |
|---|---:|---|
| 条件化传播执行完成 | 4 | retained、disabled、guard-discarded、two-layers |
| ERROR | 7 | 预注册异常、缺证据、错误身份及后代控制 |
| TIMEOUT | 6 | 预注册等待、序列化/检查/接收与最终发布截止 |
| RESOURCE_LIMIT | 1 | 1 byte RSS 限额的受控拒绝，不是实际 GPU OOM |

四个正常调用分别约 2.566、2.579、2.795、2.971 秒；这些仅是冷启动控制成本，
不是性能比较。正常调用重检 10 条支持界。归档审计还重检发布超时调用已留下的
2 条界，共 12 条；**那两条仍不构成预算内接受**。不因离线可检查就升级终态。

原关系控制的结果不变：保留 guard 时 0 个 ReLU 二元因子，disabled 和丢弃 guard
时各 1 个。两层来源仍有残余不稳定性。没有增加行数、时限、范围精度或选择新对象。

R1、R2 的 18 调用及源码快照保留。它们的当时控制均通过，但随后只读审查发现需
加强的截止、身份和前缀审计，因此最终接受以 R3 为准，不覆盖早期记录。R3 加固：

- 清理未确认保持 `CLEANUP_INCOMPLETE`，不被一般 TIMEOUT 覆盖；禁止后续启动。
- 成功需有有限的截止前退出观察、有效 PID、无存活后代及确认的清理。
- 已锚定输出即使随后超时也必须保留；缺包、缺侧、错误性质、错误来源均拒绝。
- 最终发布的故障标记绑定进成本账；NaN 成本或迟到 API 返回不能接受。
- 参考差分报告绑定 SHA；后代控制必须实际启动并观察其存活，不能以创建错误替代。

## 有界清理的确切边界

[新 CPU 原语](../scoped_proof/owned_bounded.py)在每个阶段内预留清理时间，
用 `waitid(...WNOWAIT)` 保留 leader PID 到发信号后，再做有限 `wait(timeout)`。
清理后重新观察本调用进程组；无关 sentinel 在控制中不受影响。

保证是 **leader 已回收、原进程组无存活成员**。非 leader zombie 单独记录，不宣称
代替 init 回收所有孤儿；本轮 descendant 控制的最终 live/zombie 清单均为空。
不支持逃逸 session，不保证内核不可中断任务或 GPU driver 的资源释放。
文件系统操作也不是硬实时可中断的，所以最终接受必须检查实际返回时钟。

成功标签是 `CHECKED_PROPAGATION_EXECUTION_CONDITIONAL_ON_TRUSTED_LOWERING`：
仍信任网络/guard lowering、共享因子含义及旧浮点传播。这是给定 HZ 的检查界参与
真实传播的完整执行控制，不是 G1 完成或整个网络严格认证。

## 验证与下一步

12 项新控制包括固定故障到达、原结果差分、独立重检、缺义务、污染、完整成本、
迟到返回、有界回收及无关进程隔离。67 项回归是传播 14、批量支持 16、设备候选 15、
导航 9、历史评审交接 4、历史证明账本 9；不包含新的 native/GPU 实验。

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -B \
  -m scripts.run_hz_propagation_supervision_controls audit \
  /data1/Kane/MOE/baseline_runs/hz_propagation_supervision_20261001_r3
```

下一步是把有界 owned-process 清理**单独接入设备监督**，控制截止和失败时的资源
归还，再按资源准入冻结实机 CPU/GPU 同算法控制。当前接口容量仍是有限合成范围，
不能直接用它启动真实/full-size 请求。native 支持比较、实际模型净收益、完整同源
证明和外部优势保持 OPEN；六个目标门均未因本轮工程控制关闭。

本阶段三个本地归档合计约 7.6 MiB，失败和变异控制保留；未删除任何文件、未安装
依赖、未触碰其他用户任务。旧核心 619 项来源绑定及历史账本检查仍通过。
