# 会话观察器修复控制结果

2026-10-20，一次冻结控制完成，四组全部通过，独立保存记录审计为
`PASS_SAVED_OBSERVER_CONTROLS`。这关闭了本次观察器修复的控制阶段，未运行算法性能对照。

- 执行提交：`6f5521f7f05988f59562ee5791f9bc3bcf5a20e6`。
- [冻结](hz_owned_observer_freeze_20261020.json) SHA-256：`3219b0c0d1fbe0a14f1ae9d7e39b9f9b590d00bb4101ee5286edc8c01d717fe5`。
- [完整紧凑记录](hz_owned_observer_records_20261020.json)、[保存记录审计与文件哈希](hz_owned_observer_audit_20261020.json)。
- 运行目录：`/data1/Kane/MOE/results/owned_observer_20261020_r1`；相邻 `.stdout.json`、`.stderr.json` 保存导出和时间标记。

## 四组结果

| 控制组 | 子情形数 | 结果 | 实测行为 |
| --- | ---: | --- | --- |
| 确认退出 | 4 | PASS | 同身份终止或确认消失可排除；活态 12 KiB RSS 保留 |
| 不确定时拒绝 | 8 | PASS | 活态缺 RSS、缺 status、身份改变、权限／解析／长度错误、截止均拒绝 |
| 两阶段错误保留 | 2 | PASS | 实际执行保留 ValueError 与清理 PermissionError，最终列表未知；错误字段有界 |
| 部分观察与禁止升级 | 2 | PASS | 部分扫描清除旧列表；另一实际执行清理成功后仍为 ERROR |

第一条实际执行的 `CLEANUP_INCOMPLETE` 是预期故障结果，不是一次成功运行。
随后控制发现该子进程已不可 `waitpid`，记录 `ALREADY_NOT_WAITABLE`、退出码 `null`，只读观察无存活成员；不把后处理回填成原收据清理成功。
第二条实际执行的 leader 已回收、退出码 -9、无存活会话成员，但保留原执行 ERROR。
结束时自有直接子进程为空、FD 为 4→4。确定性读取控制不是实际内核竞态发生率实验，也没有证明 R3 只有这一种根因。

## 成本和范围

| 项目 | 记录 |
| --- | ---: |
| 标准库宿主预置 | 0.030383 秒，单列 |
| 请求起点至结果导出前 | 0.389143 秒 |
| 请求起点至 stdout flush 后 | 0.393321 秒 |
| 自有会话加根进程采样 RSS | 31,072,256 字节 |
| 运行制品及 stdout／stderr | 12,103 字节 |
| 进程退出码 | 0 |
| 未执行控制组 | 0 |
| 算法 Work | 0，未调用算法 |

原 300 秒、8 GiB AS、2 GiB 采样 RSS、128 MiB 制品及 8M Work 额度均不变。
项目导入、绑定、准入、两次子进程创建、观察／错误／后处理、保存读回和接收在同一计时区间内。
“算法 Work 为零”不表示观察或控制免费。完整冷启动和最后进程退出的外部精确时长仍未测量；不使用工具的墙钟摘要替代内部成本记录。
离线审计未重跑控制，不用于补在线接受。新增运行制品很小，本阶段没有删除任何历史失败、冻结来源或模型。

## 后续决定

此次有限授权已消费，不重跑这四组或旧 pidfd 八组。R3 的失败、零接受位置及迟到 controller 证据保持原样。
原算法代码、来源、数值门、两臂配置、资源上限及空算法批准表未改变。
下一步应回到已实现的端点中点假设：在新的执行冻结和单独授权后完成原有限 CPU 对照，以精确下界、完整义务 gain/loss 和完整成本判断继续或停止该机制。
不再为本修复增加监督层或重复控制，也不自动运行真实模型／GPU。G1–G6、完整来源准入、外部领先及真实同源正证仍未完成。

复核命令仅重读记录：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/audit_owned_observer_control.py /data1/Kane/MOE/results/owned_observer_20261020_r1 /data1/Kane/MOE/results/owned_observer_20261020_r1.stdout.json /data1/Kane/MOE/results/owned_observer_20261020_r1.stderr.json 0
```

冻结绑定执行时的文件版本；未来修改代码后应在对应提交读取，不用新版本替换旧证据身份。
