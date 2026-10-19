# 身份清理有限修复：一次控制与保存记录复审结果

2026-10-20，已授权的一次控制完成：4/4 组、20 个子情形通过，退出码 0。
只读复审为 `PASS_SAVED_IDENTITY_CLEANUP_CONTROLS`；复审没有重执行控制。
本结果关闭现有 RSS 观察与成员清理耦合的有限修复，不是算法效果或完整容量证据。

- 执行提交：`8cbe3cee46ceb84754ec5364a2e414c9576644f6`，在执行前已推送。
- [协议](hz_owned_cleanup_control_20261020.md)、[冻结](hz_owned_cleanup_freeze_20261020.json)。
- 冻结 SHA-256：`461a2b5289f79ae9efab5b8828247f50aa7e5b63de03a992914c0ca63f05cbce`。
- [完整紧凑记录](hz_owned_cleanup_records_20261020.json)、[审计与原文件哈希](hz_owned_cleanup_audit_20261020.json)。
- 原目录：`/data1/Kane/MOE/results/owned_cleanup_20261020_r1`，导出保存在相邻 `.stdout.json` 与 `.stderr.json`。

## 结果与接受边界

| 控制组 | 子情形 | 结果 | 实际检查 |
| --- | ---: | --- | --- |
| 内存诊断 | 5 | PASS | 活态缺 RSS 继续拒绝，前后身份／status 读状态有界保存；部分观察不保留旧清单 |
| 身份清理 | 7 | PASS | 不读取 status、不提供 RSS；异属、身份改变、权限、截止及部分扫描不误接受 |
| 信号绑定 | 4 | PASS | 打开 pidfd 后核对 SID／启动 tick；异属与复用不发信号，所有句柄释放 |
| 实际执行 | 4 | PASS | ERROR、CLEANUP_INCOMPLETE、TIMEOUT、COMPLETED 四种预期结果分别保留 |

第一条实际执行先观察到自有后代，再注入缺 RSS。清理路径确实看到该成员，发送两次
pidfd 信号，最终没有存活会话成员；原执行仍为 ERROR，保留 `live process RSS unavailable`。
清理成功没有补成运行成功，也没有补出内存样本。

双错误执行同时保留执行 ValueError 与清理 PermissionError，原收据为
`CLEANUP_INCOMPLETE`，成员清单 `null`／未知。后处理发现直接子进程已不可 waitpid，
记录 `ALREADY_NOT_WAITABLE`、未知退出码，并只读确认没有存活成员；不修改原失败。
该故障会话的采样 RSS 字段为 0，表示没有成功的会话内存样本，不代表实际零内存。

第三条实际跨过预设短 deadline，清理完成后仍 TIMEOUT。正常短子进程返回 0，
仍通过原执行内存观察；它不是用身份清理代替资源接受。结束时 FD 4→4，无自有直接
子进程。清理清单的 zombie 项是回收前观察，不声称全部孤儿 zombie 或逃逸会话已检查。

## 完整计费与未测量项

| 项目 | 记录 |
| --- | ---: |
| 标准库测量宿主可观测预置前缀，单列 | 0.020288 秒 |
| 请求起点至结果导出前 | 0.438376 秒 |
| 请求起点至 stdout flush 后 | 0.440739 秒 |
| 所得自有会话与根进程 RSS 采样最大值 | 43,319,296 字节 |
| 结果／接收文件及 stdout／stderr 总体积 | 21,251 字节 |
| 控制未执行后缀 | 无 |
| 算法 Work | 0，未调用算法 |

项目导入、身份核对、一次资源准入、全部控制、四次子进程执行、清理、保存、读回、
接收和 stdout 导出在同一 300 秒预算内。原 8 GiB AS、2 GiB 采样 RSS、128 MiB
制品及 8M Work 上限没有提高。控制 Work 为 0 不表示控制免费。

全冷启动、最后 stderr 成本戳写出和最终进程退出的精确外部耗时仍未知；采样最大值
不等于瞬时完整内存峰值，身份清理期间不伪造 RSS 观测。工具墙钟摘要不替代这些记录。
离线复审成本另列，不补在线接受。没有删除历史失败、冻结来源、checkpoint 或环境。

## 研究决定与下一边界

本次授权已经消费；不重跑本控制或增加监督层。R3／R4 的失败与零接受位置保持原样。
此修复解决了已知的清理耦合，不识别 R4 缺 RSS 的唯一原因，也不保证下一算法运行不会
再次因未知资源观察拒绝。四个正常算法对照仍未执行，没有新下界、完整义务或证书增量。

下一项有研究价值的工作是回到既已实现的端点 best-dual 中点假设，而不是再造基础设施。
需单独批准一个有限阶段：绑定本次修复，建立新冻结与新目录，完成静态核对／一次新鲜准入，
只执行原七位置 CPU 批次（原 22 项组件控制、四个正常两臂请求及两项外层故障），随后
审计归档；每位置 300 秒、整批 2100 秒、现有来源／方法／额度／全部检查不变，意外失败
即停、不重试。不包含真实模型或 GPU，不恢复 R4 的已消费批准。

成功与否看精确下界、完整义务 gain/loss 和全部成本；控制通过不是创新成果。
若该机制没有精确增量或达不到已冻结的继续标准，就停止该机制，不扩大比例、迭代、
样本或预算追成功。算法批准表仍为空，G1–G6、完整来源准入及外部竞争均未因此完成。

只读复审命令（在执行提交上使用绑定版本，不重新运行控制）：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/audit_owned_cleanup_control.py /data1/Kane/MOE/results/owned_cleanup_20261020_r1 /data1/Kane/MOE/results/owned_cleanup_20261020_r1.stdout.json /data1/Kane/MOE/results/owned_cleanup_20261020_r1.stderr.json 0
```
