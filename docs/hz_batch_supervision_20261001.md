# HybridZ 批量支持的硬预算控制

2026-10-01。[协议](hz_batch_supervision_design_20261001.md)与
[固定配置](../configs/hz_batch_supervision_20261001.json)先于实现，在
`17c761e13` 提交并推送。本阶段将已有 CPU 多目标支持接入独立监督版本，
没有修改候选算法、原 HZ、查询、精确接受公式或生产数值门。

[R3 完整归档审计](hz_batch_supervision_20261001_r3.json)通过：14 项控制、
20 次固定调用（4 正常、16 故障），另有两次明确标记的时钟注入控制。
4 个正常批次的 8 条精确支持界与上一阶段逐值一致。
这不是 4 个正 MoE 证书；正常控制包括非正界和二元连续松弛。

## 交付与接受边界

[监督入口](../scripts/hz_batch_support_supervised.py)绑定调用方的 batch、完整查询
清单、方向、协议及本次 invocation。每次在预算内重新创建合成 SparseHZono、
导出、生成候选、序列化，再以独立进程运行原精确检查器，最后接收完整清单。
父进程锚定实际 payload 与 checker stdout；不能使用 worker 自报 hash 代替。

生成、检查、接收使用同一个绝对预算，子进程及其后代受 owned-process 截止监督。
父进程哈希、清理、接收、终态发布及实际 API 返回时间均计费。正常控制 30 秒，
故障使用冻结的 1–30 秒预算，接口上限仍为 300 秒；没有真实请求执行。
RSS 为进程组加父进程的采样上限，不是 OS 硬内存配额；父进程同步 I/O 仍可能延迟
返回，但过期不能被接受为完成。不能由这些小控制保证任意大对象的准时终止。

保存的终态本身不足以证明在线完成，审计还要求调用方实际观察到的 API 结果。
离线数学复查不能将 TIMEOUT 改为成功。只有全部三阶段成功、身份与义务清单完整、
未过期时才返回 `CHECKED_GIVEN_HZ_SUPPORT_EXECUTION`。
3/4 部分候选仍保留 required=4，拒绝整批；不按接收前缀缩小分母。

16 个故障覆盖生成、准备、检查、接收、序列化、晚发布、异常、缺候选、错目标、
错调用身份、改写已锚定输出、缺 stdout、残留子进程、启动失败与资源超限。
测试不仅比对 ERROR/TIMEOUT 名称，还验证停止阶段和故障见证确实出现。
重绑后的成本、终态、清单篡改及缺失外部完成观察均拒绝。

## 成本与结果

| 正常控制 | 完整支持义务 | API 总秒数 | 含导入的生成／检查秒数 |
|---|---:|---:|---:|
| 双侧 guard | 4/4 | 2.9632 | 1.4656 / 1.4206 |
| 非正性质 | 1/1 | 2.8487 | 1.4091 / 1.3617 |
| 等式耦合 | 1/1 | 2.8131 | 1.4245 / 1.3090 |
| 私有二元松弛 | 2/2 | 2.7792 | 1.3461 / 1.3573 |

接收子进程约 0.067–0.071 秒，父进程其余费用及发布也包含在 API 总时间。
整套 R3 控制耗时约 47.715 秒，包括故意阻塞及拒绝控制；不能作为吞吐指标。
这里没有模型传播、训练或原生 fallback，不能用此表宣称真实 MoE 加速或 GPU 优势。
普通 ACT 环境导入仍加载依赖；不是可搬迁的独立 `python -S` 数学检查包。

## 修订与失败保留

全部本地目录在 `/data1/Kane/MOE/baseline_runs/`，未覆盖前轮：

- `hz_batch_supervision_20261001_r1`：11 项控制通过，但终态取钟与故障可观察性
  仍有加固项，未作为最终归档接受。
- `hz_batch_supervision_20261001_r2`：13 项中 1 项报错。`receive_delay` 在检查
  阶段先超时，没有到达目标故障；完整记录保留，不能用相同 TIMEOUT 标签充数。
  逐项复核另发现启动失败见证错误地期待异常 repr 含路径；改为明确记录并检查
  实际启动 executable 与异常类型。预算、顺序、查询和算法均未改。
- `hz_batch_supervision_20261001_r3`：14/14 通过，所有固定故障确实触发。
  两处状态／成本取钟统一为同一采样，避免截止边缘“API 完成但审计拒绝”的不一致。
  新进程独立重读全部清单并复查已存精确界，通过后才形成紧凑报告。

R1、R2、R3 的 summary SHA-256 分别为：

```text
fd79d351feeb899b9dc207370dcf0513ae1bbd6a1ffffe7cba4abcbe7435eb43
e1e37829286083ce16b9c4a869d11235a06869ff344b91ec9a29870c3a26b09e
d23f38bc35fb9b3f528110eb1580c29884f266c6bd5803b2ac7f9cdda6ae5e21
```

[CPU 算法回归](hz_batch_supervision_cpu_regression_20261001_r1.json)另行通过 16 项，
目录 `hz_batch_supervision_cpu_regression_20261001_r1`。
原 row-wise 精确检查核 15 项回归也通过，目录
`hz_batch_supervision_rowwise_regression_20261001_r1`；其 `test_outcome.json` SHA 为
`c178d38002ca84505b2ce7e3b816bf6c8de59ced47513e1b943cf4b61617b1b2`。
回归复查旧 75 个目标 LP，不运行新的原生求解。两位只读 AI 审查推动了截止／故障
见证修复；这不是独立人类技术评审。
导航与历史交接另通过 13 项控制，248 个链接有效、619 个冻结源码绑定及
4,632 行历史交接不变；七次旧真实证明调用的失败账本重算一致。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_batch_supervision_controls \
  /data1/Kane/MOE/baseline_runs/hz_batch_supervision_20261001_r3 --check
```

## 仍未完成与下一门

这是**给定 HZ 的 CPU 支持执行**控制。上游 network→HZ、guard 和因子来源仍受信任，
路由覆盖、完整输出证明、生产传播接入及原生回退不在本次交付中。
没有新真实输入、GPU kernel、native LP/MILP、checkpoint 或训练。
G1/G2 不因监督控制通过而结案，外部比较和旧来源缺口结论不变。

下一步单独冻结同算法 CPU/GPU 候选合同：精确接受仍在 CPU，全部传输、同步、
检查、失败与回退计费；先实现和控制，再按资源准入决定是否执行 GPU。
设备忙不能强行抢占，未获准入不假报 GPU 控制通过。生产 guarded 支持接入与
H2 表示消融另外推进，不自动重跑全尺寸容量或解封旧真实对象。

本阶段五个本地归档合计约 7.73 MB（分配块），未删除任何文件；共享盘约 694 GiB
可用。继续使用 `-B`，无需扩大清理范围或删除失败证据。
