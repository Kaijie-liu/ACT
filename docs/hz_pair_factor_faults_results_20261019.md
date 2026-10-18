# 两项因子污染控制结果

**两项均通过，旧 R1 整批 FAIL 原样保留。** 本次仅补齐旧批次未执行到的 factor
missing／permuted 故障覆盖，没有重跑性能对照、产生输出证书或修改生产算法。

## 执行与结果

按[冻结协议](hz_pair_factor_faults_20261019.md)，在干净且已推送的
`cbd8456537661af84c15a28e7c87f0d484aa291d` 执行一次。新目录为
`/data1/Kane/MOE/results/pair_factor_faults_20261019_r1`。
[169 文件冻结清单](hz_pair_factor_faults_freeze_20261019.json)与
[紧凑归档](hz_pair_factor_faults_results_20261019.json)绑定执行代码、制品身份、逐项记录及成本。

同一既有多因子合成来源的 source／forest hash 与 R1 完全一致。固定 pair `{0,1}`
选中 19 个因子，每项重新检查来源并建立独立 Work、Owner 和 Scope。

| 故障 | 实际拒绝原因 | 含来源检查及清理的 Work | 阶段耗时 |
| --- | --- | ---: | ---: |
| 首个合法 factor 的 forward 映射置零 | `scope membership binding` | 11,091 | 0.03440 秒 |
| 同时交换两个 forward rank 与 reverse ID | `canonical traversal rank/order` | 11,095 | 0.02997 秒 |

两项都检查了污染前后字节，完整走过节点检查后在因子检查拒绝；节点映射已封存，
因子映射未封存，`traversal_ready=False`。随后四个映射缓冲区全部释放，
`live_bytes=0`、累计分配等于累计释放，没有接受或候选下界调用。

来源捕获／构造另耗 0.03327 秒、6,839 Work；全部计入约 **0.439 秒**的批次。
终态写入前字段为 0.438137 秒，写入后的监督器返回记录为 0.438567 秒且 `late=false`；
两者分别保存，不用写入前的字段冒充包含最后写出的完整耗时。
worker 完整耗时约 0.302 秒；差额包含父监督启动、哈希核查、监测和归档，不隐去。
制品共 **61,176 字节**，进程组采样 RSS 峰值 **30,183,424 字节**，无超时、资源停止、
迟到终态或报告解析错误。owned payload 峰值各为 4,796 字节，不等于进程内存。
原 8M Work、128 MiB arena／制品、4096／8192 位、8 GiB 地址空间、2 GiB 采样 RSS
及 300 秒限制全部保持；没有模型、GPU、求解器或依赖安装。

启动前可用内存约 98 GiB、所在磁盘余量约 310 GiB；ACT 约 52 GiB、results 约 3.4 MiB。
本批逻辑文件体积不足 60 KiB，文件系统分配空间约 284 KiB；未删除数据或干预其他任务。

## 审计与停止边界

独立记录审计输出 `PASS_TWO_FACTOR_FAULT_RECORDS`，核对执行提交、冻结文件、完整制品
清单、恰好两个预期拒绝、实际字节污染、阶段计费与全部释放。离线审计约 0.525 秒，
不计为在线成功的一部分，也不是重新执行测试或独立重证网络。

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B \
  scripts/audit_pair_factor_faults.py \
  /data1/Kane/MOE/results/pair_factor_faults_20261019_r1
```

旧 R1 仍为 11/12、FAIL，新批为 2/2；不拼接成旧批全通过。默认关闭不变，本次没有
新增性能测量或证书。至此停止该生命周期控制分支，不继续刷小源控制。
完整来源检查下界 `13,922,736 > 8M` 与 membership 限制仍未解除；完整真实请求、
实机 GPU、外部净优势及 G1–G6 均未因此完成，后续研究须回到这些实质缺口。
