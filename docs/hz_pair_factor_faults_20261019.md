# 两项因子污染控制的独立冻结

用户于 2026-10-19 明确批准：仅修复并另行执行 R1 未完成的 factor missing／permuted
两项控制。R1 整批 FAIL、原测试及冻结记录原样保留；不重跑节点污染、性能四臂、候选
消费、oracle 或搬迁。生产实现不修改，默认关闭不变。

## 修复与验收

新测试采用 R1 已有的 `eight_private_six_outputs`，以同一标准库工厂重新捕获和构造。
固定来源 SHA-256 为 `3a9df8651ea6781098377244b410a1cfa821bc4f431fbf1d067fa5c4a76cb360`，
forest 身份为 `cfd36bba9a2702c13a4d80e2f093cc4a764f77e85d66a19e12ac51c37d3ef157`。
这不是新增训练模型或更换数学对象；构造及检查成本仍全部计入。

仅测试 pair `{0,1}`，每项使用新的独立 forest、Work、Owner 和 Scope。保留
`selected > 1` 前置断言及原污染操作，核对实际修改的字节。缺失映射应触发
`scope membership binding`，自洽置换的双向映射应触发 `canonical traversal rank/order`。
必须由完整 `prepare_traversal` 走到因子检查拒绝：节点映射已封存、因子映射未封存、
`traversal_ready=False`，并完整释放自有缓冲区。不以其他异常代替预期拒绝。

## 单次执行和成本

先冻结脚本、依赖、协议及旧 R1 归档身份并提交，干净分支执行一次：

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B \
  scripts/run_pair_factor_faults.py --run \
  /data1/Kane/MOE/results/pair_factor_faults_20261019_r1
```

仅两个测试方法、一个既有合成来源，无候选、模型、GPU 或优化器。沿用 R1 监督器的
硬截止、部分报告与终态语义，但独立冻结和目录，不通过修改旧模块全局变量切换配置。
整个批次最多 300 秒，含启动、导入、捕获、构造、两次来源检查、故障检查、释放及归档；
各 Work 仍受 8M 限额，arena／制品各 128 MiB，存储／工作位数 4096／8192。
worker 8 GiB 地址空间硬上限、进程组 2 GiB 采样 RSS 门不变。离线记录审计另计，
不能补成在线通过。超时、异常或缺项保留失败，不自动重试。

审计核对冻结、执行提交、全部制品身份、恰好两项错误、实际污染字节及完整成本与清理。
这是保存记录审计，不是重新执行检查器或独立证明网络下界。两项通过也只闭合 R1
遗漏的故障覆盖；不能拼成 R1 全通过、宣称新证书／加速、推进真实模型准入或关闭 G1–G6。
结束后停止这一生命周期控制分支；完整来源 `13,922,736 > 8M` 和 membership 拒绝仍在。
