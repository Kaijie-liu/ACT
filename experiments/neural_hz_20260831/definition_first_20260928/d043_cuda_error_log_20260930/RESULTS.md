# CUDA 驱动初始化失败的单次日志结果

完整 3759 项兼容性测试通过，但随后唯一一次 torch.cuda.is_available 返回 false。新增驱动日志把错误定位到 cuInit 返回 CUDA_ERROR_OUT_OF_MEMORY；它没有报告是哪一种资源不足，不能据此认定是设备内存、主机地址空间或驱动缺陷。本版本已执行并关闭，不修改后重跑。

## 实际执行与证据

唯一 RUN 为 [d043_cuda_error_log_20260930_v1](../../results/d043_cuda_error_log_20260930_v1/exit.json)。监督器 session 20483 已终结，exit=1；worker 已由监督器 wait 回收，没有等待中的任务。日期 2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。

完整 collection 和 execution 合计 56.13721905462444 秒，3759 tests、169 files，精确 node IDs 与 D038 相同，无 failure/error/skip。全部源和输入漂移为空，生产 provenance 漂移为 false。既有兼容性测试包含旧数学 fixture 和求解调用，不能将其说成整个流程无数学执行。

诊断 worker 仅执行 Torch import 和一次 torch.cuda.is_available，没有显式 cuda.init、tensor、数值 kernel、网络、solver、设备计数追加调用或 CPU fallback。Torch 2.8.0+cu128，runtime 元数据 12.8；NVML 替代路径与 LD_PRELOAD 均未设置。worker 墙钟 4.053243050351739 秒，内部记录 3.6598545350134373 秒，监督流程总时长 71.75736698508263 秒。

[原始 stderr](../../results/d043_cuda_error_log_20260930_v1/worker.stderr.log) 的新增驱动行是：

```text
[18:40:45.404][139199028942656][CUDA][E] Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuInit
```

随后仍出现 Torch 的 cudaGetDeviceCount Error 2 警告。diagnostic_completed=true 表示调用返回且日志保留，不表示初始化成功；availability_confirmed=false 和 all_stages_passed=false 是本次执行结论。

## 资源观察与未证明的原因

AS 上限保持 17179869184 bytes，CPU affinity 为 [0]，运行库单线程。worker RSS high-water growth 为 567660544 bytes，tracemalloc peak 为 67182171 bytes、metadata 为 38012352 bytes；相应 1GiB 观测门通过。监督器观测门也通过，且在封存日志后核算。

Torch import 后、可用性调用前 VmSize 为 4785758208 bytes；调用后为 4865265664 bytes。记录的是实际已保留的自身地址空间，不包含失败的映射请求，不能由小于 16GiB 推断地址空间上限没有影响。没有采样设备 context 或设备峰值；observed_context_bytes=null，combined_physical_gate=unknown，不能将未知填为零。

因此本次比 D017 多得到的证据是驱动 cuInit 错误位置，而不是根因或可行修复。没有使用 ptrace、注入库、提升预算或改变配置重试。当前不继续重复初始化诊断，优先完成非凸关系的数学与真实适用性研究；GPU 数值资格仍是未完成项，不能用 CPU 数学测试替代。

## 资格与保管

gpu_computation_completed、native_HZ_admitted、source_census_qualified、complete_physical_qualification 均为 false；formal_gain=0。正式 1870/2413（1063 CERT 与 807 validated ADV）、13 家族逐例保旧和独立 CIFAR100 25、TinyImageNet 36，共 61/400 不变。

旧九个 tracked 文件保持 3806 insertions、57 deletions。仅新 D043 源与结果目录有新增，未改旧模型、冻结证据、生产默认，未 commit/push。按 write-page 技能区分已执行事实、未测资源和因果推断。本诊断终态不等于整体研究阻塞，Goal 保持 active。

关键摘要如下；exit.json 另绑定全部日志和 pytest 保留文件。

```text
5998070330294b5238df38966d088c8468aecc72fb3205ba5a0da2315a42109e  freeze.json
227050c6b8d573c5e3f3d9bd1111b2ef61753b6c8463eb056e81d39172ed4e9f  RUN exit.json
6f359b0e4d1007e36c94130887cd336fbe177a8f40711c1aa748af8afa0997ef  RUN preregistered.json
36e7843b0b4ac4665a35afbd6e5812270439a54d1177dd0824b8e3208be652af  RUN worker.json
8e21084ec99879a77678dddd8dd9194b899c680ffa4a6aa1e566833c909e0d5d  RUN worker.stderr.log
```
