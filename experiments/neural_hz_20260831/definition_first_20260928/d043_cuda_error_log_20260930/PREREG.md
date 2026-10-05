# 同预算 CUDA 可用性路径错误日志诊断

本版本的诊断 worker 只诊断原 CUDA 可用性检查路径，不运行神经网络或数学候选。监督流程仍先执行完整继承兼容性测试，其中包含旧数学 fixture 和求解调用；这些不是本轮 CUDA 数值工作负载或设备资格。目标是使用驱动提供的错误日志补充 D017 的 cudaGetDeviceCount out of memory 证据。它不是 D017 或 D020 的重跑，不通过 ptrace、预加载绕行或提高资源上限初始化设备。

## 文档依据和局限

NVIDIA 官方支持 CUDA_LOG_FILE=stderr，将覆盖到的 API 错误信息写入标准错误。r570 及以上驱动支持该机制；既有冻结驱动文件名为 libcuda.so.580.126.09。Torch 文本元数据为 2.8.0+cu128、CUDA12.8，不能把设备工具显示的驱动支持版本当成 Torch runtime 版本。[官方说明](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#cuda-log-file)

该机制未覆盖所有 API，空日志不证明无错误，也不排除虚拟地址预留冲突。普通 OOM 返回不能单独区分主机地址空间、设备内存或其他初始化资源。日志只能支持它实际报告的原因；不把后来的设备空闲量当作旧失败时的遥测。[Error Log Management](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#error-log-management)

## 单次执行与冻结

新结果路径为 experiments/neural_hz_20260831/results/d043_cuda_error_log_20260930_v1，必须独占创建。首次 --enabled 即消耗本版本；成功、失败、超时或前置门失败均不修改后重跑。所有旧 main、worker、writer 和结果目录只读，不调用旧执行路径，不修改旧 helper 的 globals。

首跑前由根代理冻结 PREREG.md、run_diagnostic.py、init_worker.py 三个绝对路径及 SHA256。freeze.json 的 schema 为 d043_frozen_v1，required_tests=3759，required_test_files=169，inventory_sha256=62cba6b99355a0c72ab878a76f7500418f776f130204dd00ead264dbba4d4828，worker_api=torch.cuda.is_available_once。先认证源文件再加载任何 helper；新 worker 在 Torch import 前核对自己的冻结身份。

继承 D038 的实际完成清单，不从旧 D025 重新拼装测试。authority 为：

```text
06252697d664a213c7967a6e726d99fd6217dd5d7311e59ee202b38f4083ff04  D038 preregistered.json
62cba6b99355a0c72ab878a76f7500418f776f130204dd00ead264dbba4d4828  D038 inventory.json
b1f4c6555e538a52c84c0fd78a918fdc2c65677cf1d515ec9e771b97d8e700cd  D038 exit.json
```

D038 manifest 的全部源、输入、GPU 依赖、decoder 身份和三模型 provenance 保留；前后检查全部身份，不因本轮不加载模型而删掉人口。可复用通过摘要认证的只读 sha/load/limits/memory 及 bind/drift/provenance/select_sources/bind_decoder/gpu_dependencies 辅助函数，不能复用其入口或旧写入器。

## 完整兼容性门

完整 3759 tests、169 files，精确 node IDs 同 D038。collection 和 execution 合计不超过 60 秒，不允许 failure/error/skip。使用 /data1/Kane/miniconda3/bin/python、assertions 开启、禁 bytecode、CPU1、AS16GiB、运行库单线程；pytest 时 CUDA_VISIBLE_DEVICES 为空，缓存与临时文件全部位于新 RUN。只有全部测试、身份和 provenance 门通过才启动 worker。

本轮没有新数学或日志解析组件测试；这 3759 项是完整兼容性门，不冒称覆盖新增监督逻辑。新代码在冻结前做静态审查，日志按原样保存，不从消息文本自动推断根因。不为增加测试余量减少旧测试或放宽 60 秒。

## Worker 的唯一 CUDA 路径

worker 固定原 GPU UUID 为 GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc，保留 CUDA_MODULE_LOADING=LAZY，启用 CUDA_LOG_FILE=stderr。

唯一执行序列为 import torch，记录版本与自身内存，再调用 torch.cuda.is_available() 恰好一次并记录返回值及日志。无论成功或失败，都不追加 cuda.init、device_count、tensor、kernel、synchronize、ctypes cuInit、模型或 solver；没有 CPU fallback 和重试。此范围是可用性调用所触发的初始化路径，不是显式 context 初始化或设备数值资格。

记录本次影响加载和检查路径的环境变量，包括 PYTORCH_NVML_BASED_CUDA_CHECK、CUDA 加载选项、LD_LIBRARY_PATH/LD_PRELOAD 等必要允许列表；不记录完整环境或凭证。不得让 NVML 替代所要观察的原 runtime 检查路径，不使用注入库作为绕行。旧运行未保存的环境值不补造，因此不声称两次实验严格只改变一个变量。

worker 保持 CPU1、AS16GiB、单 runtime 线程和 240 秒硬墙钟。stdout、stderr 各自写独占的新 RUN 日志，沿用 16MiB 每文件上限；原始错误文本保留。worker.json 上限 65536 bytes。所有缓存、CUDA 日志和临时文件只写新 RUN。

supervisor 和 worker 分别保留 RSS high-water growth 加 65536 不超过 1GiB、tracemalloc peak 加 metadata 加 65536 不超过 1GiB 的 host 检查。记录自身 VmSize/VmRSS/VmHWM，不查询或操作其他进程。设备 context 和峰值若未测量则为 null/unknown，不记为零，不宣称 aggregate 或完整物理资格。此诊断不运行数值 kernel，原数值组件的 work/entry/位宽门没有被替换或放宽。

## 失败和证据保存

可用性返回 false、异常、超时、日志超限、worker 缺记录、资源或身份失败均保留原始日志和 exit.json，停止本版本。不把部分日志记成初始化成功，也不改变 GPU 配置重试。监督器在异常路径也记录终态和可获得的前后身份、资源信息；首次结果不覆盖。

始终保持 gpu_computation_completed=false、native_HZ_admitted=false、source_census_qualified=false、complete_physical_qualification=false、formal_gain=0。日志收集完成、API 返回、GPU 可用性与设备计算资格是不同事实；即使 is_available 返回 true，也不等于 GPU 算术或端到端加速通过。

## Provenance 和不变目标

日期 2026-09-30；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。旧九个 tracked 修改保留为 3806 insertions、57 deletions。没有旧档写入、生产集成、commit 或 push。按 pages:write-page 分开记录诊断目的、执行边界和未获资格；本地文档，不发布外部 Page。

正式 1870/2413（1063 CERT、807 validated ADV）、13 家族逐例保旧与独立 CIFAR100 25、TinyImageNet 36，共 61/400 均不变。全部因子、bits、EQ/LE、共享身份、decoder、fail-closed 以及原晋级门不变。当前 Goal 继续 active；本诊断不是新 Neural-HZ 定义，也不是可以代替数学与真实网络突破的成果。
