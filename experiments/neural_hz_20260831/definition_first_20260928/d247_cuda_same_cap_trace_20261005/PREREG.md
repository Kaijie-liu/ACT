# 同预算 CUDA 可用性路径跟踪诊断

本版本只诊断CUDA可用性路径，不执行Neural-HZ新规则、模型、GPU算术或solver。D245完整数学回放已通过；D246发现的结构限制另行记录，不能用本诊断替代域创新。历史D020因ptrace启动检查被拒而没有得到trace，D043仅定位cuInit返回OOM。本轮只读自身/proc状态为Seccomp=0且设备管理查询可见GPU，是新建独立诊断的环境依据，不是已解除全部权限限制或已查明OOM根因的证据。

不改变系统权限、sysctl、驱动、库或预算；不附加其他进程，不移除strace安全选项。仅跟踪本监督器自行创建的worker及其子进程。旧D017/D020/D043/D245源码和结果全部只读，不执行旧main或worker。新版本首次执行即消费，失败不修补原版或重跑。

## 冻结与完整兼容性门

新源目录definition_first_20260928/d247_cuda_same_cap_trace_20261005，唯一RUN为results/d247_cuda_same_cap_trace_20261005_v1。默认关闭，唯一命令：

~~~text
/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d247_cuda_same_cap_trace_20261005/run_diagnostic.py --enabled
~~~

冻结前仅静态读写/审查，不import、AST、compile、collection或试跑新文件。冻结源精确为PREREG.md、run_diagnostic.py、collection_contract.py、init_worker.py四份绝对路径及SHA256。freeze schema为d247_cuda_same_cap_trace_v1，required_tests=4209，required_test_files=224，new_test_names=[]，new_evidence_files=[]，worker_api=torch.cuda.is_available_once。这是没有新数学测试的诊断版本，不冒称完整旧测试覆盖新监督代码。

完整继承D245实际成功的4209项、224文件、有序nodeids、7752份来源、14份输入、依赖/decoder及provenance。先确认原成功receipt、冻结和所有历史工件身份，保留原映射，不用receipt替代本轮重新执行。D245原manifest/exit/inventory分别为a6aee5aa1feb1ea175cd08f522540af8ba54dabb90e176409a8f00f55758ce82、22fdb218cbdd61c671e810117144ca0aa92d712d416311cf718a633fad5a78ed、bfdad3342f98966306bf0d772ae3936cb298221cdd24ce15675f91f7859530e1。

沿用D245同进程collection检查、importlib模式、禁外部plugin自动加载及完整JUnit。测试导入、collection、执行及JUnit合计60秒，CPU0、单线程、AS16GiB、CUDA隐藏；failure/error/skip必须为0。原所有writer仅重定位到本RUN。新增D245的_record_file原firstlineno56、文件SHA9b9ade26bb01be96649131132d3a66d9c3006973dfc09d582e5921a1001f92f8；全部20项照常执行，summary仅写本RUN/inherited_d245_controls。既有D020四项parser测试也在完整人口中，不修改它们。

只有完整兼容性门通过且身份/provenance再次一致，才启动跟踪。没有更小的诊断人口或提高60秒门。

## 唯一自有 worker 路径

认证/usr/bin/strace的SHA为28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001。使用：

~~~text
strace --kill-on-exit -q -f -ttt -T -s 256
       -e trace=mmap,mremap,brk,ioctl
       -o NEW_RUN/syscalls.trace
       FROZEN_PYTHON -B NEW_SOURCE/init_worker.py --enabled
~~~

从跟踪进程创建起，整个自有进程组限240秒；超时或异常只清理该已知自有进程组，不寻找或终止其他任务。观察超时只轮询原handle，不新建实验。AS16GiB和每个日志文件16MiB的RLIMIT_FSIZE在strace/worker启动前施加；不放宽虚拟地址限制去“验证假设”。完整CPU门及跟踪墙钟分别记录。

worker在Torch import前核对自身、runner和四份冻结源、解释器及注册身份；完整历史依赖闭包由监督器前后核验。固定GPU UUID为GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc，CUDA_MODULE_LOADING=LAZY，CUDA_LOG_FILE=stderr。禁止LD_PRELOAD注入及PYTORCH_NVML_BASED_CUDA_CHECK替代runtime路径，不记录完整环境或凭证，仅记录明确允许的加载/线程/缓存配置。

唯一CUDA入口为torch.cuda.is_available()一次。无论返回真假或抛异常，都不追加cuda.init、device_count、tensor、kernel、synchronize、ctypes cuInit、网络或solver，不重试、不fallback。bool true只是可用性结果，不等于显式context初始化、数值成功或GPU就绪认证。

保留导入前、导入后、可用性调用前后事件及epoch/monotonic时间；异常也保留调用计数、开始及异常结束时间。api_calls和api_returned明确标记，避免把Torch导入阶段的syscall错误归到可用性调用。

## 原资源边界和解释

监督器、worker分别从运行开始/导入前记录RSS高水位增长和tracemalloc，维持1GiB加65536字节预留的两项host门；记录VmSize/VmRSS/VmHWM。CPU严格为0，五个运行库线程变量均为1，所有缓存/TMPDIR只写本新RUN。worker.json独占保存且不超过65536字节。

设备context和峰值未测则为null/unknown，不能写0。strace物理峰值未独立测量，不能声称进程合计或完整物理资格。既有数学的256M总work、200M单操作、64M逻辑entries和512位门保持；本worker不执行候选数值工作，不能把CUDA初始化内部当已认证数值工作或零开销GPU。

复用认证D020/trace_summary.py的summarize，只读解析不改旧源。16MiB文件、4096字节行、1024事件上限及保守完整性语义保持；unfinished/resumed或无法识别行不会被静默忽略成完整trace。实际原始日志始终保留。parser不自动限定CUDA时间窗，报告必须结合worker事件和pid；不能把导入前ENOMEM归为CUDA调用。

failed_reservation_exceeds_total_as_cap只说明实际观测的失败mmap/mremap请求大于整个AS16GiB，不能在该上限内满足；不证明唯一根因，不授权提高预算。普通ENOMEM、ioctl错误、设备现在空闲量或不完整trace不能证明特定资源根因。任何完整trace也不授权未观测事件不存在的断言。

## 保存及终态口径

独占自动保留preregistered.json、inventory.json、tests.log/xml、全部继承证据、worker stdout/stderr、syscalls.trace、worker.json（若产生）、trace_summary.json和exit.json；失败或超时也保存能取得的工件及哈希。没有worker记录就记missing，不捏造事件。前后检查完整来源、输入和工作区provenance。

diagnostic_completed要求完整4209/224通过、原parser认定trace完整、worker实际调用一次且返回、worker和监督器host门及前后身份通过。availability_confirmed单列表示cuda_available为true且worker没有其他失败。一次正常返回false的OOM诊断可以完成证据收集，但不等于CUDA可用；异常、缺记录或不完整trace保留阳性观察并拒绝完整诊断资格。诊断完成不授予任何GPU计算或Neural-HZ能力资格。

TRUE flags为diagnostic_only、diagnostic_solver_free、fixed_component_lp_controls_registered、worker_stage_registered。mathematical_stage_only、domain_definition_changed、new_component_solver_free、negative_audit_only、solver_rescue_registered、new_set_class及全部实际模型/native/GPU/完整物理/新域/新能力资格为false。worker_launched在freeze/manifest为false，在exit按实际记录，不伪造未执行。formal_gain=independent_e0_gain=new_benchmark_solves=0。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。不改生产/default/旧结果，无commit/push。正式1870/2413及独立61/400不变。文档归档技能用于区分环境观察、执行诊断、数学资格和正式成绩；GPU、smooth、Transformer及全量能力目标保持active。
