# 真实适用性与执行缺口研究记录

上一轮D245的4209项完整数学回放是progress。本轮完成宽输入必要条件、全父对线性扫描证书、同层来源共享结论及尾项扩展的独立红队审查，改变了下一实现选择，仍属progress。结论见[THEORY.md](THEORY.md)：不把新增四产品作为已证明的新精度来源，不为了当前正控增加这些列。

## 实际来源与 GPU 边界

只读原D241三份完整source JSON，没有解码新模型或传播界：

| 模型 | 首层 ReLU 标量数 | 父层标量数 | 子层标量数 |
| --- | ---: | ---: | ---: |
| CIFAR100 large | 65536 | 65536 | 65536 |
| CIFAR100 medium | 14400 | 8192 | 8192 |
| TinyImageNet medium | 46656 | 25088 | 25088 |

这些是逻辑人口，不是crossing数或已绑定native位数。输入/参数/BN区间和拓扑已保存，不等于中间可靠界或actual H已建立。

执行源码审查未找到已经认证的完整GPU证明路径。hybridz_tf.py在ASSERT仍会以lazy_affine_reached_terminal丢弃未消费的隐式H；HZSolver声明supports_gpu=False，消费既有HZono/SparseHZono并转CPU。原常规LP/MILP终端仍属于原允许边界，这里没有将它重新定性为禁止算法；只是不能把它称作已完成GPU路径。TorchLPSolver是Adam penalty可行点搜索，不证明不可行，不应拿来作为本研究的能力救援。现有Torch affine/Conv和implicit matvec也没有可直接转移的完整外向舍入资格。

历史D017和D043只能说明CUDA初始化失败；D043定位到cuInit返回OOM，未区分主机虚拟地址、设备内存或其他资源。D020在ptrace启动检查被拒，不是另一份GPU OOM。

本轮做了只读系统诊断：设备节点存在；nvidia-smi返回RTX PRO 6000 Blackwell Max-Q、驱动580.126.09、97887 MiB总量、34 MiB已用，UUID仍为GPU-f491c2c6-a093-590a-6b8a-13b5f76aadcc。当前自身/proc状态为Seccomp=0、Seccomp_filters=0、NoNewPrivs=0、CapEff=0；/usr/bin/strace的SHA仍为28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001。这些读取并非同一原子快照，也不是CUDA初始化/算子成功、旧故障时内存或设备峰值的证据。没有运行Torch/CUDA检查，没有改变系统权限、驱动或预算。

当前环境与旧ptrace拒绝有不同的可观察状态，值得新建D247独立同预算诊断。D247使用新源码/新RUN，先完整数学兼容性门，再仅跟踪自己启动的worker；不对旧版本重试、不附加其他进程、不改安全设置、不提高AS16GiB。该诊断的执行结果另行归档，不能提前写作已完成，也不代替Neural-HZ定义研究。

## 保存范围与成绩

本轮配置为paper_only加read_only_metadata。未执行新数学候选，没有GPU算术、真实网络验证、shadow或全量回放；当前最后已通过组件仍为D245。正式1870/2413、独立CIFAR10025加TinyImageNet36等于61/400不变，三个新增收益字段均为0。

分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。旧D245 ARCHIVE46项读回校验一致。只在新隔离目录新增文档；生产、默认路径、历史结果和/data1/Kane/HyZor未修改，无commit/push。

依文档归档技能，将已证定理、被红队否定的贡献主张、真实来源缺口和GPU观察分开保存；没有创建外部Page。Goal保持active，GPU、smooth、Transformer、新家族与2413满分目标均未完成。
