# 共享池化投影的研究记录

本轮得到条件明确的共同投影与反向见证定理，但没有新的 Neural-HZ 能力。它保持原 LP 强度且可能增加 nnz，故只归档为支撑结论，不进入候选实现。用户强调的主目标始终是从 HZ 数学定义出发提出强大的非凸 Neural-HZ，而非算子压缩、helper 或工程换名。

## 来源和执行身份

日期 2026-10-04 Australia/Sydney；归档时间锚 2026-10-04 06:07:26 UTC。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked binary diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。现有生产代码与其他工作树改动均未撤回或改写。

本轮只读前一轮档案、D123 接口限制、D105 适用性、D014/D028 的相关已有结论片段、未解结构清单和生产 MaxPool 源码，并读取一手论文相关章节。关键身份见 ANCHOR_SHA256SUMS；对 D152 此次仅重新核对文件身份并引用先前记录，不声称本轮重新执行其研究。

fiber_kernel_review 核对单窗口投影及局部费用；aligned_domain_equivalence 核对重叠窗口共同区间、反例和行数；current_helper_audit 检查已有相关研究和真实适用性。主线程逐式推导共同投影及 nnz 费用，并将原标签条件和无外部 q 消费者条件纳入定理。以上均为纸面审查，不是程序测试。

## 真实代码缺口与尚缺的绑定

当前 act/back_end/hybridz_tf/tf_cnn.py 的 tf_maxpool2d 清空该层 HZ 缓存并返回 interval MaxPool；其稀疏分支也明确返回 unsupported_sparse_maxpool2d。tf_mlp.py 的稀疏网络分支将 MAXPOOL2D 列为 unsupported_sparse_nonlinear。此处确认的是所读代码路径，不能扩展成所有历史实验都执行了这条路径。

冻结的 formal_unsolved_structure_manifest_v1.json 包含 543 个未解条目，已排除正式 1870 解。在这份清单中，MaxPool 算子计数非零的只有 cgan:19、cgan:20，均为 UNKNOWN，共用以下模型的已存元数据：

```text
cgan_2023/onnx/cGAN_imgSz32_nCh_3_small_transformer.onnx
stored bytes: 272784587
stored SHA256: 10be6af09db7f6cd116a8b820bb93121b80a6b845df77c0347db4c443b354e35
IR 4, opset 9, nodes 319
MaxPool 4, Relu 13, AveragePool 10
```

以上是历史元数据，不是本轮重新计算模型哈希。算子计数不证明 ReLU→MaxPool 邻接，也未提供窗口、padding、tie-break、indices 输出和完整消费者关系。未做模型解码或动态加载，不能据此声称定理适用于这两例，更不能推出 CIFAR/Tiny 覆盖收益。D105 已指出相同来源缺口；不把重新找到它计成新适用性发现，也不借此放宽旧执行包的文件预算。

## 未执行事项和正式记账

没有候选源码、import、AST/编译、collection、数值程序、模型推理、LP/MILP、GPU、shadow 或 replay，也没有启动后台任务。最近执行人口仍为 D158 的 4032 项、212 文件。没有候选可以继承旧测试资格；后续执行仍需新的预注册、源码冻结、一次性版本和全部原验证门。

正式基线仍为 1870/2413，即 1063 CERT 与 807 validated ADV；新增 0。独立 E0 仍为 CIFAR100 25、TinyImageNet 36，共 61/400；新增 0，两套账不相加。本轮未回放，不能声称重新实测保住了这些结果。所有逐例/逐家族零回退、具体 ADV 验证、fail-closed 和原资源边界保持不变。

## 后续研究边界

这一投影是支撑成果，不是下一轮默认主线。不为已知等价消元新建完整实现框架，也不把两例池化缺口等同于强 Neural-HZ。真正的新候选必须给出域元素和具体化、非凸关系如何跨算子保留、与 HZ/CZ/已有混合符号域的实质区别，以及健全性和同口径完整费用。小例优于逐门松弛还不够，须与同一信息下的强参照比较。

持续保留 CIFAR/Tiny、13 家族、新家族、GPU、smooth/Transformer 和长期满分目标。不用 helper、attack/PGD、BaB/split/backward/dual rescue 代替域能力；普通终端决策与独立具体见证验证的边界不变。Goal active，不报告总体完成或外部阻塞。

按文档归档技能区分证明、已有研究、真实适用性缺口和未执行状态。仅新增本隔离目录和续接文件，旧模型、结果、冻结候选与档案只读；未创建外部 Page，未 commit/push。定理见 [THEORY.md](THEORY.md)。
