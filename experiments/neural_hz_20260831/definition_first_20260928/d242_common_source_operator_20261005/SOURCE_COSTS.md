# 完整来源前缀的显式 CSR 成本与 GPU 资格边界

2026-10-05，分支 redu-hz。本记录只读取 D241 已完成的三个来源工件及既有实现；以下整数公式和乘加均为纸面核算。没有执行候选、模型、数学计算程序、GPU、求解器或新诊断，没有修改 D241 或历史参数/结果。

结论限于一个明确方案：每个原始标量端口保留独立物理槽，把每个原 Conv 的定义等式全部展开为同时保留的普通 float64-data/int32-index CSR。该方案的 large 前缀，仅卷积右侧非零项就需要 1,323,279,360 bytes，已经超过 1 GiB；逐门复制改为批量构造不能消除这份单次存储。此结论不是所有 HZ 表示的下界，也不是一次实测失败。隐式算子块可能避免这份展开存储，但其可靠数值、完整原谓词消费和终端路径尚未取得资格。

## 1. 完整固定来源及证据字段

人口仍为三个既定原模型、各自原 property，从原输入到第三个原 ReLU；large 截止 Relu11，其余两者截止 Relu13。没有删掉 shortcut、空间边界、零收益行或任何通道，也不把后续实验改成只有 medium 或几个窗口。

| 工件 | 原模型 | 输入 NCHW | 前缀 Conv 节点 | 截止节点 |
| --- | --- | --- | --- | --- |
| [source_0.json](../../results/d241_residual_source_binding_20261005_v1/source_0.json) | CIFAR100_resnet_large | 1,3,32,32 | 0,3,6,9 | Relu11，port136 |
| [source_1.json](../../results/d241_residual_source_binding_20261005_v1/source_1.json) | CIFAR100_resnet_medium | 1,3,32,32 | 0,3,6,8,11 | Relu13，port132 |
| [source_2.json](../../results/d241_residual_source_binding_20261005_v1/source_2.json) | TinyImageNet_resnet_medium | 1,3,56,56 | 0,3,6,8,11 | Relu13，port132 |

三份工件本轮只读核对的 SHA256 为：

```text
source_0.json 9494cb57122809d61f62a7c75f7ee4533c8b41e9683cda86a668321fbef20dc5
source_1.json 5374d47e99db2a34e2099603b469dcb50d828050e536900e1d832267bf8222e0
source_2.json ccac731af636a70c1b6210fb8ea6bdd6108fd72ab0949547263fee1bd48b7bed
```

原模型 SHA256 依次为：

```text
large  5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16
medium aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4
Tiny   234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776
```

可复核字段为 `prefix_nodes` 中 `kind="conv"` 的 `node.index`、`input_shape`、`output_shape`、`weight_shape`、`strides`、`pads`、`dilations`、`group`、`auto_pad` 和 `weights_ref`；再以原 initializer 名字查 `initializer_descriptors` 的 `scalar_count`、`zero`、`nonzero`。所有本页卷积均 group=1、dilation=(1,1)、auto_pad=NOTSET，batch=1；各 Conv 无显式 bias 输入。BN/Add/后续消费者仍完整存在，只是不计入下面的卷积右侧存储下界。

以下名字在对应工件中的 `zero` 全为 0，`nonzero=scalar_count`：

| initializer 原名字 | large 标量数 | medium 标量数 | Tiny 标量数 |
| --- | ---: | ---: | ---: |
| conv1.weight | 1,728 | 1,728 | 1,728 |
| layer1.0.conv1.weight | 36,864 | 73,728 | 73,728 |
| layer1.0.conv2.weight | 36,864 | 147,456 | 147,456 |
| layer1.0.shortcut.0.weight | 不在 large 前缀中 | 8,192 | 8,192 |
| layer1.1.conv1.weight | 36,864 | 147,456 | 147,456 |

因此本页不是先假定训练权重稠密：非零性来自完整逐标量来源审计。D241 的原始参数总人口还包含 BN，分别是 113,344、380,864、380,864；不能把这些共享参数数目当成显式卷积矩阵的 nnz。

## 2. 有效 tap 的逐轴整数公式

对输入单轴长度 H、输出长度 O、kernel 长度 K、stride s、前侧 padding p、dilation d，定义

```text
T(H,O,K,s,p,d)
 = sum_{o=0}^{O-1} #{k in {0,...,K-1}: 0 <= o*s-p+k*d < H}.
```

二维轴独立，当前 square Conv 的右侧有效连接数恰为

```text
P = C_out * C_in * T^2.
```

一般非 square 时用 `T_height*T_width`，group 非一时还须按真实分组改写；本页不将当前三源公式泛用于这些情况。padding 的常量零位置不计非零连接。每个有效 kernel tap 对一个不同的原输入位置，当前独立物理槽方案没有同列抵消；非零有限尺度归一化也不把非零原权重变为数学上的零。

本页用到的逐轴和为：

- stride1、3-tap、pad1，H=O=N>=2：`T=2+3*(N-2)+2=3N-2`。
- stride2、3-tap、pad0，每个输出窗口完整有效：`T=3O`。
- stride2、3-tap、pad1，H=2O-1、O>=2：`T=2+3*(O-2)+2=3O-2`。
- stride2、1-tap、pad0：每个输出一个有效位置，`T=O`。

这些是每个原 Conv 各自的 padding 计数，不把 Conv/BN/Add/Conv 简化成一个全位置通用 5x5 核。边界双层 padding 不能这样合并，相关原端口与反例见 [D241 THEORY 第7节](../d241_residual_source_binding_20261005/THEORY.md)。

## 3. 三个模型的全部前缀 Conv

表内 padding 为四侧相同的单值，kernel/stride/dilation 的两轴也相同。空间形状为输入 H 到输出 O；通道为 C_in 到 C_out。

### CIFAR100 large

| 原节点 | 通道 | H→O | kernel / stride / pad | T | 精确有效连接数 P |
| --- | --- | --- | --- | ---: | ---: |
| Conv0 | 3→64 | 32→32 | 3 / 1 / 1 | 94 | 64*3*94² = 1,696,512 |
| Conv3 | 64→64 | 32→32 | 3 / 1 / 1 | 94 | 64*64*94² = 36,192,256 |
| Conv6 | 64→64 | 32→32 | 3 / 1 / 1 | 94 | 64*64*94² = 36,192,256 |
| Conv9 | 64→64 | 32→32 | 3 / 1 / 1 | 94 | 64*64*94² = 36,192,256 |

```text
P_large = 1,696,512 + 3*36,192,256
        = 110,273,280.
```

Conv6/BN7 与首 bank 的 port127 在 Add8 合流，Conv9/BN10 才到 Relu11；Add8 的 port133 还供原后续 Add14。此完整来源和活消费者不因本页只计算 Conv 存储而被删掉。

### CIFAR100 medium

| 原节点 | 通道 | H→O | kernel / stride / pad | T | 精确有效连接数 P |
| --- | --- | --- | --- | ---: | ---: |
| Conv0 | 3→64 | 32→15 | 3 / 2 / 0 | 45 | 64*3*45² = 388,800 |
| Conv3 | 64→128 | 15→8 | 3 / 2 / 1 | 22 | 128*64*22² = 3,964,928 |
| Conv6 | 128→128 | 8→8 | 3 / 1 / 1 | 22 | 128*128*22² = 7,929,856 |
| Conv8，shortcut | 64→128 | 15→8 | 1 / 2 / 0 | 8 | 128*64*8² = 524,288 |
| Conv11 | 128→128 | 8→8 | 3 / 1 / 1 | 22 | 128*128*22² = 7,929,856 |

```text
P_medium = 388,800 + 3,964,928 + 7,929,856 + 524,288 + 7,929,856
         = 20,737,728.
```

### TinyImageNet medium

| 原节点 | 通道 | H→O | kernel / stride / pad | T | 精确有效连接数 P |
| --- | --- | --- | --- | ---: | ---: |
| Conv0 | 3→64 | 56→27 | 3 / 2 / 0 | 81 | 64*3*81² = 1,259,712 |
| Conv3 | 64→128 | 27→14 | 3 / 2 / 1 | 40 | 128*64*40² = 13,107,200 |
| Conv6 | 128→128 | 14→14 | 3 / 1 / 1 | 40 | 128*128*40² = 26,214,400 |
| Conv8，shortcut | 64→128 | 27→14 | 1 / 2 / 0 | 14 | 128*64*14² = 1,605,632 |
| Conv11 | 128→128 | 14→14 | 3 / 1 / 1 | 40 | 128*128*40² = 26,214,400 |

```text
P_Tiny = 1,259,712 + 13,107,200 + 26,214,400 + 1,605,632 + 26,214,400
       = 68,401,344.
```

medium 与 Tiny 的 port121 同时供 Conv3 和 Conv8；主支 Conv6/BN7 与 shortcut Conv8/BN9 在 Add10 合流，再经 Conv11/BN12 到 Relu13。完整原图的其他消费者仍按 D241 保留，不能只提取主支路。

## 4. 显式原生 CSR 的下界究竟说明什么

假设每个有效连接在同一存活状态中有一个 float64 coefficient 和一个至少 int32 的普通 CSR column index，且每个原 Conv 被保留为独立局部定义等式。仅卷积等式右侧有

```text
bytes >= (8+4)*P = 12P.
```

| 模型 | P | 仅 data+int32 indices 的 12P bytes | 与 1 GiB=1,073,741,824 bytes 的关系 |
| --- | ---: | ---: | --- |
| large | 110,273,280 | 1,323,279,360 | 已超过 |
| medium | 20,737,728 | 248,852,736 | 此项尚未超过，不能据此判整轮通过 |
| Tiny | 68,401,344 | 820,816,128 | 此项尚未超过，不能据此判整轮通过 |

尚未计入每条 EQ 的目标槽系数、CSR indptr、RHS、列界、BN/误差载体/Add/原 ReLU 图、原输入 decoder、元数据、读出、完整参数、装配临时量、复制和任何设备/主机共存。若使用 int64 indices，单项成本还会增加。该表是解析存储下界，不是 RSS、实际运行时间、吞吐或已执行的 native admission 结果。

相反，也不能据此断言所有 HZ 或所有同源接入都不可行。下界不覆盖共享 stencil/隐式算子 EQ、压缩索引、其他精确编码、已证明的变量消元或不同的存活生命周期。它也不把逐节点 IEEE 中间结果的“精确”等同于原模型实数语义。若选择另一表示，必须保留并消费同一个共同源、原 bits、EQ/LE、decoder 和所有活消费者，证明其数值合同并完整计费，不能只把该矩阵藏到终端再当作免费。

固定三源不需要同时存活，不能把三个 `12P` 相加冒充逐模型峰值下界；同样不能通过只做更小的 medium 或局部窗口绕过 large 的原注册人口。原 512-bit、fail-closed 及完整物理门没有改变。

## 5. 批量构造可消除复制，但不消除关系

零中心物理槽可以表达 `t_j=M_j*zeta_j`，以局部 EQ 连接前驱同源槽；Conv 按原 kernel/geometry 生成一整块关系，BN/加法和 ReLU 各按完整节点人口统一生成。预先确定列/行段，一次批量装配或记录不可变 block 引用，无须每新增一门复制整个旧 H。普通显式实现的生成量仍为 O(P+N)，N 包括所有物理槽、误差载体、原相位和局部行人口；没有把完整参数校验、可靠界及输入身份算成免费。

每个原门只能有一份实际预激活及误差载体，所有消费者共享。BN 可按通道使用一个共同误差上界 E，但不能把所有空间位置绑成同一个加性 epsilon：`(a-a_hat)*t_s+(b-b_hat)` 随 s 变化。每位置的 epsilon_s 有共同真实输入扩展，且同一位置的全部消费者必须复用它。这样的外包不声称保留全部 BN 参数相关性或理想性。

若采用 implicit Conv EQ block，可复用既有卷积几何及源身份语义，避免 P 项常驻展开；实际正向/转置作用、可靠界、D240 完整 r/e、原谓词求值和终端消费仍有成本。没有完成这些路径之前，不称完整 native H、已认证后继消费或 GPU 资格。本页不引入 D228/D229 Bank、几何相位池、query-support 菜单、split、BaB 或额外求解器。

## 6. GPU 现有代码：可借布局，不能继承可靠资格

本轮只读核对如下代码：

- `act/back_end/interval_tf/tf_cnn.py:746` 的 `_conv_bound_pair` 用正负权拆分、batch stacking、两次 Torch Conv 和相加；`:109` 继续加 bias。可借批量布局，但没有该归约及偏置运算的向外误差证书。
- `act/back_end/utils.py:130` 的 `affine_bounds` 是普通 matmul 与相加；没有据此建立 D241 原始参数/BN 到可靠 GPU E/L/U 的合同。
- `act/back_end/hybridz_tf/exact_linear_op.py:336` 的 `ImplicitConv2DOp` 保留 NCHW、kernel、stride、padding、dilation、group；`:616` 的 matvec 仍是 NumPy/CPU 普通浮点乘加。其 exact 是算子表示/几何语义，不等于 GPU directed-rounding 资格；`:593` 的 `to_csr_reference` 明确是测试参考展开器。
- `act/back_end/hybridz_tf/tf_cnn.py:115` 的 `SparseHZAffineExpr` 保留共享 nonconvex frame，但不是已实现的隐式 predicate-HZ terminal 接口，也不证明原始 BN 的可靠数值绑定。
- `act/back_end/solver/neural_hz.py:69` 的 CPU `_outward_row_interval` 有累计误差补偿；它不是 GPU Conv 内核，不能直接据此为另一归约后端授予资格。

同一边界已在 [D041 THEORY 的 GPU 查询接口节](../d041_existing_gate_congruence_20260930/THEORY.md) 保存：历史 device-affine 速度或位级对齐不能自动变成当前可靠包络证明。GPU 初始化历史失败也不证明今日硬件不可用；本轮没有启动新的初始化或 GPU 诊断。

因此下一数值实现若走 GPU，必须明确并验证实际算术、归约、系数及偏置舍入误差，可靠地产生 carrier E 和原门 L/U，保留同一 H 的非凸图与全部原关系。IntervalTF 不能整体替代 HZ；GPU 只能承担已认证的域内算术/界证书及关系块作用。传输、临时 buffer、CPU/GPU 共存、隐式块终端展开或矩阵作用、原 decoder 和可靠停止均未完成，不能报告全路径收益。

## 7. 研究记账

D241 的完整 raw 参数与拓扑事实被复用，尚未补足其明确为 false 的 activation bounds、actual native phase binding、actual model verification 和 D240 执行资格。D242 本页只新增解析成本结论与实现边界，没有新数学测试、真实模型改进、GPU 运行、shadow 或正式回放。正式 baseline、独立 E0 与全部旧工件不变，formal gain 为 0。
