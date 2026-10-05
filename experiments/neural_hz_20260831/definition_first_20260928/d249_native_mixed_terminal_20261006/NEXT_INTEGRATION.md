# 原生混合关系的未完成接入边界

本阶段的终端运输是 D245 数学关系走向实际验证的一个组件，不是新的数学
创新声明，也不是完整 Neural-HZ 目标的替代。下述项目均仍需独立证据。

## 实际来源认证

D064 的实际 EQ/LE 认证能证明给定 SparseHZono 中的门关系，但不能证明它
来自指定 ONNX 模型，也不能把任意调用方 decoder 当成原始输入证书。
D241 的三个真实来源记录仍没有 complete child affine map、native phase
binding 或 propagated activation bounds 资格。必须绑定完整实际 frontier、
模型与性质哈希、所有原相位、全部消费者及具体输入映射，不能用小型 fixture
替代三个完整目标，不能关闭原惰性路径来制造较容易的入口。

本组件在原生 latent 基底使用固定 Gram。固定 carrier 不经 EQ 认证便仍按
[-1,1] 支撑余项，这可能导致 mixed guard 不成立。若不成立，应记录真实
结构覆盖不足；不能改 tau、交换多个方案、用 terminal margin/LP 状态选择
另一个方向，也不能把 carrier 误差或者非零 compact 误差设为零。

## 在线前向和生命周期

原生产 `_sparse_cont_slots_for` 以 `(frame, layer)` 登记，已有 MATMUL 消费者；
它立即写高水位且只按槽数核对复用，不是关系语义的 staged commit。
生产 rebase 可以保留整数 frame_id 同时清理登记；release 会重置编号。
因此无论局部 n_cont、对象 id 还是 generation++，单独使用都不能证明
旧 skip 与新辅助列不会混淆。本组件不向这些 cache 安装。

后续完整接入需固定整批结构人口和独立 relation-role 身份，用实际谓词及
完整源形式的语义指纹绑定 lease，从全 frame 高水位一次预约。候选与所有
相关 deferred/precomputed 消费者必须原子发布；失败不发布半状态，已花
费用不退款。仍存活的同-frame 分支要求旧列保持 identity embedding，且
高水位不可回收；重排必须换真正 frame 并证明所有消费者和 decoder 的映射。

当前 `sparse_hz_fast_bounds` 只读中心和生成元绝对值，不读 EQ/LE。
正常终端消费新增关系并不证明下一 ReLU 的 bounds 已增强。前向关系消费
必须保留同源、相位兼容的信息及可靠的 bounds 证书，不能把向外舍入后
任意辅助解解释为真实条件矩，也不能把已有单门凸包换名作新能力。

## 终端和资源

原 `_lower_hz_milp` 的普通浮点 signed 到 0/1 平移需精确审查。以 stored
float64 .1 和 .2 为二元系数，EQ 右端平移就可能不精确。这是普通系数下
的转换问题。本组件只验收精确转换，不改旧 solver，不声称所有非二进制
有理证书均能通过终端。未来若扩展支持，需单独证明输出、EQ 与 LE 各自的
安全处理；仅对 RHS 做 nextafter 或仅沿用 signed 行证明不充分。

CPU 数学回放不授予 GPU 或完整物理资源资格。D247 的 strace 自检被 EPERM
拒绝，CUDA worker 并未运行，不能推断 GPU 算法可用性或不可用性。没有环境
证据变化，不重复同一失败诊断或放宽 AS/权限。后续 GPU 工作仍要涵盖完整
传播、谓词、终端与证据费用，不把单段 tensor 运行当作全面加速。

## 并行数学排重结果

共同来源本身不保证逐源 capacity 可以廉价无损合并。普通共享来源
`f_k=x_k+epsilon*z` 仍可出现所有阈值模式；零辅助列精确 hypograph 的
指数片段不能凭来源标签消失。这只是特定投影表示的边界，不是一般 HZ
复杂性下界。若整个 H 能证固定次序 `Q1/U1<=...<=Qk/Uk`，可用 k+1 个
前缀仿射界精确表示 sum-min；不用前缀变量会支付二次 nnz，使用前缀变量
须支付其 EQ、存储及终端费用。它是已有容量的精确消元，尚无新的强参照收益。

对有界共同多面体 `P={x:Ax<=b, Ex=f}`，条件源 `t=alpha*x` 的凸包已有
perspective 描述 `At<=b*alpha, A(x-t)<=b*(1-alpha), Et=f*alpha`。
共享 EQ 加多行 capacity 如果没有新的跨事件兼容性，仍是这类条件 RLT；
独立复制给每个门只得到完整单门 hull 的交。此结论用于排重，不声称解决
原始整数网络，也不作为新论文创新。本轮没有为这两项重复原语另建组件。

整体目标保持从定义出发的强非凸 Neural-HZ、实际网络净增益、GPU、smooth、
Transformer 和新家族。正式1870/2413与独立61/400仍是不可混合的保留底线。
