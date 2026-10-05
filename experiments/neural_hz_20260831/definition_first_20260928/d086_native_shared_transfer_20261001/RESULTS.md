# 原生共享幅值上行组件通过完整测试

默认关闭的候选已在实际 SparseHZono 谓词上实现共享父相位幅值上行，并保留完整连续及二元残余。一次冻结执行通过3821项测试、182个文件，零failure/error/skipped。它将D085的纸面公式落实为可执行原生关系变换；不是新域新颖性证明、预训练网络收益或GPU加速成果。

## 数学与实现结果

原整数HZ的每个状态都有新增辅助量的规范扩展，原谓词、输出和全部原相位保留，因此新系统投影回原坐标恰等于原HZ。系数先精确合并、再逐项向外支付binary64误差；浮点辅助量不被假定为唯一精确乘积。所有选定门从实际EQ/LE核验，compact非零图误差拒绝；不依据ONNX名称或求解状态推断相位。

定向普通块 r=ReLU(u+v-3/4)、t=ReLU(u-v+1/4) 验证了严格关系：联合上行加delta<=alpha排除原逐门LP允许的点，精确间隙3/16。该检查使用Fraction及原谓词，不启动LP，不是一个具体网络ADV，也不声称超越已有联合约束。

无残余的双后继增加3个连续量、14条LE；加入一个原连续项及一个原上游bit的宽残余控制增加6个连续量、22条LE。四项新增测试同时覆盖整数见证扩展、零点全部原父标签、child零点双标签、原输入列前缀、exact=False保留、假slot/phase拒绝、非二进制有理系数舍入与buffer/nnz完整字段。所谓宽残余控制仍是合成小块，不是实际宽CNN。

## 唯一执行与证据

命令为 `/data1/Kane/miniconda3/bin/python -B experiments/neural_hz_20260831/definition_first_20260928/d086_native_shared_transfer_20261001/run_math.py --enabled`。结果仅写 `../../results/d086_native_shared_transfer_20261001_v1`。会话78018终止，exit_code=0；没有本轮遗留后台实验。

| 观察 | 结果 | 作用范围 |
| --- | --- | --- |
| 完整测试 | 3821 passed，182文件，13 warnings | D081全部3817项/181文件完整继承，新增4项 |
| pytest报告用时 | 47.76秒 | 不是速度基准 |
| 测试子进程总时间 | 49.18574175611138秒 | 启动、导入、收集、执行、JUnit合计，原60秒门不变 |
| 监督器总时间 | 60.19086084514856秒 | 另含身份检查与封存，不宣称全部监督工作小于60秒 |
| 漂移 | source=[]，input=[]，provenance=false | 旧源码、模型输入及生产身份未改变 |
| 监督器主机观察 | traced_peak=14133245，metadata=4728592，reserve=65536 bytes，RSS高水位增量=0 | 保持旧1GiB门，不是子进程或完整候选物理峰值 |

CPU1、单线程库、CUDA不可见、AS16GiB及完整有序人口检查均保持。13条既有warning为TypedStorage弃用1条及record_property/xunit2提示12条，未隐藏或忽略失败。所有六源在候选AST、导入、收集、执行前冻结，无数值预跑，无失败后修改重跑。静审只修正了新测试对费用字典字段的预期，保留全部成本字段；该修正发生在冻结之前。

freeze.json SHA256为 `8e16ba6422e40ed2b1242037fff605567bfea81e67c4f3827543c4751e0797ae`；exit.json SHA256为 `22974fc6669b1db95ae986181c3ac29fecfc4e0ab3b4e773f8064c9405814e6f`。来源、依赖、baseline provenance及完整历史合同由唯一RUN的preregistered.json保存，执行前后再次认证。没有修改旧freeze、失败版本或历史结果。

## 资格边界与下一行动

component_tests_passed、mathematical_component_gate_passed、all_stages_passed为true，最后一项只指本次注册的组件阶段。actual_model_binding_qualified、actual_phase_column_binding_verified、native_HZ_admitted、GPU和完整物理资格仍为false，formal_gain=0。

下一项应是实际同结构块的原生绑定与完整费用预注册，不再仅增加同类toy测试：通过隔离HybridzTF观测入口捕获真实pre/post读出、原(s,eta,z)及run身份；宽层完整保留其余原门与项，不能改变已冻结旧人口。稳定门、共享门别名、deferred/lazy路径必须分别有真实来源证据。当前a,b结构参数尚没有模型级统一selector；返回数组独立也不等于模型/decoder来源已认证。

新连续列目前只由本函数分配，不能未经协调就把结果塞回生产共享frame。生产 `_sparse_relu_slots_for` 用 frame registry 与当前hz列宽的最大值分配新槽；若其他缓存支路先使用旧列宽，存在复用新辅助列的风险。真实集成须先认证/协调同一frame内全部分配及活跃支路，不以相同frame整数证明跨运行身份。本轮没有修改这一生产分配器。

新行尚未实现GPU批量生成，完整原/新HZ、Fraction、构造临时、证据、终端降回和decoder成本仍待计费；buffer计数不能替代这些费用。下一层条件输出relay也未实现，不能把舍入后的辅助量当精确乘积直接续用。此候选与原HZ加同样行等强，实际贡献是可检验的共同关系接入，不宣称已完成定义创新。

## 保存与正式记账

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。上一轮文献复核提供了分组提升已有先例的研究证据，未产生可运行候选；本轮获得新的冻结源码及实际执行证据，属于progress。

新工作仅在本目录及唯一RUN；原9个tracked文件仍为3806 insertions、57 deletions。没有生产默认变更、模型/LP/MILP/GPU运行、shadow、2413/400回放、commit或push。正式1870/2413（1063 CERT +807 validated ADV）与独立CIFAR100 25、TinyImageNet36，共61/400，均不变、不相加。Goal保持active；组件通过不等于整体目标完成。

write-page技能用于分别记录数学结论、实际执行、费用与未获资格，仅保存和读回本地文本，不发布外部页面。
