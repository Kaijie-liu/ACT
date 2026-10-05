# 一般混合读出的数学与真实来源研究合同

本候选在 D119 已通过的共同曲率组件基础上，检验任意有符号权重的仿射消费者及真实原始卷积读出的标量后果。先完成完整数学测试，再执行三模型来源 pilot；默认关闭，不是生产验证器，不改变正式成绩。D119 的特殊比例后继不能代替本候选。

## 数学元素和普通消费者

保留原 D112 System 的连续列、全部 signed binary、EQ/LE、frame 和输入含义。复用 D119 对真实原门的四行 guard 认证。给定完整 readout，从其原输出列抽出组权重，自动计算 rest；不接受调用者手填遗漏其他 fan-in 的 rest。若指定 child，必须认证其原门且 child.g 等于该 readout。

对严格递增 knots 0=t_0<...<t_end=1，内部 residual r_i=g_i-(1-t_i)f-t_i h。令 p=ReLU(f)、q=ReLU(h)、S=2p+2q-f-h，A=(w_0+sum w_i(1-t_i))p+(w_end+sum w_i t_i)q。固定核 K(s)=sum w_i(min(t_i,s)-t_i s) 在 0、1 和所有 knots 上的最小/最大值为 L/U。生成

```text
V_upper = rest + A - L S + sum secant(w_i r_i)
V_lower = rest + A - U S - sum secant(-w_i r_i)
readout - V_upper <= 0
V_lower - readout <= 0.
```

每个 secant 覆盖相应带权实际残余的完整 box，不对残余总和取 ReLU。可选 child 行为 child.q-secant(V_upper)<=0，其中 secant 的范围来自 V_upper 自身，统一支持全正、全负和跨零。旧列和谓词前缀保持不动。拒绝未认证 guard、frame、资源、位宽或 child 绑定，不部分提交。

以上是 D118/D119 已推导的有效关系与一般消费者，不宣称新公式或完成了新域。新行对原整数状态全有效；加强连续查询松弛，不改变原整数具体化。

## 真实来源人口与统一分组

沿用 D025 已冻结的三模型和各自第一份原性质：CIFAR100 large、CIFAR100 medium、TinyImageNet medium。完整继承其 direct next-ReLU 分支、四角及整数中心、全部输出通道、全部 canonical Conv 槽。预期消费者为 320、640、640，共 1600；每行完整 576 槽。此人口在本次执行前确定，不取决于标签、margin、历史结果或本次增益。它是有限真实结构试验，不是全 bank、400 或 2413 回放。

每个 next-Conv 的空间 kernel offset 下，输入通道按原 canonical 顺序连续五门分组，最后四门也完整处理。组内使用等距 knots。padding 组贡献严格为零，保留其原槽记录，不制造相位。其他稳定、跨零、零系数及无收益组都计算和记录，不过滤。原 side consumers 记录不变，动态合流不冒充 direct next-ReLU。

统一重用冻结 source_binding_v2 与旧输入盒、Conv 几何和 interval kernel，从原始三模型/性质 bytes 建立来源；不按模型切换缓存读取路径，不调用旧 main/writer。区间系数覆盖同一个原固定参数，不是新增自由 latent；不将 midpoint 当 exact Form。源 residual 先按相同原输入 ID 合并，再计算可靠区间；此包围可能保守丢失系数相关性，不宣称保持全部 BN 符号消去。原输入、门端口、channel/空间身份和所有原相位保留为证据，尚不认证 native HZ。

## 低成本真实消费者诊断

数学 API 实现完整仿射包络。来源 worker 计算它的保守标量后果，避免每个消费者重新展开所有输入项；这不是把整个 HZ 改成区间域，也不冒充 native 仿射传播已完成。

对每个真实原 Conv 权重组，用相同 f/h 原区间、相同残余补偿和原 fan-in，比较共同核 L/U 与独立缺陷界 L_ind=sum min(0,w_i)t_i(1-t_i)、U_ind=sum max(0,w_i)t_i(1-t_i)。端点项化成 a*ReLU(z)+b*z，在其认证区间上由两个端点及包含时的零点求界。对带权 residual 的 ReLU 取可靠区间上界。组 bounds 汇总后包含原 Conv bias，并通过相同原 post-affine/BN interval 得到真实 next preactivation 及 ReLU bounds。

两组参考在每组都与相同的原单门区间界相交，再完整汇总 Conv；共同关系不能以删掉已有基准信息取得改善。严格 Fraction bound 改善只是此有限来源与读出算法的精度证据，不是与完整 source hull 比较，也不是 CERT/ADV。既有原 H、相位和其他关联始终不被替换。

来源诊断的健全性链是：实际固定参数落在所有认证区间内；共享输入差得到实际 r_i 的覆盖；共同核包含所有理想缺陷；实际带权增量处于 min(0,lower(w_i*r_i)) 与 max(0,upper(w_i*r_i)) 之间。对保留真实关系 p=ReLU(f)、q=ReLU(h) 的端点项取矩形外包范围，再加同一补偿，必覆盖真实组读出。组之间即使共享输入，区间和仍是健全外包，只可能丢失精度。相同的原 BN 区间变换与 ReLU 单调性继续保健全；它不需要把原整数相位放松成域元素中的连续相位，也不证明额外的跨组精确性。

## 完整成本与停止范围

保持 256M whole work、200M nested/model、40M global evidence、64M retained entries、512-bit rational、worker 240 秒、CPU1、AS16GiB、单线程、无 CUDA，以及两项 1GiB 内存门加 65536 reserve。完整继承 D119 测试人口和来源/dependency/provenance 身份。所有 source/cache/group/receiver/record 工作必须预付，interval arithmetic 沿用旧逐操作计费，kernel template 的纯有理数步骤须另有完整保守预付。

原三模型解析预付共 53703433 work；全局 evidence 和 summary 再预付 40065536。全 bank 共 98816 个 direct 消费者、56918016 槽，旧 receiver_interval 的已知必付工作超过预算，因此本次不是该全 bank 算法失败后的缩人口重跑。本次固定人口仍须承担全部本身费用，不能假设一定通过。

每个完成模型的证据包括原来源/参数哈希引用、原输入盒、全部原源 ID 和 residual 区间前提、完整统一模板的可重构定义及每一消费者正负界、原相位数量和 no-gain 记录。模板可引用原完整权重切片、固定 knots 和算法，不必重复序列化可推导的 L/U 字典；每个输出通道的全部模板跨五窗口复用，完成该通道后释放，模板构建和活跃峰值仍完整计费。复用 D047 的原 packet 引用语义与 D025 有界 ledger/encoder；引用不冒充没有参数占用，活跃 packet/raw bytes/box/caches/模板/记录都在内存账中。不得为过门只保留成功行、缩 fan-in 或静默跳过失败模型。

任一失败停止本版本，保留部分证据；不修改重跑，不提高门槛。只有完成全部预注册人口才报告 source 资格。所有 native/model/GPU/full physical/shadow/full replay 资格仍分开；本试验正式新增恒为零，完整 Goal 保持 active。

日期 2026-10-02 Australia/Sydney；redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。源码和精确执行人口将在任何候选导入或数值运行前单独冻结。
