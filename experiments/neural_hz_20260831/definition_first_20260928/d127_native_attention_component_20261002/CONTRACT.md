# 原生 Attention 同源查询的数学组件合同

本组件实现 D124 NATIVE_ATTENTION.md 已证明的固定 query、仿射 score 与 value、共同分母的方向阈值查询。它推进所需 smooth 和 Transformer 原语，不撤销普通 CNN/PWA 的优先研究。候选默认关闭；本阶段不执行模型、source census、GPU、shadow 或完整回放，不声明完整新域、新颖性、实际提速或新解。

## 非凸语义和原身份

原生元素保留同一个 D112 System 对象、全部原连续列、signed binary 身份、EQ/LE、frame，以及每个 token 的 score Form 和联合 value Forms。语义是原 System 上的共同函数图：所有输出共享相同 score、token 输入和分母。不是为每个输出方向创建独立输入，也不以临时二维投影替换整个 HZ 为凸域。

本组件固定 query 后只支持连续源上的仿射 score/value；形式直接依赖二元列时拒绝，而不是把二元列放松成连续值。其他原 bits 和谓词仍完整保留。形式常量与系数必须是通过 512-bit 门的精确有理数，不接受未经认证的浮点模型系数。构造不删除或 pivot 任何二元因子，也不认证原模型的 Softmax、BN 或参数绑定。

build 的 enabled 缺省为 False，关闭时不读取输入、不构造或修改系统；显式 True 才创建 immutable native fiber。至少一个 token 和一个 value 通道；frames 为每 token 一个 frame，均须等于原 System.frame。帧不匹配、形式越界、形状或资源失败均不部分提交。certify 重核 fiber 元数据，不因直接构造或 replace 对象就信任伪造的精确性标签。具体输入重构继续依赖保留的原源和原 decoder；本组件不返回可直接记账的 ADV。

## 阈值查询及精确性范围

给定一个联合 value 方向 w，将各 token 的值先按同一源合并为 v_i，再计算

```text
Z=sum_i exp(s_i)v_i / sum_i exp(s_i),
F(t)=sum_i max_(x_i in Xi) exp(s_i)(v_i-t).
```

当 token 使用的实际源列两两不交、System 没有 EQ/LE 关联时，Xi 的乘积是被查询源的实际盒，实数定理 max Z<=t iff F(t)<=0 成立。独立但无关的原 bits 不改变此结论。若 token 共享源列或源带谓词，原生图仍共享原身份；乘积盒 F 只给可靠上界计算，不宣称精确，也不能把其正下界当成真实反例存在的证据。

方向合并后每个 token 的 (s_i,v_i) 是精确二维盒像。使用实际生成元的有理角序与叉积形成边界，边上只需端点以及 ab<0、0<lambda_star<1 的驻点，lambda_star=-1/a-(v0-t)/b。点和线段按同一优化原理处理。几何边界扫描不是输入/相位 split，也不启动多条验证路径。

所有 token 共用同一个 score shift，取共同 score 盒上界的最大值。不得对各 token 独立平移而改变权重。每个候选点的指数以认证区间求值，先做有符号乘法再取逐 token 最大值与求和。只有 F_upper(t)<=0 才返回 certified=True；否则只是未认证，不能伪记失败阈值为不安全。返回的 F interval 始终标明是平移后的 product-box 函数的区间；共同的正缩放不改变阈值符号或零点。

不同方向、不同 heads 的最大化见证不能拼接。本阶段没有声称多头联合精确、输入斜面已实现、后续任意层都具有低成本查询闭包，或原 LP/MILP 能精确消费指数图。未来线性 lowering 须另证并付费，原终端决策边界不变。

## 认证指数与费用

指数模块只使用有理区间、明确有向 dyadic 舍入和可证明的 Taylor 尾界。支持 [-64,64]，把正值缩放到不大于 1/2，计算 k=0..63 共 64 项及几何尾，再认证平方；负值以正值结果的倒数包围。正值核心使用 72 个二进制小数位，负值倒数最终用 168 位 dyadic 以保持 exp(-64) 的严格正下界。指数范围外或位长超界 fail closed，不启用浮点替代或新救援路径。常量与循环上限在冻结中固定，不随测试结果修改。

数值证明采用正项单调性：若 rl<=r<=ru，项递推的向下/向上舍入分别夹住 r^k/k!。第一个遗漏项被 term_hi*ru/64 上界，后续项比不超过 ru/65，故尾和上界为该首项除以 1-ru/65。正区间上的平方保持序，倒数交换两端，再向外舍入。普通 float exp 不参与认证。保证的是数学指数函数的区间，不是具体框架浮点 Softmax 的误差合同。

几何 O(sum d_i log d_i)、阈值扫描 O(sum d_i) 只描述实数结构工作；指数认证、多精度位长、源合并、所有 value 方向、共享系统引用、查询证据、终端转换和主机设备开销另计。本阶段采用原 64M retained-entry 与 256M 查询工作上限及 512-bit 有理数门；计数通过不等于完整物理内存、GPU 或真实性能资格。

查询重核的条目上限取 min(原 max_entries, 本次 max_work)，包括保守工作空间预留；它可以在算术计数耗尽前拒绝。重核随后按条目量收取符号工作，几何、指数和合并共享同一算术计数器。这些计数不是多精度 bit complexity 或 wall-clock 保证；物理资格仍未授予。

临时二维 zonotope 是查询几何，不是域退化。共享 native 图保留真正的指数相关性；有限实现给外包认证，而非宣称有理数精确表示 exp。

## 比较和全局限制

普通正控包括 D124 的满二维 score/value 反相关例以及 t=5673/10000 的内部驻点例。后者相对明确的精确概率图加独立乘积与 energy 参考存在严格差；不扩大为击败所有完整 RLT、Taylor 或既有 Transformer 验证器。测试必须同时覆盖退化像、共享源导致 exact_product=False、公共 shift、旧 bits/EQ/LE/frame 原样保留及不确定/资源失败。

全部 D125 的 3881 项数学测试保持，不重跑旧来源 worker。旧 HZ 同证据参考、实际模型绑定、全 GPU、逐结构 shadow、逐家族及全部 2413/400 回放仍是后续要求。正式 1870/2413 与独立 CIFAR10025/Tiny36=61/400 均零新增，默认与历史账本不变。

2026-10-02 Australia/Sydney；redu-hz；HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。只写新隔离目录，不改生产或旧冻结文件，无 commit/push。
