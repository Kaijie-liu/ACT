# 同源差值和伴随幅值的前向关系组件验证

本轮把 D042 的通用种子、固定混权分解和 ReLU 增量规则实现为默认关闭的有理数数学组件。研究对象是有限关系如何跨算子组合，不是存储压缩或新的求解器。该组件属于已知条件约化积及差分传播机制的具体生成方法；数学正控不等于已经完成 Neural-HZ 定义创新，也不等于真实 CIFAR 或 Tiny 验证收益。

## 数学范围和认证责任

输入保留原共同 frame、原相位的 0/1 视图、同源读出及其已认证上下界。原 HZ 的连续盒、全部二元因子、EQ/LE、共享赋值和 decoder 均不替换、不删改。本组件不加载或重建原 HZ，不声称形式身份 token 或调用者提供的范围已经自动获得真实模型证书；它只在所列前提成立时生成充分结论。生产接入前仍必须完成源绑定和数值界认证。

种子保存 q=ReLU(g)、p=ReLU(f) 对同一原相位的差值 d=q-p 和伴随 p。混权消费者采用固定左分解 Aq-Bp=A*d+(A-B)*p；共同 source/readout 的系数先按同一身份合并，随后求界。ReLU 使用增量单调性与斜率界，伴随幅值一起前向更新。不得为两个条件值执行两个网络、相位子 LP、输入子盒或搜索；对整数相位分情况的数学证明不授权运行时分裂。

PROOF.md 明确通用种子产生的 19/40 性质与旧 CONTROL 更强 17/40 性质的差别。前者不依赖后者的两个额外源不等式。目标是从种子到最终读出的完整充分推导，不以额外未付费的源支持 oracle 填补证明。

NONORDERED_PROOF.md 另登记首跑前确定的普通非支配变体：只将 f 的偏置从 -1/4 改为 -1/20，g-f 范围跨零，既存在 q<p 也存在 q>p。相同通用规则给出 t-r<=17/200+(14/25)*alpha，以及 Z=t-r+(14/25)*(q-x)<=129/200。预先指定旧点的 Z=163/250，严格差为 7/1000，且仍满足对应的完整前缀/最后单门 hull 加全部六对全局差分行。该变体与原控制一并放在相同十项数学测试中，不增加或事后选择测试函数。129/200 是充分安全界，不声称真实最优值或已获具体 CERT。

## 默认关闭和完整代价边界

生成操作要求显式 enabled=True；缺失、矛盾或不同 frame/anchor 前提拒绝生成，而不是推断 SAFE 或删除原相位。纯 Fraction 组件保留 512 位上限和显式输入规模界；所有测试的有限规模、算术和关系数必须可核查，不能使用未注册的极端大矩阵。原 whole_work_cap=256000000、branch_work_cap=200000000、evidence_prepaid_work=40000000、retained_entry_cap=64000000 仍为后续物理组件的完整门，并不因本轮纯数学资格而免除。

配对关系一般生成两条 LE，原坐标下至多六个非零系数；若伴随关系也物化，再增加两条、至多四个非零系数。宽 latent 读出展开、源认证、历史观察、slack、位宽、终端转换与求解、decoder 等仍需全部计费。不能把单锚活动前沿的八个端点每坐标当成全程常数空间。本轮没有 GPU 算术、完整物理存储或真实模型资格。

## 一次完整数学和兼容性执行

沿用 D038 已通过的实际 3759 tests、169 files，保留其完整 node IDs；增加 test_relational.py 中恰好十个无参数化的顶层测试，总计 3769 tests、170 files。十项范围为通用种子包含性、混权系数、同源消去、伴随量必要性、ReLU 跨零、零点原 bit 两种选择、身份边界、通用严格正控、真实取等以及后继块组合；具体分组以首跑前冻结的函数名单为准。不得执行完后改变函数人口或选择通过子集。

新 RUN 为 experiments/neural_hz_20260831/results/d044_relational_generator_20260930_v1，独占创建；首次 --enabled 消耗本版本。根代理在首次执行前冻结 PREREG.md、PROOF.md、NONORDERED_PROOF.md、relational.py、test_relational.py、run_math.py 的绝对路径及 SHA256，以及十个测试函数名单。schema 为 d044_frozen_v1。静态 AST 解析只检查源码结构，不导入候选或执行数学测试。

collect 和 execute 合计仍为 60 秒；任何 failure/error/skip、人口漂移或超时均不通过。不单跑新测试，不把它们移到额外 worker 的 240 秒窗口，不事后延长预算或重复失败版本。采用固定 /data1/Kane/miniconda3/bin/python，assertions 开启、CPU1、AS16GiB、运行库单线程、禁 bytecode，pytest 的 CUDA_VISIBLE_DEVICES 为空。缓存、临时文件与日志仅写新 RUN。

本轮不注册独立 worker，worker_stage_registered=false、worker_launched=false；继承的 240 秒 worker 上限不使用。旧兼容性测试含已有数学 fixture 和 solver 调用；不把这些当成本候选的新基准求解。

D038 的 1GiB RSS high-water growth 与 tracemalloc peak 加 metadata 加 65536 检查保留给监督器。pytest 子进程沿用 CPU1、AS16GiB 与完整时间门，但未测全进程聚合物理峰值；不得据此声称候选完整物理门通过。该边界与旧 D038 一致，不是删除旧物理要求。

## 来源身份和结果记账

认证 D038 的 preregistered.json、inventory.json、exit.json，再继承其全部源、输入、decoder、解释器、4417 项 GPU 依赖文件及三模型 provenance；GPU 依赖仅只读枚举与摘要检查，不初始化设备。允许复用摘要认证的 D038、D015、D017 只读绑定、drift、provenance、memory 和枚举辅助函数；不调用旧 main、worker、writer，不改旧 globals。执行前后检查完整来源及生产身份。

```text
06252697d664a213c7967a6e726d99fd6217dd5d7311e59ee202b38f4083ff04  D038 preregistered.json
62cba6b99355a0c72ab878a76f7500418f776f130204dd00ead264dbba4d4828  D038 inventory.json
b1f4c6555e538a52c84c0fd78a918fdc2c65677cf1d515ec9e771b97d8e700cd  D038 exit.json
```

异常或超时也保存终态与可获得的日志、JUnit、来源漂移和资源记录。成功仅指完整数学组件与继承兼容性门通过。始终 native_HZ_admitted=false、source_census_qualified=false、gpu_computation_completed=false、complete_physical_qualification=false、formal_gain=0。

日期 2026-09-30；分支 redu-hz；commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式基线 1870/2413（1063 CERT 与 807 validated ADV）、13 家族逐例保旧及独立 E0 CIFAR100 25、TinyImageNet 36，共 61/400 不变。仅新增隔离文件，不改旧模型、数据、失败版本、结果、生产默认或 Goal，不 commit/push。依照 pages:write-page 分开说明证明、假设、实验资格及收益；本地文档，不发布外部 Page。
