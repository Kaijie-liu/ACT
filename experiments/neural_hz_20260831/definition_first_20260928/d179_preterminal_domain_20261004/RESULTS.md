# Neural-HZ 真实窄接口与精度损失边界

三个真实模型都存在 D178 候选需要的完整预终端仿射链，出生幅值分别为4096、2048、6272维，下一非线性接口分别为100、100、200维。本轮从原模型属性和完整图递推确认了这些尺寸，不再只依赖线性层权重形状。与此同时，定义推导说明窄接口不能自动保住所有相位的精度。尚无新CERT、ADV或实际速度收益。

## 唯一诊断结果

冻结后仅执行一次 [launch_structure.py](launch_structure.py)，新结果保存在 [总回执](../../results/d179_preterminal_domain_20261004_v1/result.json)。三模型的全部30个ReLU逐项报告，每模型恰一条满足同一完整链规则；其余27项有拒配原因，没有按验证结果选择人口。

| 模型 | 出生 q 形状 | m | 实际后继宽度 r | 保守幅值数 p | 候选理论行数 | 原逐门理论行数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR100 large | 1×256×4×4 | 4096 | 100 | 101 | 10210 | 16384 |
| CIFAR100 medium | 1×128×4×4 | 2048 | 100 | 101 | 6114 | 8192 |
| TinyImageNet medium | 1×128×7×7 | 6272 | 200 | 201 | 16562 | 25088 |

直接证据分别是 [large](../../results/d179_preterminal_domain_20261004_v1/model_0.json)、[medium](../../results/d179_preterminal_domain_20261004_v1/model_1.json)、[Tiny](../../results/d179_preterminal_domain_20261004_v1/model_2.json) 的 matches 与完整 nodes/consumers。large为Relu53到Relu59，另外两份为Relu51到Relu57，均通过Conv、BN、Add、Flatten、Gemm。原父skip分别为175、167、167，身份与出生前的Conv输入相同，未当作新q的identity消费者删掉。

三个匹配链的Conv均为3×3、stride1、dilation1、显式四侧pad1、group1；Flatten axis1；Gemm transA0、transB1、alpha=beta=1。BN均单输出，epsilon等属性保存为精确FLOAT的hex值；原scale、variance等数值未解码，故仿射系数有效性与可靠B仍未取得资格。上述只是整图单样本batch=1的结构绑定。

尺寸解释使用已认证本机opset12所选schema，并核对官方版本规范：[Conv11](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11)、[BatchNormalization9](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9)、[Flatten11](https://onnx.ai/onnx/operators/onnx__Flatten.html#flatten-11)、[Gemm11](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-11) 与 [Add7](https://onnx.ai/onnx/operators/onnx__Add.html#add-7)。未调用checker、shape inference或网络forward。

表内p=r+1是保守追加mass的设计计数，不是实现已压缩。原相位槽位需求仍为m，诊断没有绑定实际HZ phase身份。原4m是未作稳定相位简化的逐门四行图，不是实测生产成本；候选2m+18p+2r也未计共同继承谓词、完整系数/source展开、历史bank、终端和decoder。C的密集条目上界依次为413696、206848、1260672，不能由少行推断少nnz、低显存或提速。

## 这对域定义意味着什么

[定义检验](DEFINITION_TEST.md)将损失精确定位为 C D_beta(t-d)，其中 t-d 属于 ker[C;C D_tau]。固定父赋值和整数相位时，裸球包精确的行空间条件及方向误差公式来自D160既有几何；本轮没有将其重报为新理论或未知相位求解算法。

由rank[C;C D_tau]<=2p，本次真实尺寸给出该核的维数下界3894、1846、5870。这是符号秩上界的推论，不是测得的数值秩。下界大不证明误差大、相位可达或完整native含伪点：原caps可能删去方向，实际消费者可能消去方向，参考相位等切片仍可精确。

但压缩窗口已经足以排除一种错误主张：不能仅凭这两个源观测就宣称所有整数相位都保留原精确HZ的全部信息。此结论限定于有剩余半径的裸球规则；完整native的必要性还需附加约束留下局部自由方向。原精确HZ本来就描述真实ReLU图，健全外包不可能在集合包含意义上超过该图。

因此本轮取得的是一个真实定义落点和一个必须处理的精度问题，而不是存储优化成果。下一检验对象已从“是否有窄接口”转为：完整B、共同源、原相位和caps联立时，留下的方向会不会影响下一ReLU，以及同预算下这种关系能否比原HZ更有效地验证性质。

## 下一步只服务于本体能力

先在新的隔离预注册下完成D178语义原型及定向证明测试，保留原完整数学人口；测试必须同时覆盖同源、原相位、共同见证、caps、跨下一ReLU及见证重构，而不只是图匹配或维数。然后认证真实B和共同原生前沿，统一比较原HZ、D157、D178的可消费精度与完整成本。三方获得相同可靠信息，普通终端规则及预算保持可比，不靠新的solver级联、攻击、split或rescue补能力。

若只修复D157的损失而相对强旧路径无收益，或转换/查询费用抵消了尺寸优势，记录负结论并修改定义；不靠继续扩建辅助框架拖延判断。不能直接据本次诊断启动广泛replay、开启默认或更新成绩。GPU的数值健全性及端到端收益尚待实现和验证，smooth/Transformer闭包同样未建立。

## 执行与独立复核

内部诊断7.2363秒，外部监督7.3504秒，内外退出0且未超时。这是元数据审计时间，不是候选运行速度。7327个继承source、14个input、六份新冻结文件及三份旧图记录均按合同前后认证。生产分支redu-hz、commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac、tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5前后一致。

RSS增长加reserve为101994496字节，trace peak加metadata及reserve为90766798字节，分别低于1GiB；不是完整候选物理内存证书。whole_work137323724、branch97323724、evidence1465552、metadata entries2685516均在冻结预算内。计费采用预注册的metadata规则，不与D172逐系数算术work作性能比较。

两位协作者分别独立核对数学边界与源代码/结果，一位实现只读结构绑定。冻结前修正了三个报告问题：异常后认证状态必须显式false；相位数是设计需求而非已认证身份保全；增加外部60秒监督及自动回执。所有修正均发生在首次执行前。冻结后没有编辑代码、修改预算或重跑。

运行回执为 [exit.json](../../results/d179_preterminal_domain_20261004_v1/exit.json) 和 [外部回执](../../results/d179_preterminal_domain_20261004_v1/EXTERNAL_RECEIPT.json)，四份主要产物哈希另以只读shell复核。会话已返回终态，没有本轮未完实验。此前Claude路径未恢复；其后续LOG中的重启、false CERT和N122暂停不替代正式资格，也不计作本候选收益。

正式仍1870/2413=1063 CERT+807 validated ADV，独立E0仍CIFAR10025+Tiny36=61/400，均新增0且不能相加。最后已过数学组件仍为D1584032 tests/212 files；本轮没有重跑、替换或缩减该人口。native、数学组件、真实模型验证、GPU、完整物理资格均为false。Goal active，未完成且无外部阻塞。

文档归档技能用于区分源事实、数学推论、设计要求和未过门结论。文件在本地隔离目录文本读回；未作Markdown渲染验证，不修改历史源/结果、生产默认，也无commit或push。
