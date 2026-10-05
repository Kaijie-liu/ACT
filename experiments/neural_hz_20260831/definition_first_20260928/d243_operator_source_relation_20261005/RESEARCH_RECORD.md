# 共同源相位关系的数学实现与回放结果

本轮已把 D240 的相位作用范围关系接入一个保留完整源等式、原相位、残差消费者和输入 decoder 的 owned 非凸状态，并通过唯一一次完整数学回放：4189 tests / 223 files，零 failure、error、skip。它不再依赖另造的独立四门 Block，但仍不是原 ONNX 模型接入或新的正式解。正式1870/2413与独立61/400不变，新增能力记账均为0。

## 实现和数学证据

[source_relation.py](source_relation.py)提供追加式 Source、封存 H、完整 Affine/Add/Conv 等式、BN 耦合误差、原 ReLU 及同 H 的固定 X 关系。Conv 共享完整 kernel/geometry，各输出等式仍然同时成立。BN 每个实际标量有独立误差槽，但同一标量的分叉消费者共用该槽。声明的 BN 系数区间不是来自原模型 sqrt 的新认证。

关系从实际声明源的 EQ 沿拓扑提取完整表达式，仅对 literal identity alias 作精确代换；保留全部偏置、skip、未选幅值和误差。tau 由固定 child 系数推导，不使用标签、实例身份或求解状态。没有 D240.make_block、Bank/support/geometry、LP/dual rescue、攻击或相位 split 调用。这里只复用 D228 的预算、精确算术及 Row。

新增20项测试涵盖全部声明小图位置、padding边界、groups/stride/dilation、非对称父范围正比例归一化、共享误差、原 EQ/LE、额外二元来源、原四位的16个零点标签以及原输入重构。27个固定整数点和4个非对称缩放点均得到原投影不变的扩展。这些是定理的实施检查，不是采样验证算法或原网络 ADV。

最重要的精度证据是在共同 H 中，由实际存储 LE 的非负组合加3条原 EQ，得到实际终端读出列 F<=51/25。非对称控制的旧分数可行点为 F=25641/12500，严格缺口141/12500；该证书排除所有辅助变量扩展，不是仅拒绝一组猜测乘积。完整单子门前缀及 joint children over convexified full parents 的强参照见证也通过。

这是将已证 D240 上界接入共同源状态的进展，不是进一步提高该纸面上界，不是对所有跨层 RLT 或原网络完整凸包的优势证明，也不证明新集合表达力、PLDI新颖性或新域已经完成。

## 完整成本和执行记录

主控制从19列、8EQ、54LE、97nnz变为23列、8EQ、83LE、181nnz，4个原二元位不变。关系本身追加4个连续产品、21LE、76nnz；终端显式化另付8条产品界，完整增量29LE、84nnz。原图没有从成本账中消失。

全部普通正例共用一个终身 Budget，累计505650 work、213371 entries；拒绝fixture各有事前声明预算。这些是逻辑计数，不是 Python/native/GPU 的完整物理峰值。

六源于2026-10-05 11:28:22 UTC冻结，随后仅执行预注册命令一次，session80926正常终态exit0。没有冻结前 import/AST/compile/数值试跑，没有失败后修改重跑。完整pytest阶段含导入、collection和JUnit为51.52390134520829秒，满足60秒门；pytest自身报告50.43秒、13个warnings。监督器总墙钟66.0849423147738秒，不把它误套为60秒测试范围。

监督器traced峰21996537字节、tracer metadata7735216字节、reserve65536字节，合29797289字节。RSS highwater growth为0，不表示总RSS为0。pytest另受原AS16GiB、CPU0、单线程和60秒限制；CUDA隐藏。本阶段没有取得完整候选物理或GPU资格。

运行目录为[唯一数学RUN](../../results/d243_operator_source_relation_20261005_v1)。[exit.json](../../results/d243_operator_source_relation_20261005_v1/exit.json)、[summary.json](../../results/d243_operator_source_relation_20261005_v1/summary.json)、[inventory.json](../../results/d243_operator_source_relation_20261005_v1/inventory.json)和JUnit均保留。独立只读复核确认实际4189个唯一testcase、223文件，有序人口与manifest/inventory逐项相同；原4169/222完整前缀保留，所有failure/error/skip为0。

## 不能直接晋级真实大模型的原因

本轮没有执行三个固定原模型，不能把这些小图替代完整large、medium和Tiny的来源/能力人口。D241的完整参数来源只作为只读证据绑定，其资格没有转移。

冻结实现的逐项精确算术也存在明确规模障碍：_conv_terms为每个tap至少收12 work，为每个有效tap收2 entries。D241 large中一个64到64、32x32、3x3、pad1层具有36192256个有效连接。因此，仅该层首次范围扫描就至少需要434307072 work和72384512 entries，已经超过单操作200M、终身256M及64M entries门；还未计精确乘加、其他层或任何终端消费。完整large前缀的110273280个有效连接对应至少1323279360 work和220546560 entries。

这些数是由已认证几何与本源码固定计费规则得到的静态下界，不是新模型实测、OOM记录或所有HZ/GPU实现的普遍下界。work数与D242普通CSR的字节下界恰好出现相同数字，单位和论证完全不同。不能因数学测试通过就声称本精确逐标量实现能够直接跑完整三模型；也不能减少窗口、改小模型或提高原预算来掩盖这一点。

隐式存储已保留全部等式，但没有解决可靠批量界、operator-aware终端与完整原模型绑定的成本。当前只有CPU精确数学参考实现，没有可靠GPU计算核。多个Relation仍各自扩展同一父H，没有全网产品列合并分配器；相同frame不能授权跨对象拼接辅助列。全网四门枚举调度也尚未实现。

## 审阅修订与下一行动

所有修订均在冻结前完成。静态审阅修复了公开可变列列表，避免通过无偿全表tuple复制引入新成本。普通结构/只读拒绝现在保留已封存H和已付费用；真正资源失败保持sticky。未完成Source在部分追加后出错或MemoryError时不能发布半状态；测试验证了第二BN通道失败后seal拒绝。终端8条产品界也已计入实际测试和完整账本。

下一候选应以这份源码作为同H语义与精度的数学参考，针对固定三个完整模型预注册可靠批量算子来源接入及固定拓扑选择规则，并重新核算全生命周期。不能把已经静态超预算的逐项扫描原样运行，不能只做构造压缩而回避相位关系的真实适用率和能力收益。若批量界较松导致guard不通过，应记录真实拒绝率并据此修改关系定义假设，不追加算法菜单救援。

原模型/property认证、可靠浮点/GPU算术、完整终端消费、同结构shadow、13家族及2413/400回放、零无效ADV和四并发不回退门仍未完成。smooth、Transformer和其他新家族也仍在完整目标中，不能由本ReLU组件自动取得资格。完整Goal保持active。

## 来源与归档

分支redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff SHA256保持29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，生产13文件candidate SHA256保持15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75。新manifest共7709来源身份、14输入，完整包含D240原7655来源；前后source_drift/input_drift均为空，provenance_drift=false。旧档、生产和正式账本未改，没有commit/push。

freeze SHA256为397fb28da73fcc75acb9d4981ce62d5ef774c37e49f2b227f154438f79533f51，exit SHA256为ddfab992b32ba48f1eedb09a59e5349b7e0c849ad233909e05e97a7f2e0e134a。其余源、结果和依据由ARCHIVE.sha256绑定。文档归档技能用于将数学通过、完整成本、真实模型缺口和正式成绩分开记录，未创建外部Page。

上一轮仅排查中断，未推进候选；本轮新增并冻结实现，完成一次全人口回放和共同H严格精度证据，属于progress。正式1870/2413=1063 CERT+807 validated ADV；独立CIFAR10025+TinyImageNet36=61/400不与其相加。本轮formal_gain=independent_e0_gain=new_benchmark_solves=0。
