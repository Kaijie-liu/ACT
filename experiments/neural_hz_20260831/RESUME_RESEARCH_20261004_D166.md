# 共同激活候选筛选后的 Neural HZ 续接

主线仍是从 HZ 数学定义提出强非凸 Neural-HZ，不是压缩、helper、旧图或有效行换名。本轮 [D166 证明](definition_first_20260928/d166_shared_activation_definition_20261004/THEORY.md)排除两条具体实现路线，不缩小目标。

共同未知非降 1-Lipschitz 函数的有限样本存在性，精确等价于所有 (q_i-q_j)(g_i-g_j-q_i+q_j)>=0。排序后分段线性插值给充分性；同输入强制同输出。没有额外高阶信息。ReLU 加 (0,0) 锚及 q>=0、q>=g 后恢复精确旧图；少 active epigraph 则 ReLU/2 已允许，同源点后的实际下一门会损失幅值。全部 pair 是一种 O(N^2) 二次编码，不是所有表示的下界；有全域固定顺序时只需相邻线性行。不要把点值 GPU 排序当作集合查询算法。

精确 GELU/SiLU 有 phi(t)=ReLU(t)-d(|t|)，d_G(a)=a Phi(-a)、d_S(a)=a/(1+exp a)。两种 d 非负但不单调，均为 1/2-Lipschitz，故 |d(|f|)-d(|h|)|<=min(|f-h|,|f+h|)/2。基本 even+linear 恒等式 D109 及外部文献已有，不能再当发现；D074 已有相同形状的对称余项及反代警戒。

本轮非平行三源 SiLU 正控：f=x+y/4+1/4，h=-f+z/100，J=S(f)-S(h)-f，源盒 [-1,1]^3。反射恒等式及 |S'|<3/2 直接证明 |J|<=3/200，ReLU(J-1/32)=0。两完整 source-labelled 单门 hull 的交在均值 (-1/4,0,0) 可分别混合 f=+/-1/2 与 h=+/-1/8，给伪 J>=11/192>1/32。所有支撑源点内部、原激活非零；不是 ADV 或真实 CERT。

强参照不需 defect 节点就能用同源增量得到相同 J 界：若 q,p 已存在，两条行各四个本例 source-coordinate nnz。近相反两输出保留 anchor+increment 仍是两连续量；纯 smooth 的证明 ReLU 不是免费原节点或原 bit。无任意 mixed 后继的固定宽度闭包，也无真实模型频率/成本证据。因此不实现这个包装，不扩大成 smooth 适配项目。

下一候选要提供固定实际激活的有限共同关系及真实多消费者的前向使用，先和完整同源 pair、旧可生成结构关系比较；不能把“能编译成原 HZ”本身当否决理由，但也不能拿一个弱参考分离当新域。避免重复 D118 曲率/Peano、D109 偶性、D074 余项换元和本轮未知函数 latent。

本轮仅纸面、只读文献和旧档、新文档；无候选执行、模型、GPU、shadow、回放或后台进程。最新执行人口仍 D158 4032项/212文件；之后任何新执行须新预注册、冻结及全部原门。正式 1870/2413，独立 CIFAR25+Tiny36=61/400，两边新增0；本轮未保旧回放。Goal active。

日期2026-10-04 Australia/Sydney，分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac，tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。新 [工作记录](definition_first_20260928/d166_shared_activation_definition_20261004/RESEARCH_RECORD.md)说明来源与未执行边界，旧档不改。
