# 联合容量组件的语义与边界

本组件加强同一非凸 HZ 上两个父 ReLU 与两个子 ReLU 的关系。保留全部连续因子、原 signed binary、EQ/LE、共享 latent/frame、所有消费者及具体输入 decoder。它不声明一个已完成的新集合类或新域，不替换 H，不删除或 pivot 二元因子。

输入 a、b、tau、r_bounds、joint_upper、independent_upper、e_upper、child_upper 都必须来自同一已认证源和朝向。父 xi 在 [-1,1]、qi=ReLU(xi)；子 Y1=ReLU(z+d)、Y2=ReLU(z-d)，其中 z=a·x+b·q+r，d=tau*(q1-q2)+e，tau>0。源码只检查有限 binary64、范围和精确运算，并不能认证这些数值的模型来源；调用者必须证明完整来源及原图绑定。

joint_upper 认证 sup(|r-mu|+2*max(e,0))，independent_upper 也是同一表达式的有效上界，mu 为 r_bounds 的精确中点。e_upper 另认证 e 的上界。不得只给一个小数值就把它当证明，也不能把不同模型或不同 mask 的证据拼接。

输出物理顺序固定为 (x1,x2,q1,q2,Y1,Y2)，六个精确有理系数和 rhs。原 JP 行与新行系数相同，只将旧 J+eta10minus+eta01plus 改为 THEORY.md 中的 Gstar。固定 min 保证数学上不弱于原 JP；不按实例、LP 状态、标签或 margin 选择。输出额外记录旧 rhs、共同守卫付款、改进量和明确限定的普通旧约束冗余证书。

冗余证书仅用原两个 unit ReLU 三角包络、Y1 已有上界和 Y2>=0。not_excluded 不是严格增益；来源付款下降、标量证书通过、或候选物理行 nnz 少都不是完整网络资格。

enabled 默认 False，此时不读数值、不触碰 meter。启用后复用 D265 的可靠精确标量基础，每次运算前计费并检查每个约分后有理数不超过 512 bit。资源拒绝沿原 Budget 粘滞语义，不捕获后降级或换路径。未通过数学、完整来源、native 接回、终端及全量能力门之前，原运行路径和分数不变。

数学核心只有常量个精确标量操作，最终 0 新变量、0 新相位、1 LE、至多 6 个物理系数。源行构造、权重读取、共享坐标合并、完整谓词中的展开、证据序列化及终端求解全部另计；原 D265 构造超预算没有被解决。CPU Fraction 实现不等于 GPU 实现；潜在向量化不能记为速度成果。

分支 redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。用户已有 tracked diff SHA256 为 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5；原生产 provenance candidate_sha256 为 15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75，两者作用域不同。本轮只写新的隔离研究目录及 RUN。
