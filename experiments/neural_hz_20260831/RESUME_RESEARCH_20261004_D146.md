# Neural-HZ 主线与共同误差候选的未通过项

用户最新重申：重点是提出强大的 Neural-HZ。当前 Goal 已是数学定义优先，保持 active；不要把后续工作转成 helper、存储优化、算子构造修补或弱参照实验。

[D146 数学记录](definition_first_20260928/d146_phase_affine_joint_budget_20261004/THEORY.md)保存了一个项目内具体的闭包：y=c0+G beta+r，共同 weighted L1 残差半径为 a0+a^T beta。对新 ReLU 和精确保留的 live skip，固定参考 tau 产生仍为一次相位的中心，以及 Rnext=kappa R+sum d_j|beta_j-tau_j| 的相位仿射半径。必须保留整个旧 source、guards、predicates、decoder 和 skip 的共同身份；只放松新激活块的 exact residual mapping。

结论严格限于有条件的纸面健全性。这是一般 HZ 可表达的非凸抽象正规形，不是已确立的新表达能力；active-value 等式被放松，不能称精确门图。旧历史成本没有消失，新的绝对值预算还要付列、行和界计算费用。没有新增实现、模型/测试/求解器/GPU执行或后台作业。

同源盒控制对原逐门四行 LP 有严格分离：新界 J<=27/20，旧 LP 允许 J=661/400。但已有两门 max 恒等式同样得到 27/20，所以强比较未通过，不可据此晋级、记收益或启动 helpers。完整整数 HZ 本来就不接受这个分数相位假点。

后续研究标准：域元素与具体化有明确结构意义；原二元非凸依赖、共同 latent 和具体输入可追溯；算子真正利用普通神经网络重复结构；与同信息、同预算且包含已有低成本关系的参照相比，在跨层查询上有实质收益。不要要求外包集合比同信息精确 HZ 更精确，也不要将可回编码一般 HZ 本身当成创新充分证据或否定全部创新的理由。

生产路径与历史成绩未改。正式 1870/2413，外部 CIFAR100 25 + TinyImageNet 36 =61/400；新正式与外部收益均为零。先前测试人口和资源限制不变。分支 redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked diff SHA256 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

按用户长期归档要求，仅新建本记录及隔离数学档案。文档技能用于区分证明、未证能力和来源；未写外部 Page。
