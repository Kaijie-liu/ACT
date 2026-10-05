# 六十门正负对照与完整读出账

这是有完整父域证明的纸面控制例，不是实际 CNN、数值试验或新的正式成绩。所有比较保留同一源、原 bits、decoder 和完整消费者。一次比较安装一个后继 child；同时安装两个时，双方都另付相同 child 成本。

## 非平行、满行秩的共同控制

源 x,z_1,...,z_60 独立属于 [-1,1]，前后各三十门：

```text
g_i=x+z_i/100+1/20  (i<=30),
g_i=x+z_i/100-1/20  (i>30),
a=1, b=beta_1,
M=sum q_i,
D=(sum_plus q_i-sum_minus q_i)/30,
v=(sum_plus g_i-sum_minus g_i)/30,
F=D-v.
```

这是带独立小扰动的普通 affine/ReLU/pooling 结构；producer 的行秩为 60，不是 rank3 曲线。可靠界为 g_plus in [-24/25,53/50]、g_minus in [-53/50,24/25]、v in [2/25,3/25]。

可取 R_1=0、其余 plus 的 R_i=1/50、minus 的 R_i=3/25，故 T0=209/50。令 E=sum e_i、T=(sum_plus e_i-sum_minus e_i)/30，则误差的完整二维像恰为

```text
E>=30*T, E>=-30*T.
```

因为两组误差质量分别为 (E+30T)/2、(E-30T)/2，可以分配给各组一个非 anchor 门。无需额外相位枚举。令 hM=sum g_i、uM=b*hM、uD=b*v，实际读出为 M=uM+E、D=uD+T。另安装完整非负输出 cone M>=30|D|；它只是必要条件，不还原原 q。

## 正对照：强于简单逐门 LP

产品的可靠四行编码给出 uD<=v，预算给出 E<=T0，所以即使在有限 LP 中也有

```text
F=(uD-v)+T <= 209/1500 < 1/5.
```

因而通过前向稳定性或原生精确后继变换，h_plus=ReLU(F-1/5)=0，严格余量为 91/1500。不声称只给后继宽界的任意 LP 自动强制其输出为零。

简单旧四行 LP 却允许 x=z_i=0、beta_plus=1/2、beta_minus=0、q_plus=53/100、q_minus=0。这给 D=53/100、v=1/10、F=43/100，真实应用后继得到 h_plus=23/100。这只是旧 LP 松弛伪点；原相位整数语义不允许它，也不是 ADV。

## 强旧关系与严格内部负例

每个固定配对 i,i+30 在整个父盒上满足 g_i-g_(i+30)>=2/25>0。已有 ReLU 有序 sector 关系因此给

```text
q_i-q_(i+30) <= g_i-g_(i+30),
D<=v, F<=0.
```

最后只需一条聚合行。前提认证与构造仍须计费；该关系不按公开标签或目标 margin 触发。它同时证明 h_plus=0 和 h_minus=ReLU(F-2/25)=0。

新候选反而允许以下严格内部赋值：

```text
x=1/20; z_plus=9/10; z_minus=-9/10;
g_plus=109/1000; g_minus=-9/1000;
beta_plus=1; beta_minus=0; b=1;
e_1=0;
e_i=333/2900 for the other 29 plus gates;
e_i=9/1000 for all 30 minus gates.
```

由此 E=18/5、T=51/500、hM=3、v=59/500，M=33/5、D=11/50、F=51/500，h_minus=11/500>0。预算恰为 30*(3/25)=18/5；误差 cone 与输出 cone 均成立，后者取等号。内部 q 全部非负、满足 epigraph、inactive 输出为零且低于各自可靠上界，原源与原门符号也都严格。因此这是普通混合相位关联丢失，不是极端数值问题或漏加非负行。

## 更便宜的已有聚合对照

使用 D152/D185 已有正权四行聚合，保留全部原 guards，定义

```text
B_plus=sum_plus beta_i, B_minus=sum_minus beta_i,
h_plus_sum=30*x+sum_plus z_i/100+3/2,
h_minus_sum=30*x+sum_minus z_i/100-3/2,
v=(h_plus_sum-h_minus_sum)/30,
M=P+N, D=(P-N)/30.

P>=0; P>=h_plus_sum;
P<=(53/50)*B_plus;
P<=h_plus_sum+(24/25)*(30-B_plus);
N>=0; N>=h_minus_sum;
N<=(24/25)*B_minus;
N<=h_minus_sum+(53/50)*(30-B_minus);
D<=v.
```

这些行在新生包上由真实逐门图求和及上面的已知 sector 证明得到。它仍是有损关系，不是旧精确图；也未被本轮实现为正式路径。它证明两个 child 为零，并排除上述新候选伪点。不能由此推断两个完整抽象域之间的全局包含关系。

计费约定：共同 61 个源不计入新增连续量；所有公共源盒、bit 界和 decoder 单列保留，常数 RHS 不计变量系数 nnz。一个后继 h 的原四行图添加 1 连续、1 原 bit、4 LE、10 nnz，共同可靠预激活宽界取 [-3,3]。下面仅为完整列出的有限系统逻辑账，不是物理峰值、GPU/终端时间或字节资格。

已有聚合对照有 10 新增连续量、61 原 bits；7 EQ 包含 B 定义 62 nnz、h_sum 定义 64 nnz、v/M/D 定义各 3 nnz，共 135。133 LE 包含原 guards 的 360 nnz、8 聚合行的 16 nnz、sector 行 2 nnz、child 10 nnz，共 388。合计 523 nnz。

共享 anchor 候选直接实现为 138 LE+4 EQ、649 nnz、9 新增连续量；共享 A=sum R_i beta_i 后为 138 LE+5 EQ、593 nnz、10 新增连续量。还应给候选同等源和读出共享优化，不能只与这个未优化版本比较；最终优化账见本文件下节。

共同 child 宽界 [-3,3] 是保守可靠界：|hM|<=303/5，产品和预算给 M<=303/5+209/50，输出 cone 于是给 |D|<=3239/1500，配合 |v|<=3/25，两个 F-threshold 都落在此范围内。它不取代上面强关系导出的稳定性结论。

## 给候选同等共享简化后的公平比较

候选使用以下 10 个新增连续量：Z_plus、Z_minus、H、v、A、uM、uD、M、D、后继 h。令 H=hM；5 条 EQ 为

```text
Z_plus=sum_plus z_i,                         # 31 nnz
Z_minus=sum_minus z_i,                       # 31 nnz
H=60*x+(Z_plus+Z_minus)/100,                 # 4 nnz
v=1/10+(Z_plus-Z_minus)/3000,                # 3 nnz
A=sum_(i=2..30) beta_i/50
  +sum_(i=31..60) 3*beta_i/25.               # 60 nnz
```

共 129 EQ nnz。E=M-uM、T=D-uD 只是内联别名，不再有误差变量或两个读出 EQ。原 120 guards 仍为 360 nnz，uM=bH、uD=bv 的 8 条产品行共 20 nnz；其余行是

```text
-M+uM+30*D-30*uD <= 0,
-M+uM-30*D+30*uD <= 0,                      # error cone: 8 nnz
-M+30*D <= 0, -M-30*D <= 0,                 # output cone: 4 nnz
M-uM-A-(209/50)*b <= 0,
M-uM+A+(209/50)*b <= 209/25.                # budgets: 8 nnz
```

再加同一 child 的 4 LE/10 nnz，合计 138 LE/410 nnz。E>=0 已由误差 cone 蕴含，M>=0 已由输出 cone 蕴含，不另列冗余行。统一 child 粗界为 [-3719/1500,2999/1500]，包含于 [-3,3]。

| 相同完整读出和一个后继 | 新增连续量 | 原 bits | LE | EQ | 总系数 nnz |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原逐门四行图 | 64 | 61 | 244 | 3 | 793 |
| 原逐门图加已知 sector 行 | 64 | 61 | 245 | 3 | 795 |
| 共享 anchor 候选，充分共享与内联 | 10 | 61 | 138 | 5 | 539 |
| 已有分组四行加已知 sector 行 | 10 | 61 | 133 | 7 | 523 |

前两行尚可通过共享组和简化，不把它们当成本最优对照。关键比较是后两行：在同样连续量和原 bits 下，已知对照不仅系数更少、总行数 140<143，而且证明两个指定性质，新候选不能证明第二个。这一结论不依赖把候选未共享的 593 nnz 当最优账。这里只筛选当前定义与控制例，尚未测量完整实现的存储、查询或速度，也不声称已知对照在所有性质上支配候选。
