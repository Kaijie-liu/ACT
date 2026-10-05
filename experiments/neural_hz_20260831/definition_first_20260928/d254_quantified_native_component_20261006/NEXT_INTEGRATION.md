# 真实网络接入中的重基底坐标缺口

本轮源码分析定位了一项具体覆盖问题：当前组件按原parent幅值列提取child系数；生产重基底保留原图，却使child改读新image列，因而可能把实际非零系数读成零。下面给出最小反例和一个可反证的后继接口，不重建完整卷积图、不关闭lazy/rebase，也不将这项支撑修复称作新抽象域。

## 原生生产入口

act/back_end/hybridz_tf/tf_cnn.py中_try_phase_selective_exact_relu在约805行通过正常_sparse_apply_relu得到SparseHZono core，约844行将它与stable-positive lazy expression合成。约1565行的正常RELU路径返回该core和expression。因此谓词来源并非必须由一张额外声明图重建。

但core单独并非完整输出状态：stable读出仍在expression，其他skip/deferred消费者、全frame宽度及decoder也必须保留。D254 capture要求实际SparseHZono及显式Gc/Gb（native_quantified.py的_charge_hz/capture）；它尚不能以完整lazy消费者接口代替输出矩阵。上述入口只是避免不必要重建的候选位置，不是已合格adapter，也没有证明整个前缀资源可行。

sparse_hz_rebase_image_exact（solver_hz.py:555）保留旧列、二元位和EQ/LE并追加image linking EQ。安装rebase后，hybridz_tf.py约790行会清除同frame的_sparse_relu_slots缓存；Add还可能合并/重排行。因此只有最终slot cache、对象id或frame整数，不能恢复完整门身份。后继应认证出生行内容、列追加世代、实际RHS与所有消费者，而非仅保存行号。

## 发生在普通重基底下的最小反例

取独立父源f1、f2属于[-1,1]，原门q_i=ReLU(f_i)=(1-eta_i)/2。对输出可靠界[0,1]调用原rebase，新增kappa_i，存储等式为

~~~text
eta_i+kappa_i=0,
q_i=(1+kappa_i)/2.
~~~

这是二进制有理数的精确实例，无需数值特例。所有原相位和门谓词均保留。后继读出为

~~~text
g1=q1-q2+t/8,  g2=-q1+q2+t/8,  t in [-1,1],
d=(g1-g2)/2=(kappa1-kappa2)/2.
~~~

D254沿用的qcoeff只检查parents[i].slots[1]对应的旧eta列；child存储行只有kappa，故两项都读为0，固定tau=0并拒绝。实际沿原link代回得到d=q1-q2、tau=1、r=t/8、e=0。这个拒绝是表示覆盖缺口，不是错误CERT或健全性漏洞。没有执行此反例，也未证明固定三个真实模型确实触发此情形；证据为源码和独立纸面复核。

## 统一的同源读出归一化合同

只使用同一frame中可逐项认证的生产追加link，形式为

~~~text
a_j*kappa_j + sum_k c_jk*xi_k = R_j,   a_j != 0,
kappa_j = (R_j - sum_k c_jk*xi_k)/a_j.
~~~

kappa_j必须是该link行中唯一的本次新增连续image列，其余支持均为本次rebase之前已有列；同一次rebase可以包含多条这样的行和多个image列。原二元列可被读取但不作pivot、不删除。对完整登记的parent/child读出按追加DAG逆序统一代换；保留完整原HZ、所有列、二元位、EQ/LE、输出和decoder。这是读出规范化，不是消除原因子的投影算法，也不是一般循环等式消元。

每一步由H内已有EQ给出同值关系，归纳得任意原可行赋值上f=N_H(f)。因而在规范化读出上证明的相位关系可拉回同一H；它不提高原整数集合表达力，不构成另一项新域创新。任意循环、无法核实来源的行、行内含多个本次新增列、数值/位长/资源不合格均拒绝，而不猜测等价。

实际存储常数必须逐项保留。若旧读出为c+G*xi、新读出为m+r*kappa，而实际存储link为G*xi-r*kappa=R，则

~~~text
m+r*kappa = (c+G*xi) + (m-c-R).
~~~

生产link_rhs使用浮点center-hz.c，不能默认它等于两个存储数的精确差；最后的完整常数偏差必须进入r/e。该代换证明的是stored H中的同值性，不会自动认证原ONNX/BN/前向浮点模型。

单epoch展开工作至少覆盖原读出出现数，以及每次image出现所对应link的全部支持；乘、除、加、合并排序、证书、位长和全部消费者逐项计费。多epoch只有认证追加DAG才递归成立。共享缓存可以避免重复遍历，但不保证最终展开nnz或全网成本小。旧显式Conv超账和旧稠密reader失败不因此被取消，也不授权删掉收费。

## 下一项有明确成败条件的工作

为原正常生产前缀提供默认关闭的birth/link内容记录和完整消费者描述。先证明并测试上述精确代换，再在固定的CIFAR100-large row117、CIFAR100-medium row30、TinyImageNet-medium row73三个完整来源中核验：

完整登记的原门，经已有link统一归一化后，能否恢复正确parent幅值系数，并在原预算内取得完整r/e？

不能用有利窗口、关闭rebase、仅core或不同性质快照代替完整来源。若没有原生门、link格式或来源无法认证、仍超账，记录真实拒绝。已完整保留的非零m-c-R不因非零本身而拒绝。若通过，仍需补全模型/性质哈希、可靠L/U及BN对应、旧terminal精确signed转换、原生前向关系消费和全部原晋级回放。尤其sparse_hz_fast_bounds目前不读取EQ/LE，关系写进终端不代表下一ReLU自动获得更紧界。

本合同尚未实现/预注册真实worker，没有新模型或GPU运行。它给出接入研究下一动作，不把组件或接口完成缩成整体Neural-HZ目标。
