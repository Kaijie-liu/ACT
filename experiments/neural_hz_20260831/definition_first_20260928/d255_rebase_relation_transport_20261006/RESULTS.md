# 重基底后关系恢复的验证结果

完整数学回放通过4257项测试、227个文件，零失败、零错误、零跳过。生产重基底使旧提取器遗漏关系的构造反例已经执行；新规则能沿实际存储link EQ恢复同一个定量守卫关系。这里只取得组件的关系运输资格，不是新抽象域完成、真实网络提分或GPU结果。

## 实际证明与覆盖

主输入在同一个SparseHZono中保留五维输入、两个父门的全部连续因子及二元位，调用实际sparse_hz_rebase_image_exact，将9个连续因子变成11个，再追加两个child。最终旧H有15个连续因子、4个原二元位；父/child门本身是按原生公式构造的测试门，不是从三个真实模型前向捕获的门。

旧D254提取器在该重基底状态上因tau=0拒绝，新规则精确归一化实际存储读出，恢复tau=1、a=(7/8,7/8)、b=(1/16,1/16)、r属于[-3/32,3/32]及e=0。归一化后29条关系与未重基底的同结构关系一致。保留全部image列和link EQ，不pivot或删除二元列，不更换普通终端路径。

实际存储LE的非负组合加原EQ和每条真实link EQ的−2倍，给完整显示读出F<=265/128，即2.0703125。证明中辅助列完全消去，原可行分数点F=8509/4096与该上界相差29/4096。阈值1061/512下，新证书给−1/512、旧分数点给21/4096的margin。这些分数点不是ADV。

两次固定普通LP观测旧上界2.75、新上界2.05859375（527/256）。后者数值与一个已核原整数点相等，但本轮精确证书仍为265/128，不声称已证明较强数值界的紧性。候选组件不调用solver、攻击或其他救援；两次LP只作为原普通终端对照。

27个固定原整数点和16种全零合法标签均有canonical延拓并保持五维decoder。两次实际重基底、原binary参与link、无pivot常量link、EQ重排后的内容重绑、完整旧谓词/输出和全局宽度均通过。存储常数偏差−2^-55被保留，没有把浮点减法当成实数精确减法。非法private依赖在代换前后检查，篡改、资源失败及失败整批均fail closed。

## 成本与运行边界

单关系新增6个连续因子、29条LE、104个非零项；完整谓词为6 EQ、37 LE、154 nnz，最终21个连续因子和4个原binary。两条image列及其link EQ均计入完整状态。共享Budget最后为work=1938672、entries=620307；该数对应本轮固定组件人口，不与不同测试人口的D254数值比较成提速。

原生产rebase与测试输入构造未计入候选Budget逻辑work，但包含在完整pytest时间中。完整实际前向的构造、可靠bounds、所有live消费者和物理峰值仍须单独过门，不能用该逻辑账单替代。原256M/200M work、40M evidence、64M累计entries、512-bit和60秒pytest门均未放宽。

CPU0单线程、CUDA隐藏。完整pytest进程52.684899秒，日志51.53秒；监督器总67.333102秒不属于该pytest时限口径。监督器traced peak=21930743 bytes、tracer metadata=7663088 bytes、RSS高水位增量0；这些不代表pytest完整物理峰值、真实网络或GPU资格。

## 执行与独立读回

唯一RUN为experiments/neural_hz_20260831/results/d255_rebase_relation_transport_20261006_v1，会话2298实际退出0。freeze时刻2026-10-05 17:07:21 UTC（悉尼2026-10-06 04:07:21）。分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；tracked binary diff仍为29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。

执行后独立核验JUnit实际4257个case、227个文件及零failure/error/skipped；inventory前4241项与D254有序人口完全相同，后16项与预注册一致。manifest认证7886份来源身份、14份输入；旧7841来源映射及原输入全部保留。34份运行工件哈希、六冻结源码及manifest/freeze交叉身份均通过；source_drift和input_drift为空，provenance_drift=false。

freeze SHA256为d92f7bc30be05bceb1a8efbb061616fc95f7c0938f65f98a5617839def7f5f69；exit为d1e86c7c166d867b89b60a4c1021253bddfcc6f7853f938599f3b0c58b2dfd27；summary为4d3408a64297469f53dba851af4526a7f517f856040dbb7d8f97fd26b37bcc68。manifest为49a4af4dcea963fa2bf719761a1e97f86db31585ffdbbde9e5ccfa3f5b153714。

summary在测试结束时保持rebase_native_transport_passed=false，不自行授予全门资格；监督器完成全部后验检查后，exit才置该标志为true。旧quantified_native_transport_passed和native_mathematical_transport_passed不转授新run，均为false。

## 未完成事项与成绩

实际三模型完整前缀、同结构shadow、13家族、完整2413、独立400和四并发不回退尚未执行。实际模型、在线安装、GPU、完整物理、新域和新能力资格均false，domain_definition_changed=false。正式1870/2413（1063 CERT+807 validated ADV）、独立CIFAR10025和TinyImageNet36，共61/400均不变，所有gain为0。

本轮解决的是同H关系在生产坐标变换下的可识别性，不是扩大整数集合表达力或一般等式代换的新颖性。下一实质工作必须回到完整真实来源的生命周期和可执行接入，不能用继续累加构造样例代替Neural-HZ能力突破。
