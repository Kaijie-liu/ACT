# Neural HZ 定义原型完成数学门后的恢复记录

2026-10-04，分支redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。Goal active，未完成。上一轮用户方向核对是no progress；本轮完成定义原型和唯一一次完整数学资格运行，是progress。没有外部阻塞，没有后台待跑实验。

## 当前可靠状态

- 正式1870/2413=1063 CERT+807 validated ADV；独立E0为CIFAR10025+Tiny36=61/400；新增都为0，不能相加。
- 最新数学组件变为D180：4056 tests/213 files，含完整4032/212旧人口加24新项，零失败、错误、跳过。13条警告，测试子进程49.781453秒，supervisor64.236344秒。
- D180参考观测域默认关闭。共同t同时满足C/H源观测，canonical delta保留共享源和相位；固定三前向证书，未新增外部验证helper/rescue。仍有合法非参考相位失真，未证明强旧路径净收益或新颖性。
- 最新真实CNN结构证据仍D179，仅图/形状绑定；最新既有系数诊断D172属于另外的ViT结构，不是D179三份CNN的B。
- 真实模型资格、完整物理资格、GPU、smooth/Transformer、shadow和完整回放全部仍未过门。

## 不得更改或重跑

本候选六个冻结文件和freeze均已消费；结果目录experiments/neural_hz_20260831/results/d180_reference_observed_component_20261004_v1已经存在。不得编辑、再导入作未登记诊断、重跑main或换参数再试。未来继承须新版本预注册；不得减少4056/213成功人口或放宽历史冻结门。

冻结SHA256 aeb91c29cfc7db49c50abbca90b4dbd3cd84a647967b833fc09307d38772ecfb。

结果manifest SHA256 a82fd1d847efc1c4c7272c51b10c67aa68f76ad0312e3409fe43832a1980dc24。

inventory SHA256 4fc672ed90d24e5e53273952cf8bbbfc4e27f333b7c38542eb1fb59bc4cda3b2。

exit SHA256 e7ce8dc68c9d5c2735c535f08d78f7c048fab173774c9d511fe00b5457b251c4。

新manifest source7353/input14，全部历史来源、项目闭包、生产provenance前后无漂移。tracked diff SHA256仍29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5，与runner的production candidate digest不是同一口径。会话42255已exit0，不应等待或重启。

## 下一动作及主线边界

先读[结果与边界](RESULTS.md)、[数学合同](CONTRACT.md)和D179的RESULTS/DEFINITION_TEST，然后预注册真实完整预终端bundle的系数及原生前沿绑定，检验新定义在实际mixed weights、偏置、共享residual和下一ReLU上的精度与完整成本。完整B是强制前提；不能以raw-source skip替代原父激活，也不能删identity消费者。

既有32源/64相位有理实现只作数学参考，不是大型CNN后端。后续适用规模、区间系数、GPU算术/误差及完整费用须在新的合同中先确定，不能原地提高已冻上限。不要把这项实现工作变成只优化存储、测试框架或恢复旧剪枝。

比较双方必须得到同一可靠前沿和信息；原HZ强参照、D157、新域的关系/查询及终端全部计费。若没有强参照收益，修改定义而非添加攻击、split、BaB、backward/dual rescue或按实例菜单。普通原终端与原网络见证验证边界不变。

历史/data1/Kane/HyZor、生产dirty worktree、已有实验版本全部只读。所有新工作仅写新的隔离目录，默认off；不commit/push。Claude的其他验证路径没有恢复，也不计入本候选。本轮完整归档清单为同目录ARCHIVE.sha256，从仓库根以sha256sum -c读取核对即可。
