# 共享相位三角组件通过数学检查

本轮将上一版三门证明变成了默认关闭的可执行组件，并一次通过完整 3793 项、175 文件的数学检查。它能在固定普通三门控制上排除当前独立双门关系允许的点，最多增加四条静态行，不新增变量、不删除原 bits，也没有搜索或 split。

这是一项关系组合组件成果，不是已经完成的 Neural HZ 定义创新。经典布尔三角机制是已有知识；原 HZ 加相同行有相同强度。真实 CIFAR/Tiny 改善、GPU、完整回放和正式提分仍未取得，不能把数学正控包装成这些成果。

## 本次执行证据

唯一运行目录为 [本轮结果](../../results/d056_phase_triangle_component_20260930_v1)，源和测试六文件已在首次候选导入前由 freeze.json 绑定。执行命令为固定解释器加 -B，运行本目录 run_math.py --enabled。会话 74940 已返回 exit 0，运行终止；没有遗留 worker，也不重跑本版本。

完整收集及执行 3793 项、175 文件，零失败、零 error、零 skip。新增四项均出现在 JUnit 中；其余完整继承 D053 的 3789 项，未删减。pytest 自身报告 44.54 秒、13 条 warnings；collection 加 execution 58.023457899689674 秒，仍在原 60 秒门内。整个监督器包含认证、哈希和归档为 69.75407276302576 秒，不能把这两个计时混为同一指标。

CPU1、AS16GiB、单线程库、无 CUDA、assertions 与禁止 bytecode 均按冻结 runner 设置。监督器 tracemalloc peak 14299943 bytes、metadata 4824496 bytes，加 65536 reserve 后未触及 1GiB；RSS high-water growth 观察值为 0，不代表实际进程内存为零。pytest 子进程并未因此获得完整聚合物理峰值认证。

source_drift 和 input_drift 为空，provenance_drift=false。原 D047 三模型 census 的 whole-work 失败记录原样保留；D049/D053 原运行没有重启。所有本次数值日志、JUnit、精确 inventory 和 provenance 自动保存到独立新 RUN。

## 已检查的数学收益

两正一负控制复现一条八 nnz 行：旧假点通过六个方向的 D053 pair 行，新行排除差为 3/10。测试同时检查完整独立单门凸包见证、实际物理读出旧值308/171和新上界5/3的取等真点，以及严格扰动点仍超出824/7695。比较范围见 [CONTROL.md](CONTROL.md)，不是全组凸包、完整整数 HZ 或 benchmark 比较。

三负混权、非零偏置控制也通过：新行 2*sum(q)<=2/3+(11/5)*sum(b) 排除通过旧三条固定方向 pair 投影的点，差为7/30。全部八种边符号排列、等幅/不等幅平面公式、非点偏置支持、零预激活的全部合法相位、严格稳定与触零区别、输入顺序不变及身份/资源拒绝均进入四项新测试。

组件按原 phase ordinal 固定边方向，不按收益挑方向；严格稳定时只不生成新增三角行，原 bit 仍存在。它不向原 LP 安装稳定事实，因此没有将语义稳定性误记为生产 LP 冗余检查完成。上下界、原 source 身份、全部返回行的常数和 P 偏置都参与编译。

独立审查者静态复核了两种三角公式、负边常数、P 偏置、原身份绑定、稀疏预检、测试预期和报告范围，未发现阻断问题。审查不是另一次数值执行，也不替代真实模型认证。

## 下一步真实研究的具体约束

已有 D055 排除了固定相邻三通道选择。因此下一步不是增加更多同类小控制，而是新预注册下完整处理原三个模型的全部已定分支、窗口、通道和 slots，先支付普通可靠界，再处理全部未严格稳定门的不同三元组，包括触零门。不能将旧 Medium 的80组当白名单，也不能省掉 Large 或未完成的 Tiny。

只读接入审查已定位可复用路径：D025 census.first_form 与 receiver_interval 以及 D015 的 Conv 几何和可靠 post_affine；source 必须是第一 bank 的激活后原坐标和盒，第二 bank 参数继续使用可靠区间而非中点。D047 每对四 seed/四 clamp 的工作可以不调用，但这不代表移除了 D053 对 D049/D046 token/form 的依赖。

当前组件每个三角重建三条边，不能预记边缓存收益。下一真实算法若使用缓存，必须重新证明 certificate/context 生命周期及完整成本。旧 evidence.bounded_ledger 不接受 typed dataclass、MappingProxy 或 phase token；typed 窗口暂存与 plain 证据必须分别计费，不能将 typed roots 算零。原256M/200M工作、40M证据、64M entries、512位及时间内存门不变。

三模型全量结构 census 即使成功，也只证明真实证书适用；第三 Conv/残差后继实际参数、native phase 列/frame/decoder 身份和实际后继利用仍需认证。GPU 的可靠批量支持与行合成尚未实现，不能由“适合批处理”宣称加速。完整2413与独立400回放仍在后续。

## 状态与存档

本轮属于 progress：新增并通过实际可复用关系组件，补全了先前仅纸面成立的组合实现。数学组件资格为 true；source_census_completed、source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed、complete_physical_qualification 仍全为 false，formal_gain=0。

正式1870/2413与独立E0 61/400均不变；本轮没有重新验证全部旧解，更没有新增正式CERT或validated ADV。整体定义创新、GPU和满分目标未完成，Goal保持active。

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。九项原tracked dirty changes仍为3806 insertions、57 deletions。本次只新建此隔离目录和唯一RUN，未改生产、旧源码或历史结果，未commit/push。write-page技能用于分开记录已知数学、实现资格和未获收益；本地Markdown读回确认，不声称外部Page渲染。SHA256SUMS绑定源、冻结清单、报告和运行工件。
