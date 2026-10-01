# HybridZ 设备候选的统一预算控制

2026-10-01。[先行协议](hz_device_supervision_design_20261001.md)和
[固定配置](../configs/hz_device_supervision_20261001.json)在 `5bbb0b3cb` 冻结并推送。
本阶段完成设备候选入口的 CPU／模拟故障监督，不执行实际 CUDA、真实模型或原生求解。
候选内核、精确接受公式及旧生产入口均未改；不能据此宣称 GPU 加速或完整 MoE 证明。

## 接入和保证范围

[新监督器](../scripts/hz_device_supervised.py)使用四阶段流程：准入、生成、独立检查、
完整接收。调用方锚定请求、batch、完整 query 清单；父进程生成 CUDA 上下文，绑定
同次 invocation、UUID、produce 截止及 allocator 限额。父进程自己锚定 admission、
payload 和 checker stdout，失败终态也复核已发布前缀。

[准入模块](../scripts/hz_device_admission.py)只读解析 GPU UUID、显存、利用率和计算
进程，要求两个间隔观察同时满足空计算进程、利用率不超过 5%、空闲至少 4096 MiB。
这不是资源预约或互斥保证。当前监督入口显式拒绝非 stub 的 CUDA 请求；仅 producer
设备路径可见目标 UUID，CPU／检查／接收进程隐藏 CUDA。

本轮 GPU 拒绝由明确标记的合成资源记录触发；初始化／OOM／同步／回传故障仍是
模拟，不是驱动或稀疏 CUDA 的实测。启动前实际只读资源快照显示设备利用率约 90%，
有其他用户进程；没有启动 CUDA 或干预这些任务。

所有准入、导入、创建／导出、候选、序列化、精确检查、父端 hash、接收、进程回收、
终态发布及 API 返回均计费。候选成本是 producer 阶段的子成本，不能再次相加。
失败缺候选时仍有父阶段耗时和事件前缀，不记为零。晚到、缺义务或错误设备元数据
不能得到完成状态；正常完成的非正支持界不代表 UNSAFE 或完整 SAFE。

## 固定结果

[最终独立归档](hz_device_supervision_20261001_r3.json)通过。15 项新测试覆盖 22 次
固定调用，另有两次明确标记的时钟注入控制，不把注入时钟当墙钟性能数据。

| 22 次固定调用的终态 | 数量 | 含义 |
|---|---:|---|
| CHECKED_GIVEN_HZ_DEVICE_SUPPORT_EXECUTION | 4 | 完整给定 HZ 支持执行，非完整 MoE 证明 |
| ERROR | 10 | 预定异常／污染／缺输出被拒绝 |
| TIMEOUT | 6 | 预定截止被触发，不能接受已有前缀 |
| RESOURCE_UNAVAILABLE | 1 | 模拟繁忙准入，无 producer 启动 |
| RESOURCE_LIMIT | 1 | 1 字节控制上限故意触发主机 RSS 拒绝 |

| 正常来源 | 完整支持目标 | 本次 API 秒数 |
|---|---:|---:|
| 双侧 guard | 4/4 | 11.1560 |
| 非正性质 | 1/1 | 2.8517 |
| 等式耦合 | 1/1 | 2.8453 |
| 私有二元连续松弛 | 2/2 | 2.9822 |

八条精确界与原设备内核 CPU 控制逐值一致，审计独立重读原始矩阵和证书，并重算
该差分。全部失败的触发位置和见证均核验，不能只拿同名 ERROR／TIMEOUT 充数。
控制套件耗时约 63.5221 秒，含故意阻塞；不作为速度比较或真实请求成本估计。
首个正常调用比 R2 更慢，但仍在原三十秒预算内；没有据此追加调参或推断唯一原因。
22 次调用采样的最高父子主机 RSS 为 753,147,904 字节，不是 OS 硬配额。

[设备内核回归](hz_device_supervision_device_regression_20261001_r1.json)另有 15/15
通过及独立审计；逐行精确检查核 15/15 通过，重查 75 个旧 LP、72 个对偶和 3 个盒
事实，没有新求解。工程导航、历史保存与交接检查另为 9/9、4/4。

## 审查修复和保留记录

全部本地证据位于 `/data1/Kane/MOE/baseline_runs/`：

- `hz_device_supervision_20261001_r1`：12 项通过，22 个调用保留；只读审查发现
  失败前缀审计和准入父端复核的阶段截止尚需收紧，未作为最终接受归档。
- `hz_device_supervision_20261001_r2`：15 项通过；父端准入复核纳入阶段最终取钟，
  所有已锚定前缀在失败时也检查。新增时钟注入、失败前缀污染和发布截止控制。
  后续审查发现同时删除阶段和终态两个锚点仍能绕过前缀检查；该版保留、被 R3 取代。
  summary SHA-256：`900cebd016c4ecf065219b01b5b51bae2388dbbcf0b6e188a850c36cc14355d9`。
- `hz_device_supervision_20261001_r3`：15 项通过；已完成且未报接收错误的阶段必须有
  两侧输出锚点，补上同时删除控制。最终独立审计通过，正常界与义务数不变。
  summary SHA-256：`9065fda798e12b945a5dd28e402d33bba70ef593b801b2ce7709058bce7996f5`。
- `hz_device_supervision_device_regression_20261001_r1`：设备算法回归；
  summary SHA-256：`18f890b0abbf800719f0a218476ff7dd59415866e2da564f04fa3faa7db679e4`。
- `hz_device_supervision_rowwise_regression_20261001_r1`：精确核回归；
  test_outcome SHA-256：`c178d38002ca84505b2ce7e3b816bf6c8de59ced47513e1b943cf4b61617b1b2`。

R1 summary 为 `8693e326d1a64b7ff7a908e69ed887f8d162012bcc3393c1fe0440f0a2348fc1`。
没有覆盖失败、改变数学门、预算、目标或迭代。五个本地目录约 8.70 MB 分配空间，
未删除文件；收尾文件系统余量约 690 GiB，不是项目占用统计。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -B -m scripts.run_hz_device_supervision_controls \
  /data1/Kane/MOE/baseline_runs/hz_device_supervision_20261001_r3 --check
```

## 实机和生产路径仍有的门

现有 owned-process 执行器在 killpg 后使用无 timeout 的 wait，且没有证明 CUDA
驱动释放已完成。它可拒绝晚到成功，但不能承诺任意驱动阻塞时有界返回或显存及时
回收。**物理 CUDA 准入保持关闭**，下一项应是单独的有界回收与未确认清理状态，
再冻结微型实机控制。不得杀他人进程、reset 设备或以空闲显存代替准入。

另一个独立且可在设备繁忙时推进的 G1 接口是实际
[`_guarded_support_query`](../act/back_end/hybridz_tf/tf_mlp.py)：先只考虑 LP 连续松弛，
对相同 guarded HZ 的选中神经元生成双侧目标，独立检查后向外转换有理数界，再让
原生路径在剩余预算内回退。该函数目前缺调用方 request/domain/guard 身份，需要
从真实入口传递，不能由 proposer 自造。当前批量接口只控制到 128 因子、256 约束、
8 目标；不提高上限冒充全尺寸能力，也不能给 fallback 新的一份时间。
这仍是下一步接入要求，本轮没有修改传播或跑真实请求。

G1–G6 均 OPEN；本阶段只补完整执行的一段。旧外部不利比较、历史 23 增益来源缺口、
真实证明失败和全部封存边界不变。只读 AI 审查不是独立人类技术审阅。
