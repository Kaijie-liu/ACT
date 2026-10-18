# 端点中点有限控制的冻结候选

2026-10-19；起点 `5e8b2784bbdb6c3403d86d49f1f329829c326f7b`，
`feat/moe-route-verification`，起始工作区干净、远端同步。

**冻结候选已生成并经静态核对，但真实可信调用者未绑定，不能执行。**
这是[外层接口准备](hz_endpoint_observer_preparation_20261019.md)之后的身份冻结工作，
不是控制结果、资源准入或新执行授权。端点中点搜索的实际效果仍未检验。

## 候选身份与不变项

[候选 JSON](hz_endpoint_midpoint_freeze_candidate_20261019.json)的规范字节为 77,475 字节，
SHA-256 为 `bb7bdadda356202f8dec7132782ac6e4e73dd445571094eb6179eac2e3faf107`。
[静态审查记录](hz_endpoint_midpoint_freeze_review_20261019.json)绑定来源、环境、代码、
批准表原值及拟议值；其状态为 `NOT_READY_CALLER_HOST_UNBOUND`。

| 顺序 | 位置 | 来源及作用 |
| --- | --- | --- |
| 1 | components | 原 22 项组件；guarded、mixed、half 固定声明 |
| 2 | guarded_baseline | guarded，原逐列搜索 |
| 3 | guarded_midpoint | 同一 guarded，末次端点中点更新 |
| 4 | mixed_midpoint | mixed，中点更新 |
| 5 | mixed_baseline | 同一 mixed，原逐列搜索 |
| 6 | midpoint_block | guarded，中点接缝阻塞故障 |
| 7 | partial_candidate | guarded，部分候选故障 |

第一位置沿用第三位置的 profile，合计六份唯一配置。两臂仍是 CPU/float64、单线程，
129 次评价、128 次 forward；全部来源、符号、残差、零角色及最终接受检查保留。
每位置仍为 300 秒、8M Work、128 MiB 制品、8 GiB AS 硬限额及 2 GiB 采样 RSS 停止门；
整批为 2100 秒，单次最多 60 秒的准入计入其中。七位置和故障次数不增加，不升限重试。
完整 heap 或物理强实时完成保证不在合同内。

来源身份来自已存归档，对 manifest 与全部 tensor chunks 逐字节复核，未调用来源工厂：

| 来源 | 成员数 | 总字节 |
| --- | ---: | ---: |
| guarded | 10 | 5,198 |
| mixed | 16 | 8,616 |
| half | 16 | 8,073 |

现有 `act-py312` 的六份主要环境文件共 469,974,229 字节已重新核对；版本为
Torch 2.9.1+cu128，但本协议仍为 CPU-only。环境文件身份不是整个 OS 的可信证明。
代码绑定包含 60 个 native、62 个 component、74 个 batch 文件，包括实际调用顺序函数。

## 批准表仅作投影

实际三份批准表均保持原字节。native 仍有 26 份旧 profile，endpoint 批次批准表为空。
审查记录只列出将来另获批准时所需的 26+6=32 份 native profile 投影及其
`da5260bcf34f4fb4fd1c1bb6592402d67e2f8658b59aa000a9006aa352a9858c` hash。
freeze 绑定的是这个未来投影，而不是激活前旧表；批准表不在代码闭包内，因此没有循环 hash。

投影不是许可，也没有写入任何生效表。候选和未来 native 表必须是排序、紧凑、UTF-8、
**无末尾换行**的规范 JSON；只比较解析后对象不够。实际执行入口还会验权，当前必拒绝。
不运行会重新生成来源并修改批准表的旧 `freeze_native_controls.py`。

## 仍缺真实调用者

[绑定的调用顺序](../scripts/endpoint_midpoint_caller.py)只有：调用 `execute()`，
待其真正返回，再调用 `observe_return()`。默认关闭，没有 CLI、循环、写文件或额外计时器。
固定未来目录为 `/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r1`，当前不存在。

但函数不是一个已运行的可信服务。现有 `execute()` 从函数入口计时，不能覆盖新起 Python
解释器及其导入；本次没有确定符合该前提的实际 host 身份。因此 `host_identity=null`、
`execution_ready=false`，不能新起 Python 后把它称为已存在服务，也不能拿 API 代码 hash
替代执行环境及起点的绑定。

下一项最小决策是确定真实调用者及其启动计费边界：若确有可用的预存可信服务，绑定其身份；
否则另行明确把新调用者启动纳入原预算的有限方案。两者均须保持全部计费和原限额。
若涉及执行代码改变，生成新版本冻结，不改写本候选。之后才另取这七位置的执行授权。
此处停止，不继续增加监督层或控制数量，也不自动启动服务或原生计算。

在线接收只指可信 host 的返回后内存观测；外部持久归档单独计费，不能将迟到或缺失接收
补成在线成功。目录文件独自仍只能达到 `CONSISTENT_PENDING_CALLER_OBSERVATION`。
未来运行还需新鲜资源准入；本次磁盘约 293 GiB 可用不是运行时准入。

## 可复核的静态检查

以下命令只读文件、AST 和字节，不导入被测项目运行时，不执行控制或候选：

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_endpoint_freeze_candidate.py
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_endpoint_batch_static.py
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/check_native_integration_static.py
```

首项逐字节重建候选及审查记录，返回 `PASS_STATIC_CANDIDATE_NOT_EXECUTION_READY`；
批次检查为 11 个入口、4 个 bootstrap、74 个依赖；native 为 12 个入口、60 个依赖。
只读 AI 审查核对了 V2 合同与批准表投影，并指出规范 JSON 尾换行不一致问题，已修正。
这些是准备检查，不是原生故障控制通过，也不是独立人类审阅。

检查器有意绑定当前未激活状态；将来合法变更批准表后，它会报告 preparation drift，
不应将该旧准备检查重新用于认定新执行失败。新的执行以其独立冻结和审计为准。

本阶段没有控制、来源工厂、原生候选、真实模型或 GPU 运行，没有新正证书或性能结果。
完整注册来源仍受 Work/membership 门拒绝，G1–G6 保持 OPEN。
