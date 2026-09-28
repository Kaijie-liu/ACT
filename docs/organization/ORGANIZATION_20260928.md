# 工程整理记录（2026-09-28）

## 范围与保留

起始科学版本 `8ab784898d467992cff587c953e84a770b70a5fb`。
用户确认已有 `act/pipeline/log/pipeline_tests.log` 删除是有意的；单独提交并推送
`711dae7a8`，其内容可从前一提交恢复。后续整理以这个干净版本为基线。

本次仅做导航、交接分层和维护控制。没有训练、求解、重开封存输入、修改接受门、
移动代码包/证据/环境、去重数据或清理缓存。没有联系作者或发布评审制品。

## 具体交付

1. 本机 `/data1/Kane/MOE/README.md` 总入口（仓库外，本地保留）。
2. 仓库 [PROJECT_INDEX](../PROJECT_INDEX.md)：按任务找实现、协议、结果、论文和作者基线。
3. [当前交接](../CODEX_HANDOFF.md) 从 4,632 行压缩为 96 行；
   [原始全文](../CODEX_HANDOFF_HISTORY_20260928.md) 字节完全一致，仍在同一个目录，
   不破坏历史相对路径的上下文。SHA-256：
   `9681affe28c4b400c444a77df74255bd37c58ffc3801a10e397ba4bffe9a005a`。
4. [完整目录分类](DIRECTORY_CATALOG.md) 与 [机器可读规则](layout.json)：
   81 个已跟踪顶层目录划为 8 组；工作区全部现有条目也有分类。
   检查器拒绝新出现但未分类的目录，不推断它们可删除。
5. [VS Code 工作区](../../MOE.code-workspace)：主仓库、文档、论文、指导、作者运行五个入口；
   设置仅在打开该工作区时生效，不改全局配置。环境与大型证据保持原位。
6. [导航检查器](../../scripts/check_project_layout.py) 与
   [控制测试](../../scripts/test_project_layout.py)，可用既有 Python 的 `-I -S` 运行。

目录平铺的实际代码路径没有改名；这是有意的兼容策略，而不是声称已经完成代码架构重构。
若以后要做物理迁移，必须单独核对 import、冻结配置、证据引用与运行中的任务。
历史 R5 等评审清单仍是当时的身份快照，不因本次交接整理而覆盖更新；较早的实时
清单检查命令不应冒充对当前版本的身份验证。本次维护回执单独保存。

## 验证

环境：`/data1/Kane/miniconda3/envs/act-py312/bin/python`，未安装依赖。

| 检查 | 结果 |
|---|---|
| `scripts/check_project_layout.py --workspace` | 全部目录已分类、导航有效、619 个冻结实现绑定不变 |
| `scripts/test_project_layout.py` | 9 个控制通过：分类、错误路径、重复 JSON、历史身份、坏链接等 |
| `scripts/summarize_proof_closure.py --check` | 已归档七次真实调用账本未变，仍无完整正证书 |
| `scripts/test_proof_closure.py` | 9 个账本控制通过 |
| `scripts/test_submission_review.py` | 12 个审阅包/搬迁/篡改控制通过 |
| `scripts/test_review_handoff.py` | 4 个原评审身份与链接控制通过 |

共 **34 个控制通过**；不是重跑全工程全部测试，更不是独立重证所有网络结论。
只核查整理所涉及的导航、历史保留、实现绑定、账本与评审包兼容性。
详见 [工作区清单与检查回执](inventory_20260928.json)。

## 后续使用

- 以后从 [工程总入口](../PROJECT_INDEX.md) 或当前交接进入，不从四千行历史里猜下一步。
- 实验结论仍由已有日期化结果与最新裁决提供，本次没有复制一套新“科学状态”。
- 任何删除/迁移候选都需单独确认；本次没有回收存储空间的承诺。
