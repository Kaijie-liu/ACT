# 跨层幅值审查结果与稳定源负结论

单次冻结执行通过全部组件与完整存档审查。更重要的研究结论是：廉价独立残差界留下的 267 条未决边全部来自严格稳定激活的父源。若比较查询保留这些源应有的 alpha=1，新增四类行均已被原界蕴含。因此不将这个廉价逐边版本作为该存档块的能力提升路线；下一步研究必须利用额外共同源关系。没有新增正式 CERT 或 ADV。

## 冻结执行证据

2026-09-30，redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。只执行一次 run_reference.py --enabled，结果在 results/d038_descendant_census_20260930_v1。统一执行会话 53164 已返回 exit_code=0，supervisor_exit=0；没有重启、重跑、补测或修改冻结文件。

完整 3759 tests / 169 files 全通过，无 skip/failure/error，13 条既有 warning 保留。collection 与 execution 合计 58.0179413985461 秒，未改变 60 秒门。worker 完成全部 320 接收行、184320 canonical 槽位，其中 102400 真实边和 81920 padding；没有人口删减。每个 ordinary_bounds 与旧存档完全一致。全部旧 source/input drift 均为空，production provenance 未变。

初始统一冗余判据留下 267 条 potential edges、396 条 potential rows、1243 个坐标 nnz，四类行计数为 [99,75,68,154]。这些名称始终代表 UNRESOLVED，不是非冗余或收益。

未决边分布于 64 个接收行。五个原位置依次的边数为 69、50、82、37、29，对应行数为 103、76、108、59、50。已保存完整 320 行的全部 masks，完整证据为 527429 bytes。

## 执行后的只读源分类

在候选结束后，用 jq 将 complete.json 中每个非零 row_mask 按 branch、position、canonical slot 与原 complete_0.json 的 window.source_bounds/original_phases 连接。根据认证 Fraction 下端点的分子严格大于零判定 strict_active，上端点分子严格小于零判定 strict_inactive，严格跨零单列；这只是既有证据的事后描述性审查，不修改初始 kernel、冻结成功条件或 masks，也不重新执行数学候选。

结果：267 条未决边、396 条行全部为 strict_active，涉及 180 个不同原 source phase 身份。crossing、strict_inactive、zero_boundary 的未决边数都是零。分类使用原界的严格符号，没有数值容差或按结果改规则。

对 alpha=1，R1 是 r>=0，R3 是 r-h>=0；R2 是 r>=alo*q+L，由 r>=h、h=a*q+v、a>=alo、q>=0、v>=L 推出；R4 是 r-h>=-ahi*q-H，由 r-h>=-h 及 h<=ahi*q+H 推出。因此这些余下行也冗余。

这是条件明确的负结论：实际生产相位列以及 alpha=1 是否已经绑定尚未检查，不能声称已证明生产 LP 添加这些行必定无效；更不能由此删掉原 bit。它证明的是相对保留已认证稳定相位事实的比较系统，该存档块没有剩余增强。结论不外推到整网、其他层、medium 或 Tiny，也不否定更强的相关残差证书和分组关系。

## 完整实测成本

worker 15.081705855205655 秒；supervisor 全程 92.24650327302516 秒。whole work 106398000 / 256M，branch work 66332464 / 200M，evidence work 30554952 / 40M。含临时 reserve 的 retained numeric entries 为 1215250 / 64M。

worker RSS high-water growth 为 555872256 bytes；tracemalloc peak 为 239792521、tracer metadata 为 121949968 bytes，分别加 65536 reserve 后仍在各自 1GiB 门内。supervisor 的两项门也通过。二者独立测量，没有 aggregate 物理资格。事后的 jq 证据汇总不在这些执行时间内，没有借此发布性能倍数或 GPU 声明。

## 资格边界和下一步

all_stages_passed=true 只涵盖完整组件测试与这个只读存档审查。source_census_qualified、actual_phase_column_binding_verified、native_HZ_admitted、gpu_computation_completed 和 complete_physical_qualification 均为 false。没有模型 forward、LP/MILP、shadow、逐家族或全量回放。正式 baseline 仍为 1870/2413（1063 CERT、807 validated ADV）；独立 CIFAR100 25、TinyImageNet 36，共 61/400，不相加。

本轮 Goal 分类为 progress：完成真实结构负证据，排除了直接铺开廉价逐边规则的依据。不是仅重述计划或等待。Goal 继续 active；不能因本组件成功宣布整体完成。

并行数学研究转向共同源认证的关断组：用同一原相位控制一组激活的零面，保留成员全部原 bits，研究其混合仿射与后续 ReLU 的组合。其纸面结果另存 D039，不借本次组件资格晋级。保持原 GPU、完整回放及零无效见证要求，不放宽任何资源或权限。

前序 D037 三个草稿哈希保持不变，未执行；其新增六测试文件是未执行的静态草稿。新 D038 在冻结后没有源码修改。历史模型、日志、结果和九个 tracked 修改保持原状，没有 commit/push。使用 pages:write-page 将原预注册结果、事后源分类、条件性冗余证明和未获资格分开记录。

## 证据身份

```text
8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73  freeze.json
b1f4c6555e538a52c84c0fd78a918fdc2c65677cf1d515ec9e771b97d8e700cd  results/d038_descendant_census_20260930_v1/exit.json
18d324ee9bed4791c2099a3b0fee4066e4aedafac3506bc539eccf4b91478a31  results/d038_descendant_census_20260930_v1/diagnostic.json
4201138a29b4b33557ba0a3e57df4c15b83bf8b98634c876d11a39b10f9e9c66  results/d038_descendant_census_20260930_v1/complete.json
fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0  results/d025_interval_capacity_20260930_v1/complete_0.json
```
