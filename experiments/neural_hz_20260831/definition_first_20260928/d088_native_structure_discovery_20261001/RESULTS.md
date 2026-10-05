# 原生关系发现的数学通过与归档入口失败

本版本数学组件门通过，真实归档诊断未通过。两者不是矛盾：归档阶段在加载历史数据之前发现导入依赖身份清单不完整，因而 fail closed；它尚未扫描真实门关系，不能据此判定候选适用或不适用。

## 已执行结果

唯一数学运行位于 results/d088_native_structure_discovery_20261001_v1。3825项测试、183个文件全部通过，没有 failure、error 或 skip。单pytest进程从启动到退出48.437424598261714秒；监督器总59.397102205082774秒。source/input drift均为空，provenance未漂移。数学组件资格为true；真实模型、完整物理、GPU及正式收益资格仍为false。

随后只执行一次显式归档诊断。worker在3.34404269233346秒报告 ValueError，原因是 `unbound project import: /data1/Kane/FSE/ACT/act/__init__.py`。archive_loaded、archive_authentication_completed、archive_census_completed、transformed、all_groups_applied均为false。此时branch工作为0，whole工作78609946；没有读取或转换被选归档，没有调用终端求解器。

外监督记录worker_exit=1、timeout=false，子进程总3.9236146714538336秒，自动保存worker.json、worker.log和supervisor.json。主机观察未越门、source_drift和identities_unchecked均为空，不改变这次入口失败结论。日志中的Gurobi license警告不是此次异常原因。

## 原因和下一动作

已有数学依赖清单包含部分生产模块，但未覆盖Python包初始化文件。下一版本应在导入前冻结完整生产包源库存，保留已有绑定并拒绝冲突、继续预付前后身份检查费用和执行期缺项检查。不能删除身份检查、在缺项导入后追认、缩小测试人口或修改本冻结版本再运行。

新修正版另开D090，结构算法仍复用本版本和D086；它是实际结构入口修复，不是新的域定义贡献。本版本八个冻结源码、freeze.json和结果目录保持原样；本文件只补充事后记录。

2026-10-01，branch redu-hz，commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。正式1870/2413及独立61/400不变，formal_gain=0。文档技能用于分开记录数学组件、真实入口失败与未获资格，不发布外部Page。
