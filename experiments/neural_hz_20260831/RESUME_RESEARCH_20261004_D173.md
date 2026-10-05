# Neural-HZ共同差异关系研究恢复入口

完整Goal仍active，定义优先、GPU、CIFAR/Tiny及13家族目标不变；所有保旧、二元/源/decoder、禁helper、只写新隔离档、默认关闭和完整回放限制继续。redu-hz，HEAD f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac；正式1870/2413、独立61/400均零新增。

新入口为 [D173理论](definition_first_20260928/d173_observable_contrast_fiber_20261004/THEORY.md)、[先例边界](definition_first_20260928/d173_observable_contrast_fiber_20261004/PRIOR_AND_BOUNDARIES.md)、[研究记录](definition_first_20260928/d173_observable_contrast_fiber_20261004/RESEARCH_RECORD.md)。旧[D172恢复入口](RESUME_RESEARCH_20261004_D172.md)及其唯一系数诊断保持冻结。

## 新正向证据

对任意完整U，令P=I−11ᵀ/m、t=mean(r)、y=Ur、g=Vq+c。单条repeated-ReLU共同QC的精确投影是

```text
e=y−tU1−UPg/2,
e∈range(UP), eᵀ(UPUᵀ)†e≤||Pg||²/4.
```

保全部原γ、共同q、guards和C=[U;mean]的4(n+1)条active/inactive caps，不保m个隐藏r。完整ReLU图是该集合的子集，不能称整个投影精确。新幅值n+1，旧m；所有消费者必须覆盖，不能只挑性质方向。

固定w、a=Uᵀw、κ=sum(a)可前向发布

```text
wᵀy−κt−(Pa)ᵀg/2≤R||Pa||/2, R≥sup_H||Pg||.
```

不需要近负转置、相反列、LP或优化权重。P用均值归约，不必物化；R必须对整个当前父域认证。

普通三门控制g=(x+s/4+1/4,x−s/4+1/4,x+h/4+1/4)、U=(1,−2,11/10)，源[-1,1]³，得J=y−3s/8−2h/15≤2/3<7/10。因此下一ReLU恒零。

同源分数点γ=(1/2)³、r=(5/8,3/10,5/8)给J57/80，下一1/80。它通过retained-source单门hulls，且通过同C、参考gbar=.25·1、global E59/16的完整双分支矩阵透视松弛及所有caps；有共同lift u=(1/2,7/40,1/2)、v=−u、u+v=0，能量849/400<E。这是强于明确全局预算参照的查询进展，不是integer native假点或真实反例。

## 不可省略的负边界

共同grounded sector等价于 ||P(r−g/2)||²≤||Pg||²/4+m t(μ−t)。mass caps给t≥max(0,μ)，故grounded已蕴含contrast，不能把等权all-pair QC当新增原理。

同C、零reference、pointwise E=||g||²的D156包含于mean+grounded投影；不是现行global-E组件的支配关系。混合相位g=(1,2,−1)、γ110、U=(1,−1,1/2)允许新输出−9/10，但点态D156拒绝，真实−1；完整caps仍通过。保持这个强对照，不只展示正控。

费用尚未过：2m+4(n+1)+2k行；axis k=n,m=2n时10n+4对比旧8n。三门控制16对12行。新颖性、真实CNN/ViT参数收益、native→terminal完整精度和GPU尚无资格。

## 下一问题与执行边界

继续研究普通CNN局部共同变化能否给可付的分组对比预算，以及与原phase双分支如何共享同一载体。先解决共同见证、完整消费者和全费用；别堆all-pairs、别把patch源独立化、别把global改pointwise的收益归给新名字。相反列D020替换只是窄结构支撑，不把主目标改成它。

本轮paper-only，无新实现或数值。最后组件仍D1584032 tests/212 files；D172五块19.423秒只读系数诊断保持原记录，不重试。新候选另行预注册、冻结并依原门晋级。没有helper、模型/solver/GPU运行或后台实验，无commit/push/default改变。

tracked diff SHA256仍29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5。本轮ARCHIVE.sha256与ANCHOR_SOURCE.sha256从仓库根校验，旧档及生产文件未改。
