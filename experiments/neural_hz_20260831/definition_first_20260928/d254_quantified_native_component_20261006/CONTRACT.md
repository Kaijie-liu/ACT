# 原生定量守卫关系的组件合同

本候选实现D252的统一定量守卫公式，默认关闭。它是同一原非凸HZ上的原生谓词关系，不调用第二个验证器、bound generator、attack、split、backward/dual或LP状态救援。完整定义、证明和比较范围以只读D252/THEORY.md、CONTROL.md、CLIP_BOUNDARY.md为依据。

沿用D249的detached native snapshot生命周期：完整旧连续/二元因子、EQ/LE、全部输出、global factor widths、frame和输入decoder原样保留。捕获复制，不修改生产状态。规范化两父x∈[-1,1]，读取同一H上两个完整child的均值z和半差d，固定Gram恢复a，固定差分系数决定tau；完整r/e及所有系数误差不遗漏。

唯一数学变化是总是计算两个不同相位事件的完整z界及四个eta。Relation.event_bounds顺序为((L10,U10),(L01,U01))；tail_errors顺序为((eta10minus,eta10plus),(eta01minus,eta01plus))，其中eta_minus=max(-L-tau,0)，eta_plus=max(U-tau,0)。mixed_bounds仍记录事件界的并集。

保留原六产品和24条完整MC、三条X容量。两条defect统一为：

~~~text
 Y1-Y2-K <= 2*tau*p2+2*max(Ue,0)+eta10minus*alpha1+eta01plus*alpha2,
-Y1+Y2+K <= 2*tau*p1-2*min(Le,0)+eta10plus*alpha1+eta01minus*alpha2.
~~~

旧guard内eta全0，精确有理关系逐行恢复D249。guard外不选择第二路径，不换tau，不增加clip变量或交相位位。未证来源、非零未覆盖门误差、秩、身份、可靠界、舍入、资源或见证失败仍fail closed。旧guard外由新的健全数学合同覆盖，不是事后放松运行门。

每关系仍新增六连续因子、29 LE、零binary；实际nnz按完整source展开计量，不能把96独立坐标上界说成native固定成本。向外舍入沿用只读D086，每个存储系数误差和RHS补偿都验证。普通terminal lowering保留全部因素，必须通过原精确signed转换审计，不新增替代solver路径。

在精确有理关系中，原位整数时六产品唯一。向外舍入后的任意辅助解不保证仍等于精确产品，不能将这些辅助值当成真实条件矩继续传播；每个原H点的canonical真实产品仍须满足全部存储行，decoder只读取保留的原坐标。分数点只检验查询松弛，不是ADV。本候选不是新集合类、完整新域、生产在线安装、GPU或实际模型资格；domain_definition_changed=false。原D249及全部旧冻结实验只读。

保持原共享Budget和sticky resource failure：whole work256M、branch work200M、evidence预付40M、retained64M entries、512-bit有理数。旧snapshot、全局宽度、来源、辅助、证据、终端和反向重构都收费。每条关系零新增行不代表此前被拒关系安装后全网零开销。

组件过门后仍需固定三个完整真实来源、同结构shadow、13家族、全部2413、独立400和原四并发不回退。任何组件正控均不更新1870/2413或61/400，不默认启用。
