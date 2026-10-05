# 跨领域综述的证据范围

日期 2026-09-30。分支 redu-hz。commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac。

用户本轮请求借鉴 HZ 以外的研究。配置为只读历史文档、原始论文相关章节核验、已有数学推导比较和隔离综述记录。三位协作者分别研究组合域、约束网络、多神经元及依赖保持；根代理核对引用、范围和最终判断。没有模型读取执行、候选导入、测试收集、求解器、设备初始化、shadow 或正式回放。

目标权威文本为 GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md，SHA256 为 0fd93920413f63ad6b3b65849ea4d857c416c6bf4ab56263c63ff73416ebe80c。目标服务状态仍 active；正文中的早期 paused 服务注记不代表现状。

正式 baseline 为 1870/2413，1063 CERT 加 807 经具体网络验证的 ADV。独立外部 E0 为 CIFAR100 25、TinyImageNet 36，61/400。它们是既有口径，不是本轮新跑或重新认证的结果。formal_gain=0。

历史和冻结证据只读。本轮仅新增当前目录，不修改 production 文件、旧源码、模型、日志、结果和默认配置，不 commit 或 push。工作树原有九个 tracked dirty 文件保留；它们不属于本轮改动。

## 文献核验范围

REVIEW.md 在相应结论旁给出原始论文链接及节或命题。核验的是这些相关部分，不声称逐字通读全部文献或完成系统性文献穷尽。Gulwani 与 Tiwari 的作者 PDF 由协作者读取；根代理的同 URL 抓取超时，另核验作者页面及检索到的原文摘要，不谎报成功抓取全文。

Anderson 文献采用 arXiv 1811.01988 的 37 页五作者版本，§5.2 的单门表述是 Proposition 12，分离是 Proposition 13。此前不同版本的命题编号不混用。

本轮以已保存共同源审计和混合源候选为项目证据；没有从公开标签、实例身份、LP 状态或最终 margin 选择实现规则。报告中的数学等式是纸面说明，不是测试通过声明。

## 草稿保留

D049 的理论、实现、测试与执行器保持为未冻结未执行的草稿。下列哈希记录本轮查阅时的版本，不是执行冻结授权：

```text
a868cb40ac62dd0e86fd7de8e78b9ac8d0254933137936fd53ffbbed0abe4cd3  d048/RESEARCH_AND_PROOFS.md
47783c2c7da97acb1b68fb2fd37981919f543f3c223c351f58680f3def874110  d049/THEORY.md
dfd9402b84a784e93d4f01f6f26d03885a453dcec8e4a563223b58dd39e85a09  d049/CONTROL.md
a57f7edd504fe0432548b18336d3f42415d2fc84e6c4129b8da290ac62f1a85a  d049/mixed_source.py
c88f1fb7b80ff0c41357cf3460e6ff5d073953925ad1ff843686e94720c1ccb1  d049/test_mixed_source.py
```

d048 指同级 d048_source_information_audit_20260930 目录，d049 指同级 d049_mixed_source_envelopes_20260930 目录。没有将静态同行审查视为数学执行过门。
