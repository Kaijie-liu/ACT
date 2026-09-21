# Condensed-paper reference checks (2026-09-21)

Only primary paper/publisher/project sources were used for the condensed
related-work claims. No cited code was installed, no benchmark was run, and
no published speedup was copied into our own empirical tables. These checks
do not establish exhaustive novelty or replace human full-paper reading.

| Key | Verified source | Claim used in the short paper |
|---|---|---|
| hz | [Ortiz, Vellucci, Koeln, Ruths; arXiv:2304.02755](https://arxiv.org/abs/2304.02755) | Hybrid-zonotope representation of ReLU networks predates this work. |
| dnnv | [Shriver, Elbaum, Dwyer; arXiv:2105.12841](https://arxiv.org/abs/2105.12841) | Property reduction and verifier interoperability are established ideas. |
| metamoe | [SAIV proceedings table of contents](https://link.springer.com/book/10.1007/978-3-032-32357-6), chapter DOI 10.1007/978-3-032-32357-6_8 | Authors and pp.167–190 confirmed; compositional MoE verification is related, not newly invented here. Direct chapter retrieval timed out in this pass. No new router metric or full-text claim is inferred from that failure. Proceedings event is SAIV 2026; final venue bibliography year must follow the publisher export. |
| crown | [Zhang et al.; NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html) | A static-network bounding framework; our plain configuration is not its entire tool family's capability. |
| clip | [Zhou et al.; NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ffa977364ab7046c803da0e04dbb2832-Abstract-Conference.html) | Constraint-driven domain/intermediate-bound improvements within BaB. |
| fastcert | [Das et al.; arXiv:2608.19351](https://arxiv.org/abs/2608.19351) | Cross-query intermediate-template reuse, distinct from our within-request property facts. |
| conflicts | [Elsaleh, Davis, Wu, Katz; arXiv:2603.12232](https://arxiv.org/abs/2603.12232) | Query refinement supports learned-conflict reuse, implemented in Marabou. |
| torchlean | [arXiv:2602.22631](https://arxiv.org/abs/2602.22631), [author project](https://leandojo.org/torchlean.html) | Source/execution/verification semantics is a related research problem. Metadata author lists differ across sources/versions; the draft uses George et al., without asserting a full reconciled author list or equivalence to our checker. |

Before venue submission: obtain final version-specific bibliography exports,
read the directly competing papers beyond their abstracts, and review whether
additional classical abstract-interpretation/envelope/certificate references
are needed. The current eight references are a focused review draft, not a
claim of complete submission bibliography.
