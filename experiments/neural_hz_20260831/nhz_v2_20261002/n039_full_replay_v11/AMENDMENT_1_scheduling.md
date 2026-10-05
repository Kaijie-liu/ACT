# N039 v11 amendment 1 (scheduling only)

Scheduling amendment (about 1 hour after launch; no code change, FREEZE unchanged): a
third worker wC (`--families linearizenn_2024,tllverifybench_2023,cora_2024`) was added,
running the same frozen runner and code. Worker wB will be stopped when it reaches
linearizenn_2024, so every row is computed exactly once. Rows of a family are taken only
from the worker assigned to it here (wA: cersyve, cgan, metaroom, dist_shift, safenlp,
sat_relu, malbeware; wB: acasxu, relusplitter, vit; wC: linearizenn, tll, cora).
