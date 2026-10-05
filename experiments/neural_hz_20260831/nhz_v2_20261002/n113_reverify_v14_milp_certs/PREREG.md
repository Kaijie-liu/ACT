# N113 re-verification of MILP-excluded CERTs (N039 v14 and E0 N109) with path v7.2

Trigger: LOG N111/N112 (false CERT on E0 CIFAR row 164 from a numerically spurious MILP
infeasibility). Every CERT of v14 and N109 whose exclusion came from a MILP stage is re-run with
path v7.2 (terminal v7.11 / v8.5: numerical safety width 1e-5 on every unit row; otherwise the
v14 code), same budgets, same engine n020.1. LP-only CERTs (stage B, our own rigorous
arithmetic) are not affected and are not re-run. Outcomes: a CERT that is reproduced stands as
re-verified; a row that is no longer excluded is withdrawn from the v14 vector. This is a
re-verification of specific claims, not a new single-path replay; the v14 vector is reported
with the withdrawn rows removed and marked.
Workers: wA (safenlp, sat_relu, malbeware, cersyve, metaroom, cgan, dist_shift), wB (acasxu,
relusplitter, linearizenn, tll, cora, vit), E0 (cifar100 rows). Three processes, nothing else.
