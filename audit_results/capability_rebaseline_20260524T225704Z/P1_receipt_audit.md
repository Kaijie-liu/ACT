[ACT] Auto-detecting project root: ../ACT
[ACT] Gurobi license found: ../ACT/modules/gurobi/gurobi.lic
# Receipt audit  run_root=/data1/Kane/ACT/audit_results/capability_rebaseline_20260524T225704Z
# official labels available: 2647 (benchmark,iid) keys

## Receipt audit summary
  total FAL receipts:                       87
  sha all 3 match (model+spec+x*):          87/87
  fresh ORT zero_tol_holds:                 56/87
  verdict consistent (recorded vs fresh):   56/87
  HARD violation (official=unsat AND fresh zero FAL): 0
  per-bench breakdown:
    sat_relu                       49
    safenlp_2024                   20
    acasxu_2023                    15
    tllverifybench_2023            2
    malbeware                      1
  CSV: /data1/Kane/ACT/audit_results/capability_rebaseline_20260524T225704Z/p1_receipt_audit.csv

AUDIT PASS
