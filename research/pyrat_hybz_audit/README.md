# PyRAT HyBZ audit plan

Observed 2026-05-29: `/data1/Kane/pyrat/results_pure_hybz/vggnet16_2022/results.csv` has 14 verified + 1 falsified on vggnet16 GPU, while ACT baseline has 0/18 (all timeout/resource).

But PyRAT strict is NOT automatically acceptable:
- `run_pure.py` disables PGD/DeepFool/random-sample concrete falsifiers.
- However config uses `split=True` on vggnet16, so may violate No BaB/input split.
- PyRAT emits warning: `HybZonotopes use a solver so it cannot be sound. Sound mode changed to False`.

Audit experiments:
1. `split=False` config on vggnet16 spec0/spec1/spec2: if still verified/falsified, improvement comes from single-pass HyBZ domain, not BaB.
2. `con_z` only vs `con_z+hyb_z`: identify whether hybrid binary part is essential.
3. If single-pass HyBZ works, port the specific representation idea to ACT with sound LP/HiGHS/outward checks.
