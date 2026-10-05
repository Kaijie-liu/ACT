# N039 v14 amendment 1 (scheduling only; no code change; FREEZE unchanged)

Worker wA (safenlp, sat_relu, malbeware, cersyve, metaroom, cgan) finished. A worker wE
(`--families tllverifybench_2023,dist_shift_2023,cora_2024`) takes its slot, so four workers keep
running, matching the baseline's four-way concurrency. wC is stopped by PID after it logs
linearizenn_2024 59 and wD after vit_2023 199, so neither starts a family assigned to wE. Final
assignment: wA safenlp, sat_relu, malbeware, cersyve, metaroom, cgan; wB acasxu, relusplitter;
wC linearizenn; wD vit; wE tll, dist_shift, cora. Each family's rows come from exactly one
worker. The E0 replay N102 starts only after all of wB, wC, wD, wE have exited.
Sat Oct  3 02:40:46 PM AEST 2026 amendment 1: wA exited; wE (tllverifybench_2023, dist_shift_2023, cora_2024) started, pid 1435334; concurrency stays at 4
Sat Oct  3 05:04:20 PM AEST 2026 wD stopped after vit_2023 199 (cora and tll rows from wE only)
Sat Oct  3 07:59:06 PM AEST 2026 wC stopped after linearizenn_2024 59 (dist_shift rows from wE only)
