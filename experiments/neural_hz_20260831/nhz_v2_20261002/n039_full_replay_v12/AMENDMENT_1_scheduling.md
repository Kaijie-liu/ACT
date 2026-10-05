# N039 v12 amendment 1 (scheduling only; no code change; FREEZE unchanged)

When worker wA finishes, a worker wD runs `--families cora_2024` with the same frozen runner and
code. Worker wC is stopped by PID after it logs tllverifybench_2023 31 (its last TLL row), so it
never starts cora_2024. Rows of cora_2024 are taken from wD only; each family's rows come from
exactly one worker (wA: cersyve, cgan, metaroom, dist_shift, safenlp, sat_relu, malbeware;
wB: acasxu, relusplitter, vit; wC: linearizenn, tll; wD: cora).
Sat Oct  3 07:56:46 AM AEST 2026 wA finished; starting wD (cora_2024)

Amendment 2 (scheduling only; same frozen runner and code): worker wD finished cora_2024. Workers
wE (`--families vit_2023`) and wF (`--families tllverifybench_2023`) are added. wB is stopped by
PID after it logs relusplitter 219 and wC after linearizenn_2024 59, so neither starts its next
family. Final assignment: wA cersyve, cgan, metaroom, dist_shift, safenlp, sat_relu, malbeware;
wB acasxu, relusplitter; wC linearizenn; wD cora; wE vit; wF tll. Each family's rows come from
exactly that worker.
Sat Oct  3 08:44:46 AM AEST 2026 amendment 2: wE (vit_2023) and wF (tllverifybench_2023) started
