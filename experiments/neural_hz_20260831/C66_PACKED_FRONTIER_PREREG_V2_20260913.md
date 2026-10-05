# C66 v2: ordinary Mapping member collision repair only

C66 v1 is terminal after7 failed/1533 passed of1540 tests. Its ndarray member
`values` shadowed Mapping.values(), breaking the complete independent source
observer and four recurrence checks. No full-source preflight or real generator
ran. The frozen v1 implementation/tests/result/exit remain unchanged.

This separately versioned candidate renames only the internal storage members
`keys` -> `index` and `values` -> `buffer`, restoring the standard keys()/values()
mapping interface. The tests' internal storage/weakref assertions follow those
new member names. Every prior behavioral assertion and all1540 cases remain;
the ordinary recurrence test additionally exercises dict(mapping) and keys().
The66-file suite retains every62-file C65 prerequisite, with the four C66 test
modules pointed at the corrected implementation. The broken frozen v1 modules
remain immutable negative evidence, not silently edited or omitted failures.

All mathematics, accepted inputs, complete100965 optimum, physical boundaries,
extra1024 work, unchanged prices/caps and every preflight/proof/restore obligation
are EXACTLY C66_PACKED_FRONTIER_PREREG_20260913.md. That full v1 card is a governing
dependency of this v2 supplement. No alternate algorithm, easier target, dropped
check, threshold change or interpretation of the observed failure as success.

Exclusive output results/c66_packed_frontier_20260913_v2. Freeze all v1/C65 sources,
the v1 negative results and new v2 files before tests. Same1540 exact checks,
60s qualification/240s workers/60s restore, CPU1/GPU0,16GiBAS,256M/200M,
64M entries,both1GiB transient,radix16384/131072/16M. All stages/logs/results/
exit/source/artifact hashes auto-retained; any new failure closes v2 unchanged.
No native/LIVE/solver/default/history/score/commit/push action. Formal1870/all13
and separate E0 CIFAR25/Tiny36 unchanged; full goal ACTIVE.
