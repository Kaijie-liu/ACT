# C105 complete-source diagnostic audit

C105 supervisor and worker are terminal, both exit0. Session77691 was polled to
completion, not restarted. Supervisor48.897325s; worker41.732416s inside60s.
All1822 frozen source entries were rechecked against current files with no drift;
the exit record binds preregistration, worker log and full result. Production
provenance is unchanged. This is diagnostic PROGRESS, not benchmark progress.

The same ordinary nonconvex expression(c=16,k=32,h=12) was fully generated twice
from original source, keeping both complete outputs. Unprofiled14.586910s,
profiled27.100323s. Complete source/report/owners/inverses fingerprint matches:
99598220fab3d8174410906c57a1a54b5eb97b49562917b0e5bafee2e3bb2fb4.
Original expression unchanged,19,192,640reachable numeric bytes/2,454,548entries.
Both C41 transient tests pass for each generation and full comparison. Plain
RSSgrowth53,710,848B, trace20,743,802+866,464B; profiled13,680,640B and
20,599,726+939,584B; comparison0 and3,315,743+573,664B, all below1GiB.

The complete profile retained381entries and7,696,260calls. Its61,618,848logical
observation units fit the64M reserved allowance. Full binding cost9,990,048fits
64M. The one256M aggregate reservation prepays two64Mgenerations plus64Mprofile
and64Mcomparison; no per-arm full256M budget, no resets/refunds. Actual generation
report whole20,689,996/branch22,624,860 in each lower64Mcap. These are existing
logical resource rules, not a claim to count every CPU or allocator instruction.

Profile attribution (overlapping inclusive intervals, NOT additive):

| Function | Inclusive seconds | Self seconds | Calls |
| --- | ---: | ---: | ---: |
| complete lift |27.066987|1.466397|1|
| circuit install |18.351198|4.015375|1|
| exact circuit construct |12.423544|7.882244|25|
| exact circuit row emission |3.620239|1.976093|19200|
| owned definition encode |2.492731|0.048834|11906|
| original restricted row |1.578497|1.330543|9600|
| exact quotient fold |1.560239|0.402132|1|

This fixture activates25circuit tiles/16000auxiliaries and is circuit-heavy.
The actual C104 target activates4tiles/7357auxiliaries; its complete circuit
interval is139.469083-132.553413=6.915671s. Even eliminating that whole interval
cannot supply the37.059264s needed before BASE merely to leave45s. Therefore do
not prioritize a circuit micro-optimization based on this fixture's ranking.
The next C106 change targets complete original Conv/CSR/shared-sum defining-row
construction and retains the old encoder/quotient/circuit semantics.

Artifacts: results/c105_source_profile_20260913_v1/{preregistered.json,result.json,
worker.log,exit.json}. Full function/callee facts remain in result.json, not just
the table. Exit SHA5753eacf9d25973e1fea129261bed264bf7a0bb89565a0782051d9ad2a25085c;
result SHAaee8608cdb3a1190a231773e6930b0b5fc256380995019dcc639fc5b367dbb54.
No new numerical candidate qualified by C105;2941 unchanged tests reused only
after the exact C104 source/ABI/provenance chain matched. No target/solver/point,
formal gain0;1870/all13/full2413 and separateE0CIFAR25/Tiny36/full400 unchanged.
The full Neural-HZ/PLDI/generalization goal remains ACTIVE.
