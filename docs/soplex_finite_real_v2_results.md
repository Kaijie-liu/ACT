# Finite original-four SoPlex diagnostic — completed, admission-limited

Execution commit `f47a6eed5779578a375b1af160d3fc7ebbc6527f`; 2026-09-20.
Read `soplex_finite_real_v2_summary.json` and `soplex_finite_real_v2_review.json`.
Raw directory: `data/moe/results/soplex_diagnostic_real_20260920_v2`.

## Result

**4/4 executed exactly once, 4 LIMIT at candidate admission, 0 independently
checked feasible U.** All original exact LP input/readback comparisons passed.
All four native processes returned 0 and *reported* optimality. Those reports
are untrusted and no printed objective is promoted to a feasible upper bound.
No job reached packaging or the original-LP feasibility checker.

| Job | Full request s | Nested solve phase s | Solver peak RSS KiB | Oversized serialized coordinates | Maximum integer decimal digits | Terminal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| input220_p0 | 10.781 | 5.788 | 153404 | 92 / 7397 | 13748 | LIMIT |
| input222_p1 | 10.481 | 5.733 | 149656 | 157 / 7682 | 14843 | LIMIT |
| input230_p2 | 44.813 | 37.327 | 308864 | 641 / 9482 | 19290 | LIMIT |
| input232_p0 | 17.812 | 11.236 | 220836 | 362 / 9095 | 18303 | LIMIT |

The receiver stopped at `rational token length` before allocating oversized
integers. A separate **lexical** scan of saved output corroborates serialized
numerator/denominator sizes exceeding the frozen4096-bit admission contract;
it compares decimal strings to the bounded4096-bit maximum, without parsing
the enormous integers or invoking a solver/checker. This is not a claim about
the smallest possible feasible witness, canonical fraction reduction, or the
bit complexity required by the original LP. It does not prove the points valid.

Point files are 2,650,685 / 3,925,701 / 25,270,387 / 8,330,218 bytes, all below
64MiB. Thus the observed stop is **not** the point-file byte cap, native timeout,
8GiB address-space cap, or old Python20M operation count. The stored candidate
format exceeded the registered admission policy. No cap was raised and no
decimal rounding, point repair, alternate basis or second call was attempted.

## Costs and evidence boundary

Total timed request cost83.8868s; controller batch83.9278s; guardian lifecycle
84.8760s (includes resource monitoring/waiting); automatic postterminal audit
4.9237s. These are nested scopes, not additive speedup estimates. All four
resource gates passed on their first poll; V1's earlier600s censored resource
wait remains a separate failed execution identity, not charged to these LPs.

Per-phase load/export/import/readback/solve times are in the review. Capture
stopped exceptionally: completed duration remains null with observed elapsed
cost retained. Package/check/review phases were not entered and remain null,
not zero-cost successes. GNU-time RSS is per native process, not a simultaneous
tree total or address-space measurement. Publication and parent overhead are
charged. No source network/HZ/F0 generation was repeated or included: these are
the original **supplied LP** diagnostics, not end-to-end MoE verification times.

Automatic original-coordinate readback/terminal/cost audit: **PASS,0issues**.
Archival scan rechecked128 saved artifact hashes and preserved all four slots.
It performed0new native queries and0feasibility checks. Therefore “audit passed”
does **not** mean four LP feasible points were independently certified.
Three lexical-reporting controls PASS: exact4096-bit boundary, very long
decimal text without integer parsing, and malformed numeric token rejection.

The detached queue completed without relying on an open tool-launch session.
Seven lifecycle/identity controls passed beforehand, including launch-tree
death, controller SIGKILL with owned orphan cleanup, and no-resume postmortems.
The original worker, receiver, deadlines and mathematical checker are unchanged.
Private tmux's normal end is expected after completion; completion.json is the
terminal evidence, not heartbeat freshness after all work has ended.

## Scientific disposition

This finite study is **closed at the registered limits**. Mature native
generation returned candidate files promptly on these four LPs, but the current
admission contract prevented exact independent feasibility checking. This narrows
the observed execution bottleneck; it does not establish LP infeasibility,
nonpositive exact optima, network UNSAFE, or an intrinsic solver limitation.

Historical checked lower bounds remain unchanged. Without accepted U, the
relative contributions of weak lower-bound proposals and LP relaxation gaps
remain unresolved. Do not increase the bit cap or reopen custom elimination
automatically. Following Advice/dd.md, retain this bounded limitation and return
to complete MoE proof capability and paper/artifact integration. The next research
decision should specify which complete output obligations it would improve;
another internal arithmetic counter reduction is not itself that outcome.
