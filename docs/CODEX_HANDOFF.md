# MoE project handoff

LATEST EXECUTION READINESS (2026-09-20): finite SoPlex execution addendum READY.
Read `docs/soplex_execution_v1.md`, execution freeze and readiness review FIRST.
18/18 new controls +21/21 unchanged checker regressions PASS; release review
rechecked10 analytic LPs with relocated python-I-S checker,0issues/0real queries.
Exact candidate receiver, owned-tree 218/298/300 supervision, partial costs,
resource gates and four-slot fail-stop implemented in separate soplex_execution/.
Original scientific freeze/LPs/checker/act-py312 unchanged. Preparation001's
extreme-timeout regression failure and002's namespace inventory error retained;
003 passed. No feasibility overclaim and no changed checker threshold.

User authorized "补齐外层监督、候选输出接收与终态成本审计，再执行".
NEXT AFTER THIS PREPARATION COMMIT: launch `python -m soplex_execution.supervisor`
under act-py312 (new reserved real result directory only),then independent
`python -m soplex_execution.audit docs/soplex_finite_real_v1_review.json`.
Before every job obey original resource gate (currently load/core above0.5),
no retry/new LP/longer budget. Archive all four outcomes, close finite study.
Do NOT restart custom elimination or infer network UNSAFE from LP feasible U.

LATEST PREPARATION (2026-09-20): original-four SoPlex sparse input fidelity is
COMPLETE; finite scientific protocol frozen, **NOT execution-ready/launched**.
Read `docs/soplex_large_import_v1_results.md` FIRST, then import attempt001,
fresh review and `docs/soplex_finite_comparison_v1.md`/freeze/review.
All4 original LPs pass exact rational readback:1,161,639 nonzeros,including67
abs<=1e-9 coefficients, every objective/box/row side unchanged. Offset explicitly
omitted natively and retained in original checking objective. Reader has no
optimize/basis/candidate calls.4 real imports,0 real solves,0 new feasible U.
Preparation31.936s; independent24-artifact/all-coefficient review7.298s,PASS0.

Controls attempt00227/27 PASS (6 new+21 original regressions). Includes11 analytic
read-only imports and4 legacy analytic SciPy solves;0 SoPlex optimization.
Attempt001 retained;002 fixes overly broad zero-call metadata, not outcomes.
Old source/results, installed compatibility probe and act-py312 unchanged.

Finite protocol: same jobs220p0/222p1/230p2/232p0,one full-original-LP exact
SoPlex attempt/job, native basis choice allowed;218/298/300 clock,8GiB AS,
explicit output/bit limits,all original checks,no retry/tuning/expansion.
Status PROTOCOL_FROZEN_NOT_EXECUTED,execution_ready=false,output absent.
NEXT: implement/validate the narrow external-path execution addendum (rational
CLI-output admission, unified deadlines/resources, partial terminals and costs),
bind its hashes,then separately authorize launch. Do not mislabel this protocol
freeze as a tested supervisor. Do not resume custom arithmetic development.
Historical4LIMIT/zero checked feasible U unchanged; writing must not wait for U.

LATEST PRIORITY / COMPLETION (2026-09-20): follow PI's `Advice/dd.md`, not the
historical arithmetic-to-supervisor NEXT chain below. Pause custom arithmetic
expansion. User approved isolated SoPlex installation and exact-I/O controls.
Read `docs/soplex_compat_v1.md` FIRST, controls attempt001, fresh review attempt001
and installation inventory. SoPlex8.0.3 pinned13e2ab2467e0 installed outside ACT
at `/data1/Kane/MOE/envs/soplex-8.0.3/bin`; act-py312 unchanged, no package install
there. First static-link failure retained, flags repaired without upstream edits.

10/10 compatibility tests PASS with10 analytic LP queries:9 exactly feasible
points checked by relocated python-I-S original-LP checker,1 no-point unresolved.
Rational model readbacks before/after agree, including tiny signed coefficients.
Fresh saved-evidence review111 artifacts/10 isolated checks,PASS0issues;
mutation and deadline rejection pass. Unchanged checker21/21 regressions PASS.
Native status is not proof; no dual/optimality/full-network guarantee claimed.
This is LP syntax capability, not general MPS compatibility or a real-study
supervisor. Real queries0; original four remain4LIMIT/zero checked feasible U.

NEXT: prepare a FINITE mature-generator compatibility/freeze for the SAME four
original LPs, preserving exact coefficients and final independent LP checking.
SoPlex full-LP basis choice is different from a fixed-basis arithmetic comparison.
First bind large sparse export/readback and wall-time/memory/output/check cost;
do not compare external tools by Python visit counts. No launch registered by
this control stage, no extra samples/time, no automatic amortized supervisor.
SPEX remains optional/uninstalled. Refocus paper on complete MoE proofs, method
effects and explicit trusted base; unresolved arithmetic must not block writing.

Latest research (2026-09-20): immutable source/plan validation amortization is
COMPLETE **AT CONTROL SCOPE ONLY**. Read
`docs/amortized_basis_v1_results.md` FIRST, then protocol, controls attempt002
and fresh review attempt001. Separate `amortized_basis/`: **106/106 PASS
(20 new+86 regressions)**; fresh review341 artifacts/92 exact saved systems/
28 new-mode differentials,13 unresolved retained,0issues. Fourteen moved
`python -I -S` original-LP checks:11 feasible,3 expected rejections (including
regression fixtures). New four-mode analytic LP retains all acceptance checks.

Four fixed modes share owned immutable tuple/integer source data, full admission,
numerical work and per-use source/scope/owner/generation guards. Only repeated
source binding and plan validation are amortized; external imports always get
full checks. Caller/export mutations cannot alter owned tuples. Late admission
deadline rolls back the receipt, not cost. Plan invalidation remains charged;
modular/exact residuals and original-LP checks are never bypassed.

Sparse1024 control operations: repeated258718,source-only248482,plan-only244386,
both234150. **24568 operations saved(9.496%)**, exactly10236 source+14332 plan;
numerical/residual operation counts unchanged. NOT a matched real speedup or
evidence that plan reuse beats no reuse: old V1 no-reuse reference212648 remains
lower, with a different admission contract. No real/native queries or expansion.

Common new-arm repair: V1 dynamic peak recorded plan storage but its live-cap
comparison omitted it; analytic cap3 returned with peak7. New inclusive check
LIMITs at4. Frozen code/results unchanged; no mathematical false-certificate claim.

NEXT: control stage closed; keep optional. Any outer-supervisor integration,
durable partial-evidence accounting and same-version no-plan comparison are
separate work, NOT DONE. Do not expand real diagnostics, reset/increase budgets
or remove checks. Original real result remains4LIMIT/zero checked feasible U.

Latest research (2026-09-20): separate cross-prime symbolic plan reuse and
diagnostic instrumentation are COMPLETE **AT CONTROL SCOPE ONLY**. Read
`docs/plan_basis_v1_results.md` FIRST, then protocol, controls attempt004 and
fresh review attempt002. `plan_basis/` has **86/86 tests PASS (22 new+64
regressions)**; fresh review228 artifacts/60 exact-system residuals/49 new-arm
saved differentials,11 unresolved retained,0issues. Nine moved `python -I -S`
original-LP checks: seven feasible,two expected rejection. One new LP uses
three primes/two plan replays before its independently checked feasible U.

Reuse is symbolic only; all modular values are recomputed. Exact source/RHS/
scope identity and immutable schema are checked; corruption rejects. Modular
cancellation or zero planned pivots invalidates a plan and pays for dynamic
factorization of the same prime, without resetting shared budgets. Old+new
plan storage during fallback is charged. Both arms check modular residuals;
full rational residuals and unchanged original-LP acceptance remain mandatory.
New map/elimination/back-substitution counters and reconstruction first-failure/
successful-prefix records do not alter historical missing diagnostics.

**No net operation improvement in the registered sparse1024 control:** reuse
249499 vs no-reuse212648 (+17.33%), same solution/three rounds. Repeated binding
and validation cost outweigh avoided schedule work. Timings are descriptive
analytic controls, not a real paired speedup. Do not promote this to default
or rerun the four real LPs on the strength of correctness tests alone.

NEXT: separately investigate amortizing immutable per-request source/plan
validation while retaining checks, if commissioned. Later integration into a
new unified-budget supervisor and any real diagnostic freeze remain separate,
NOT DONE. No cap increase, production hook or new native/real-LP calls here.
Prior real outcome remains4LIMIT/zero checked feasible U. Earlier source-bound
control receipts remain historical; current review binds attempt004.

Latest execution (2026-09-20): the user authorized the four frozen modular
diagnostics; execution and archival review are COMPLETE. Read
`docs/modular_diagnostic_v1_execution_results.md` FIRST, then its JSON archive,
`docs/modular_diagnostic_v1_execution_review.json` and review controls.
Executed once at clean pushed HEAD `ab4a70728241c9d34c6a813b909e50064274d20c`.
**4/4 LIMIT, zero complete original-LP checks, zero checked feasible U.**
Fresh structural review: 1980 artifacts, PASS/0 issues; four post-run control
tests PASS. No extra solve, retry, alternate basis, cap/time increase or changed
acceptance rule. Frozen readiness records below remain historical, not current
instructions to repeat integration or launch.

All four basis structures and assembled-system hashes match the previous
primitive run. Every stop is the shared 20,000,000-operation cap (observed
20,000,001), before any complete reconstructed vector. CRT merged80/77/49/49
primes, with modulus2400/2310/1470/1470bits; recorded field products <=60bits.
No4096-bit cap or request timeout was reached. This removes the observed old
large-row-product stop in this execution, but does NOT produce a feasible point
or establish solution bit size, LP infeasibility or network UNSAFE. Zero exact
residual rejections means no complete candidate reached that check.

Provided-LP publication clocks total95.460309s; batch141.006109s includes
post-terminal audits45.543369s. Preflight3.926427s, final-summary audit45.409580s,
archive47.829198s and fresh review48.022604s are separately recorded. Unreached
package/check costs remain null, not zero. Historical network/HZ/F0 generation
was not rerun or charged; these are not complete MoE request timings.

NEXT: frozen execution scope is closed. Use saved journals to characterize
repeated finite-field work and incomplete reconstruction before proposing a
separate arithmetic change. No automatic rerun, expansion, increased caps or
claim that all128 primes would succeed. No production acceptance change.

Latest readiness (2026-09-20): unified-budget modular supervision and a
separate real-diagnostic freeze are COMPLETE, **NOT EXECUTED**. Read
`docs/modular_diagnostic_v1_readiness.md` FIRST, then
`docs/modular_supervised_v1_results.md` and linked control/review/freeze records.
The user authorized integration/controls/freeze, not the real launch this turn.

`modular_supervised/`:107/107 controls PASS (36 integration+7 native+64
arithmetic regressions). Fresh review2624 artifacts/27 terminals/six moved
python -I -S checks,PASS0issues; five feasible,one expected rejection.
Cyclic prime/field/CRT/reconstruction/exact-residual journals preserve bounded
partial evidence, not proof. Full three-prime instrumentation differential
matches the frozen arithmetic. Original218/298/300 clocks,4096 bits,128 primes,
one native<=10s/one basis unchanged. Only journal capacity is2048 bounded events
to accommodate cycles; every write is charged. Missing/censored costs null.

`modular_diagnostic/`:15/15 batch/readiness controls PASS; fresh review2166
artifacts/three ledgers/12 rows/four moved checks,PASS0issues. Freeze and
selection review bind unchanged original jobs220p0,222p1,230p2,232p0, runtime,
LP/source/statement/property identities and all new execution/audit sources.
Read-only static size/import compatibility passes; new modular growth and
real efficacy remain UNMEASURED. Output
`data/moe/results/modular_diagnostic_real_20260920_v1` is absent.

NEXT: only after explicit execution authorization, launch the frozen four
diagnostics once via `modular_diagnostic.run launch --execute-frozen`, then
independently archive and report all outcomes/costs. No new design guidance is
needed for this frozen scope. No retry/resume, alternate basis, cap/time
increase, threshold change or implied network verdict. Prior primitive REAL
results remain4LIMIT/zero checked feasible U. Older "next integrate/freeze"
notes below are now historical; they are not new work to repeat.

Latest arithmetic research (2026-09-20): user requested a scheme avoiding large
intermediate products, **controls first**. Read
`docs/modular_basis_v1_results.md` FIRST, then its control receipt and fresh
review. Separate `modular_basis/` implements bounded small-prime sparse solving,
CRT and rational reconstruction gated by exact original-equation residuals.
**64/64 controls PASS (18 new +46 regressions); fresh review PASS, 0 issues**:
30 successful systems, eight unresolved retained, five relocated `python -I -S`
original-LP checks (four feasible and one expected rejection), 127 artifacts.

Analytic positive: primitive cross product4201bits fails unchanged4096 cap;
new method recovers(1,1) using one prime and <=59-bit field products, then full
original LP checker accepts feasible U=-1. This is NOT real-LP efficacy or a
network proof. Intrinsic-large-answer control still LIMITs during a trial exact
residual at4167bits (round69); bounded reconstruction is deliberately incomplete.
Real reconstructions/native queries0; sealed source/results unchanged.

NEXT: separately integrate this candidate-only interface into a unified-budget
outer supervisor; test deadlines, partial evidence and full costs before any
new real diagnostic freeze. Do not rerun the four real LPs, increase4096 caps,
claim infeasibility/UNSAFE, or treat modular candidates as feasible. No production
hook in this stage. Earlier primitive execution below remains the latest REAL
result; its "next arithmetic question" is now addressed only by analytic controls.

Latest execution (2026-09-20): user explicitly authorized the four frozen
primitive diagnostics. Executed once at clean pushed HEAD
`17a7976ad90daa96960ec372472cb32938f99a9d`. Read
`docs/primitive_diagnostic_v1_execution_results.md/json` FIRST, then fresh
execution review. **4/4 LIMIT, zero original-LP checks/feasible U.** No retry,
cap/time increase, new sample, algorithm fallback or frozen-source modification.

All four new basis structures and assembled-system hashes match old V2.
Row clearing completes with entry integer maxima144/139/143/154bits. Every stop
is elimination row_product, at4112/4097/4113/4116bits against4096, before
subtraction/content normalization. Recorded pivots3760/3050/4939/4081, fill0,
no time/operations/live-entry exhaustion. The raw-product gate failed; final
solution size and later cancellation remain unmeasured. Native negatives are
untrusted, not checked U, LP infeasibility or network UNSAFE.

Supplied-LP clocks13.489612/12.660985/18.821466/18.017880s, total62.989942s.
All four costs/journals complete for their LIMIT outcome, but no point/proof.
Package/check durations null. Batch105.226850s includes requests plus
post-terminal audits42.185560s; final summary42.377780s and archival work
separately disclosed. Archive211 raw hashes; fresh record reconstruction and
saved-journal derivation do not independently certify elimination arithmetic.

NEXT decision: this study is closed. Primitive rows show no endpoint gain here;
do NOT rerun unchanged or extend caps/time. A separately scoped candidate-
construction approach avoiding the observed large intermediate products is the
next arithmetic research question, with analytic controls and the unchanged
full original-LP checker. No new algorithm or real diagnostic is authorized by
this archival stage. Do not infer intrinsic LP-relaxation failure or model
unsafety without checked evidence. Earlier readiness text below is historical.

Latest preparation (2026-09-20): user requested comprehensive readiness, not
execution. Read `docs/primitive_diagnostic_v1.md` FIRST. New separate batch
namespace `primitive_diagnostic/`; frozen `primitive_supervised/` untouched.
Same four original LPs now independently frozen and selection-reviewed for
primitive arithmetic: status FROZEN_NOT_EXECUTED, real output directory absent.
NO real solve/reconstruction occurred. Do not confuse readiness with efficacy.

15/15 preparation controls PASS (11 new +4 archive regressions), on top of the
sealed101-test integration. Fresh review2086 artifacts,3 batch ledgers/12 rows,
4 relocated python -I -S checks,PASS0issues. Attempt001's wrong analytic offset
expectation is retained; only assertion corrected to exact U=i-1/3, no changed
inputs or acceptance. Read controls attempt002 and fresh controls review.

Freeze `docs/primitive_diagnostic_v1_freeze.json` binds original jobs220p0,
222p1,230p2,232p0, all sources, runtime, controls/reviews and old V2 archive.
Read-only size/import compatibility passes; new fill/bit growth unmeasured.
Output `data/moe/results/primitive_diagnostic_real_20260920_v1` must be absent.
One basis/native attempt<=10s, original218/298/300 clocks and4096-bit cap;
no retry/resume/fallback. ERROR stops and retains later unstarted rows.
Costs keep request/attempt/batch nesting and partial durations null. Archive
compares basis geometry/system hash before any arithmetic attribution; no
matched timing claim. Native objective remains untrusted, no network verdict.

NEXT: readiness is complete; actual launch is separate `primitive_diagnostic.run
launch --execute-frozen` after clean pushed commit, then fresh archive process
and documentation/commit/push. This turn deliberately did NOT launch. No new
research guidance required within this four-LP freeze; ask before extending
scope/caps or changing algorithm in response to results. Historical V2 remains
4LIMIT/zero checked feasible U; primitive real efficacy remains unknown.

Latest completion (2026-09-20): separate primitive-integer supervision integrated
under user's deadline/partial-evidence/full-cost instruction. Read
`docs/primitive_supervised_v1.md`, controls attempt001 and fresh V1 review FIRST.
101/101 controls PASS (32 integration +7 native fidelity +62 regressions).
Fresh review1677 artifacts/23 terminals/all costs, five moved python -I -S
checks (4 feasible,1 correctly rejected),PASS0issues. No real LP reconstructions.

New optional namespace `primitive_supervised/`, frozen primitive arithmetic and
native fidelity V2 reused unchanged. Original clock218/298/300, one native<=10s,
one basis/attempt, all structural and4096-bit caps unchanged. Bounded identity
journals retain phase/operation/bit-limit progress; partials are NOT proofs.
Constructor subphase and serialization costs nested in whole supplied-LP clock;
missing/censored durations null, no double counting. No full-MoE timing claim.

Cutoff, row-clear/elimination/backsub stalls, serialization failures, native
partial returns, checker deadlines, mutations and exact instrumentation
differential pass. A completed check may reject feasibility; negative LP
objectives are not network witnesses. Historical sources/results untouched.

NEXT: integration gate passed; consider separately freezing same four original
LPs with this new identity, fresh directory, unchanged caps and once-only roster.
This stage DOES NOT freeze or launch that real diagnostic, retry V2, add a
portfolio, change native/SAFE acceptance, expand samples or reopen holdouts.
Real efficacy and intermediate-swell versus necessary solution size stay open.

Latest completion (2026-09-20): separate bit-growth-control component researched
under user's explicit instruction. Read `docs/primitive_basis_v1.md`, controls
attempt002 and fresh review FIRST. `primitive_basis/` uses denominator clearing,
whole-row (including RHS) gcd normalization, cross-gcd integer elimination and
late exact back-substitution. Same structural pivot rule/caps and original LP;
4096-bit gate unchanged, raw integer products checked before cancellation.
No frozen implementation modified, no real LP reconstruction/native rerun.

62/62 controls PASS (16 new +46 regressions); 24 rational differential systems.
Fresh review114 artifacts,26 successful systems checked against original exact
equations,5 unresolved controls retained,5 relocated python -I -S LP checks
(4 feasible,1 correctly rejected),PASS0issues. Initial61-test pass retained;
first review's execution-metadata comparison error retained and repaired with
an explicit flags-plus-mathematical-fields regression, no tolerance changes.

Positive control: old method LIMIT at4096; primitive rows solve(1,1), max integer
3002bits, original LP checker confirms U=-1. Negative control: old method solves
zero, new LCM hits5036bits and stops. Intrinsic4201-bit-answer control still LIMIT.
Thus this is an optional candidate, NOT a universally better/default solver.
Synthetic4096-row control completes; not a real LP efficiency result. Constructor
remains CANDIDATE_ONLY; independent full original LP checking remains mandatory.

NEXT: separately integrate optional arithmetic into original-clock supervision,
test cutoff/partial/error/cost behavior, then consider a fresh bounded real
protocol. No V2 retry, cap increase, automatic algorithm portfolio or changed
native/SAFE acceptance is authorized by this completed component study. Real
efficacy and intermediate-swell versus exact-solution-size remain open.

Latest completion (2026-09-20): V2 four real diagnostics executed once at clean
pushed HEAD `6556b14ace261c2488bb404a30f5730bcfd266ea` and archived. Read
`docs/fidelity_supervised_real_v2_execution_results.md/json` FIRST, plus fresh
archive review. **4/4 LIMIT (rational bit budget), 0 checked feasible U.** All
four kOk imports and complete before/after intended binary64 readbacks match;
all four basis hints map. No retries, extra time, input or policy changes.

Original LP constants max bits <=154, but elimination exceeds frozen4096-bit
cap after3760/3051/4939/4081 pivots. Fill insertions0 each; live entries<=59773;
no wall-clock/native cap exhaustion. Thus the previous import blocker is closed
for these submissions; current arithmetic growth stops exact reconstruction.
Intermediate swell versus necessary exact solution size remains unseparated.
Native Optimal/negative objectives are UNTRUSTED, not checked U or network UNSAFE.

Whole supplied-LP clocks12.910765/12.269820/18.183476/17.706483s;total61.070543s.
All phases/serialization/in-request review charged. Package/check not reached,
missing duration null. Native calls4;total timing and separate resource/preflight/
post-terminal/final audits retained. Not full-network timing. Archive121 hashes,
original source-bit/basis inventory and nine controls; archival PASS is not proof.
The earlier frozen V1 ERROR+3 unstarted and all failed control attempts survive.

NEXT: no rerun of V2. A separate exact-arithmetic development contract is needed
to study denominator-cleared/fraction-free or checked modular reconstruction,
starting with controls and visible bit-growth failure metadata. Do not simply
raise cap or accept approximate feasibility. Keep original LP and checker,
no new samples/training/CROWN search. New arithmetic may still fail or reveal
an inexact basis. This evidence updates the next question; no further real
diagnostic is registered by the completed V2 freeze.

Latest stage (2026-09-20): separately authorized native fidelity V2 implemented.
Read `docs/fidelity_supervised_v2.md`, controls attempt003 and fresh V2 review.
158/158 controls PASS; 973-artifact/16-terminal review, five relocated isolated
checks PASS, zero issues. Eight separate archive controls pass. Failed attempts
001 (stale mapping schema) and 002 (fault harness import + legacy 20-ms startup
race) remain retained. No frozen V1 source or result was changed.

V2 fixes supported small_matrix_value=1e-12, rejects entries at/below that floor,
requires kOk and full intended binary64 model readback, saves native logs and
import records. No warning bypass, coefficient pruning or scaling. Original
rational LP reconstruction/checking remains mandatory; float import fidelity
is NOT rational model equivalence. One basis/native attempt, native<=10s,
proposal218/check298/publication300 seconds and sparse caps remain unchanged.

Same four original LP diagnostics are separately frozen under
`docs/fidelity_supervised_real_v2_freeze.json` and selection review. Fresh output:
`data/moe/results/fidelity_supervised_real_20260920_v2`. No real optimization at
this stage. Current user authorizes continuation within a 20-hour work window,
not expanded per-request caps. NEXT after clean pushed freeze: launch once,
fresh audit, archive all four outcomes/costs and commit/push results. No retry,
new input, alternate basis or acceptance change after seeing results.

Earlier guidance asking for authorization is superseded for this bounded V2
continuation only. Historical freezes and failures below remain sealed.

Latest completion (2026-09-20): first TWO native-warning follow-ups completed.
Read `docs/native_import_analysis_v1.md`, attempt002 and review FIRST.
Read-only input220 submission inventory +8 minimal analytic import controls,
5 unit controls PASS; fresh16-artifact/8-case/2-exact-witness review PASS,0 issues.
Attempt0017-case pass retained.15 analytic imports total;0 real imports or
optimizations. Frozen sources/options/results remain unchanged.

Saved real matrix246558 entries contains16 entries at−3.6294188569593725e−10,
all A row2(zero-based),columns0–3,32–35,64–67,96–99. Runtime default
small_matrix_value=1e−9. No invalid CSR/duplicates/inverted boxes/large values.
Saved rational→binary64 conversion reproduces;12084 matrix rounding changes
are distinct from the16 small entries and do not justify native deletion.

Analytic passModel/readback confirms magnitude<=1e−9 is dropped with kWarning,
even presolveOFF;nextafter above threshold retained,kOk. Other fields unchanged
in controls. Positive/negative tiny-coefficient witnesses show deletion can add
OR remove feasible points; not generally equivalent or a safe outer relaxation.
Original real import was NOT repeated: full real readback/all warning causes
still unavailable. No native basis/LP efficacy conclusion.

NEXT decision: separately authorize fidelity-preserving native-import V2 with
explicit supported small-entry policy and full readback controls; if needed,
equivalent scaling is a separate design. No silent warning acceptance, V1 option
edit or real rerun. A filtered-model-as-untrusted-hint alternative needs a new
identity/mapping contract. Scientific direction is clear; interface change and
fresh experiment authorization are needed, not samples/time/training changes.

Latest completion (2026-09-20): user authorized the four frozen sparse diagnostics.
Batch launched ONCE at clean pushed HEAD23dc688f9; stopped per frozen ERROR rule.
Read `docs/sparse_supervised_real_v1_execution_results.md/json` FIRST.
**1 ERROR +3 NOT_RUN_AFTER_ERROR, denominator4;0 completed checks/checked U.**
Not four successful diagnostics. No retries, new inputs, time or policy changes.

Input220/p0 loads, then `ok(h.passModel(model))` raises native API status
HighsStatus.kWarning. This precedes h.run(), model readback, raw native return,
basis map and exact construction. prepared.json/native input survive. Detailed
warning cause unknown (output_flag=False); don't assume harmlessness or pin it
on tiny coefficients without further evidence. Native count/time remain null
in frozen accounting; traceback+source explain why optimization was not reached.
Other3 jobs unstarted, not omitted or charged as completed zero-time requests.

Input220 publication2.393508s=load0.414452+capture1.516251+residual0.462805.
Separate preflight2.622400s,resource0.000091s,post-audit0.312447s,final-audit0.791455s.
Machine archive/fresh saved-record reconstruction PASS; execution remains
AUDITED_WITH_ERRORS.4 archive controls pass. No real basis efficacy or LP/model
safety conclusion. Historical inexact-primal results and all freezes unchanged.

NEXT: do NOT launch/resume this freeze again. Separately investigate native model
import warning and coefficient/constraint readback semantics before designing
any new version. No warning bypass, cap change or manual continuation of skipped
jobs. The launch instruction below is historical and superseded by this result.

Latest completion (2026-09-19): LARGE SPARSE SINGLE-BUDGET SUPERVISION integrated
separately in `sparse_supervised/`. Read `docs/sparse_supervised_v1.md`, controls
attempt002 and review FIRST.122/122 PASS(27 new+95 regressions);fresh-process
854-artifact/15-terminal-cost review plus4 moved isolated checks PASS,0 issues.
Old interfaces, caps, sealed results and production MoE acceptance unchanged.

One original300s supplied-LP clock: load/capture/map/sparse construct by218s,
package/check by298s,publication300s; native still one<=10s call. New plan/capture/
construction schema+policy bound. Owned cutoffs, LIMIT, errors, late publication,
mutation and denominator controls pass. Valid `raw_native.json` still records
one call+duration if post-native readback fails/is cut off; missing/malformed
returns remain null. All phases/serialization/cleanup charged; component times
nested, not added again. Historical network/HZ/F0 generation excluded explicitly.

Full supervised synthetic4096-x/8192-E system checksU=−4096/3 in2.6143s in this
control; not a real LP timing forecast. Float-collapsed distinct rational E
control completes checker but remains NOT_EXACTLY_FEASIBLE,upper=null.
CHECKED_LP_DIAGNOSTIC is not network SAFE or necessarily a feasible-point success.

SEPARATE REAL FREEZE: `docs/sparse_supervised_real_v1_freeze.json`, selection
review. FROZEN_NOT_EXECUTED; same4 unchanged source obligations input220/p0,
222/p1,230/p2,232/p0, unchanged LP bytes. All static shapes fit new caps; real
basis mapping/rank/fill/bit growth/runtime unmeasured. No real native call or
reconstruction, no output directory, no rerun of old diagnostics in place.
New output reserved: `data/moe/results/sparse_supervised_real_20260919_v1`.

NEXT: only after an execution decision, clean pushed feature branch and resource
gate, run `act-py312 python -m sparse_supervised.study launch`;then fresh audit
and archive all4 denominators. No retries/alternate basis/cap increases or new
property selection. Prior L is context only; new bundle has no dual/gap/optimality
claim. A checked feasible U<=0 constrains this LP, not the original network.
Old four inexact outcomes remain sealed. Do not interpret the freeze as execution.

Latest completion (2026-09-19): LARGE SPARSE ORIGINAL BASIS component developed
separately under explicit user authorization. Read `docs/sparse_basis_v1.md`,
attempt004 and review FIRST.95/95 PASS(16 new+79 regressions),4 new native
analytic captures,0 real-network LP calls/reconstructions. Fresh-process review
36 artifacts/4 mappings/5 isolated original-LP checks PASS,0 issues. Attempt001
conversion-wrapper failure preserved;002/003/004 passes preserved. Old sources,
caps, frozen results and production runner unchanged.

New `sparse_basis/`: column incidence + minimum-degree heap sparse elimination,
singleton-column fast path, bounded fill/heap/bit/operation growth. Explicit
E_residual coordinates allow native basic E rows WITHOUT dropping equalities:
Ex+e=h requires e=0; Ax+s=b requires s>=0. Generator remains CANDIDATE_ONLY;
unchanged full original-LP checker enforces every constraint. Float-collapsed
distinct rational RHS control correctly stays NOT_EXACTLY_FEASIBLE,upper=null.

Native4096-variable/8192-E-row control maps4096 basic E residuals and checks
U=−4096/3. Synthetic9500-variable/6500-row/356015-entry triangular control
checks U=−6500/3 after relocation;6500 singleton pivots,zero fill,1179547 events.
These are structured synthetic controls, NOT real LP efficacy/timing evidence.
New component caps16k variables/rows,1M input entries,2M active+pivot entries,
1M fill,200k heap,4096 bits,20M work events; native still one<=10s call, fixed
highspy1.14.0 options. All old64-size contracts remain frozen.

NEXT: separately bind this new schema/policy/raw-capture layout into a NEW
single-budget outer supervisor and terminal/cost audit; control partial capture,
native/exact/check cutoffs before any real freeze. Existing basis_supervised
still uses the old small interface. Do not treat component controls as production
integration, real-LP basis compatibility, or permission to relabel old runs.

Latest completion (2026-09-19): REAL LP COMPATIBILITY READ-ONLY REVIEW completed.
Read `docs/basis_compatibility_v1.md` / JSON / audit FIRST. All30 archived
nonpositive weighted obligations fail current limits; fresh-process original
file/dimension reconstruction PASS,0 issues;6 analytic compatibility controls
PASS. No native calls, reconstruction, rank/bound checks, cap changes or new
selection. Decision **NOT_FROZEN_INCOMPATIBLE_CURRENT_LIMITS**; no real diagnostic
frozen/launched. This supersedes the pending compatibility check below.

Variables7397–9482 vs64; augmented E+A rows4331–6416 vs64; stored CSR entries
236502–348915 vs8192. Even initial scalar visits264165–383778 exceed200000;
full-rank pivot storage minimum4331–6416 exceeds4096. These are necessary static
limits, not actual fill-in/time measurements. Bit growth/rank not inferred.

Four old native records have no bound row/column basis statuses, version/options
or submitted/readback snapshots; mapping UNDETERMINED_NO_BOUND_BASIS_CAPTURE.
Do not infer unsupported E-row status or a valid basis from solver success.
Old scipy-highs metadata is not the new fixed native adapter protocol. All four
sealed inexact-primal outcomes remain unchanged; no network safety conclusion.

Next needs a separately scoped large-sparse original-basis/anchor design and
mapping/degeneracy/operation/fill-in controls, not simply raising caps or adding
time. Meaningful real freeze remains blocked pending that new research decision.

Latest completion (2026-09-19): SINGLE-BUDGET BASIS EVIDENCE SUPERVISION added
in `basis_supervised/`. Read `docs/basis_supervised_v1.md`, attempt002 and
`docs/basis_supervised_v1_review.json` FIRST.73/73 PASS (19 new+54 prior);
fresh-process10-terminal/cost review and moved isolated analytic recheck PASS,
0 issues. Attempt00171/71 preserved. No old modules/freezes changed, no real LP
calls/reconstructions, no new experiment freeze. This supersedes the missing
outer-supervision prerequisite below, NOT the small-size/mapping restrictions.

One original300s supplied-LP clock: load→native capture→basis map→exact construct
by218s; pack and isolated full original-LP check by298s; publication300s.
Native remains one call,max10s,highspy1.14.0 fixed options. Owned process cleanup,
no retry/overwrite; raw capture before mapping; unsupported mapping stops intact.
Missing timings/native counts null, interrupted windows censored, native and
construction nested not double charged; late publication revokes acceptance.
ERROR stops ordered roster without dropping later denominators.

Complete analytic path checks(1/3,2/3),U=−4/3,not network UNSAFE. New controls
include4 completed native analytic captures; native/exact stalls are explicitly
synthetic faults, not measured optimization difficulty. Full outer TIMEOUT/error
and partial evidence costs audited; no formal network or performance claim.

Next requires a distinct real-compatibility/freeze decision, not automatic launch:
64-variable/64-equation and sparse/bit caps stay; basic E-row maps remain
unsupported. Do not raise caps, repair sealed points, or rerun the four frozen
real diagnostics. Network→HZ/guard/route exclusion/F0 lowering trust unchanged.

Latest completion (2026-09-19): ACTUAL NATIVE BASIS ADAPTER controls in
`native_basis/`. Read `docs/native_basis_v1.md` + attempt001 FIRST.54/54 PASS
(7 new+47 regressions);6 native analytic captures,0 real-network LP calls or
reconstructions. Existing highspy1.14.0, no install. All sealed sources and four
real diagnostic outcomes unchanged. Native adapter now exists for its declared
restricted mapping; outer-supervised integration is still NOT implemented.

Fixed simplex/presolveOFF/scaling0/threads1/parallelOFF, actual option readback;
records submitted float model + before/after readback, version/binary hash,
raw basis/point/status BEFORE mapping/checking. Original rational LP unchanged.
Structural basic→x; finite bound status→exact anchor; <= row basic→positive
slack; <= row upper→zero slack. Real control reconstructs(1/3,2/3), isolated
check U=−4/3 for objective−x−1. Inactive row yields slack1; fixed/permuted cases
pass. Native optimal status is not exact proof.

Actual redundant-equality case yields basic E-row variables: current schema
cannot map them, so capture preserved and UNSUPPORTED_MAPPING returned. No row
dropping, LP-infeasibility claim, free-status guess or presolve-map assumption.
Next: new single-budget owned outer capture→map→construct→pack→check pipeline,
analytic deadline/partial/error/cost controls for supported cases. Unsupported
E-row mapping and large sparse support remain separate design work, not cap
increases or hidden runtime changes. No new real experiment frozen/launched.

Latest completion (2026-09-19): ORIGINAL-COORDINATE SPARSE BASIS/ANCHOR interface
added in `exact_basis/`. Read `docs/exact_basis_v1.md` and attempt001 FIRST.
47/47 controls PASS (12 new +35 regressions). Old modules/frozen identities
unchanged; real LP reconstructions0, real solver calls0. Four sealed real
diagnostics stay unresolved. This supersedes the missing mapping-interface
work below, NOT the missing native adapter or production supervision.

Manifest binds original E/A rows and hashes, named x/slack columns, complete
disjoint basic/nonbasic partition and exact anchors. Uses Ex=h, Ax+s=b; no
unmapped presolve/scaling or opposite slack signs. Sparse exact assembly and
elimination record fill-in/peak elimination nnz/operation counts; singularity
is unresolved, not LP infeasibility. Constructor only emits CANDIDATE_ONLY.
Full frozen original-LP checker validates every constraint; relocated python
-I-S checks (1/3,2/3), U=−2/3. A different anchor gives negative slack and is
correctly rejected. Missing/duplicate coordinates, drift, limits, permutation,
fixed bounds and rank deficiency controls pass. Caps remain analytic-small.

Next: tiny native-basis adapter controls with submitted-model and solver-option
identity, original structural/slack mapping, degeneracy and unknown-map rejection;
THEN single-budget owned outer capture→construct→pack→check supervision before
any new real diagnostic. Do not simply lift caps or silently repair sealed
points. No complete-network proof, performance or real effectiveness claim.

Latest completion (2026-09-19): EXACT FEASIBLE WITNESS construction research
added in `exact_primal/`, analytic-only, no production integration. Read
`docs/exact_primal_v1.md` and numbered receipts FIRST. Latest attempt00335/35
PASS (13 new +22 regressions); attempt002 one wrong RHS fixture failure kept,
corrected47/35→67/35 only. All sealed sources unchanged. Real LP reconstructions0,
real solver calls0; four archived diagnostic outcomes remain unresolved.

Untrusted active-face hinting + sparse rational elimination yields only
CANDIDATE_ONLY. Full unchanged lp_sandwich checker establishes feasibility/U.
Control3x=1 reconstructs exact1/3; moved python-I-S checks U=−2/3. No dual or
optimality claim. Wrong active guesses, violated unselected constraints,
identity/objective mutations, deadlines/limits correctly rejected or unresolved.
1e−8 hint radius is generation-only, NEVER a feasibility tolerance. Exact
binary coefficient semantics preserved. Max64variables/512rows/8192nnz,
4096bits/200000operations/one attempt; not suitable for the real large LPs.

Research next: explicit sparse basis/anchor identities and original-LP mapping,
degeneracy/fill-in/bit caps + independent full check; then separately supervised
costed controls before any new real diagnostic. Do not silently raise caps,
repair sealed native points, install external solvers or claim network UNSAFE
from an LP feasible point. See doc for established SoPlex literature and scope.

Latest completion (2026-09-19): FOUR FROZEN LP DIAGNOSTICS EXECUTED AND SEALED.
Read `docs/lp_diagnostic_v1_execution_results.md` / JSON FIRST. Execution HEAD
9fdb77beac9c82530a2a1114a0c9ff75e7fc5ac1;4/4 once,4 native calls, all exact
checks complete. Automatic audit + fresh-process archive PASS,0 issues;
aggregate mutation test PASS. No ERROR/TIMEOUT/retry/repair/config drift.
This supersedes freeze-only instructions below. DO NOT launch again.

All4 remain UNRESOLVED_CANDIDATE_VS_LP_RELAXATION. Native status0/optimal on
all4, but primal NOT_EXACTLY_FEASIBLE:0 checked upper bounds,0 checked optima.
New exact L equals old exact L on all4 (display −1.92552,−3.53151,−1.36549,
−9.70755). Max equality violations2.82e−15,1.54e−15,1.64e−15,7.26e−10;
inequality/box violations also retained. Small violations are NOT ignored.
Negative candidate objectives close to L are not valid uppers or certified gaps.
No complete-network SAFE/UNSAFE. The old30 remain unresolved;26 not queried.

Total supplied-LP diagnostic cost25.562252s:load1.707815,propose17.934386,
package0.555540,isolated check4.712545,residual0.651965. Native1.269790s is
nested in proposal, never added twice. Historical propagation/range/F0 lowering
excluded; no end-to-end MoE speed claim. Wait/preflight/audits separately saved.
No reason from these four runs to add solver time or enlarge the query set.
Next possible research is a separately scoped exact-feasibility witness
interface with analytic controls, NOT silently repairing these frozen points.
Network→HZ/guard/route exclusion/F0 lowering trust and numerical gates unchanged.

Latest completion (2026-09-19): OUTER-SUPERVISED LP DIAGNOSTIC V1 implemented,
control-tested and FROZEN ONLY. Read `docs/lp_diagnostic_v1_results.md`, protocol,
freeze and selection review FIRST. New namespace `lp_diagnostic/`; old
`lp_sandwich/` and every earlier frozen source unchanged. Latest attempt003
41/41 PASS (15 supervision/batch +26 prior regression tests); attempts001/002
preserved. Fresh-process source/ordered-selection review PASS,0 issues.

Four jobs: input220 pair{1,2} p0;222 {0,1} p1;230 {0,3} p2;232 {0,1} p0.
First nonpositive obligation in original order per previously observed input;
unchanged archived weighted LPs, NOT a new holdout. Real solver calls0 and
`data/moe/results/lp_diagnostic_20260919_v1` NOT CREATED. Do not claim execution.

One original300s clock includes load/import/native capture/retained precheck/
pack/isolated check/admission. Native cap60, proposal deadline218, work watchdog
298, late publication300 invalidates acceptance. Owned-tree cleanup only.
Immutable terminal roster retains four denominators; ERROR stops, TIMEOUT and
checked-unresolved continue. Full diagnostic time/phase windows/residual
reconcile; native/component nested, missing cost null, waits/audits separate.
It is supplied-LP cost, NOT full MoE cost; historic propagation is excluded.

Next bounded action is explicit launch of these four frozen diagnostics AFTER
clean commit/push. No rerun-all30, point repair, tolerance/range/representation/
cache/order change. U<=0 from an exactly feasible point limits this LP only,
not model UNSAFE; L>threshold is LP-only, not full-network SAFE. Missing/inexact
primal leaves attribution unresolved. Old30 remain unresolved until new data.
Network→HZ/guard/route exclusion/F0 lowering trust is unchanged. This entry
supersedes the missing-outer-watchdog prerequisite below, not prior results.

Latest completion (2026-09-19): EXACT LP PRIMAL/DUAL diagnostic component added,
opt-in and NOT production-integrated. Read `docs/lp_sandwich_v1.md` and numbered
control receipts first. Attempt00125/25 PASS; attempt00226/26 PASS includes
multidimensional rational LP. New namespace `lp_sandwich/`; all old freezes
unchanged. Real-request solver calls0; the30 nonpositive rows remain unresolved.

Standalone stdlib checker validates exact box/inequality/equality feasibility,
objective and signed dual+residual lower bound; combines L≤optimum≤U only if
both witnesses pass. U≤0 establishes this LP's obstruction, NOT model UNSAFE.
L>threshold is LP-only evidence, NOT complete MoE SAFE. Native success ignored
as proof; float1/3 equality control correctly rejects an approximate point.
Point/status/objective and all native marginals/residuals retained before check.
Relocation python-I-S, hash/property/sign mutation, timeout/no-retry controls pass.

This is an interface/control release, NOT an actual-request proof or timing run.
No default/cache/representation/precision/budget/query-order change. CLI has
POSIX timeout; candidate capture needs an OUTER watchdog before any real use.
Next bounded step: separately freeze a small unchanged-LP diagnostic, including
native capture, full cost/terminal accounting and isolated checking. Do not rerun
all30, repair primal points, enlarge budgets or interpret missing primal evidence
as LP impossibility. Network→HZ/guard/route exclusions/lowering still trusted.

Latest completion (2026-09-16): SOURCE-CACHE ATTRIBUTION execution FINISHED,
sealed, independently archived. Read `docs/source_cache_ablation_v1_execution_results.md`
and JSON FIRST. Execution HEADf855c0b713fe9874aeb7d43913a54ff64063865e,8/8 once;
automatic final audit + fresh-process archive PASS,0 issues. No ERROR/TIMEOUT,
retry, sample/order/precision changes. This supersedes freeze-only instructions
below. DO NOT launch this study again.

matrix_only and both:4/4 complete checks,4 UNKNOWN_NONPOSITIVE,36/36 obligations
checked (6 positive,30 nonpositive,0 missing),0 complete positive requests.
All36 exact bounds and all4 complete checker results equal; requests, common
facts, route sets and router/joint-expert source bytes agree.116 proposals/arm,
348 exact dual evaluations/arm, all checks retained. All4 inputs single pair.

Full cost matrix_only836.820s vs both848.045s; means209.205 vs212.011s.
Both-minus-matrix differences+0.849,+2.613,+5.015,+2.748s,median+2.681s;
both1.341% more costly in this small observed cohort. Source decode/freeze/copy
9.597→19.753s; decode116→84 but84 freezes/32 hit copies;78 source evictions.
CSR work essentially unchanged, matrix cache0 evictions; native solver~28.3s
both. No automatic default switch; matrix-only is the better observed option,
not proof source caching is universally useless. Do not tune capacity/order.

Raw data remain local, old freezes/outcomes unchanged. New archive aggregate
mutation control passes. Numerical/trusted-lowering boundaries unchanged.
All30 nonpositive obligations remain unseparated candidate-vs-relaxation; no
new representation change or solver search is authorized by this result.

Latest completion (2026-09-16): sealed30 NONPOSITIVE obligations analyzed without
new solving. Read `docs/nonpositive_v1_analysis.md` / JSON and review JSON FIRST.
36/36 exact saved candidate bounds reconstructed:6 positive,30 nonpositive;
all30 remain UNRESOLVED_CANDIDATE_VS_LP_RELAXATION. No saved primal feasible point
or independently checked LP upper/optimality bound. PROPOSED is not optimality
evidence; nonpositive lower bound is not proof of LP impossibility/model unsafety.
Gate/difference ranges, per-property blockers and exact dual decomposition are
archived. All30 difference enclosures cross zero; input232 gate remains[0,1],
others half intervals. These are NOT isolated causes or permission to tighten.
Residual box terms−13.73..−4.35 mainly come from continuous factors: mandatory
dual accounting, NOT removable numeric padding.22 positive pre-box subtotals
are not valid bounds. No representation/budget/order/training change justified.

Separate SOURCE-CACHE ATTRIBUTION follow-up fully integrated and FROZEN ONLY:
read `docs/source_cache_ablation_v1_results.md`, protocol, freeze and review.
110/110 controls PASS including two analytic full chains, moved isolated checks,
cache flag tampering, deadline/cost/error/denominator controls; parent unchanged.
Same observed inputs220,222,230,232, same checkpoint/tensors/2/255; no expansion.
New namespace `source_cache_ablation/`: matrix_only(sourceOFF,matrixON) versus
both(sourceON,matrixON); same tail cacheON, all checks, support-first order,
300s total/298 watchdog/60 proposal cap/80 tail reserve.8 alternating requests,
fresh directory `data/moe/results/source_cache_ablation_comparison_20260916_v1`.
Separate-process parent-selection/hash reconstruction PASS. Real calls0, output
directory NOT created. No automatic launch/default change. Next scoped action
is this frozen comparison if execution is requested, not precision work.
Keep old reuse study sealed; query ordering remains separate. Network→HZ/guard/
route-exclusion trust unchanged. This entry supersedes potential-next prose below.

Latest completion (2026-09-16): frozen UPSTREAM REUSE timing experiment FINISHED
and independently archived. Read `docs/reuse_supervised_v1_execution_results.md`
and JSON first. Execution HEAD805e19372730e41e2cc93f17cf2419f41ac2d858,8/8 once,
no retry/tuning/order change; final audit + fresh-process archival review PASS.
No ERROR/TIMEOUT. Raw data stay local; methods/freezes unchanged. This completed
entry supersedes all pending-launch instructions below. DO NOT launch again.

reuse_off:4/4 checker completions,4 UNKNOWN_MISSING_EVIDENCE,21/36 output rows
missing. reuse_on:4/4 checker completions,4 UNKNOWN_NONPOSITIVE,0/36 missing;
6 positive rows but30 nonpositive. Complete positive requests remain0/4 each.
All4 inputs have one legal pair: this is not a route-changing SAFE result.
Mean full cost276.391→212.251s; all4 on-minus-off differences negative,
paired median−66.947s; total reduction23.21% in this small engineering cohort.
More complete evidence, NOT more completed checkers or new full certificates.

Measured exclusive CSR access/parse cost472.78→76.23s is the dominant reduction.
Native linprog21.05→28.15s; queries95→116, weighted15→36, exact dual evaluations
285→348 (3 per query retained). Source decode/freeze/copy7.79→19.65s: don't
claim both caches individually help. On has more packaging/checking work and
all costs remain charged. Four paired requests/common facts/routes agree;
all4 router and joint-expert HZ source pairs are byte-identical. Archive exact
reconstruction and aggregate mutation control pass;0 new solves during review.

Seal study, keep reuse opt-in. No expansion, extra time or same-row rescue.
Potential next decision: separately isolate source-cache overhead vs matrix
reuse, and separately study nonpositive LP evidence. Order optimization stays
a different ablation. No automatic next experiment is frozen by these results.
Network→HZ/guard/route exclusions still trusted; no deployed-float SAFE claim.

Latest completion (2026-09-16): UPSTREAM REUSE FULL-FLOW integration controlled;
timing study FROZEN, NOT EXECUTED. Read `docs/reuse_supervised_v1_results.md`,
protocol, controls_attempt002.json, freeze and selection_review JSON files.
96/96 controls PASS; two actual analytic capture→proposal→portable-check→outer
requests agree exactly, relocation passes python -I -S, original-start clocks,
real watchdog/late-publication/censored-log accounting covered. Failed harness
invocation and attempt001 are retained. Existing sources/freezes unchanged.

New namespace `reuse_supervised/`: reuse_off vs reuse_on upstream source/CSR
caches; identical single full tail check with tail cache ON, all proof checks,
same support-first ordering and300s/298s/60s/80s policy. No clock refund at tail.
New ordered clean-only inputs220,222,230,232 (902 excluded historical indices),
same E4/C10 conv epoch89 and2/255,4×2=8 requests. Separate-process selection
reconstruction PASS,0 issues;0 real verification calls. Result directory
`data/moe/results/reuse_supervised_comparison_20260916_v1` not created yet.

Next if executing: clean pushed freeze, then `nice -n 10
/data1/Kane/miniconda3/envs/act-py312/bin/python -m reuse_supervised.study launch`.
No retry/resume, resource-gated single CPU worker/thread, interleaved arms,
ERROR stops with all remaining slots retained. Evaluate completion and full
cost jointly; no claimed speedup yet, and no erased missing/nonpositive states.
Order optimization remains a separate ablation, not mixed into this study.
This supersedes the previous full-flow integration TODO, not old results.

Latest completion (2026-09-16): UPSTREAM SOURCE/EXACT-MATRIX REUSE candidate
implemented, opt-in/default OFF;86/86 controls PASS. Read
`docs/upstream_reuse_v1.md` and `docs/upstream_reuse_controls_attempt001.json`.
New namespace `upstream_reuse/`, no old frozen source changes. Current source
bytes/hash and every source/property/range/construction/dual check retained;
no verdict caching. Four option combinations match original analytic manifests
and request conclusions exactly;3 exact dual evaluations/successful proposal
retained. Old all-supports-before-weighted order remains unchanged. The initial
two fixture-path test errors are documented, not a failed research experiment.

This is an upstream proposal adapter, NOT yet a full-flow or real-model timing
result. Existing production/portable workers are not rewired. No sealed8-request
rerun, expanded samples, extra time or query-order ablation. Nested exclusive/
inclusive timers are ready but synthetic clocks are not speed measurements.
Next gate: opt-in full-flow integration with original-start total deadline,
portable tail/terminal controls, then separately freeze timing if authorized.
Keep schedule optimization separate; mathematical and numerical gates unchanged.

Latest completion (2026-09-16): saved-log evidence-generation COST ANALYSIS
FINISHED. Read `docs/upstream_generation_cost_v1.md` and JSON first. No new
models, solver calls, proof replays, samples or budget changes. Old sources/
results remain frozen. `upstream_cost_analysis/` reconstructs phase clocks and
source-path work, with2 controls and exact saved-analysis reconstruction.

Eight proposal phases total1118.334s: recorded query windows444.122s, outside
windows674.212s(60.29%). These windows INCLUDE preparation, two exact dual
evaluations and some serialization; they are NOT native solver time. Gaps mix
postchecks, next-source reads/prechecks, rational construction and I/O. Current
logs cannot give separate checking/construction/serialization seconds; fields
remain null. All212 entered queries PROPOSED, none recorded solver failure.
Frozen successful paths imply636 exact dual evaluations during generation;
tail parser cache does not cover these upstream calls. Count is static, not
profile timing, and is NOT authorization to remove acceptance checks.

Support-order barrier: all supports finish before ANY weighted query. At
handoff207 has15/18 properties with all four range certificates but0 weighted;
214 has7/27 but0 weighted.209 has9 range-ready,6 weighted;211 all9 weighted but
nonpositive. Readiness is metadata, not a new SAFE proof or evidence that a
different order would close all obligations. Known support reads~410–810MB/arm;
weighted209/211 repeatedly decode one joint source and write~242–248MB/arm.
Logical bytes are not measured disk time; don't call this an I/O bottleneck.

Next separately scoped candidate: upstream immutable decoded-source/exact-CSR
reuse with unchanged per-query checks, cache identity/pollution/differential
controls, and nested timer categories BEFORE a new benchmark. Do not silently
enable it now. Support-order interleaving is a separate ablation, not bundled
with reuse. No expand/time/profile rerun of the sealed8-request study. This
entry supersedes the previous saved-log analysis task, not its experiment data.

Latest completion (2026-09-16): frozen full-upstream comparison FINISHED and
independently reviewed. Read `docs/upstream_portable_v1_execution_results.md`,
execution_results.json and source_comparison.json FIRST. Published execution
HEAD17c2ec9f31753a8f40620a756e39ed436a8cff65 ran8/8 once, no retry/reselection/
code change, original300s total/298s watchdog/80s reserve. Both final roster
audit and separate-process archival review PASS;0 errors,4 retained timeouts.

double_check:1/4 completed independent checks (211 nonpositive),3 TIMEOUT.
single_check:3/4 completed checks (207 missing18/18,209 missing3/9 with6
nonpositive,211 all9 nonpositive),1 TIMEOUT(214). Both0 conditional positives.
New completion gains207 and209, no loss. Mean full request cost286.543 vs
276.629s; paired median single-minus-double−3.874s.211 completes252.058 vs
220.137s;207 single finishes297.940s close to cutoff. These are completion/
cost gains, NOT new SAFE or a broad speedup. Total charged requests2252.686s.

Identity nuance: request/routing (excluding only branch elapsed)/common facts
agree for4/4.3 of7 joint-HZ source pairs differ in stored matrices; all router
sources agree.211 exact output bounds differ slightly though both nonpositive.
Do not say every full upstream proof was byte-identical or assign every time
difference uniquely to removal of precheck. Cause of source variation not
isolated; no effect-driven reruns. Raw hash inventory and derived clarification
retain these findings without modifying frozen records.

Next bounded work should target evidence-generation cost/coverage, using
saved logs FIRST. Proposed-query recorded time is only~41–70s versus~99–175s
proposal-phase time; the difference includes checking/construction/serialization,
not automatically native-solver difficulty.207/209 lack necessary certificates;
211 complete nonpositive is a separate relaxation/evidence-strength limitation;
214 times out even single-check (capture121.6s,pack41.3s). No new budget/cache/
range change or larger cohort is authorized by these results. Keep the current
study sealed. Any optimized upstream path needs its own controls and freeze.
Network→HZ/guard/route exclusion trust remains; no deployed-float SAFE claim.

This completed entry supersedes the pending launch instructions below.

Latest completion (2026-09-16): full upstream→portable-tail integration is
CONTROL-TESTED and a small NEW-input comparison is FROZEN, NOT RUN. Read
`docs/upstream_portable_v1_results.md`, protocol, controls_attempt002.json,
freeze.json and selection_review.json.78/78 controls PASS, including actual
analytic capture/proposals→both tails; original72 regressions unchanged.
All earlier method, source and experiment freezes still verify.

New `upstream_portable/` compares same independently charged capture + reserve
handoff + cache ON, with double_check(V2 duplicate precheck) vs single_check(V3
sole isolated full check). Same300s total/298s watchdog,60s proposal cap,80s
tail reserve. Whole supervisor owns upstream and tail; no reset/refund at tail
entry. Loading/propagation/support/export/proposals/I/O/check/admission all
charged. Missing/nonpositive/timeout remain distinct; trusted lowering unchanged.
This is an engineering tail ablation, NOT matched monolithic or CROWN results.

Frozen NEW convolutional inputs207,209,211,214, same epoch89 checkpoint and
2/255;4 inputs×2 arms=8 requests, ordered clean-only, no route/bound selection.
Separate-process input reconstruction/exclusion audit PASS. Actual verification
calls=0. Old20-input results and offline114 remain sealed, never resumed.
Output `data/moe/results/upstream_portable_comparison_20260916_v1` does NOT
exist yet. Next execution after clean pushed freeze:
`nice -n 10 /data1/Kane/miniconda3/envs/act-py312/bin/python -m upstream_portable.study launch`.
One CPU thread/worker, resource gate, interleaved arms, no retry/resume; ERROR
stops and retains remaining roster as NOT_RUN_AFTER_ERROR. Final8-slot summary
recomputes terminal/clock/request bindings; run audit after completion and
archive results separately. Do not retune, replace inputs, expand budget, or
interpret completed nonpositive checks as SAFE. Eight300s caps total40min
before resource waits/archival audit; this is a bound, not a runtime prediction.
The current turn completed integration/testing/freeze only, not this launch.

This supersedes the previous "prepare/control-test full upstream" next task.

Latest completion (2026-09-16): V3 unverified-pack→sole isolated full check
is FINISHED. Read `docs/single_check_v3_results.md` and saved114 JSON first.
72/72 controls PASS; all legacy/math/cache/source/execution freezes unchanged.
Same complete checker, numerical gate and trusted lowering; no result oracle
in packaging, no duplicated mathematical precheck. Invalid proofs may pack
but cannot pass the sole checker. Extra result audit verifies structure, not LP
mathematics. Whole-driver/phase/watchdog/publication controls remain fail-closed.

Executed published6a7c32695 ONCE on unchanged archived input114, with simulated
220s prior work and original300s total/298s work deadline. Observed tail54.880s
(pack15.642,isolated check38.437,other0.801); request clock274.880s,25.120s
slack. Thus this saved case fits the unchanged80s tail reserve; no actual
upstream computation was performed. Result exactly unchanged UNKNOWN_NONPOSITIVE,
3 positive/6 nonpositive, no precheck files, no proposals/new model requests,
no old TIMEOUT promotion. Package10.323MiB; source/artifact/terminal audits pass.
DO NOT rerun single_check_saved114_20260916_v3 or edit its results.

Next: prepare/control-test a separate full upstream→V3 development execution
under the same300s/80s policy BEFORE choosing/freezing a small real comparison.
Actual upstream runtime and across-request budget feasibility remain untested.
Report complete checks vs nonpositive/missing/timeout, not promised SAFE gains.
No cohort/model/holdout run was launched. This completion supersedes pending
V3 instructions immediately below and V2's duplicate-check bottleneck entry.

Current bounded V3 stage (2026-09-16): user authorized removing the duplicate
full precheck in a separate opt-in version, NOT changing the sole complete
checker or80s reserve. Read `docs/single_check_portable_v3.md` and
single_check_v3_controls_attempt001.json.72/72 controls PASS, zero errors/
failures/skips; V2 and all earlier frozen source/execution identities unchanged.
`single_check_portable/` packages unverified evidence, then invokes the same
full isolated rational checker once. V3 forbids expected_result in metadata;
pack success is never proof success. All source/property/range/dual/coverage
checks remain in the sole authoritative check. Extra result-structure audit
does not re-prove LP bounds. Original300s total/298s owned cutoff unchanged.

After clean commit/push, one FIXED offline check pending:
act-py312,CPU one thread,nice10, `python -m single_check_portable.replay`.
Same archived input114, no new proposals/models/inputs. New root
data/moe/results/single_check_saved114_20260916_v3. Pass original_start=
tail_start-220: simulated220s upstream gives only80s total/78s work for the
tail. Do not retry, extend time or substitute on failure. Expected saved
UNKNOWN_NONPOSITIVE,3 positive/6 nonpositive, original TIMEOUT unchanged.
Compare/aggregate saved original only AFTER runtime acceptance; no result
oracle inside packaging/checking. This is offline budget feasibility, not
real upstream runtime or new verifier SAFE. Archive before any new study.

Latest completion (2026-09-16): portable optional cache V2 and bounded saved
evidence integration are FINISHED. Read `docs/cached_portable_v2_results.md`
and saved114 JSON first.61/61 controls PASS; old method/execution hashes
unchanged. New V2 opt-in bundle binds code/cache mode; analytic relocation
after removing original sources, python-I-S isolation, mutation, inherited
clock, owned watchdog and terminal/publication controls pass. Same300s total,
298s work; `supervise()` plus `audit_outer()` required, never inner files alone.

Executed published bdae1da5d ONCE: saved rank0/input114 evidence tail90.497s
including outer publication. Precheck35.032s,pack15.695s,isolated check38.882s,
other0.888s. Package10.326MiB. Exact UNKNOWN_NONPOSITIVE preserved:3 positive/
6 nonpositive,267/290 cache hits,23 parses,cache cleared. Final review and
archival/hash accounting PASS. No proposals, new model requests or old TIMEOUT
promotion; no new real-request study launched. DO NOT rerun the replay root.

Important next-step boundary:90.5s is OFFLINE TAIL ONLY, excluding upstream
propagation/proposals, and exceeds the old80s tail reserve on this observed
case. Do not claim production300s feasibility or speedup. Next separately
scoped engineering question is duplicate full precheck+isolated-check cost,
while retaining authoritative independent full-obligation checking and terminal
gates. No change to precheck semantics/reserve is implemented or pre-approved.
Real-request experiments still require a separate execution/selection freeze.
This completed entry supersedes pending replay instructions immediately below.

Current implementation stage (2026-09-16): optional exact cache now has a NEW
portable V2 bundle and owned whole-tail supervisor in `cached_portable/`.
Read `docs/cached_portable_v2.md` and controls_attempt001.json.61/61 controls
PASS, zero skips/errors/failures; all old freezes/source identities unchanged.
Cache defaults OFF; V2 metadata pins mode, policy and code. Legacy V1 packages
and production/cohort paths are untouched. Relocation after deleting analytic
source, python-I-S isolation, cache/no-cache exact differential, semantic/code
mutations, partial proofs and deadline controls pass. Both analytic modes
inherit a simulated270s prior cost, using only the remaining28s work budget.
Whole-driver watchdog includes precheck→packing→isolated check→candidate I/O;
same300s total/298s work. Late/failed output and late terminal writes cannot
promote positives. `run()` candidate records alone are NOT final admission;
use `supervise()` plus `audit_outer()`. No real-request study launched.

One fixed OFFLINE saved-proof integration check is pending after clean commit/
push: act-py312,CPU one thread,nice10,no GPU, `python -m cached_portable.replay`.
Same archived rank0/input114 and unchanged evidence; expected
UNKNOWN_NONPOSITIVE,3 positive/6 nonpositive. New root:
data/moe/results/cached_portable_saved114_20260916_v2.300s includes precheck,
packing, isolated verification and inner terminal I/O; no upstream computation
or solver queries. Do not retry/retune/substitute or promote historical TIMEOUT.
This offline integration cost is NOT production end-to-end acceleration.
Review and archive outcome, then separately decide real-request integration.

Latest completed engineering stage (2026-09-16): exact matrix parsing reuse
controls AND offline timing are finished and archived. Read
`docs/exact_matrix_cache_v1_results.md` and its timing JSON FIRST.41/41 controls
PASS;6/6 saved checks reproduce UNKNOWN_NONPOSITIVE,3 positive/6 nonpositive.
Executed frozen published HEAD b3de034cb. Original/uncached/cached median
checker seconds54.942/77.859/33.902; cached vs original38.30% lower,1.621 ratio.
Whole-process medians55.982/78.904/34.961s; cached37.55% lower. Peak RSS rises
850.324→1127.680MiB (+277.355MiB,+32.62%). Single saved input114,two runs/mode,
descriptive checker-only result, NOT end-to-end acceleration/new SAFE.
267/290 cache hits,23 parses,cache cleared at return; all exact results equal.
Separate accounting/hash review passes; old request/terminal/source unchanged.
No solver query, no old TIMEOUT promotion, no new real-request experiment.
DO NOT rerun the completed immutable benchmark/controls-receipt writer below.
Next separately scoped opportunity: identity-bound portable optional checker
integration plus relocation/deadline/full-budget controls, before any new
real-request study. No production rollout or new cohort was launched here.
This completion supersedes the pending launch instructions immediately below.

Previous prelaunch engineering record (2026-09-16): optional exact CSR parsing
reuse is implemented in `exact_matrix_cache/`; read
`docs/exact_matrix_cache_v1.md` and its controls JSON.41/41 controls PASS,
zero skips/errors, including cache pollution, content collision, request and
property identity, unchanged dual checks, deadline failure and uncached exact
differential. All old method/execution freezes checked before/after controls.
Cache is request-local and immutable, keyed by canonical content and request;
only matrix parsing is reused, NEVER a source validation or proof verdict.
Original production/portable code is unchanged; this is an opt-in adapter.

Timing is PENDING: after publishing this implementation/control/protocol
commit, run `python -m exact_matrix_cache.benchmark` ONCE in act-py312,
one CPU thread/nice10/no GPU. Six fresh-process saved-proof checks in fixed
reference/uncached/cached/cached/uncached/reference order, input114 only.
No checkpoint, dataset or solver query; original TIMEOUT is not promoted.
Output: data/moe/results/exact_matrix_cache_benchmark_20260916_v1.
Do not retry/retune based on timings. Each offline check must reproduce the
saved UNKNOWN_NONPOSITIVE,3 positive/6 nonpositive result exactly. This stage
supersedes the older "no implemented cache" opportunity below, not the sealed
cohort results. Production rollout/new verification needs a separate protocol.

Latest engineering completion (2026-09-16): optional reserve handoff revision
is implemented separately in `evidence_handoff/`; read
`docs/evidence_handoff_v1.md`, controls.json and profile.json.36/36 analytic
and regression tests PASS, zero skips/errors. Old source/method/execution
freezes verified before and after tests; original cohort records unchanged.
Only proposal-grant expiry against the fixed80s reserve is caught and handed
to unchanged precheck; true300s exhaustion, invalid evidence and I/O failure
remain failures. Committed obligations survive; missing ones remain UNKNOWN.
Portable partial-proof check and actual analytic worker→driver→isolated-check
→audit chain pass. Same60s proposal cap,300s total, thresholds and trust base.
New driver emits candidate terminals, requiring the caller's298s owned-process
watchdog. No real-cohort launcher/outer protocol is registered for this version;
the old supervisor still uses the old driver. No new real-model run was made.

One fixed saved-evidence profile: first archived precheck,rank0/input114,
parent archive SHA67cf5602..., NOT input98/new bound search. Unchanged checker
reproduces UNKNOWN_NONPOSITIVE (3 positive/6 nonpositive) in102.852s with
cProfile, outside original budget; original terminal remains TIMEOUT.61 loads
of61 files cost2.600s;23,257,441 rational conversions,232 sparse-entry parses,
166 identity calls expose repeated exact computation, not repeated reads of
one pathname. Cumulative times overlap and profiling is NOT a speed comparison.
No solver/checkpoint/data load in this saved-proof profile. Raw run preserved
under evidence_handoff_profile_20260916_v1; compact result committed separately.

Next bounded engineering opportunity: immutable parsed-source/matrix reuse
keyed by checked content identity, maintaining all per-reference property,
frame/guard/factor-order and dual checks. First use stale/cache-mutation and
uncached differential controls; no implemented cache or claimed speedup yet.
Keep handoff version fixed, retain all20 original TIMEOUTs and nonpositive
bounds. Do not rerun/retune the old cohort or claim the handoff adds SAFE.
Any new real-request study needs a separate execution freeze and terminal audit.

Latest completion (2026-09-16): the frozen general-evidence new20-input cohort
is FINISHED AND ARCHIVED; do not launch it again. Read
`docs/general_evidence_execution_v1_results.md` and its hash-bound JSON first.
60/60 terminals, original final audit PASS0; separate read-only archival review
also PASS0,60 fresh terminal/source reviews and17 full-model UNSAFE replays,
105.661s separate review cost, ZERO new solver queries. Original records unchanged.
Executed HEAD a0248e697; same selection db3043fb..., conv epoch89,2/255/300s.
Matched:0 SAFE/10 UNSAFE/10 TIMEOUT,mean220.680s. Evidence:0 conditional
positives/20 TIMEOUT,mean283.089s. CROWN:0 numeric positives/7 UNSAFE/13 UNKNOWN,
mean4.081s. Common facts20/20 equal;15 single/5 multiple-pair inputs.
All17 UNSAFE runs cover12 distinct inputs; no cross-arm result promotion.

Saved evidence logs:13 local prechecks (7 full but nonpositive,6 missing),
135 obligations=3 positive+70 nonpositive+62 missing. Seven requests have no
saved complete precheck. These are hash/count-checked historical observations,
NOT freshly re-proved bounds or completed isolated request checks. Outer stops:
3 precheck,9 package,4 isolated check;4 other exits are internal proposal-budget
exhaustion near220s.519 recorded proposal wrappers all PROPOSED, not proof that
weighted MILPs were optimal. Exclusive solver/serialization costs unmeasured.
The80s reserve can expire after preflight during construction and raise at
grant, aborting before partial precheck; analytic clock control reproduces the
reachable path. No production repair or new run was made. Seven accounting
controls pass. Input98 success did NOT generalize under this frozen setup.

Disposition: close this configuration with the negative result; do not expand
samples, retune reserve/gates or resume requests. Any future engineering study
must be separately identified: first test graceful reserve-boundary handoff and
profile repeated source/check costs on controls or saved evidence, preserving
all proof obligations and acceptance gates. No new real-model run is queued.
Upstream network→HZ/source/guard/route-exclusion trust remains; high-accuracy
strict and cross-architecture route-changing positive claims remain unachieved.
This completion supersedes every launch/pending statement immediately below.

Current authorized execution (2026-09-16): user explicitly requested completion
and freezing of the cohort supervisor/final auditor, THEN launch the unchanged
20-input/60-request study. `evidence_cohort/` now implements this execution-only
layer; no frozen act/scripts/moe_evidence/portable_proof source was edited.
Read docs/general_evidence_execution_v1.md, its controls and freeze JSON.
25 tests PASS (owned cross-session descendant cleanup, fail-stop roster,
resource cap, distinct-grade statistics/conflicts and actual analytic outer
driver/terminal review, plus method regressions); fresh clean reconstruction
PASS0, same selection db3043fb... No selected endpoint queried by preparation.

After this commit/push, launch ONCE with act-py312:
`python -m evidence_cohort.run --launch`. User authorization is present.
Root:data/moe/results/general_evidence_cohort_20260916_v1;
launch log:data/moe/results/general_evidence_launch_20260916_v1/supervisor.log.
Check these paths before any launch: if present, inspect, never resume/replace.
Global lock; one CPU thread/request, nice10, no GPU; original configs2/255/300s.
Whole-driver clock includes startup/serialization; owned descendants killed
at298s work limit, terminal inside300s; late files never promote a TIMEOUT.
ERROR stops; unknown/timeouts continue. Final separate-process per-request
reviews and cohort aggregation run automatically, with separate audit costs.
No automatic raw-data commit or scientific-success announcement. Archive final
results only after review. Preserve all grade/conditional trust distinctions.
This entry supersedes the earlier "no launch authorized yet" statements below;
it records preparation and launch instructions, NOT a completed60-run result.

Latest completion (2026-09-16): general-evidence preparation AND new20-input
conv freeze COMPLETE; docs/general_evidence_v1_preparation_results.md is the
current summary. Selection sha db3043fb124703e8123e5326eda853dc0d67d45e9104ca316487a631603daea7;
separate-process clean reconstruction PASS0;5,696 exclusions sources/878 used
indices. New indices114..205 (explicit roster in selection),60 planned queries,
ZERO new verification queries executed.36 analytic/regression tests PASS.
Unchanged matched V2 and original CROWN versus general conditional evidence,
same checkpoint2/255/300s. No old result upgrade, input98 re-query or new SAFE.

Next authorized preparation: bind/test cohort-level lock, resource gate,
roster/fail-stop handling and final three-arm aggregation to THIS frozen20
selection and protocol; then freeze/publish execution identity before launch.
The per-request runner/audit exists; a full60-cohort supervisor/auditor is not
yet registered. Preserve indices/settings, do not run endpoints piecemeal or
use old input98-only launcher. No full launch was performed in this turn.

Current completion (2026-09-16): general weighted-top2 conditional evidence
interface and per-request budgeted runner/terminal checker implemented OUTSIDE
frozen act sources (`moe_evidence/`). See docs/general_evidence_v1.md and
docs/general_evidence_v1_controls.json:36 controls/regressions PASS, no skips.
E/C and explicit linear properties derive from request; all tie-legal pairs,
partial scoped reuse, missing obligations, invalid bindings, check timeouts,
portable -I -S and real analytic multi-pair capture/pipeline covered. Same300s;
no floating F0 trust, no production gate change, no new real-model certificate.
Old input98 frozen source identities remain unchanged (458 files checked).

Next in this stage: freeze20 NEW conv clean-correct inputs under
docs/general_evidence_v1_protocol.json, separately reconstruct selection;
matched V2/evidence/original CROWN, fixed2/255 and300s. Freeze only, do NOT
execute selected verification endpoints in this stage. Full cohort supervisor,
lock/resource/roster and final three-arm aggregation require their own frozen
execution gate before later launch; do not use the old input98-only runner.
All earlier narrower results below remain historical, not current API limits.

Latest completion: optional same-budget input98 development COMPLETE under
`d944a9ed3`. Read docs/optional_evidence_dev_v1_results.md and review.json.
Original matched V2 TIMEOUT295.697s; opt-in evidence CHECKED_CONDITIONAL9/9
in185.317s INCLUDING capture29.274, propose83.736, precheck32.432, pack9.381,
independent check30.225s.26 fresh proposals, no old bounds/census; fact views
equal. Separate terminal/package/portable re-audit PASS0 (31.564s outside run).
New portable bundle7,181,522bytes; same min0.1772745273844. Single observed
positive control/single pair; NOT general speedup, production integration,
route-changing or deployed-float proof. No input16/ACT-only reruns, no follow-on.
All three requested stages now delivered: portable proof; read-only five-case
cost/failure analysis; separately frozen optional-budget development comparison.

Current authorized development: docs/optional_evidence_dev_v1.md and frozen
JSON, ONLY observed input98. Two arms: unchanged production matched V2 and
opt-in pre-F0 evidence; one300s clock each includes startup, capture, proposals,
local aggregation, portable packing and independent -I -S check. No old proof
or census reused; no input16/ACT-only queries. Postselected engineering smoke,
not confirmation.80s explicit proposal reserve for checking, terminal reserve2s;
production acceptance unchanged. Freeze/tests/commit/push/live publication gate
before run_optional_evidence_dev; new root optional_evidence_dev_20260915_v1.
Run once, preserve ERROR/TIMEOUT, then separately audit and archive. No follow-on.

Latest read-only analysis: docs/proof_closure_costs_20260915.md/.json separates
three ACT-only checked-nonpositive cases, conv16 checked-nonpositive7/9 and
conv98 complete-conditional9/9. Saved raw inventories rehashed; no new solves.
Historical native solver-limit TIMEOUTs are a distinct evidence layer.
Exclusive propagation/serialization were not measured: null, not invented0.
Next authorized step is a NEW optional one-request-budget evidence development
mode, not relaxing production optimal-status or refining these old bounds.

Latest completion (2026-09-15): portable input98 proof accepted under
`f9dc8f2dd`. Read docs/portable_conv_proof_v1_results.md and its compact review.
Copied OUTSIDE checkout; python -I -S checks9/9 with exact archived result,
without model/data/history/solver reads. Four mutations reject, including
semantic mutations with rehashed transport.56 logical files428.19MB reduce
to7.18MB bundle (32.10MB deduplicated uncompressed); pack9.26s/check30.22s.
Trusted upstream HZ/guards/exclusions unchanged; no new solve or gate change.
Current user sequence: next read-only five-case unresolved/cost analysis,
THEN separately freeze optional same-request-budget evidence development.
Do not treat old179s as a comparative speedup or automatically remove status0.

Latest completion (2026-09-15): input98 PRE-F0 rational request proof COMPLETE
under `2a8e11d76`. Read docs/conv_pre_f0_r2_results.md and
act/pipeline/moe/results/conv_pre_f0_review_20260915_r2.json.9/9 positive:
8 independently reconstructed rational McCormick LPs+1 scoped interval fact,
minimum0.1772745273844 (reuse); minimum residual1.61557502934. Checked router
order r1-r2 in[0.70535119,0.85031138] gives lambda1 in[1/2,1].26 proposals,
two isolated checks and fresh review; all-stage179.01s, not benchmark timing.
Floating F0 construction no longer trusted. Network/input→HZ and expert/source
binding, guards and route exclusions still trusted; NOT deployed-float SAFE,
NOT route-changing (single pair), NOT high-accuracy AdvMoE closure. Original
TIMEOUT and prior weaker-boundary positive evidence unchanged. No input16 query.
R1 ERROR25.67s/zero proposals preserved. Correction to early failure wording:
the rejected entries were numpy.float64 (Torch→NumPy), not Torch scalars;
the exact Python +/-1/0 conversion repair is unchanged.24 tests cover both
types, actual property API, construction mutations and aggregation. No further
query, bound refinement or production-gate integration is queued.

Pre-F0 R1 execution `e9999be19` stopped with ERROR at the first difference
export: real Torch property scalars rejected by rational(), no LP proposal or
new output bound. Failure preserved and independently inventoried in
act/pipeline/moe/results/conv_pre_f0_failure_20260915_r1.json. Current R2 repair
only canonicalizes verified +/-1/0 property scalars to Python numbers and tests
the real Torch-scalar boundary. See docs/conv_pre_f0_r2.md; new source freeze,
commit/push and root conv_pre_f0_rational_20260915_r2. All inputs/ranges/budgets
and mathematical rules unchanged; no production gate change. Do not reuse R1
root or delete its failure. R2 result is not yet known in this preparation entry.

Current authorized stage (2026-09-15): input98 ONLY, same nine properties and
materialized2/255 box, pre-F0 rational-construction proof. Read
docs/conv_pre_f0_r1.md and scripts/conv_pre_f0_protocol.json. Fresh shared expert
HZ, fresh router-order and difference LP evidence, dyadic0/half/1 gate ranges;
existing independent rational McCormick checker. No floating F0/gate routine
or weighted-property MILP, no input16 rerun. New raw root
data/moe/results/conv_pre_f0_rational_20260915_r1. Tests/freeze/commit/push and
live remote equality precede launch. Outcome may be UNKNOWN under this more
conservative gate policy; no refinement/retry or production acceptance change.
If all nine close, remove only the floating F0 construction trust assumption;
upstream network→HZ, source binding, guards and route exclusions remain trusted.
This records preparation, not completion; no later query queued.

Latest completion (2026-09-15): ALL obligations of conv inputs16/98 checked
under frozen execution `0ed860053`, remote publication confirmed BEFORE root
creation. Read docs/conv_request_sign_lp_r1_results.md and
act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json.
18/18 generated/checked,16 positive. Index16:7/9 positive; class5-minus7
LB-0.2737518580 and5-minus9 LB-0.1255899474 leave UNKNOWN. Index98:8 positive
residual LPs+1 independently checked scoped interval fact,9/9 positive,
minimum0.1772745273844: CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_F0_LOWERING.
Two isolated checker processes and fresh review agree; all6 candidate pairs
per request accounted, but route exclusions/upstream HZ/floating F0 remain
trusted. This is NOT the stronger prior pre-F0 rational-McCormick proof path,
NOT deployed-float SAFE, NOT route-changing (both single pair). Historical
TIMEOUTs and production acceptance unchanged. No retries, no weighted-property
MILP. All-stage proof-study costs98.35/60.13s, not a benchmark speedup.19 tests
pass. No follow-up automatically queued; requested study is complete with one
conditional request proof and one nonclosed request, not2/2 success.

Current authorized stage (2026-09-15): freeze ALL required output obligations
of the SAME two observed conv requests16/98. Read docs/conv_request_sign_lp_r1.md.
New one-shot script run_conv_request_sign_lp generates fresh scoped interval
proofs and all residual supplied-F0-HZ LPs, then checks exact rational evidence
and complete route/property accounting. Upstream lowering, route exclusions and
floating F0 construction remain trusted. Production gates/old TIMEOUT unchanged.
Tests, freeze commit AND confirmed remote publication required before run-root
creation. One CPU thread, no GPU/retraining; no further query queued. This entry
is preparation, not a claim of successful complete-request evidence.

Latest completion (2026-09-15): two sign-sufficient evidence controls COMPLETE
under `e269821cf`. Read `docs/conv_sign_lp_r1_results.md` and
`act/pipeline/moe/results/conv_sign_lp_review_20260915_r1.json`. Independent
Python -S checks and fresh review agree: index16 first property LB3.690808268875,
index98 first property LB5.712379468352;8,078/6,803 factors with1,668/1,243 binary
factors relaxed. No weighted-property MILP executed; routing/support capture
still paid. This is CHECKED_POSITIVE_SUPPLIED_F0_LP, two properties, ZERO full
requests. Original TIMEOUTs and production status0 gate unchanged. Floating
F0 construction, network→HZ and guards remain trusted; not the prior stronger
pre-F0 rational-McCormick contract. Documented deviation: initial GitHub push
failed500; run launched after local freeze commit but before successful remote
push retry. Evidence audit PASS, execution protocol explicitly records that
publication-order deviation. Do not rerun/relabel to erase it. Next separate
scope could check ALL obligations of these same two observed requests, with
explicit conditional trust and complete aggregation. No follow-up queued.

Current authorized research (2026-09-15): independently checkable sign-sufficient
lower bounds. Read `docs/conv_sign_lp_r1.md`. New two-control protocol freezes
first matched F0 properties at index16/{0,3}/5-minus0 and98/{1,2}/0-minus1.
Capture original recipe's supplied F0 HZ BEFORE the property MILP, export root
continuous LP, propose duals then independently check in two Python -S processes.
This still trusts floating F0 construction and upstream lowering; not a full
request or deployed-float proof, not the stronger pre-F0 rational-McCormick path.
Tests/freeze/commit/push precede execution. Output
`data/moe/results/conv_sign_lp_20260915_r1`; no retries or following queries.
Production status0 gate and all full V2 outcomes remain unchanged. This entry
records preparation, not a new checked positive result.

Latest diagnostic completion (2026-09-15): full V2 results archived/pushed in
`7b1ab7269`; separate read-only obligation analysis is in
`docs/conv_full_v2_obligations.md` and
`act/pipeline/moe/results/conv_full_v2_obligations_20260915.json`. All frozen
raw hashes unchanged; no new solve/forward query.419 returned F0 property
queries ALL status1 solver-limit UNKNOWN, not completed crossing-zero bounds.
All32 ACT timeouts entered F0 after exact routes. Adaptive queried192/207
pair-properties,1 interval-reusable,14 not reached (inputs26/48/95/106);
matched170 union queries cover314/315,1 reusable.54 stored full-objective
duals are positive but UNACCEPTED. Inputs16 (9/9) and98 (8/8 plus1 interval
fact) have diagnostic sign coverage in both arms, remain TIMEOUT, single-pair,
NOT newly certified. A scalar MILP dual is not an independent proof. Native
limits are compliant but local return overrun reaches43s; do not remove outer
watchdog. Next possible separate protocol: independently checked sign-sufficient
bounds on these observed controls; separately analyze scheduling long tails.
No gate relaxation, extra solve, holdout or follow-up is queued/authorized by
this diagnostic. See retained parser-draft note; R1/full V2 remain unchanged.

Latest completion (2026-09-15): full convolutional V2 COMPLETE and freshly
reviewed under execution `2d2477e4b`. Read `docs/conv_full_v2_results.md` and
`act/pipeline/moe/results/conv_full_v2_review_20260915.json`. All90 requests
complete; automatic/separate/fresh audits agree PASS0;30/30 fact pairs equal,
35 UNSAFE method-run replays (18 distinct inputs). Adaptive0 SAFE/17 UNSAFE/
13 internal TIMEOUT; matched0/11/19; CROWN1 numerical positive/7 UNSAFE/
22 UNKNOWN. No outer timeout. Adaptive's six additional decisions are ALL
counterexamples, not certificates. CROWN-only positive index98 is not formal
SAFE; its witness index113 is missed by both ACT arms. No work remains queued.
User requests archival then saved-log analysis of unresolved obligations;
do that read-only, in separate derived artifacts, without new solving,
retuning or replacing frozen results. Earlier "not started" entries below
are historical. Full V2 sources/protocol and all raw bytes stay immutable.

Current explicit authorization (2026-09-15): PI requests the90-call full V2
three-arm experiment on the ORIGINAL30 unexecuted conv inputs. Read frozen
`docs/conv_full_v2.md` and `scripts/conv_full_v2_protocol.json`. New scripts
bind both ACT arms to unchanged budget V2; CROWN frontend/backend unchanged.
Preflight independently rechecks the existing V2 ACT smoke, original CROWN
records and original clean-only selection. Old R1 remains FAIL. Tests, freeze
review, commit and push BEFORE launch. Run once via
`python -m scripts.run_conv_full_v2` in act-py312, durable session
`moe-conv-full-v2`, log `data/moe/results/conv_full_v2_pipeline.log`, result root
`data/moe/results/conv_three_arm_full_20260915_v2`. Inspect live state before
launch; this entry records preparation, not that90 requests are complete.
No resume/retry/config change or extra query; ERROR stops and inventories
unattempted requests. Two separate-process final audits plus FULL_SUMMARY.json
are automatic; reviewed Git archival is separate. Do not edit the checkout
while running. No more work is queued after this experiment. Statistical unit
is30 inputs; CROWN numerical positives are separate from HZ-policy SAFE.

Latest completion (2026-09-15): frozen V2 outer-supervised OLD-input two-arm
smoke COMPLETE under `bd8610238`. Read `docs/conv_budget_smoke_v2_results.md`
and `act/pipeline/moe/results/conv_budget_smoke_review_20260915_v2.json`.
Automatic/separate/fresh audit PASS0, V2 gate PASS;4/4 complete packages,
2/2 common-fact pairs equal,1 full-model UNSAFE replay. Outcomes remain1 UNSAFE
+3 TIMEOUT,0 SAFE. Old3 outer kills become0, with complete internal TIMEOUT
packages at295.580/295.581/296.275s.5,993 journal events,2,593 returned native
calls,37 UNKNOWN property records; no passed-allocation violation. Native
return still overruns a local deadline by up to54.956s (input1 adaptive Tier1),
so native hard timing is NOT solved; owning300s watchdog remains necessary.
74 tests pass. All54 current parent artifacts unchanged. No worker/full90
running or queued. This requested stage is finished; do not rerun it. A full
comparison requires a separately frozen V2-bound protocol and authorization,
not promotion/mixing into old R1 (which remains FAIL). Do not change frozen
V2 sources/protocol; new findings belong in separate follow-ups. Accounting
conformance is not new SAFE coverage or an independent network-bound proof.

Current authorized freeze (2026-09-15): `docs/conv_budget_smoke_v2.md` freezes
four OLD-input requests (0/1 x adaptive/matched monolithic), same300s total,
V2 reserve5s inside300 and unchanged numerical gates. New outer runner
`scripts.conv_budget_smoke_v2` and terminal/journal auditor are implemented.
Test/commit/push BEFORE launch; then execute once with act-py312 at
`data/moe/results/conv_budget_smoke_20260915_v2`. No retry/resume/full90 path.
Audit must retain late/partial packages and compare full snapshot facts, bind
journal identity to each complete package, and recheck journals via Python -S.
Success requires a complete package in EACH arm, not a positive certificate.
Independent archival review follows execution. R1 FAIL is not overwritten.
Do not edit frozen V2 sources or its protocol document during/after execution;
write the result in a separate completion document and compact archive.

Latest completion (2026-09-15): budget/partial-terminal V2 implementation and
source-defined controls are COMPLETE, execution `e9cde6945`. Read
`docs/budget_contract_v2.md` and
`act/pipeline/moe/results/budget_contract_v2_controls_review_20260915_r1.json`.
Both ACT arms preserve toy baseline SAFE;4 packages pass structural audit and
separate python -S journal checks match (adaptive80 events/2 property results,
monolithic62/1).66 focused tests pass. All40 old smoke and13 timing artifacts
unchanged; no trained-conv query/full90 started and no control worker remains.
This is an opt-in runtime budget adapter, not new numerical evidence or a
default algorithm change. Next deliver the separate frozen outer supervisor
and terminal+journal audit for both ACT arms on old smoke inputs, then test and
record any new execution under that protocol. R1 remains FAIL. Do not just run
the new worker without a300s owning watchdog, or mix V2 outcomes into R1.

Latest implementation (2026-09-15): explicit opt-in budget/partial-terminal V2
is implemented for both adaptive and matched monolithic under `scripts/`.
Read `docs/budget_contract_v2.md` and `scripts/conv_budget_contract_v2.json`.
Local grants retain absolute deadlines through construction; native limits
are rechecked after durable READY publication; both arms reserve5s INSIDE300s
for terminal work.25%, support configs and numerical gates unchanged. Property
and replay records are durable and separately labelled, never request verdicts.
Stdlib-only checker validates budget accounting; it is not a bound proof.
All frozen ACT/old wrapper files remain unchanged; this is an explicitly bound
runtime execution adapter, not an observational-only patch or default change.
New conv worker requires policy/source identity in its request. Real-model V2
smoke has NOT run; a separately frozen outer supervisor+terminal audit remains
required. Full90 is not authorized. First commit/push and run source-defined
toy controls via `scripts.run_budget_v2_controls`, then archive/review them;
do not confuse toy conformance with repaired conv smoke or new SAFE results.

Latest completion (2026-09-15): the separately frozen F0 timing diagnostic has
finished ONE old input0/matched-monolithic request under execution `fd69daa1a`.
Read `docs/conv_f0_timing_r1.md` and
`act/pipeline/moe/results/conv_f0_timing_review_20260915_r1.json` first.
Terminal outer TIMEOUT300.069s; automatic/separate/fresh audits PASS0 issues,
2055 durable events, all40 original smoke artifacts unchanged. Pair propagation
45.017s (42.158s support within it), encoding8.341s, union construction14.014s;
eight property-native calls208.612s all status1, last native call right-censored
(19.138s exposure, no result). Native entry at280.931s had19.069s request budget
left but received stale20.629s: measured1.560s mismatch after construction.
This supports a separately versioned budget-accounting/partial-terminal repair,
not a claim that fixing it makes the properties positive. Missing solver bounds
are unavailable, not zero. Preserve observer/old source hashes and read the
documented incidental-caller-context and draft-aggregation caveats. The final
observer/review/lifecycle/F0 suite passes43 tests; no solver rerun for review.
Old smoke remains FAIL; no full90, no workers and no automatic rerun. Next
decision: account construction in local/global native budgets and durable
per-property terminal progress, equally for both arms, with new version/tests
before any new execution. Do not alter support,25%, numerical gates or samples.

Current next-stage implementation (2026-09-15): user authorized durable F0
substage timing, then ONE old smoke request at the unchanged300s cap. Read
`docs/conv_f0_timing_r1.md` and `scripts/conv_f0_timing_protocol_r1.json`.
Diagnostic-only wrappers under scripts preserve all frozen ACT/old wrapper
hashes. Fixed input0/matched monolithic; no full90, retry, support ablation,
budget or policy changes. Test/commit/push first, then execute
`python -m scripts.run_conv_f0_timing` in act-py312 and independently audit.
Raw output `data/moe/results/conv_f0_timing_20260915_r1`; check whether it exists
before launch. An existing directory is not permission to retry. Old smoke
remains FAIL. This entry precedes execution; later review must report measured
substage times and censored spans rather than assuming all F0 time is solving.

Latest completion (2026-09-15): conv three-arm smoke R1 completed6/6 under
execution `c192bca4d3c3161abe3ae816471894be24e3d908`. Read
`act/pipeline/moe/docs/conv_three_arm_r1.md` and
`act/pipeline/moe/results/conv_three_arm_smoke_review_20260915_r1.json`.
Separate re-audit exactly matches automatic audit: PASS,0 issues,1 complete
HZ package,2 complete CROWN records,1/1 full-model UNSAFE replay,2/2 common-fact
pairs equal. Adaptive:1 UNSAFE+1 outer TIMEOUT; monolithic:2 outer TIMEOUT;
CROWN:2 UNKNOWN. No SAFE or numerical positive. Three outer timeouts and all
snapshots preserved; total observed requests1021.96s. Thirty focused tests pass.
SMOKE GATE FAIL: monolithic has no complete non-error record. Supervisor is
STOPPED_REVIEW_REQUIRED, no worker remains, no full90 started/queued. Do not
run the full cohort or extend budgets/replace inputs to bypass this gate.
The outer scheduler and terminal auditor are implemented; this is a frozen
budget conformance failure, not missing orchestration. Stop positions are
monolithic F0 (two) and adaptive Tier2 F0 (one), not established root causes.
Any next investigation must be separately scoped and preserve this failed R1.
Post-archive test repeat exposed an exists/read /proc reaping race in the
original test (not the executor). Frozen wrapper hashes remain unchanged.
Use `scripts.test_conv_three_arm_lifecycle` for subsequent controls:31 focused
tests pass, plus two11-test repeats. Failure and test-only repair are retained
in `results/conv_smoke_posttest_review_20260915_r1.json` under the pipeline.
The following prelaunch entries are historical and do not authorize rerunning.

Latest implementation (2026-09-15): conv three-arm outer orchestration and
terminal audit are now integrated under `scripts/`, preserving the frozen ACT
source inventory. Read the execution-wrapper section of
`act/pipeline/moe/docs/conv_three_arm_r1.md`. Thirty focused tests pass, and the
separate clean-only freeze audit remains PASS. User explicitly authorizes the
six smoke requests ONLY; run `python -m scripts.run_conv_three_arm` with
act-py312 after commit/push. Output `data/moe/results/conv_three_arm_smoke_20260915_r1`.
Single owned process group, shared lock, resource gate, 300s complete request,
fail-stop ERROR and retained timeout/partial snapshots; independent terminal
audit and conformance gate. No positive-count requirement. No full/pipeline
option exists: 90 full requests require a separate instruction after smoke.
At this implementation commit no smoke has yet run. Inspect live artifacts
before launching; never overwrite or resume an existing root. The older freeze
status below records historical preparation, not current missing wrappers.

Latest preparation (2026-09-15): convolutional three-arm R1 protocol and selection
are FROZEN, NOT EXECUTED. Read `act/pipeline/moe/docs/conv_three_arm_r1.md`,
`configs/conv_three_arm_protocol_r1.json`, `configs/conv_three_arm_selection_r1.json`
and `results/conv_three_arm_freeze_review_20260915_r1.json` under the MoE pipeline.
Same selected conv epoch89, E4/C10;30 new clean-only selected inputs, epsilon2/255,
adaptive versus matched-reuse monolithic versus ACT-fronted plain CROWN.
832 recorded previous indices from5,416 artifacts excluded; smoke indices0,1
are disjoint. Exact materialized tensors and source/method/model hashes bind all
arms. Separate-process clean reconstruction PASS,0 issues;24 focused tests pass.
90 full calls +6 smoke calls,300s each, input-blocked rotated order, one worker;
positive evidence levels are not interchangeable. No route or bound query ran.
The E4 worker adapter is implemented, but outer three-arm orchestration and final
auditor are NOT_YET_INTEGRATED; do not launch the old three-model batch scripts.
Next implement/test those wrappers without changing the frozen decisions, freeze
their execution identities, then obtain an execution instruction and run audited
smoke before full. No outcome-driven budget/recipe/cohort change or old holdout.

Latest completion (2026-09-15): convolutional family full-shape compatibility,
supervised training and independent landing audit are COMPLETE. Execution
`6d2e6d299`; read `act/pipeline/moe/docs/conv_training_results.md` and
`act/pipeline/moe/results/conv_training_review_20260915_r1.json` first.
Frozen seed17 ran100/100 epochs; selected epoch89 by earliest validation maximum:
validation68.08%, test67.06% on the full5000/10000 respectively. All100 immutable
checkpoint hashes/metadata and exact validation/test metric replay pass the
separate audit. Selected checkpoint SHA256
`f5781a792f844a68de941f1a6b314d0e627ad5dd30e262088e6c1864d6bc5289`;
local path `data/moe/results/conv_training_seed17_20260915_r1/checkpoints/epoch_089.pt`.
Supervisor/landing status LANDED_AUDITED; no training worker remains. Training
used an immutable Git source export, not later edits. No retry, reselection,
dependency change or production numerical-policy relaxation.66 related tests pass.
The following preparation now freezes three-arm trained-family verification
(E4, not E8); see the newer top entry for scope and remaining execution work.
Training is not a new certificate or a cross-architecture verifier win. Do not
retrain for a better accuracy or reopen sealed AdvMoE/backend studies.

Earlier prelaunch completion (2026-09-15): full-size convolutional family gate
and supervised training implementation. Read
`act/pipeline/moe/docs/conv_family_r1.md` and
`act/pipeline/moe/results/conv_pretraining_review_20260915_r1.json`.
Full-shape R1 passed ACT sparse retention but external CROWN failed on a default
dtype omission; preserve it. R2 sets the declared float64 default, same model,
box and backend, and passes ACT + plain CROWN conformance. All five ACT components
retain SparseHZ; all six static pair expressions pass; external pair{0,1}
returns nine finite ordered bounds. No positive-bound acceptance requirement.
CUDA two-batch smoke passes finite gradients, actual router updates, exact
checkpoint replay and one optimizer continuation. This is not production state.
`conv_training_supervisor` and `conv_training` implement source-snapshot training,
immutable epochs, full validation selection and independent final metric replay.
Nine new supervision/training unit controls and four factory tests pass.
Production100-epoch training was authorized for the next launch and is now
completed above. Check live supervisor before launching another job.
Three-arm trained-model verification remains separately scoped and unlaunched.

Latest completion (2026-09-15): Advice/bb.md fixed ACT-only rational transfer,
execution `fbc48d6a9`. Read `act/pipeline/moe/docs/request_lp_act_only_results.md`.
All three fresh generations complete; 115 exports independently rechecked.
Positive obligations5/9,26/27,15/18, but complete positive requests0/3 (UNKNOWN),
not115 independent network proofs. No old proof facts reused; complete worker
generation/check costs and substantial evidence sizes archived. Do not tighten
gate ranges or add retries to force these selected cases positive.

Manuscript now has abstract, introduction and discussion; method03 follows the
actual schedule and evaluation08 centers on four primary experiments. Historical
tables remain in `paper/appendices/historical_evaluation.md`, not erased.
`scripts/run_moe_proof_demo.py` is a source-defined no-download example that
runs the standard verifier and fresh rational proof plus a Python -S checker.
It passed from outside the checkout; see `paper/artifact_quickstart.md`. This is
not yet a tested clean-container distribution of the empirical model artifacts.

Second-family scope is now fixed in `docs/conv_family_r1.md` and its training
config under the MoE pipeline. Versioned conv factory/checkpoint loader is
implemented, with small conv/pool controls. At that earlier stage training
supervision, full-shape external conformance and three-arm evaluation remained
next; see the newer status above. A non-dyadic singleton sparse control
exposed inconsistent independently rounded bounds and failed closed; no
numerical gate was relaxed. Dyadic point and nondegenerate sparse-box controls
pass. Do not interpret small compatibility controls as certified model quality.

No experiment remained running at that earlier completed-stage handoff. Existing
high-accuracy strict-certification and cross-architecture outcome goals remain
unachieved; there is no acceptance guarantee.

Historical prelaunch (2026-09-14): method section03 now describes
the actual scoped-fact/route-complexity/residual-obligation algorithm, with the
legacy-only F0 trigger removed from the current version's description. The new
generic rational-request generator and fixed three ACT-only protocol are in
`act/pipeline/moe/docs/request_lp_act_only_r1.md`. Tests and freeze precede new
queries. Do not substitute cases or interpret LP UNKNOWN as an unsafe model.
The next separate workstreams are a materially different moderate-scale model
family and a reviewer-runnable model/request artifact. Neither is complete just
because the rational checker can run without Torch. No submission/acceptance
guarantee is made; existing high-accuracy/external-comparison limits remain.

Latest completion: external complete-cost comparison, execution `0de4fe1c7`.
Read `act/pipeline/moe/docs/external_pair_comparison_results.md` and linked
compact re-audit. Smoke6/full60 completed; separate re-audit exactly matches
saved audits, zero issues. Full:25 ACT packages +30 external records,5 retained
outer timeouts,9 full-model UNSAFE replays. ACT11 HZ-policy SAFE versus13 CROWN
numerical positives,8 shared,3 ACT-only(two multi-pair),5 CROWN-only. All five
CROWN-only rows are ACT solver-limit UNKNOWN. Mean complete costs138.11/4.23s;
the external path is much cheaper and has more positives overall. Report
complementarity, NOT ACT dominance or interchangeable proof levels.
Final regression:42 focused tests PASS; compact archive independently
reconstructs exactly. No dependency, solver or production-policy change.
No experiment remains running or queued at this archival completion.
The two requested deliverables are complete: external complete-request table
and rational pre-F0 construction checking. Do not retune25%, expand this cohort,
make an outcome-selected portfolio or reopen sealed searches. Next experimental
scope/model must be frozen separately; high-accuracy strict certificates and
independent full-dynamic-model external comparison remain unachieved.

Latest completion: direct rational construction R3, execution `f30f8ccf4`.
Read `act/pipeline/moe/docs/request_lp_rational_results.md` and linked review.
All 18 required output obligations independently check (15 reused, 3 residual).
The trusted base no longer includes floating F0 construction; network-to-HZ,
guard lowering and route exclusion remain trusted. Same R2 ranges, no policy
change. 28 focused tests and read-only review PASS, zero issues.
Historical registration of the now-completed external comparison:
read `act/pipeline/moe/docs/external_pair_comparison_r1.md`. Same ten observed
inputs; six old-input smoke requests gate full execution. It materializes one
shared raw input file, charges loads and the complete cross-env request, and
kills the owned process group on timeout. No numerical-policy changes.

Updated 2026-09-14. Latest 100-input confirmation execution HEAD: `bc0791976`.
Latest 30-input confirmation execution HEAD: `6ad58bc9d`.
Earlier development schedule execution HEAD: `d7ac0b0a9`.
Earlier reuse experiment execution HEAD: `751e386d8`.
Latest four-arm paired-result reference HEAD: `631f211bc`.
The earlier common-task result remains at `70869e84bb6f7023c03354a322e3e70abf393a38`.
Check live Git/process state on arrival;
do not infer running jobs from historical conversation progress reports.

## Objective and implementation

Prove complete MoE output robustness even when perturbations change routes,
using the relationships among input, router, legal guards, and expert outputs.
Demonstrate the contribution of retained relationships beyond path enumeration
or a stronger downstream backend.

The production API is `verify_staged_linf` in
`act/pipeline/moe/staged_verifier.py`. V1 supports eval-mode CPU/float64,
output-level selected-softmax top-2. Exact route analysis precedes guarded
expert-wise Tier 1 and conditional property-directed F0 McCormick fallback.
The normalized top-k proof is broader than this public implementation contract.
Read `act/pipeline/moe/docs/staged_verifier.md` and
`act/pipeline/moe/audit_staged_evidence.py` for the evidence/acceptance contract.

## Current audited facts

- Original boundary confirmatory: 36 route-changing SAFE, with 5 from Tier 1
  and 31 from F0. Original overall solved 56/100 failed its 60% gate. Closure
  is separate; do not backfill that failure.
- Earlier seed-1/2 boundary replication: 13/40 and 6/40 route-changing SAFE;
  the full registered conjunction passed 0/2. These use model-specific radii.
- New common task: the same 100 jointly clean-correct images and `2/255` for
  all three bal010 runs. SAFE/UNSAFE/UNKNOWN/TIMEOUT are 30/24/39/7,
  26/21/38/15, and 44/22/25/9. Route-changing SAFE: 8/8/7. Complete outcomes:
  54%/47%/66%. Full bundles pass 2/3; seed 1 misses the frozen 50% coverage
  threshold. The all-model conjunction is false. Audit: zero issues, 297
  complete packages, 67 replayed UNSAFE. Three hard deadlines explain the
  missing full packages; denominators remain 100 each.
- Common-task F0 SAFE additions: 12/7/15; complete resolutions 27/62, 19/50,
  31/53. These are within-run stage contributions, not a full-budget Tier-1-only
  ablation. Candidate reduction, width, and guard gates pass for each model.
- The models have about 48% clean accuracy and share one architecture/recipe.
  Common-task coverage is not certified accuracy, cross-architecture evidence,
  or a new matched baseline competition.
- Accepted AdvMoE compatibility checkpoint: 85.67% clean accuracy. CROWN and
  Lagrangian experiments supply numerical filters, not strict certificates.
  The frozen strict PyRAT pilot has two requests/four static-path queries, all
  TIMEOUT, strict SAFE 0/2. High-accuracy real-scale strict new certificates
  remain unachieved. Probe agreement does not prove full-domain export identity.
- Lagrangian development: local bound improvements but identical 2/100 complete
  positive filters and higher cost. Four-cell graph/config check did not show
  that router removal or best-bound retention explains the huge sparse-alpha
  negative bounds. No unique global cause was established.

## Authoritative reading order

CURRENT FOLLOW-ON (2026-09-14, PI approved bounded continuation): request LP
order-only R2 COMPLETED; read `act/pipeline/moe/docs/request_lp_order_results.md`
and `results/request_lp_order_review_20260914_r2.json` under the MoE pipeline.
Same old request/18 obligations, at most2 router LPs+3 residual LPs,600s outer
cap. Exact score-order facts imply dyadic gate envelopes without evaluating
sigmoid. Reuse hash-checked R1 proofs; do not resample or replace the R1 UNKNOWN.
Execution `db0b4b281`,33.13s,exactly5 new LPs. All41 stored exports/duals
rechecked;18/18 positive obligations (15 reused+3 residual),minimum~.18304675.
Status CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING,not full-network proof.
R1 UNKNOWN remains separate. No further refinement/search is queued.
The other, separate stage is COMPLETED; read
`act/pipeline/moe/docs/external_static_pair_results.md`: old index3000/seed0,
two whole-box static pairs{2,4},{4,5},nine properties each,CPU/float64 plain
CROWN,120s/pair,no retry/tuning. `StaticSelectedSoftmaxPair` preserves actual
variable weights; finite conformance and model/input hashes are checked.
NOT a complete dynamic-MoE benchmark. Execution `7e911dfab`:both pairs9/9
positive numerical margins,minimum LBs3.05086109/3.09949436,all finite probes
match. Re-audit PASS0. No formal CROWN SAFE or performance claim. No dependency
install or sealed search. Both bounded follow-on stages have completed; no
experiment is running or automatically queued. Keep R1 failures immutable.

EXTERNAL FRONTEND R1 completed separately (2026-09-14):
`act/pipeline/moe/docs/external_compatibility_r1.md`,
`act/pipeline/moe/external_compatibility.py`. Three bounded CPU probes on
pinned alpha-beta-CROWN/auto_LiRPA source, existing Python3.11 environment.
Dynamic top2, static variable-weight pair, and relational input parser; no
full BaB/holdout/performance comparison. Read
`act/pipeline/moe/docs/external_compatibility_results.md`: dynamic TopK/OneHot
rejected; static variable-weight pair returns numerical bounds; full API box
accepted but relational input halfspace rejected. Provenance audit PASS0;
execution `84352b890`. No install. No request-LP or compatibility job remains
running at R1 completion. Full-model external competition remains open; the
later order-only R2 above supersedes R1's absence of conditional positive
request evidence. These controls do not authorize expanded searches.

REQUEST LP R1 completed (2026-09-14): separate frozen index3000/seed0
control in `act/pipeline/moe/docs/request_lp_r1.md`, `request_lp_control.py`
and `check_request_lp.py`. Sparse supplied-HZ LP dual checks and exhaustive
output aggregation are implemented. Read `docs/request_lp_results.md` under
the MoE pipeline and `results/request_lp_review_20260914_r1.json`: all36
supplied LPs rechecked,18 output obligations,15 positive through scoped reuse,
three negative residual F0 lower bounds. Overall UNKNOWN, no missing obligation,
not a complete positive certificate. Execution58.63s at `5c98e5399`.
No production SAFE gate,
budget or sigmoid range is changed. External compatibility remains separate.

LATEST COMPLETION (relation R1,2026-09-14): all6 smoke +60 full calls finished;
separate re-audit exactly matches saved audits. Read
`act/pipeline/moe/docs/relation_ablation_results.md` and
`act/pipeline/moe/results/relation_ablation_review_20260914_r1.json`.
48 packages,16 UNSAFE replays,12 retained outer TIMEOUTs,30/30 common facts
equal. Shared vs independent SAFE4/4,3/2,4/2 (ten/model): +3 SAFE,+5 solved,
no losses. Two gains are multi-pair versus relaxation UNKNOWN; one is single-
pair versus solver limit.27 recorded gate pairs agree,3 one-sided,not universal
observability. This is observed-input mechanism evidence, not new confirmation.
This relation stage is complete. The separate LP and external R1 completions
above supersede its former next-step notes. No dependency installation, sample
expansion or sealed-search reopening is authorized.

HISTORICAL REGISTRATION (2026-09-14, now completed): relationship-only R1 is
implemented and preregistered in `act/pipeline/moe/docs/relation_ablation_r1.md`.
The independent arm duplicates all guarded expert factors/constraints into a
Cartesian product; it retains scoped reuse, gates, Tier1 and25% scheduling.
Read `relation_ablation.py` and its fixed config before execution. Six old-input
smoke calls gate60 calls on the FIRST10 already-observed thirty-input R2 images,
not the new100. Single CPU timing worker; no resume/overwrite or effect-size
smoke gate. The pipeline writes final structural audits but does NOT commit or
push results automatically. Check process/runtime/log files for actual launch
state; implementation alone is not an experiment result. External compatibility
and complete-request LP checking remain separate, not delivered by this stage.

LATEST COMPLETION (2026-09-14): hundred-input confirmation is complete and
independently re-audited, NOT pending. Read
`act/pipeline/moe/docs/schedule_confirmation_100_results.md` and
`act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json`.
All900 full requests, PASS/0 issues, 739 packages, 198 concrete UNSAFE replays,
161 outer TIMEOUT terminals retained. All300 common-fact pairs agree; all600
scheduled snapshots survive, including67 after kills. Separate review exactly
reconstructs both saved smoke/full summaries and the frozen execution source.
Adaptive SAFE/solved 59/89,57/81,63/86; matched50/76,47/65,59/78;
legacy46/68,45/63,50/68 (100/model/arm). Primary +23 SAFE/+37 solved, no
losses. ALL23 SAFE gains have multiple exact legal pairs: 2 Tier1,21 F0;
all21 F0 gains record scoped reuse. This is source accounting, not reuse-off
causal ablation. Legacy +40/-2 SAFE, net38, not set dominance. Both losses
are single-pair UNKNOWN_MONOLITHIC_SOLVER_LIMIT, indices4150/seed0 and4142/seed1.
Primary clustered SAFE difference +7.67pp [4.67,11.00], descriptive/unadjusted.
These results confirm scoped new-endpoint internal net benefit, not merely
development potential; no high-accuracy, cross-architecture, independent
external-tool or complete floating-point proof claim. Old2/3 failure and
thirty-input result remain separate. 54 focused tests pass; review --check
reconstructs the archived supplement. No training or optimization was rerun.

ARCHIVAL WORKSTREAM CONTRACT (R1 stages above now completed): stop same-family
sample expansion and keep25% frozen. Read separate
workstream contracts in `act/pipeline/moe/docs/post_confirmation_workstreams.md`:
relation-only sound outer-envelope ablation, one external semantic-compatibility
path, and request-level LP obligation checking. None was implemented/launched by
the archival stage; the later R1 entry above supersedes its implementation state.
None may be folded into the completed900 or replace ACT/HybridZ. No jobs remained
running at archival completion. The following
registration/launch entries are HISTORICAL; they do not override this completion.

LATEST NEXT EXECUTION (2026-09-12): PI explicitly approved SEPARATE 100 new
inputs. Read `act/pipeline/moe/docs/schedule_confirmation_100_r1.md` first.
Selection 4088--4389, 529 excluded indices / 1,847 hash-bound artifacts,
clean-only independent reconstruction PASS, zero overlap with prior thirty.
`schedule_confirmation --config act/pipeline/moe/configs/schedule_confirmation_100_r1.json --pipeline`
runs nine fresh OLD index3000 smoke requests, re-audits/gates, then900 full.
Same frozen three methods/25%/models/2/255/300sec, no math or solver changes.
Primary matched, secondary legacy, per-model SAFE/solved gains/losses; 100-input
clustered descriptive intervals (not300 independent pairs). No pooling or
retuning. Shared lock, fail-stop, snapshots survive kills, no resume/replacement.
Target tmux `moe-schedule-confirm-100-r1`, log
`data/moe/results/schedule_confirmation_100_pipeline_20260912_r1.log`.
Inspect live state; code registration is not launch or completion. Source
checkout must stay clean/frozen while running. Final audit automatic; outcome
commit/push remains a separate reviewed step. External tools, LP request proofs
and candidate-superset fallback are NOT included.
Validation: 49 focused tests pass; separate process reconstructs100 selection
and reproduces both old R2 smoke/full summaries exactly after the extension.
Thirty-input results already committed/pushed at `cd32685ad`. The100 protocol
and new selection are a separate preparation commit before any new endpoints.

LATEST COMPLETION: 30-input R2 finished 270/270; separate-process re-audit
exactly reproduces both smoke/full summaries. Full PASS, 0 issues, 190 packages,
87 UNSAFE replays, 80 retained TIMEOUTs; all 90 common-fact pairs equal.
Read `act/pipeline/moe/results/schedule_confirmation_review_20260912_r2.json`
and completed section of `act/pipeline/moe/docs/schedule_confirmation_r1.md`.
Adaptive SAFE/solved 11/22, 11/25, 11/25 (30/model); matched 11/18, 9/18,
10/17; legacy 9/16, 8/19, 7/14. Primary SAFE gains 0/2/1, no losses,
all three gains multi-legal-route. Clustered SAFE interval [0,0.0889] includes
zero. Legacy net +9 SAFE includes one loss. No general superiority claim.
PI authorizes a SEPARATE 100-new-input, 900-request experiment with unchanged
models/strategy/radius/budgets, excluding this 30 and all prior recorded HZ
endpoints. Freeze/audit selection, tests, commit/push, fresh old-input smoke,
then full. No pooling, retuning or performance-based stopping. External tools,
LP request proofs and candidate-superset fallback stay separate. No previous
job remains active at this completion; older launch notes below are historical.

R2 REPAIR BEFORE CONFIRMATION: R1 smoke stopped at its first OLD input 3000
with a represented-input identity mismatch. The selector scaled ToTensor in
float32 then cast; CLI initializes float64 before ToTensor. No new confirmation
endpoint ran. Preserve R1 failure and original selection/audit. The selector
now follows CLI initialization. R2 clean-only re-audit passes, with ALL thirty
indices unchanged, same exclusion union, and smoke request identity matching
the retained actual worker package. Read the R2 section of
`act/pipeline/moe/docs/schedule_confirmation_r1.md` and
`results/schedule_confirmation_smoke_dtype_repair_20260912_r1.json` under the
pipeline. Default config/selection now use `_r2.json`; all solver method configs,
budgets, numerical gates and analysis stay fixed. Repeat all nine old-input
smoke requests before new endpoints, at new `_20260912_r2` roots. Planned tmux
`moe-schedule-confirm-r2`, log `schedule_confirmation_pipeline_20260912_r2.log`.
Inspect live state; no code edits during execution. R1 launch notes below are
historical. Do not count the failed row or its positive package as smoke PASS.

LATEST REGISTERED CONFIRMATION (2026-09-12): PI chose 30 new inputs. Read
`act/pipeline/moe/docs/schedule_confirmation_r1.md`. Selection is frozen and
separately reconstructed: indices 4006--4086, excludes 442 earlier indices from
1,643 hash-bound artifacts; same three models, CPU/float64 batch-one clean
semantics, epsilon 2/255. No new verification endpoint used for selection.
`schedule_confirmation.py --pipeline` (act-py312) runs nine **old-input** smoke
requests, re-audits/gates them, then 270 new requests across adaptive, matched
monolithic (primary), and old strong monolithic (secondary). Each has a hard
300-second cap. Snapshot audit works after outer kills, bound to frozen request
identity; legacy has no common-prelude overhead. No resume or replacements.
No strategy, 25% fraction, numerical gates, default or prior result changed.
Planned tmux `moe-schedule-confirm-r1`, log
`data/moe/results/schedule_confirmation_pipeline_20260912_r1.log`; inspect live
state, registration alone is not launch/completion. Do not edit this checkout
during execution. After final audit, independently review, archive compact
results and update/commit/push. No automatic Git writes in the runner.

LATEST PREPARATION (2026-09-12): read
`act/pipeline/moe/docs/route_complexity_confirmation_preparation.md` first.
Common-fact snapshots persist before arm-specific solves, with atomic no-clobber
publication and hash/scope/interval checks. A real SIGKILL test retains the
snapshot without a final package. Opt-in only; no historical run changed.
The old strong configuration is preserved as `configs/monolithic_legacy_reference_v1.json`
under the pipeline. Read-only `analyze_schedule_complementarity.py` produces
`results/schedule_complementarity_phase_review_20260912_r1.json`: 50/53
monolithic-only SAFE are single-pair, 17/19 staged-only are multi-pair. Twelve
`UNKNOWN_SOLVER_LIMIT` labels mean Tier-1 `violation_region_undecided`, not
candidate failure; original data remain intact.
No new endpoint is running. Next: settle 30 versus 100 new-input scope, audit
the exclusion inventory, freeze selection/statistics, implement a separate
three-arm runner/auditor including timeout snapshots and old strong reference,
test/commit/push, then observed-input smoke before confirmation. Do not call
this preparation a frozen or launched confirmation experiment.

LATEST COMPLETION: route-complexity smoke (6) and full pairing (60) finished.
Separate re-audit reproduces both saved summaries: full PASS / 0 issues,
49 packages, 13 UNSAFE replays, 11 outer deadlines retained. Adaptive versus
matched monolithic SAFE/solved counts (denominator ten per arm/model): 7/8 vs
6/7, 3/7 vs 2/5, 4/7 vs 3/4. Adaptive gains one multi-legal-route SAFE per
model, with no SAFE/solved losses. All 22 comparable common-fact pairs match;
8 are unavailable, not equal. Mean savings are tail-sensitive; median paired
differences are under 0.26 s. This is observed-cohort engineering, not a new
holdout or general superiority result. Read
`act/pipeline/moe/results/route_complexity_paired_review_20260911_r1.json` and
the completed-result section of `act/pipeline/moe/docs/route_complexity_schedule.md`.
No experiment remains running or queued by this completion. The following
launch/pending notes are historical. Defaults, numerical gates, old R1s and
sealed searches remain unchanged; further experiments require separate scope.

Next execution frozen (2026-09-11): `configs/route_complexity_paired_r1.json`
and `route_complexity_paired.py` under `act/pipeline/moe` run a six-request
smoke then a 60-request observed-cohort adaptive/matched-monolithic pairing.
Same ranks 0--9, three models, 2/255, 300-second external caps; no new holdout.
Both arms independently compute the common facts; the audit compares actual
intervals, not just fact counts. A complete matching smoke gates the full run;
errors fail-stop and retain artifacts, no resume/overwrite path. 47 focused
tests pass. See `docs/route_complexity_schedule.md` under the pipeline for
registered acceptance and caveats. Planned session `moe-route-complexity-r1`,
log `data/moe/results/route_complexity_pipeline_20260911_r1.log`. Verify live
state: registration does not itself mean execution. Do not edit the checkout
while the pipeline runs. Final audit is automatic, final commit/push is not.
No candidate-superset fallback, external tool or request-level LP stage is
included, and the historical runs remain untouched.

Newest implementation (2026-09-11): opt-in route-complexity scheduling and
matched monolithic scoped reuse are implemented. Read
`act/pipeline/moe/docs/route_complexity_schedule.md`. Configs
`route_complexity_reuse_v1.json` and `monolithic_matched_reuse_v1.json` share
the common guarded interval prelude and total budget; single pair goes direct
weighted, multi-pair adaptive allocates 25% of remaining time to Tier 1 and
then residual F0. Monolithic can discharge the same per-pair/property facts.
41 focused tests pass. Only analytic controls ran; no trained-model scheduling
comparison is started or queued. Native calls need an external watchdog;
the internal budget is cooperative. Defaults and old R1s remain untouched.
Do not conflate the new no-support common fact prelude with the historical
support-tightened Tier-1 source when comparing costs. Candidate-superset
fallback, external-tool integration and request-level LP proof packs remain
separate future stages, not part of this implementation or its validation.

Newest completion: `act/pipeline/moe/docs/proof_reuse_engineering.md` and
`act/pipeline/moe/results/proof_reuse_paired_review_20260911_r1.json` supersede
the launch/pending notes below. All 60 reuse off/on requests finished; separate
re-audit PASS, 0 issues, 60 packages and 12 UNSAFE replays. SAFE off/on counts
are 4/5, 1/1, 3/3 (ten inputs each); sole gain seed0/rank0, no losses. Mean
seconds 60.33/41.77, 106.23/92.05, 89.18/69.30; median paired savings under
one second. This is observed-cohort engineering, not holdout, general speedup,
high-accuracy evidence or independent full-network proof. Actual stored-HZ
export checking has separately passed (see linked result). No experiment is
currently running or automatically queued by this completed stage. Reuse stays
opt-in; mathematical/numerical gates and all historical results are unchanged.
Next work needs its own scope: no implicit enlargement of this sample or
sealed searches, and no inference that independently checked LP bounds certify
upstream network-to-HZ propagation or MILP search trees.

Latest addition first: `act/pipeline/moe/docs/paired_followup.md` and
`act/pipeline/moe/results/paired_followup_full_review_20260911.json` report
the completed four-arm comparison, including monolithic's coverage advantage.

1. `act/pipeline/moe/docs/staged_multimodel_performance_bundle.md`
2. `act/pipeline/moe/results/staged_verifier_multimodel_bundle_20260906_r1.json`
3. `act/pipeline/moe/docs/advmoe_strict_pyrat.md` and its linked R3 result
4. `act/pipeline/moe/docs/lagrangian_top1_guard.md` and linked diagnostics
5. `act/pipeline/moe/docs/monolithic_f0_baseline.md`
6. `paper/evidence_table.md`, `paper/sections/08_evaluation.md`
7. `act/pipeline/moe/EXPERIMENTS.md` for historical chronology

Tracked JSON manifests point to hash-bound raw artifacts under
`data/moe/results`; those and checkpoints are generally not in Git. A fresh
clone alone is insufficient to rerun experiments. Read the exact manifest
before choosing a checkpoint, cohort, or config. Resolve disagreements by
checking raw artifacts and the recorded auditor scope, not by trusting prose.

## Historical review and execution notes (superseded, not the current queue)

1. Build the common-task, equal-total-budget paired follow-up comparing full
   staged, route-invariance plus the same weighted F0/backend, monolithic F0,
   and Tier-1-only with the whole budget available. A stable top-2 set still
   has variable gate weights. Include construction, candidate analysis and
   support in each method's total budget. Charge reused preprocessing explicitly;
   do not give one method free oracle results from the census. Interleave methods
   by input with frozen order balancing. Rerun the full method for paired timing;
   historical times cannot substitute. Report SAFE and UNSAFE separately,
   gained/lost solution sets, timeouts, and per-model results. Follow-up use of
   observed data is explicit; algorithm tuning makes it development data.
2. After establishing the comparison, evaluate scoped Tier-1 proof reuse in
   F0. Only reuse a proven property on a superset of the target pair domain,
   binding model, property, guard containment, frame, and numerical policy.
   Both expert obligations must justify skipping a mixture property. Never
   reuse variable identifiers as if independently propagated frames were shared.
3. First inspect recorded primal/dual progress for possible sign-based early
   completion. Preserve the current optimal-status acceptance gate until a
   separately checked bound contract is implemented and validated. A positive
   solver-reported dual estimate alone does not authorize a numerical-policy
   change. Start independent bound checking with a small LP obligation;
   do not describe it as a complete MILP proof checker.
4. A future external-validity target should be a preregistered different,
   moderate-scale configuration bridging bal010 and AdvMoE. Architecture,
   training and checkpoint selection must precede verification outcomes.
   This remains a proposed next model, not an already selected experiment.

The fair baseline runner/config are now implemented in
`act/pipeline/moe/paired_followup.py` and `configs/paired_followup_r1.json`.
See `act/pipeline/moe/docs/paired_followup.md` for exact semantics. Next execute
and independently audit the mandatory 12-job smoke (one-hour worker-budget
ceiling), then the 1,200-job full observed-cohort follow-up (100-hour ceiling,
plus auditing). All four methods have the same external 300-second cap,
including startup, data/model loading and route analysis. They run sequentially
with rotated ordering and one-thread solver settings. This is not a holdout or
a retrospective timing correction. No new training is authorized by this step.
Executable-source/config identity must match the smoke; documentation-only
commits are allowed. Preserve failed attempts; never overwrite a result root.

Execution started 2026-09-07 21:13 Sydney after implementation `154fce45b`
was tested (20 tests) and pushed. tmux: `moe-paired-followup-r1`; log:
`data/moe/results/paired_followup_pipeline_r1.log`. This is an active
smoke → independent audit → full follow-up → independent audit chain, with
fail-stop shell gates. Check the live log/JSON rather than inferring completion
from this handoff. Do not launch a second copy or change executable sources
while it runs. After completion, check the terminal audit, preserve all rows,
write compact tracked results and update paper/EXPERIMENTS, then commit/push.
This chain does not automatically publish or push experimental outcomes.

2026-09-08 live review: smoke completed 12/12, structural audit PASS with
11 complete packages, zero issues, and zero UNSAFE to replay. Independent
re-audit is tracked at
`act/pipeline/moe/results/paired_followup_smoke_review_20260908.json`.
Full run was live at 636/1200 rows (through rank 52), not complete or finally
audited. The active full command has no `--resume`; its parent shell records
the successful smoke/audit gate. Known defect: a first invocation with
`--resume` and a nonexistent output directory can bypass the smoke gate.
Do not use that path. Fix with regression tests after this frozen run, not by
changing executable sources midway. Smoke PASS is not evidence of superiority.
The four arms are internal ACT algorithm comparisons, not four independent
external tools. External baseline claims supplied in conversation still need
primary-source verification and pinned artifact/semantic registration before
adoption; no external tool installation or execution has been started.

## Frozen decisions and boundaries

Latest completion (2026-09-11): the four-arm full run has finished. Re-audit
PASS, 0 issues, 1,075 complete packages, 171 UNSAFE replays, all 1,200 rows.
Tracked analysis: `act/pipeline/moe/results/paired_followup_full_review_20260911.json`;
full tables: `act/pipeline/moe/docs/paired_followup.md`. Staged SAFE/solved
30/54, 26/47, 44/66 beats invariance and Tier-1-only counts on each model, but
monolithic gives 46/65, 36/54, 52/70. Staged is less expensive under the
registered schedules and has distinct route-changing certificates; it is NOT
the overall coverage winner. Both positive and negative findings are now in
the evaluation text. No MoE comparison process remains running. Fix the known
resume-entry defect next, with regression tests, without rewriting R1 data.
The recovery repair is now implemented: resume requires an existing directory
and runtime identity; full resume must pass smoke audit and code/config checks.
Three new negative-path tests and the complete 26-test focused suite pass.
R1's runtime/source hash and all outputs remain untouched. Next pursue scoped
proof reuse / independently checked bound evidence, keeping optimal-status
acceptance unchanged. No new experiment is currently running or queued by
this completion stage; do not assume automatic follow-on work.

Do not chase seed 1's 50% threshold, add seeds until a pass, overwrite any R1,
reopen Lagrangian/CROWN tuning or its locked holdout, silently expand PyRAT
budgets, or reopen init/census work. F1 remains untriggered. No third census
dataset or ViT-224 end-to-end certification. Contact is managed by PI;
publication accounts, licenses and release permissions are not supplied here.

All HZ SAFE wording is scoped to the recorded HZ/HiGHS acceptance policy.
Evidence-package integrity is not independent re-proving of SAFE. Shared
architecture replications are not stable superiority over competing methods.
The 50% gate is a registered acceptance criterion, not a baseline score.

## Local validation and handoff routine

2026-09-11 next-stage implementation: optional config
`act/pipeline/moe/configs/staged_verifier_proof_reuse_v1.json` enables scoped
Tier-1 **guarded interval property** reuse in F0. It does not reuse partial
MILP proofs (not yet exported) or relax the old acceptance policy. Independent
checking currently covers supplied finite-box LP bounds using rational
arithmetic, including interval margin arithmetic; it does not validate HZ
propagation/LP lowering or MILP trees. Read
`act/pipeline/moe/docs/scoped_f0_proof_reuse.md`. Next run the retained analytic
controls, then freeze a separate observed-cohort engineering comparison before
making any real-model efficiency claim. No old R1 rerun/closure is authorized.

Retained controls completed at `cceadd326`, result
`act/pipeline/moe/results/scoped_proof_reuse_controls_20260911_r1.json`:
reference/reuse SAFE, F0 rows 2/1, both audits 0 issues, exact supplied LP bound
3/4 independently rechecked. Full focused suite: 34 passing tests. Opt-in only;
no real-model comparison or HZ-to-LP proof-export check has been executed yet.

Next stage now registered in `configs/proof_reuse_paired_r1.json` (under
`act/pipeline/moe`): observed common ranks 0--9, three trained models, reuse
off/on, equal 300-second caps, 60 jobs / five-hour maximum worker budget.
Run `hz_lp_real_control` first for seed0/rank0, audit via `audit_hz_lp_real`,
then start `proof_reuse_paired`. Both write new roots. See
`docs/proof_reuse_engineering.md` under the MoE pipeline. Real HZ export checking
is limited to the given stored HZ -> continuous binary relaxation -> rational
LP bound chain; upstream network propagation remains trusted. Do not call this
a high-accuracy strict full-model certificate. Runner has no resume/replacement
path and stops on code drift or execution errors. Check live state before
launching; no implicit permission to overwrite a failed root.

Real HZ export completed at `f5ea8457d` and passed independent audit, zero
issues: `act/pipeline/moe/results/hz_lp_real_export_20260911_r1.json`.
Given-HZ support bound ~9.485071885777709, 3,075 factors / one relaxed binary,
seed0/index3000/pair{4,5}/score4-score0. No full output or network propagation
proof is claimed. Next launch target is tmux `moe-proof-reuse-paired-r1`, log
`data/moe/results/proof_reuse_paired_20260911_r1.log`, running the frozen
60-request comparison with an automatic final audit. Check live state: do not
infer completion from the launch target. It stops on a dirty checkout or source
drift between requests, so do not edit during execution. Commit/push compact
paired results only after audit; there is no automatic result push.

Use `conda run --no-capture-output -n act-py312 python -m unittest` with the
relevant test modules. Last targeted run passed 11 tests in
`act.pipeline.moe.test_freeze_staged_multimodel_bundle` and
`act.pipeline.moe.test_staged_verifier`. `pytest` is not installed in this env;
the evidence-audit tests are inside `test_staged_verifier`, not a standalone
`test_audit_staged_evidence` module.

Keep this file current after each completed stage. Record evidence links,
remaining questions and the exact next action; preserve previous scientific
endpoints. Commit and push completed stages. Only one session should write to
this checkout at a time.
