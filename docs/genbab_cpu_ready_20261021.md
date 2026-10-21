# B2 inventory reception and nonlinear CPU smoke

## Current decision after user correction

R6 **passes the authorized execution smoke gate**: nonlinear branching was
enabled and24,498 nonlinear domains were visited. A numerical positive was an
extra requirement introduced by the implementation, not by the registered goal.
The original FAIL/UNKNOWN artifacts below remain unchanged; they do not block
the approved B1→B2 sequence. R6's bound agrees numerically with the unguarded
minimum `1/2-tanh(1)`. This motivates a guard/representation diagnosis, not a
proof that the guard was ignored or that unlimited time could never help.

The separately authorized [two-control script](../scripts/genbab_guard_semantics.py)
retains the same input, experts, gate operation sequence, constants and tie semantics.
One cumulative≤300s batch allocates≤140s per control including reception/cleanup:
(i) the original AND specification with input splitting, naive branching and
built-in PGD before verification; (ii) the equivalent ReLU maximum of the three
unsafe atoms with the prior nonlinear BaB configuration. Strict maximum>0 is the
safe property, including boundary ties. UNKNOWN is a result, not a readiness
failure. Any claimed attack must violate the full original AND inside the box.
Positive results require successful outer reception; a file surviving timeout
does not replace the terminal. No B1 run overlaps these controls.

The compatibility snapshot remains explicitly modified. Read-only comparison
with upstream release commit`b58d83ccb32469fc7ae9304fa1d33568ea76b00e`
still finds the three depth/ambiguous-ReLU all-split conditions; no directly
applicable upstream fix was identified, so this batch does not upgrade versions.

## Historical protocol and unchanged attempt records

This is the user-authorized repair of the existing isolated installation, not a
second deployment. The failed R1 inventory audit is preserved. Transitive
distribution metadata is now a list of name/version/location records; actual
torch, auto_LiRPA, verifier API, packaging and wheel paths are recorded separately.
No dependencies or files in the pinned author checkout are changed; the later
explicit compatibility snapshot is separate and fully bound below.

The fixed synthetic source is `x in [-1,1]`, router `(x,-x,-1/2)`, pair `{0,1}`,
experts `E0=1/2-x`, `E1=1/2+x`. The graph retains `sigmoid(2x)` and the product.
Its weighted scalar output is `F=1/2-x*tanh(x)`. The pair guard is `|x|<=1/2`,
where `F>=1/4`; the unguarded box contains negative values. The unsafe formula is
the conjunction `F<=0`, `r2-r0<=0`, `r2-r1<=0`, including equality ties.

The committed script fixes float64 CPU alpha-CROWN plus nonlinear BaB, uniform
binary nonlinear branching and no input-split/attack substitution. Acceptance
requires a numerical positive, nonzero native visited-domain count, branching
point logs and a graph linking split candidates to sigmoid/multiplication.
An incomplete-only positive does not cover the smoke; do not tune the example
after observing that outcome. Ordinary metadata/path/serialization failures may
be repaired under the approved policy, retaining attempts and costs.

The existing owned-session supervisor charges imports, construction, solve,
reception and cleanup within one 300-second control budget. Each managed process
has 8GiB address-space limit; RSS and wait4 usage are telemetry, not equivalent
group-memory limits. New control artifacts stay within 128MiB. Native output is
saved to a file, not silently truncated to the supervisor's 16KiB console.
The interpreter-start interval before the controller's first timestamp is not
claimed as measured full cold start. No real request, model checkpoint or GPU.

Implementation: `scripts/genbab_cpu_ready.py`; ordinary controls:
`tests/test_genbab_cpu_ready.py`. This is baseline compatibility work, not a new
ACT algorithm, benchmark win or independently checked network certificate.

## Compatibility repair observed before the completed smoke

`genbab_cpu_ready_20261021_r2` imported and checked the installation, then actually
branched nonlinear domains. At the second BaB round, the pinned backend's fused
decision-precompute extraction attempted to mask a `[1,1]` intermediate tensor
with a length-2 mask. The 6.783214s attempt and original log remain unchanged.
The first attempted configuration repair (`branching_decision_precompute=false`)
did not help: R3 failed after 5.509465s at the same line. Further source inspection
shows `stage_solve.py` hardcodes precompute on, ignoring that public switch.
R2 and R3 remain failures. The directly implicated pruning path is instead
disabled via `bab.pruning_in_iteration=false`: optimization retains the full
batch before the extractor applies its mask. No domain or check is discarded.
The model, specification, nonlinear branching method, dtype and acceptance stay
unchanged. This is an explicit compatibility configuration, not a performance
search. Each retry uses the remaining control budget, not another 300 seconds.

R4 removed that shape exception and reached six visited domains, but returned
UNKNOWN after 6.807640s. Its original FAIL is retained. Source inspection locates
a separate ReLU-style `depth == tot_ambi_nodes` exhaustion rule in `stage_solve`
and the corresponding decision mask in `decision_precompute`; a third location
would initialize a leaf LP. A sigmoid/product interval is not made exactly
linear merely by reaching that depth; the log already repeats the same split
node. The isolated compatibility snapshot disables these depth-count shortcuts
only for the configured nonlinear method. Ordinary ReLU behavior, domain
coverage, bound acceptance and hard timeout are unchanged; no leaf is declared
SAFE by the patch. All three exact textual edits and every snapshot file are
checked against the pinned author commit. The author checkout and environment
remain untouched. This must be called a compatibility-adapted GenBaB path.

The first snapshot precheck (requested R5 directory, not created) encountered a
tracked directory symlink and incorrectly tried to read it as a regular file.
It exited before any native call (shell-reported wall time 0.014745s). The
inventory now compares symbolic-link text separately; imported auto_LiRPA still
resolves to the original pinned checkout. This is tooling metadata, not a failed
network obligation or a second deployment.

## Saved results and decision

[Records](genbab_cpu_ready_records_20261021.json) bind all four native attempts.
The installed-distribution and actual-import checks pass. The numerical smoke
does **not** pass its fixed positive-example criterion; all original FAIL
terminals remain FAIL. Saved-only structural audits pass for the failure records.

| Attempt | Outer measured seconds | Verifier outcome | Actual domains visited |
| --- | ---: | --- | ---: |
| R2 | 6.783214 | extractor shape exception | 2 |
| R3 | 5.509465 | same exception; switch ignored upstream | 2 |
| R4 | 6.807640 | UNKNOWN at depth-count stop | 6 |
| R6, isolated depth compatibility | 271.159864 | UNKNOWN at remaining budget | 24,498 |

R6 returned normally, with no execution/cleanup error, and retained the analytic
guarded source unchanged. Its last global numerical lower bound is
`-0.2615941559557649`; this is not a feasible upper bound or a counterexample.
The log also repeats effectively zero-width branching intervals with no global
bound improvement. That observation is not proof of a unique root cause or of
the original model being unsafe. Do not tune this case or extend its budget.

Cumulative measured controlled execution, including the failed symlink precheck,
is **290.274928s**; the calendar span including source repairs/commits is
**645.282470s**, not a contiguous 300-second end-to-end batch. Each native attempt
uses its remaining share of the control allowance. Offline saved-only audit
commands took another 0.753172s and do not repair an online verdict. Source
snapshot extraction took a separately observed 0.791677s. Interpreter startup
before each first timestamp remains outside the measured scope as disclosed.
The four run directories contain 16,944,418 bytes; the compact archive is charged
additionally and the total remains below 128MiB. The 111MiB source snapshot is
deployment/compatibility storage, not request evidence.

R6 complete-sample session-plus-controller RSS reached 4,785,143,808 bytes;
wait4 direct-child maximum was 4,728,172,544 bytes. Neither is an aggregate hard
memory limit. Per-process 8GiB address-space limits remained in force under the
approved development policy. Cleanup confirmed no live owned session members;
it does not claim all zombies or unrelated processes were reaped.

**Research ledger:** zero real requests, real obligations or real attribution
queries; four synthetic verifier calls, zero positive smoke results. This
closes the metadata ambiguity and establishes actual nonlinear execution,
not successful guarded verification, an ACT gain or external superiority.
The approved B1 attribution work remains necessary. The historical numerical
positive criterion failed; the authorized execution gate is satisfied as corrected above.
