# B2 inventory reception and nonlinear CPU smoke

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
