# Pre-registered Survey of Certified Robustness for Routed Models

## Status and purpose

This document freezes the protocol before systematic searching begins. The
survey asks whether formal robustness claims for mixture-of-experts and related
data-dependent routed models are operationally reproducible: can the theorem be
numerically instantiated, can its assumptions be checked on the released
artifact, and are the required model and certificate assets available?

The motivating ICML 2025 RT-ER case was known before preregistration. It is
therefore labeled a **motivating/calibration case**, not a held-out discovery.
All primary proportions will be accompanied by a sensitivity analysis that
excludes it.

The protocol does not cap the corpus at 5--10 papers. Every record satisfying
the frozen criteria is included. This prevents a desired corpus size from
becoming a hidden selection rule.

## Research questions

- **RQ1 — Numerical instantiation.** Is the claimed theorem evaluated with a
  certified radius, certified accuracy, verified property count, or another
  per-model numerical certificate?
- **RQ2 — Bound provenance.** Are all constants and bounds required by the
  theorem computed by a sound method, estimated empirically, or left
  unspecified?
- **RQ3 — Artifact availability.** Are the trained checkpoint and certificate
  implementation publicly available from an official source?
- **RQ4 — Machine-checkable assumptions.** Can every artifact-specific theorem
  premise be decided from released code, parameters, and data?
- **RQ5 — Semantic alignment.** Do the released model's routing, gating, expert
  output, tie, and perturbation semantics match those used by the theorem?
- **RQ6 — Operational outcome.** Under the five-leaf ACT decision tree, is the
  result formally instantiated, benchmark-vacuous, applicable, blocked by an
  assumption, or extended by a route-changing certificate?

## Population and time window

The search covers work published or posted from 2017-01-01 through the frozen
cutoff 2026-08-29. The unit of analysis is a **paper family**: conference,
journal, workshop, and arXiv versions of the same work are merged, with the
latest peer-reviewed version primary and version-specific artifact differences
retained.

Peer-reviewed papers and public preprints are eligible. Publication status is a
stratification variable, not an inclusion shortcut.

## Inclusion criteria

A paper family is included only when all conditions hold:

1. the evaluated model contains input-dependent expert, path, module, or model
   selection, including sparse/hard MoE, soft/dense MoE with expert-specific
   weights, or an explicitly routed composition;
2. the paper claims a formal, certified, verified, or provable robustness/safety
   guarantee under bounded input perturbations;
3. the guarantee applies to the routed/composed model output, not solely to an
   isolated expert or router;
4. a theorem, algorithm, or executable procedure defines the claimed guarantee;
5. full text is publicly obtainable by the cutoff date.

## Exclusion codes

- `E_ATTACK_ONLY`: empirical attacks/adversarial training without a certificate;
- `E_ROUTER_ONLY`: guarantees only router stability, not model-output safety;
- `E_EXPERT_ONLY`: certifies isolated experts without routed composition;
- `E_NO_DYNAMIC_ROUTING`: fixed ensembles or static model combinations;
- `E_NO_BOUNDED_INPUT_PROPERTY`: no bounded-input robustness/safety property;
- `E_POSITION_OR_SURVEY`: no original theorem or certification algorithm;
- `E_DUPLICATE_VERSION`: merged into another paper family;
- `E_NO_FULL_TEXT`: full text unavailable at cutoff;
- `E_OUTSIDE_WINDOW`: outside the frozen date range.

Each excluded full-text candidate receives exactly one primary exclusion code
and optional secondary notes. Records are never silently dropped.

## Search strategy

### Sources

Search all of the following, recording query time, result count, exported IDs,
and source-specific syntax:

- DBLP;
- ACM Digital Library;
- IEEE Xplore;
- PMLR;
- CVF Open Access;
- USENIX;
- SpringerLink;
- OpenReview;
- arXiv;
- Semantic Scholar or OpenAlex as a cross-index completeness check.

One-hop backward and forward snowballing is performed for every included paper.
Snowballed records face the same criteria and are marked by discovery source.

### Frozen concept query

Source syntax may change, but the Boolean concept groups may not:

```text
("mixture of experts" OR "mixture-of-experts" OR MoE
 OR "conditional computation" OR "dynamic routing"
 OR "expert routing" OR "routed model")
AND
(certif* OR verif* OR "formal guarantee" OR "provable robustness"
 OR "robustness bound" OR "certified radius")
AND
(robust* OR adversarial OR perturb* OR safety)
```

Search title, abstract, and keywords where supported. Exact per-source strings
are saved verbatim. Syntax repairs are allowed only when a database rejects the
query; both failed and repaired strings remain in the log.

Known papers mentioned before preregistration serve only as **retrieval
sentinels**. Failure to retrieve a sentinel triggers a documented syntax or
index-coverage audit before screening. Sentinels are not automatically included
and criteria are not changed to force their inclusion.

The disclosed pre-registration sentinels are the already resolved Zhang et al.
ICML 2025 dual-model paper and the user-nominated labels `MetaMoE (SAIV 2026)`,
`Puigcerver et al. 2022 Lipschitz analysis`, and `Kada et al. 2025`. The latter
three are bibliographic leads, not verified identities or claims; exact title,
venue, version family, and eligibility are resolved only after the protocol is
committed.

## Screening and deduplication

1. Deduplicate by DOI, arXiv ID, normalized title, and explicit version links.
2. Title/abstract screening applies the frozen population criteria.
3. Full-text screening assigns inclusion or one exclusion code.
4. Merge paper versions and select the primary bibliographic record.
5. Record every transition in a flow table: retrieved, deduplicated,
   title/abstract excluded, full-text assessed, full-text excluded, included.

Final prevalence claims require two independent screeners. They record decisions
before seeing each other's labels, resolve disagreements with a written
adjudication, and report raw agreement plus Cohen's kappa. If only one screener
is available, the output is explicitly labeled a single-reviewer pilot and no
ecosystem prevalence claim is made.

## Data extraction schema

Every included paper family receives evidence locations (page, section,
theorem, table, repository path, release, or commit) for:

| Dimension | Frozen coding |
|---|---|
| theorem numerically instantiated | `CERTIFICATE_EVALUATED`, `FORMULA_ONLY`, `ATTACK_ONLY`, `AMBIGUOUS` |
| constant/bound protocol | `SOUND`, `EMPIRICAL`, `MIXED`, `UNSPECIFIED`, `NOT_APPLICABLE` |
| official checkpoint | `PUBLIC`, `AUTHOR_PROVIDED`, `DECLINED`, `NO_RESPONSE`, `NOT_CONTACTED`, `NOT_APPLICABLE` |
| certificate code | `PUBLIC_OFFICIAL`, `PAPER_FORMULA_ONLY`, `PARTIAL`, `NOT_FOUND` |
| assumptions machine-decidable | `ALL`, `PARTIAL`, `NONE`, `AMBIGUOUS` plus undecidable premise list |
| artifact/theorem semantics | `CONSISTENT`, `GAP`, `AMBIGUOUS`, `NO_ARTIFACT` plus mismatch list |

Additional fields include model/dataset/norm/radius, router and gate type,
top-k/tie semantics, expert architecture and output semantics, theorem IDs,
claimed status, checkpoint/code URLs and hashes, license, artifact commit,
contact status, and the ACT five-leaf outcome for each constants provider and
output-semantics reading.

`NOT_FOUND` means the protocol searched the official paper, supplement,
repository, releases, README, and linked project pages without locating the
asset by the cutoff. It does not mean the asset cannot exist privately.

## Constant and semantics audit

For each theorem, enumerate every numerical dependency before classifying its
protocol. Sound analytic or optimization upper bounds are distinct from sampled
gradients, attacks, fitted constants, and unreported values. An empirical value
never inherits the word "certified" from the enclosing paper.

Map theorem objects to concrete artifact operations:

- continuous gate weights versus hard dispatch;
- weighted combination versus selected-expert output;
- raw logits versus probabilities or margins;
- top-k ordering and tie behavior;
- input preprocessing and perturbation units;
- global versus local assumptions and the domain over which they hold.

If multiple plausible readings exist, extract each reading separately rather
than silently selecting the favorable one.

## Official artifact audit

Only author/institution repositories, publisher supplements, official project
pages, and author-provided artifacts count as official. Third-party checkpoints
may be listed separately but cannot turn a missing official checkpoint into
`PUBLIC`.

Training/model source and certificate source are separate fields. A public
training repository does not imply a public verifier. Pin branch, commit, URL,
license, release identifiers, checkpoint hashes, and the audit date. External
repositories remain outside ACT and are never copied into ACT when licensing is
absent or incompatible.

## Author-contact policy

Contact occurs only after full-text and official-artifact extraction is frozen
for that paper. Use a public corresponding-author address and one standardized,
neutral request covering only missing checkpoint, certificate code, constant
protocol, or semantic clarification.

- initial message at day 0;
- one follow-up no earlier than day 14;
- close as `NO_RESPONSE` at day 30;
- use `DECLINED` only for an explicit refusal;
- record `NOT_CONTACTED` when no public channel exists or contact was not
  authorized;
- never convert silence into refusal or evidence against correctness.

No message is sent without explicit user authorization. Raw correspondence and
addresses remain in a private, untracked directory under
`/data1/Kane/MOE/ACT/data/moe/survey/private`; committed artifacts contain only
paper ID, dates, status, requested asset classes, and a redacted factual
summary.

## Analysis and reporting

Report corpus counts and proportions for every frozen dimension, stratified by
publication status and year where sample size permits. Since the corpus is the
complete result of the frozen search rather than a random sample of papers,
descriptive proportions are primary; no sampling confidence interval is used
to imply population randomness.

Required outputs:

- search and screening flow;
- included/excluded paper-family manifest with reasons;
- six-dimension evidence matrix;
- per-paper assumption/semantics map;
- asset/contact-status table;
- five-leaf operational outcome table;
- sensitivity analysis excluding the motivating ICML 2025 case;
- a limitations section covering database coverage, terminology drift,
  inaccessible artifacts, reviewer judgment, and version changes.

Wording is artifact-centered:

- "no numerical certificate evaluation was located";
- "the constant-computation protocol is not reported";
- "the released artifact executes hard dispatch while the theorem assumes ...";
- "applicability is not established for this artifact."

Do not write that an author "cannot provide" an artifact, that a certificate is
unsound, or that a theorem is invalid without the stronger evidence required
for those claims.

## Reproducibility, amendments, and execution gate

Machine-readable protocol fields live in
`act/pipeline/moe/configs/certification_gap_survey_protocol.json`. Search exports,
deduplication decisions, evidence rows, contact-status records, and summaries
will be written under `/data1/Kane/MOE/ACT/data/moe/survey` and hashed. Public
evidence rows may be committed; copyrighted papers, private correspondence, and
external repositories are not.

Any post-search protocol change creates a dated amendment with rationale and
reports results under both the original and amended rule where possible. It
never overwrites this preregistration.

This stage freezes the protocol only. Systematic search, author contact,
environment creation, model training, certificate execution, N1/N2, F1, and
baseline verification have not started.

## Execution closure amendment: 2026-08-30

The frozen source-native retrieval could not be completed by the execution
cutoff, and no authorized institutional exports were supplied. The study is
therefore closed as an **audited case series**, not a systematic prevalence
study. This changes reporting scope rather than retroactively changing the
search protocol.

The closed case series contains eight adjudicated included paper families and
48/48 primary-source-located evidence cells. One-hop snowballing produced 13
non-seed candidates and no new included family under the frozen criteria. These
facts support a qualitative statement that operational certificate gaps exist
in the audited cases. They do not support ecosystem prevalence, search recall,
or a claim that no other eligible work exists. The machine-readable closure is
`act/pipeline/moe/results/survey/case_series_closure_20260830.json`.

Author contact was later authorized, and the user reports personally sending
the neutral RT-ER clarification stored at
`act/pipeline/moe/docs/icml2025_rt_er_author_disclosure_draft.md`. The actual
sent date has not yet been recorded; until it is, the Day-14 reminder and
Day-21--30 public-issue windows remain unscheduled. The earlier failed agent
Gmail attempt is not treated as delivery.
