# Pinned external frontend compatibility R1

Execution `84352b890`, protocol `external_compatibility_r1.md`. Three fixed
CPU probes completed without dependency changes. All actual auto_LiRPA imports
resolve to the pinned source submodule, NOT the separately installed wheel.
Python3.11.16/Torch2.11.0+cu130,CPU/float32; alpha-beta-CROWN
`e5c7e17bf0488843acb77b7519f59876717a49f4`,auto_LiRPA
`5a098e8f9fb5786a428a024981d833d303921f2d`. External trees remain clean.

| Task level | Observed result | Permitted interpretation |
|---|---|---|
| Literal dynamic weighted top2,E3 affine toy | Conversion rejects TopK and exporter-generated OneHot | This pinned frontend/expression is not currently a direct full-model baseline |
| Fixed pair{0,1},variable softmax weights,both expert outputs | Plain CROWN returns[0.89551866,1.10547280];25 finite probes agree exactly with original module | Static whole-box sufficient obligation is consumable numerically |
| API input box plus x0+x1<=.1 | Box control accepted; relational parse raises ValueError:single-variable constraints only | This API entry does not consume that general input halfspace |

Do NOT freeze gate weights or export only the clean route and call the static
case the same full MoE problem. PyTorch literal tie selection also does not
by itself cover ANY_LEGAL_TOPK. Probe agreement is finite, not a full-domain
export-equivalence proof; CROWN values here are numerical filters,not formal
SAFE. No complete BaB, original trained MoE or F0-relaxation competition was
run. These failures are frontend/configuration observations, not general
impossibility claims about alpha-beta-CROWN or indications of model unsafety.

Source anchors (hashes in compact result):

- `auto_LiRPA/.../operators/indexing.py`,BoundGatherElements,lines216–248:
  its interval implementation requires an unperturbed index; its existence is
  not evidence of general perturbed dispatch support.
- `auto_LiRPA/.../bound_op_map.py` and `operators/softmax.py`: runtime graph
  log supplies the observed unsupported TopK/OneHot; static graph retains the
  nonlinear weighting. Presence of an operator class alone is not a benchmark.
- `complete_verifier/api.py`,lines981–1012:_parse_input_bounds rejects
  coefficients involving more than one input variable. This does not audit
  every possible internal constrained backend entrypoint.

`review_external_compatibility.py` independently checks recorded identities,
all raw hashes,pinned source/worker hashes,import paths and finite-probe/bound
consistency. It does not rerun queries or prove bound arithmetic. Compact record:
`../results/external_compatibility_review_20260914_r1.json`,PASS,zero issues.
Next possible integration is explicitly a static-obligation backend adapter;
equivalent full-MoE comparison remains open and needs a separately justified
encoding. No additional search is inferred from this compatibility result.
