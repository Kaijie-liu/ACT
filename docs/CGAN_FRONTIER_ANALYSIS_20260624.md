# cGAN Frontier Analysis 2026-06-24

Scope: strict pure HybridZ.  ORT audits never promote UNKNOWN; relaxation
incumbents are diagnostics unless they are exact-HZ MILP witnesses.

## Current Frozen State

Frozen headline:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/`

`cgan_2023`: `13/21 = 5 CERT + 8 ADV`, `P0=0`.

Unresolved high-priority rows in the frozen frontier:

| iid | model | property | frozen class |
| ---: | --- | --- | --- |
| 8 | `cGAN_imgSz64_nCh_1.onnx` | prop 0 | representation drop |
| 11 | `cGAN_imgSz64_nCh_1.onnx` | prop 3 | representation drop |
| 13 | `cGAN_imgSz64_nCh_3.onnx` | prop 1 | representation drop |
| 15 | `cGAN_imgSz64_nCh_3.onnx` | prop 3 | representation drop |
| 16 | `cGAN_imgSz32_nCh_3_nonlinear_activations.onnx` | prop 0 | representation / sparse-MIP wall |
| 19 | `cGAN_imgSz32_nCh_3_small_transformer.onnx` | prop 0 | transformer MatMul/Softmax wall |
| 20 | `cGAN_imgSz32_nCh_3_small_transformer.onnx` | prop 1 | transformer MatMul/Softmax wall |

The first four rows fail quickly in the dense portfolio with `hz_dropped=True`,
so they are not official-wall solver timeouts.  They are representation/solver
construction targets.

## Evidence

Dense frozen artifact:

`audit_results/hz_cgan_full_driver_mem32_clean_MERGED_20260623/cgan_2023.jsonl`

Sparse structure checks:

`audit_results/hz_cgan_structure_check_20260622/`

64x64 rows build very large sparse exact-HZ states:

| iid | n_cont | n_bin | n_eq | eq_nnz | base feasible |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 247419 | 123708 | 123708 | 25499967 | true |
| 11 | 247459 | 123728 | 123728 | 25327337 | true |
| 13 | 251999 | 125998 | 125998 | 22870240 | true |
| 15 | 249355 | 124676 | 124676 | 22152194 | true |

`iid16` is no longer a pure operator-support problem after sparse TANH/Sigmoid
support:

`audit_results/hz_cgan_sparse_census_dropped_20260621/iid16_comps_witness_summary.json`

Final sparse state: `n_cont=93829`, `n_bin=46913`, `n_eq=46911`,
`eq_nnz=6981528`, base feasible by constructive center.

But query 0 with connected presolve and HiGHS cutoff still timed out at root:

`audit_results/hz_cgan_sparse_census_dropped_20260621/iid16_comps_q0_conn_m60_summary.json`

`iid19/20` expose the transformer path.  Sparse propagation now reaches
variable-variable `MATMUL`, `SOFTMAX`, and a second variable-variable `MATMUL`,
but the current relaxation is too loose for ADV and too large for easy CERT.

Important non-result:

`audit_results/hz_cgan_transformer_probe_20260623/iid19_cuts_q1.json`

This found an HZ-relaxation unsafe incumbent, but ORT replay gave
`real_unsafe=false`.  Therefore it is not a countable ADV under the pure-HZ
rule.

## Operator Audit Added

New local gate:

`scripts/hz_sparse_attention_operator_audit.py`

Artifact:

`audit_results/hz_sparse_attention_operator_audit_20260624/`

This validates the existing sparse attention relaxations on deterministic toy
boxes:

- variable-variable MatMul product-interval HZ lift;
- Softmax interval/simplex/ratio relaxation.

Checks:

- every toy box vertex embeds into the generated HZ constraints;
- reconstructed output matches the true operator at those vertices;
- LP bounds contain independently enumerated exact output ranges.

Result: both operators pass.

During this audit, a shape bug was found and fixed in
`scripts/cifar_sparse_exact_probe.py`: Softmax generated `Auc` rows but an
empty `Aub` with zero rows.  `Aub` now has the same row count as `Auc` and zero
binary columns.  This is a representation-consistency fix, not a relaxation
change.

The sparse HZ object now also checks lightweight shape invariants at
construction time: value rows, equality rows, and upper-constraint rows must
all agree across continuous and binary matrices.  This turns future malformed
operator output into an immediate error instead of a later LP/MILP failure.

Real-network smoke after the fix:

`audit_results/hz_sparse_attention_operator_audit_20260624/cgan_iid19_l45_smoke.json`

This propagated `cgan_2023 iid19` through layer 44, including
`MATMUL -> SOFTMAX -> MATMUL`, with `base_hz_feasible=True`.  The layer-44
state was `n_cont=187848`, `n_bin=11491`, `n_eq=64`, `n_ub=262144`,
`value_nnz=33554432`, and `ub_nnz=524288`.  This is still not a result, but it
confirms the attention-relaxation path and shape checks survive a real cGAN
transformer prefix.

## Next Valid cGAN Work

Do not count current transformer relaxation incumbents as ADV.

Additional sparse-MIP structure audit:

`audit_results/hz_sparse_mip_structure_audit_20260624/`

For `iid18` (`cGAN_imgSz32_nCh_3_upsample`, the medium-priority sparse-MIP
wall), both unsafe rows were audited without running a verifier verdict.  The
sparse exact-HZ path builds a much larger exact MILP than the dense summary
suggests: `n_cont=114081`, `n_bin=57031`, `n_eq=57045`, and about `16.1M`
equality nonzeros.  Exact equality substitution removes `57043` continuous
columns, but the remaining matrix is still one objective-connected component:
`114069` columns, `228142` rows, `57031` integer columns, and about `48.2M`
matrix nonzeros.  Low-cost Fourier-Motzkin candidates after substitution are
small (`6` with pair count <=16, `518` with pair count <=32) and all have
positive local row growth, so a naive generic projection is not an obvious
production path.

This mirrors the TLL structure audit: generic connected-component block
presolve is not enough once the hard sparse exact-HZ MILP is built.

Driver instrumentation update:

`audit_results/hz_cgan_driver_instrument_smoke4_20260624/`

The cGAN full driver now preserves the sparse exact-HZ branch result when the
dense branch remains the UNKNOWN winner. It also reports the row-level
`time_s` and `verify_s` as the whole pure-HZ portfolio wall, while preserving
per-branch timing in `portfolio_stage_wall_s`. A short iid8 smoke with a 62s
wall produced:

- verdict `UNKNOWN`, `P0=false`;
- dense branch `UNKNOWN` in about `12.31s` (`dense_branch_verify_s=10.14`);
- sparse exact-HZ branch `TIMEOUT` in about `48.77s`;
- row `time_s=62.06`, `verify_s=62.06`.

This is only an audit/instrumentation fix; it does not change any CERT/ADV
criterion or cGAN headline count. It makes future frozen artifacts distinguish
quick dense representation drop from time spent in the sparse exact-HZ branch.

64x64 representation audit:

`audit_results/hz_sparse_mip_structure_audit_20260624/cgan_2023_iid8_q0.json`

For `iid8`, sparse exact-HZ propagation succeeds and the base HZ is feasible,
but the exact MILP is already a single objective-connected component before any
solver search:

- before exact substitution: `371127` columns, `371124` rows, `25.99M` nnz;
- equality rows dominate: `123708` equality rows with `25.50M` nnz;
- simple upper-bound rows are small: `247416` rows with `494832` nnz;
- objective support is `512` columns but connects to the whole component.

Exact equality substitution is not a good cGAN compression axis. It removes
`123708` continuous columns, but increases the matrix to `247419` columns,
`494832` rows, and `76.25M` nnz (`2.93x` nnz). Therefore the cGAN sparse branch
intentionally does not enable `--elim-eq-subst`.

ReLU valid cuts are also not a safe cGAN-wide default.  They are sound, but on
this 64x64 row they densify the exact sparse formulation rather than helping
the solver:

`audit_results/hz_sparse_mip_structure_audit_20260624/cgan_2023_iid8_q0_relucuts.json`

- before substitution: nnz grows from `25.99M` to `76.50M`;
- upper-bound rows grow from `247416` rows / `494832` nnz to `494832` rows /
  `51.00M` nnz;
- after equality substitution: the matrix reaches `126.76M` nnz;
- the structure audit alone took about `222s` and `20.8GB` RSS.

So ReLU cuts should stay opt-in for cGAN until there is a more selective rule
that adds only locally useful cuts.  Adding all exact-valid cuts is still exact,
but it makes the hard cGAN representation wall worse.

The best pure-HZ improvement path is:

1. For iid8/11/13/15: make the sparse exact-HZ cGAN path the official cGAN
   branch for these model families, then test whether SCIP/HiGHS can certify
   any full properties under the official wall.
2. For iid16 and iid18: improve formulation-aware sparse root LP/MILP presolve;
   generic connected-component splitting and simple row scaling are not enough.
   Any Schur/block idea must exploit cGAN operator structure, not just graph
   connected components.
3. For iid19/20: keep MatMul/Softmax as sound overapprox for CERT only, but do
   not report ADV unless an exact witness path is added.  The useful next
   operator improvement is tighter sound attention cuts, validated by the new
   toy audit before production.

Any cGAN headline update still requires a one-shot frozen run with full iid
coverage and `P0=0`.
