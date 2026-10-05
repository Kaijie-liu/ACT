# C15 actual unit-splice component audit — 2026-09-10

PROGRESS: a new exact nonconvex HZ component was constructed and independently
proved, not merely censused. Goal remains ACTIVE. Formal1870/2413 and separate
E061/400 are unchanged; new CERT0, new validated ADV0, formal gain0. There was
no whole-live, terminal, family, E0 retention or full2413 replay in this run.

## Terminal evidence of this one component run

Exclusive results/c15_unit_row_splice_20260910_v1/. Inherited611 plus33 new
tests:644 passed in6.70s. Tests and worker exit0; supervisor42.998850611s,
worker33.964851513s; no source/provenance drift. One target execution, no retry,
no cap/window/default change. Both complete sealed source checkpoint dictionaries
remained reachable. Production SHA15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75
on redu-hz/f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac remains unchanged.

Constructed all268 independently selected unit row pairs; the complete sealed
C14 table was checked only AFTER construction as an external oracle. No table,
target column/id, public label or solver status entered the representation rule.
268 defining equalities and536 coefficients were actually removed. Raw global
continuous width255298 is deliberately unchanged; ordinary lowering drops the
268 newly unused continuous columns:154362 ->154094. All1350 binary factors
remain. Predicate matrix nnz10960724 ->10960188. Equality143937 ->143669,
inequality2700 unchanged. Original value/input maps and shared frame unchanged.

The independent Fraction proof checks all268 changed continuous rows,1248664
continuous coefficients and146101 unaffected predicate rows, plus all binary
payload and RHS identities. Its changed continuous coefficient count excludes
the199 unchanged binary coefficients included in the builder's1248863 affected
total-row terms; these are different denominators, not missing proof terms.
Every removed variable's box is proved redundant; the original definition is
reconstructible from the surviving row prefix and compact certificate. No old
wide definition buffer is retained in the returned component.

## Storage, work and fidelity — small gain, not a large-network breakthrough

Certificate:268 int32 columns,268 uint64 descriptors,69 nonzero float64 offsets:
3768 numeric bytes and605 numeric entries. Complete component comparison:

| Metric | Original | Spliced including certificate |
| --- | ---: | ---: |
| Unique numeric owner bytes | 134358104 | 134351152 |
| Numeric resident entries | 11147561 | 11147362 |
| Reachable Python shallow payload | 12717 | 13664 |
| Controlled bytes (numeric + Python payload) | 134370821 | 134364816 |

Net numeric saving6952bytes; Python overhead grows947bytes; net controlled
saving6005bytes, only0.0044689762%. Numeric entries decrease199. All preregistered
strict component inequalities pass, but this is NOT a whole-runtime reduction
measurement or evidence of a new solved benchmark. The result is too small by
itself to support the user's desired CIFAR/Tiny capability leap.

Standalone work253357392 <=256000000 (margin2642608): structural95786976,
exact norm19971152, scalar8576, compact emission/seals132008688, affected row
assembly4995452, row metadata586548. Construction16.906767184s; conservative
RSS growth368369664bytes; traced peak376896234+97984metadata;1GiB gate passes.
Python allocator rounding, interpreter/class globals and Torch C++ metadata
remain explicitly excluded from the shallow metric and separately RSS bounded.

Ordinary native lowering/passModel/getLp inspection5.954783889s, SciPy1.17.1 /
HiGHS1.12.0: all10960188 predicate coefficients, all row/column bounds and
integrality preserved; zero different/dropped coefficients. No optimizer or
presolve was called, no terminal verdict or concrete neural witness produced.
This is ingestion fidelity, not solving performance or capability.

## Retained hashes and an instrumentation caveat

- preregistered.json:710fe19c45c1202578214549ba5e6845dd26d301496e136b91812cfdd8717604
- component.json:9d658a04c7e406a9b3ddc87310dca5d8b745788b76dfee5b8d50d5476a854462
- exact_proof.json:2e7b50f68a2265cb0664a514acf57ac2e89ceff2eac68ace88c8a0d5211c353e
- native_ingestion.json:7c9a9194081de0225f000396743277bb3b83446d8ddbf4183b12cac9383224d7
- result.json:1284c9a1c645387d1285f1be39967ed592597995f10c5aac74aed2a2d7e1f9cd
- exit.json:002d9f5b1fac9999484639befa523e59d8e14085fc8627422e5e3a64ca32faff
- isolated spliced_hz.pickle:6b93e5f929be286d762f9c0779c0b8d505440325e9b786f943df73ff0022dc30
  (134617763 file bytes; not a complete runtime checkpoint or RAM metric).
- HZ content:ce63d102a6956a7a1f5bc4d1296a756d46f201c48e52e6557cfd2f01ecbd8898
- certificate seal:a05fe9a9bcef6dfe4a2359dfd2ec3970f4a35d8b9227ee0b34f0d4314280b080

The independent_exact_proof_saved event has elapsed_s7.035646286, which is the
proof-stage duration, not the worker-relative timestamp: the proof dictionary
overwrites that event key. Artifact order, surrounding timestamps, explicit
stage durations and final wall/exit records establish execution. Do not use
that single field as an absolute timeline; do not edit the frozen event/code.
Component native_ingestion_executed=false was saved before ingestion; the
final result and native artifact truthfully record its later execution.

## Next boundary

See C15_GENERATION_INTEGRATION_DESIGN_20260910.md. Appending this component to
C10 costs508107469 >256M and is forbidden. Success permits only generation-time
integration design with coupled work and complete live ownership, not another
solver rerun. The measured6005byte gain is retained as a reusable primitive;
it must not be exaggerated or used to bypass any promotion rung. No jobs remain.
