# Reviewer artifact readiness, not a release claim

This bounded inspection copies the **existing** input98 conditional proof to a
new directory outside ACT and rechecks it; no HZ/LP/proof is generated, and no
model, dataset or solver is loaded. It does not reopen input98 follow-up.
The source is the immutable bundle bound by
[the old portability review](portable_conv_proof_v1_review.json).

## Observed local check

- 13 distributed files including metadata, 7,181,520 bytes total; all identities
  match the old bundle and statement. The inventory is in
  `reviewer_artifact_readiness_20260921.json`.
- New retained location: `/data1/Kane/MOE/review-artifact-20260921-gc83e7zp/bundle`.
  Copy: 0.00388s. Separate `python -I -S` process, startup included: 30.9191s.
- The complete result matches the old bundle's pinned result: 9/9 positive
  obligations, minimum `199593373867685/1125899906842624`.
- Checker audit hooks prohibit outside reads (except interpreter stdlib),
  model/numerical solver imports, subprocesses and network. This is observable
  dependency isolation, not an adversarial-code sandbox or clean OS install.
- No old outcome, production timeout or source-checked negative is relabelled.
  Four earlier rehashed semantic/content rejection controls remain bound by
  the old review; this inspection does not claim to rerun them.

This positive result is still conditional on supplied HZ/source, guard and
route lowering. It does not establish the newer declared-source positive
request or deployed floating-point safety. Its cost is a local recheck, not
a production speedup or a new timing comparison.

## Access and reproduction matrix

| Operation | Available in Git alone? | Required material | Status |
|---|---|---|---|
| Rebuild main result tables | Yes | Standard-library Python and committed reviews | `rebuild_moe_main_tables.py --check`; checks recorded arithmetic, not network bounds |
| Review current and original manuscripts | Yes | Revision inventory, old manifest and Git objects | Full current Markdown inventory; original 52-file snapshot preserved |
| Check real portable conditional proof | No | Entire 13-file directory and separately authenticated bundle/statement hashes | Locally copied and rechecked; distribution not performed |
| Repeat saved-input applicability audit | No | Saved request tensor packages and existing torch decoder | No checkpoint/solver required, but private raw inputs still needed |
| Rerun empirical verification | No | Exact checkpoints, data, configs, numerical environment and runtime identities | Clean install/rerun not performed by this stage |
| Prove new source-complete output positive | No | Same-source enclosure and all positive output obligations | Scientific goal remains open; old positive bundle cannot substitute |

After authorized transfer, copy the *whole* directory, keep its identities out
of band, and run from any working directory:

```sh
python -I -S /COPY/verify.py --bundle-hash 8f35a4ba23b51bdcc829535a47880e5f6158a6fbaaf119b4f7744e0c2278606b --statement-hash 7c31f551137b33257e178c40eeea55bf4b94e3438ae00ddb3ed16e2808e56b00
```

`bundle.json` binds the checker, license and ZIP object store; its own expected
hash and the statement hash must be obtained independently, not accepted from
an untrusted self-description. No server path or model/solver dependency is
needed by that copied checker. Python3.12 from the existing act-py312 environment
was used here; compatibility with a fresh installation is not inferred.

## Still requires a separate PI decision

Reviewer access/public or anonymous release, redistribution rights for real
input/checkpoint artifacts, and a clean-environment empirical reproduction
remain unfinished. A bundled LICENSE file alone does not settle all model/data
redistribution rights. No upload, token use, permission change, dependency
installation or external contact was performed. The package/raw tensors stay
outside Git; only this inventory and compact review receipt are committed.
