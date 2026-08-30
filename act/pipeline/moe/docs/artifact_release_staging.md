# Artifact release staging

## Current status

The local release inventory is frozen in
`act/pipeline/moe/configs/artifact_release_manifest_draft.json`. It records the
currently available checkpoints, replayed witnesses, dynamic and specialized
ONNX graphs, router census, and paper figures with their paths, hashes, sizes,
and readiness state.

This is **not** yet a Zenodo record. On 2026-08-30 the user authorized a
versioned Zenodo v1 using the currently frozen artifact set, with later B1/B3
artifacts added as new versions. No Zenodo account, token, connector, creator
metadata, or deposit ID is available to this task, so no upload, DOI
reservation, or publication has occurred. The local manifest is therefore
`AUTHORIZED_PENDING_EXTERNAL_CHANNEL`, not a published archive.

## Release boundaries

- ACT source is AGPL-3.0-or-later. Checkpoints, derived ONNX graphs, datasets,
  and reproduction artifacts require a separate license review; the source
  license is not silently assigned to every binary.
- Interim RT-ER epochs and ONNX files are labeled `READY_INTERIM`; they cannot
  replace the final epoch-130 reproduction artifacts.
- Failed and permanently excluded experiment directories are not release
  candidates.
- Raw CIFAR-10 data are not uploaded. Dataset provenance and official archive
  hashes are released instead, subject to the dataset's own terms.
- Private author correspondence and addresses never enter the public archive.
- Every uploaded witness must retain its parent row identity and replay audit.

## Remaining gate before an external deposit

1. Zenodo account/channel and creator/affiliation metadata must be supplied.
2. The v1 selected-file list needs a final local hash audit and archive build.
3. Source uses the repository AGPL-3.0-or-later license; project-generated
   data/figures/witnesses use CC-BY-4.0; checkpoint/model binaries retain an
   explicit per-parent provenance/license note rather than silently inheriting
   one blanket license.
4. B1/B3 remain future archive versions and cannot overwrite v1 identities.
5. Complete witness collections and final ONNX graphs remain gated on replay
   and per-file manifests before their later release version.

The tracked draft removes the risk of discovering missing binary provenance at
submission time while preserving the rule that no external publication occurs
without approval.
