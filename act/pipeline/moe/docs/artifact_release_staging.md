# Artifact release staging

## Current status

The local release inventory is frozen in
`act/pipeline/moe/configs/artifact_release_manifest_draft.json`. It records the
currently available checkpoints, replayed witnesses, dynamic and specialized
ONNX graphs, router census, and paper figures with their paths, hashes, sizes,
and readiness state.

This is **not** a Zenodo record. No upload, external draft deposition, DOI
reservation, or publication has been authorized or performed. The manifest is
intentionally labeled `DRAFT_INCOMPLETE_DO_NOT_UPLOAD` while B1 and B3 remain
incomplete.

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

1. B1 must finish and its final checkpoints/telemetry must pass the frozen
   audit.
2. B3 and remaining baseline results must be added without overwriting earlier
   artifacts.
3. The complete witness collections must be archived with a per-file manifest.
4. Final ONNX graphs must pass ONNX checker and replay against their parent
   checkpoints.
5. Creators, affiliations, description, keywords, and per-artifact license
   decisions require user approval.
6. A final local hash audit must pass before any upload.
7. Uploading, reserving a DOI, and publishing each require explicit external
   authorization.

The tracked draft removes the risk of discovering missing binary provenance at
submission time while preserving the rule that no external publication occurs
without approval.
