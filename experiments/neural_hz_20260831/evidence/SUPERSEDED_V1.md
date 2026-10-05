# Superseded evidence-ledger v1 records

The two v1 files are retained as successful but provenance-incomplete trial
outputs:

- `cifar100_2024_evidence_baseline_v1.json`, SHA-256
  `7e9107bc9498302a40b288e9a123d2ec38acbbc4cffcb21bf27873e46d753960`;
- `tinyimagenet_2024_evidence_baseline_v1.json`, SHA-256
  `98681b166d607acf45a3c439145bfbd6e501438f7a51f9655158835256794abf`.

They independently replayed the same 25 and 36 witnesses with zero invalid
ADV, but recorded only the main validator source hash and omitted the imported
S-expression evaluator and publication helper hashes. They are therefore not
the E0 authority and must not be used for a promotion or score claim. The v2
files in `SHA256SUMS` repeat the full replay and include the three-file source
closure, Python/platform, NumPy/ORT, and deterministic session configuration.
