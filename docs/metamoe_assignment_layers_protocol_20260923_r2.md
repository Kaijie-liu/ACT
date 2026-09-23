# Same-point layer localization R2 — recorder-only repair

R1 froze at `c30de592f`, config SHA256
`d283a7948ac634c54749ec351dce923fc64a5cb597fc559f32395978b9e4e9a5`.
It stopped at the second progress publication: the shared `write()` helper
uses exclusive creation, but the diagnostic reused `layer_progress.json`.
R1 outer ERROR, stderr, receipt and first provisional INPUT observation remain.
No final identity check, output proof or accepted localization came from R1.

R2 leaves that code/config and all frozen scientific sources unchanged. Its
wrapper redirects progress to `layer_progress_{layer_id:03d}.json`. Duplicate
layer identities still fail; other result writes retain exclusive creation.
Three controls cover multiple publications, duplicate refusal, non-progress
files and invalid IDs. R1's six point-evaluation controls are unchanged.

Same old MNIST0, saved assignment, matrix/physical input, expert1, no new seed,
30 s outer / 8 GiB, zero native queries. New execution/root, recorder repair
only; no new propagation algorithm or acceptance tolerance. All earlier failed
costs disclosed separately. Partial records cannot override the outer terminal.
Freeze after controls, commit/push, run once, then saved-array/terminal audit.
