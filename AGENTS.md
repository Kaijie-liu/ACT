# MoE research working agreements

Read `docs/CODEX_HANDOFF.md` before starting MoE work. It identifies current
results, frozen decisions, and the next task; follow its evidence links before
making scientific claims. Treat older planning prose as historical when a later
audited result supersedes it.

- Work only on `feat/moe-route-verification`, with writes under `/data1/Kane/MOE`.
- At task start report branch, HEAD, and status. If the starting worktree is
  dirty or on another branch, stop and report; never discard another session's
  changes. Do not run two writers in the same checkout.
- Do not modify, checkout, reset, or push `main`; no PRs or force-push.
- Use the existing `act-py312` environment for ACT work. Do not install or
  upgrade dependencies without explicit authorization for that change.
- Each completed implementation/experiment stage includes relevant tests or
  independent audit, project documentation, commit, and push of this branch.
  Report final HEAD, remote synchronization, and worktree status.
- Preserve frozen results and failed attempts. Use new result directories for
  follow-ups; never change a threshold, denominator, or model after observing
  outcomes to relabel a failure. Do not commit raw data, checkpoints, or external
  repositories.
- All tie-legal unordered top-k sets are obligations. Expert-only violations
  and relaxation violations are UNKNOWN unless a concrete full-model witness
  validates UNSAFE. Incomplete enumeration cannot establish SAFE.
- Distinguish mathematical soundness, solver numerical policy, structural
  evidence audit, and deployed floating-point semantics. Non-outward-rounded
  CROWN positive filters are not formal SAFE; a successful JSON audit is not
  an independently checked proof of every bound.
- Contact with authors is managed by PI. Do not send messages, open disclosure
  issues, publish artifacts, or request contact dates automatically.
- Inspect existing jobs and resource use before launching long work. Never
  interrupt other users' jobs. Do not reopen frozen holdouts or sealed backend
  searches without a new explicitly scoped research decision.
