## Exporting an existing visualization

- Keep the fragment as the editable inline source. When the user explicitly asks
  to save, export, or publish a visualization that is already shown in the
  conversation, render it with
  `python3 <skill-root>/scripts/render.py <absolute-fragment-path> <destination>.html`.
- Apply this export flow only when the user explicitly asks to turn the existing
  inline source or visualization into a website. For a general website request,
  build a new responsive site in the output directory or open project, using
  Sites when appropriate, without applying this skill's guidance.
- If the visualization calls `window.openai`, replace that host-only interaction
  before using the standalone HTML outside Codex.
- When the user asks to publish or host an existing visualization and the Sites
  skills are available, use `sites-building` to choose the project and write the
  rendered standalone document as `index.html`, then use `sites-hosting`.
- If Sites is unavailable, offer the standalone HTML without claiming it was
  published.
