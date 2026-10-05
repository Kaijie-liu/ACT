---
name: visualize
description: "Create an in-conversation visualization when requested or when a visual materially improves understanding. Do not use for ordinary comparisons, inspections, project edits, or static scientific figures."
---

# Visualize

- A request for a new standalone file, website, app page, component, or other
  project change is not an in-conversation visualization request, even when the
  deliverable contains charts or interactive content.
- A request to preview, explain, or explore a proposed interface in the
  conversation is an in-conversation visualization request.
- Create a visual only when the user needs to see or explore it in the
  conversation and it materially improves the explanation. Do not create an
  inline visual merely because the request involves data, charts, or an
  interactive page.
- Use a normal Markdown table when the user asks for a table; return it directly
  and do not create a visualization file.
- Use Mermaid when labeled nodes and edges fully explain a static structure;
  return a normal fenced Mermaid block and no visualization file. Use HTML for
  dynamics, spatial motion, adjustable inputs, and other visuals.
- Avoid implementation narration. Follow the host's required skill announcements
  and progress-update policy; keep those updates concise.
- In user-facing prose, describe only what the visual helps the user see or
  decide. Keep it concise and do not repeat information already clear from the
  visual. Apart from required announcements or requested technical detail, avoid
  discussing visualization surfaces, widgets, HTML, SVG, scripts, or local files.

## Context compaction

After compaction, reload this entrypoint and the references applicable to the
current output mode. Do not load unrelated modes or previously used modes that
are no longer relevant.

## Inline HTML output contract

### File

- For each new or updated visualization, choose a concise ASCII
  lowercase-hyphenated title and write `<title>.html` in an explicitly writable,
  durable, task-owned location. Prefer the thread-scoped visualization directory
  when it appears in the writable roots. Otherwise, use the task's supplied
  `work/` directory or create an output directory under its authorized
  working directory.
- Never save inline visualization fragments to Library; they are response content, not user-facing file deliverables.
- Never add `sandbox:` links to inline visualization HTML unless the user specifically requests a download.
- Do not choose system temp as a separate fallback. Write access alone does
  not guarantee that the conversation can read the file.
- Use the absolute path on the executor that creates the file. Never assume
  `~/.codex` is writable unless its thread directory appears in the writable
  roots.
- Build the visual in the conversation. Use the open project when the user asks
  for a site, app page, component, or change to existing project files.

### Fragment

- Write only an HTML fragment: no `<!doctype>`, `<html>`, `<head>`, or `<body>`.
- Write literal markup: use `<div class="card">Hi</div>` plus a real newline,
  never `<div class=\"card\">Hi</div>\n`. Never embed the fragment in an inline
  Python, JavaScript, or shell string. Read it back; rewrite literal `\"` or
  `\n`.
- Keep CSS and JavaScript in the fragment only when base classes are
  insufficient. Load static resources only from the CDN allowlist. Never use
  `fetch`, XHR, WebSocket, or other API calls.
- Give the fragment root a unique ID and select it with
  `document.getElementById(...)`. Never derive the root from
  `document.currentScript`; scripts may sit outside the root.
- Keep visualizations under 1 MB. Aggregate, bin, downsample, reduce precision,
  or drop unused fields from large inline datasets.
- Check that JavaScript has no undefined identifiers, every queried element
  exists, and the primary interaction updates the visual. The bundled
  `python3 scripts/render.py <absolute-fragment-path> [<destination>.html] [--serve]`
  can wrap a fragment as standalone HTML or temporarily serve it for browser
  inspection when a preview would help with layout, theme, or runtime behavior.

### Content and response

- Keep the fragment focused on the visualization. Do not include explanatory
  paragraphs, formulas, instructions, or narrative callouts. Include only
  necessary labels, legends, values, and accessible text alternatives.
- Use the normal response flow. Put any necessary concise explanation outside
  the fragment, and add this visualization content reference on its own line
  where the visual should appear, using the absolute executor-side file path:

```text
visualize{"path":"<absolute-path>/<title>.html"}
```

- Add `"mode":"wide"` for a full-screen desktop app mockup, including its
  application shell. For other visuals, add it only when several compact chart
  panels must remain side by side for direct comparison and would be unreadable
  at the normal width. Never widen a single plot, map, grid, diagram, or
  timeline merely because it is dense. Keep contained mockups, dialogs, and
  mobile screens at normal width; stack separate self-contained views
  vertically. Wide visualizations render in an expandable inline surface up to
  1,024px:

```text
visualize{"path":"<absolute-path>/<title>.html","mode":"wide"}
```

- Whenever you create or update an inline visualization, include its content
  reference in that same turn's final response, even when editing an existing
  file or reusing a path shown in an earlier turn.
- The JSON object may also include a `title` when needed.
- Emit only the content reference for the fragment. Never announce it as an
  artifact, website, output, attachment, link, or download, and never add a
  Markdown link to it. Do not append a Markdown table or repeat the visual's
  data; add at most one short conclusion when the user needs an explanation.

### External resources

- The CSP allows only `cdnjs.cloudflare.com`, `esm.sh`, `cdn.jsdelivr.net`,
  `unpkg.com`, `fonts.googleapis.com`, `fonts.gstatic.com`, and
  `fonts.bunny.net`. Other origins are blocked and fail silently.

## Composition

Choose the smallest composition that fits.

- Prefer interaction detail over permanent panels, toolbars, repeated legends,
  or long stacks. Add only requested controls, use one mechanism per state, and
  never invent search, filter, or reset controls.
- Keep filters, selections, and other presentation-only interactions local. For
  drill-down actions that ask Codex to investigate or explain selected data,
  call `await window.openai.sendFollowUpMessage({ prompt, title })`, where the
  optional `title` is a concise confirmation-dialog heading of up to 250
  characters. Include the selected values and requested investigation in the
  prompt, and label the action clearly.
- Show only metrics that explain the requested behavior. Put live values in
  control headers or on the visual before cards. Treat maxima as ceilings, not
  targets. Never invent qualitative scores, status cards, or secondary fact
  grids to fill space.

## Read only the applicable references

- For inline HTML, read [Layout and theme](references/layout-theme.md). Its accessibility and theme requirements apply to every HTML mode.
- For host-provided components, controls, tables, or icons, also read [Components and icons](references/components-icons.md). Mockup-specific styling overrides remain in the mockup guide.
- For UI mockups, read [Mockups](references/mockups.md).
- For graphs or plots, read [Plots](references/plots.md), including the existing responsive, interaction, and theme verification requirements.
- For maps, read [Maps](references/maps.md), including geometry/source and rendered-output checks.
- For explainers, simulations, categorical grids, or part-to-whole layouts, read [Other layouts](references/other-layouts.md).
- Only when exporting or publishing an existing inline visualization, read [Export](references/export.md).
- Plain Markdown tables and static Mermaid responses do not require the HTML references.
