## Layout and accessibility

- Use semantic HTML, keyboard-accessible controls, and concise labels.
- Use `aria-live="polite"` for dynamic results, selections, and simulator
  updates. Use `role="alert"` for validation errors. Do not announce every
  hover or animation frame.
- Keep the top-level surface transparent and unframed, and fill the available
  conversation width. Design for 736px, or 1,024px in wide mode, and support
  widths down to 320px. Stack side-by-side content when it no longer fits.
- At every supported width, text, controls, cards, toolbars, and dynamic content
  must fit without overlap or clipping. Reflow by stacking or wrapping; use
  `.table-responsive` only when table columns cannot fit. The host sizes the
  frame to its content, so avoid fixed outer widths, other horizontal overflow,
  internal scrolling, `position: fixed`, and viewport-height layouts.
- Size every SVG from its actual container. At narrow widths, reduce ticks,
  declutter annotations, and keep visible text at least 11 screen pixels;
  never shrink a fixed-width `viewBox`.
- Keep native tab order; never add `tabindex`.
- Use native `button`, `input`, `select`, and `textarea` elements with matching
  utilities; never recreate controls.
- Keep browser or utility focus styles; never override them.
- On coarse pointers, provide non-overlapping effective targets about 44px by
  44px without breaking 320px layouts; visible icons and marks may stay small.
  Keep fine-pointer controls compact, and let shared utilities own touch sizing
  and at least 16px editable-field text.
- Keep essential content and actions available without hover.

## Typography

- Scale type with `--font-size-base`. Use normal text by default and `.text-small` only for secondary annotations; at the default scale these are 14px and 12px. Never make supporting text smaller than 11px.
- `h1`, `h2`, and `h3` are available; use one concise visible heading for a
  self-contained chart or graph, with short panel headings only when needed.
  Do not restate the prompt or add a redundant title to other visualizations.
- Use only weights `400` and `500`. Never set custom font sizes or line heights.
- Use `.tabular-nums` on changing or aligned numbers. Avoid it for editorial or
  decorative numerals.

## Color

- Make every fill, stroke, text, border, shadow, chart, and canvas color
  theme-aware. Never hardcode light or dark palettes such as white panels,
  off-white backgrounds, black text, slate strokes, or Tailwind color literals.
- Keep text readable against its actual background. Muted or secondary colors
  must retain clear contrast; never use `.text-muted` inside `.card` or another
  filled container unless its background preserves that contrast.
- Available theme variables include `--background`, `--foreground`, `--card`,
  `--card-foreground`, `--popover`, `--popover-foreground`, `--primary`,
  `--primary-foreground`, `--secondary`, `--secondary-foreground`, `--muted`,
  `--muted-foreground`, `--accent`, `--accent-foreground`, `--destructive`,
  `--border`, `--input`, `--ring`, `--blue`, `--orange`, `--green`, `--red`,
  `--purple`, and `--yellow`. Use `currentColor` inside SVG.
- Never add decorative borders, outlines, or strokes to progress tracks,
  meters, bars, stacked segments, or other filled quantitative marks. Use a
  subtle neutral or translucent track and distinguish marks with fill,
  contrast, spacing, or opacity.
- Use `--viz-series-1` for one measure or active state. Use `--viz-series-2`
  through `--viz-series-6` only for important persistent category, series, or
  status identity; never give every peer a different color by default.
  - For categorical tiles or nodes, prefer a soft low-opacity series fill with a
    neutral or transparent border; never color every outline.
  - Keep mappings stable and pair color with labels, shapes, or line styles.
  - Secondary series colors are theme-derived; never assume hues or use them
    decoratively.
- When color encodes a category or series, apply it consistently to the
  corresponding visual marks—not just the legend—and keep large-area fills
  subtle.
- Use series colors only for chart lines, marks, and legend swatches. Keep
  values, axis text, and direct labels in `--foreground` or
  `--muted-foreground`.
- Keep chart grids and inactive structure thin and neutral. Use 1-2px neutral
  structural paths; never thicken, dash, or double-stroke the whole structure.
- In each color pair, the base token is a surface and its `-foreground` token is the content on that surface. Use `.btn-primary` for high-emphasis actions; its neutral fill is supplied by the utility. Use `--primary` and `--primary-foreground` for filled selected, active, or pressed controls. Reserve `--accent` and `--accent-foreground` for subtle interactive surfaces and soft highlights. Buttons with `aria-pressed="true"`, `aria-selected="true"`, or `.is-selected` already use the primary pairing; `.nav-pills .nav-link.active` keeps selection neutral.
