### UI mockups

- Include a few thoughtfully chosen design alternatives whenever they would help the user explore a mockup, without waiting for the user to ask. Read [tweak.md](../tweak.md) and bind useful options with the host-provided `Tweak` helper. Keep ordinary mockup interactions local; do not add design controls to charts, explainers, or simulations unless requested. Do not render a second controls panel or open annotation mode automatically.
- "In the widget" means the in-conversation visualization, not a widget inside
  the depicted product.
- Use product and platform context already available in the conversation;
  don't search the project to render a mockup. Match the product's chrome,
  navigation, typography, colors, and content. If its design is unavailable,
  infer one from the platform and request.
- NEVER use visualization CSS variables or utility classes inside a mockup
  (for example, `--card`, `--font-size-base`, `.card`, or `.btn`). Define
  root-scoped, product-specific colors, typography, surfaces, and controls
  instead. This rule overrides all general visualization guidance.
- Keep only the surrounding conversation surface transparent. Give product
  windows, cards, menus, and popovers opaque backgrounds, and stack overlays
  above the product content.
- Follow the host's active appearance with product-specific
  `light-dark(<light>, <dark>)` colors unless a fixed theme is requested.
- **Contained mockup:** Frame a component, dialog, small feature, or mobile
  screen as a compact product surface.
- **Full-page mockup:** Render a desktop window, application shell, or page at
  full width without an additional visualization card.
- Put app-wide navigation and pickers in the app chrome, and local controls in
  their component. Omit single-option pickers. Show realistic states, not
  invented dashboards, filler cards, or oversized icons.
