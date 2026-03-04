# UI UX Pro Max recommendation used for website rebuild

Generated from:
`python3 /tmp/ux-skill/.claude/skills/ui-ux-pro-max/scripts/search.py "battery health dashboard developer tools AI operations" --design-system -p "Battery Health Skills" -f markdown`

## Selected outputs

### Pattern
- Name: Feature-Rich Showcase
- CTA placement: Above fold
- Sections: Hero, Features, CTA

### Style
- Name: Vibrant & Block-based
- Keywords: bold, energetic, playful, block layout, geometric shapes, high contrast
- Performance: Good
- Accessibility: Ensure WCAG

### Colors
- Primary: `#1E293B`
- Secondary: `#334155`
- CTA: `#22C55E`
- Background: `#0F172A`
- Text: `#F8FAFC`

### Typography
- Headings: `JetBrains Mono`
- Body: `IBM Plex Sans`
- CSS import:
```css
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
```

### Key effects
- large sections with 48px+ gaps
- animated patterns
- bold hover with color shift
- scroll-snap
- large type (32px+)
- transition 200-300ms

### Anti-patterns
- flat design without depth
- text-heavy pages

### Pre-delivery checklist
- No emoji icons
- `cursor: pointer` for clickable
- visible focus states
- respect `prefers-reduced-motion`
- responsive targets: 375px, 768px, 1024px, 1440px
