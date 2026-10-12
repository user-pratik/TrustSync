---
name: modern-ui-design
description: Use for any work inside web/: building or changing the TrustSync dashboard, components, styling, charts, layout or animation. Contains the design tokens, layout grid, component rules and the visual verification checklist.
---
# TrustSync UI design system
Mood: calm control room. Dark, high contrast, one colour per meaning, generous spacing, numbers in a mono font.
## Tokens (CSS variables in web/src/index.css, mapped in tailwind.config.js)
--bg #0A0E14; --bg-elev #0F1520; --card rgba(255,255,255,0.04); --card-border rgba(255,255,255,0.08); --text #E6EDF3; --muted #8B98A9; --accept #2DD4BF; --reject #F87171; --listen #FBBF24; --info #A78BFA; --idle #64748B.
Radius: cards 16px, chips 999px, inputs 10px. Spacing scale 4 8 12 16 24 32. Shadow 0 8px 30px rgba(0,0,0,.35).
Card: backdrop-blur-xl, 1px border var(--card-border), a soft radial glow in the card's accent colour at 6 percent opacity in the top-left corner.
Fonts: Inter Variable for text and JetBrains Mono for numbers, installed from @fontsource-variable/inter and @fontsource/jetbrains-mono. No CDN links: the demo must work offline.
## Layout
Shell: 72px left rail (lucide icons with tooltips), top bar with a connection chip and a source selector. Main area is a 12-column grid with 24px gaps.
Live page: video hero card 8 columns; right column 4 columns holding the Decision card and the three Trust gauges; below, Sync timeline 8 columns and Event log 4 columns.
Below 768px: the rail becomes a bottom tab bar and everything stacks in one column with the video first.
## Rules
- Colour is semantic only: teal accept, red reject, amber listening, slate idle. Never decorative.
- Every decision change animates for 200 ms (ease-out, scale and fade, framer-motion). Respect prefers-reduced-motion.
- Every data component has three states: loading skeleton, empty state with a hint, error state with a retry button.
- Text contrast on cards at least 4.5:1. Focus ring 2px in the accent colour. Every icon-only button has an aria-label. Decision changes are announced in an aria-live=polite region.
- Numbers use tabular-nums with fixed decimals: scores 2 dp, angles 0 dp with a degree sign, fps 0 dp.
- Charts: horizontal gridlines only at 6 percent opacity, 2px lines, the threshold as a dashed line labelled with its value, dark glass tooltip.
- No lorem ipsum, no emoji as icons (use lucide-react), no default blue links.
## Verification (after every UI task)
1. npm run typecheck and npm run build must both pass.
2. Open the app with the browser tool at 1440x900 and at 390x844 and take a screenshot of each.
3. Confirm: no horizontal scroll, no clipped text, no overlapping cards, 0 console errors, and ?mock=1 shows all of IDLE, LISTENING, ACCEPT and REJECT states.
4. Describe what you saw in 3 bullets. Never claim a visual is fine without having opened it.
