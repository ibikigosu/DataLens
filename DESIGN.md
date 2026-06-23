# Design system

## Scene and theme

A procurement analyst and mentor review evidence together on a bright office monitor during a structured walkthrough.
The interface uses a light, warm-neutral theme so dense tables remain readable and screenshots reproduce cleanly.

## Color strategy

Use a restrained palette with warm paper neutrals and one muted plum accent.
Semantic green, amber, and red appear only for status and severity.

- Canvas: `oklch(0.975 0.008 85)`.
- Secondary surface: `oklch(0.94 0.012 80)`.
- Primary text: `oklch(0.24 0.015 320)`.
- Muted text: `oklch(0.48 0.018 320)`.
- Accent: `oklch(0.48 0.09 330)`.
- Success: `oklch(0.52 0.10 150)`.
- Warning: `oklch(0.66 0.12 75)`.
- Error: `oklch(0.52 0.14 25)`.

## Typography

Use the system sans-serif stack for labels, prose, controls, and tables.
Use weight and modest scale changes for hierarchy.
Keep explanatory prose below 72 characters per line where practical.

## Layout

Use a wide dashboard layout with a concise header, one primary upload workflow, a compact metric row, and a two-column review area.
Use borders only where they establish a real task boundary.
Avoid nested cards and repeated identical tiles.

## Components

- Primary actions use the plum accent.
- Status always combines text with a Material Symbol.
- Severity remains visible as text even when color is used.
- Findings use one interactive dataframe with focused column configuration.
- Feedback uses an inline form tied to the selected finding.
- Model comparison uses a side-by-side metric table and explicit gate list.

## Motion

Use only native Streamlit state transitions and loading indicators.
Do not add decorative animation.
