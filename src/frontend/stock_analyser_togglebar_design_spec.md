
# Stock Analyser Frontend — Design Spec with Left Toggle Input Bar

## 1️⃣ Page Structure
| Area | Width | Behavior | Purpose |
|-------|--------|----------|---------|
| Left Toggle Bar | 240px expanded / 64px collapsed | Collapsible; toggle button on top | Collect user input: filters, dropdowns, sliders |
| Main Content | Remaining width | Flexible, scrollable | Display analysis result, charts, tables, stock cards |
| Header | Full width | Fixed, 80px height | Branding, profile, nav |

## 2️⃣ General Layout
- **Grid:** 12-column fluid grid within main content
- **Page padding:** 32px on desktop, 16px on mobile
- **Gutter:** 24px
- **Sidebar toggle button:** Top-left in header or sidebar

## 3️⃣ Left Toggle Bar — Input Components
| Component | Spec |
|------------|-------|
| **Search Input** | Rounded 8px, padding 12px, border `#E0E0E0`, placeholder gray |
| **Dropdown (Stock Sector, Market)** | Same style as search input, chevron icon right |
| **Date Range Picker** | Inline calendar icon, rounded input, `#E0E0E0` border |
| **Sliders (Risk tolerance, P/E ratio)** | Track: `#7986CB`; Thumb: `#3F51B5`; Label above thumb |
| **Checkboxes (Include small cap, include foreign stocks)** | Material style, 20px checkbox, 16px label |
| **Analyse Button** | Full-width, 48px height, background `#3F51B5`, text white, rounded 6px |

**Spacing:** 24px between components  
**Collapsed State:** Icons only, tooltips on hover, toggle icon at top

## 4️⃣ Main Content — Output Window
- **Title / Breadcrumb:** Top left: `Stock Analysis Result`, small breadcrumb below
- **Summary Cards:** Horizontal stack, each 240px wide, white bg, slight shadow
- **Charts:** Full width, up to 400px height (line chart, bar chart)
- **Table:** Sticky header, striped rows, hover row highlight `#FAFAFA`

## 5️⃣ Interactivity
- **Analyse button:** Triggers analysis + loader
- **Left bar toggle:** Smooth 0.3s width transition
- **Cards hover:** Elevation increase, cursor pointer
- **Charts:** Tooltips on hover, zoomable

## 6️⃣ Color Palette
| Purpose | Hex |
|----------|-------|
| Primary | `#3F51B5` |
| Secondary | `#7986CB` |
| Accent | `#4CAF50` |
| Error | `#F44336` |
| Background | `#F5F5F5` |
| Sidebar Bg | `#FFFFFF` |
| Border | `#E0E0E0` |

## 7️⃣ Typography
| Element | Font | Weight | Size | Color |
|----------|-------|--------|-------|--------|
| App title | `Inter` | 700 | 32px | `#212121` |
| Sidebar labels | `Inter` | 500 | 14px | `#555555` |
| Body | `Inter` | 400 | 14px | `#666666` |
| Metrics | `Roboto Mono` | 500 | 16px | `#000000` |

## 8️⃣ Code Structure
```
/components
  /Header.tsx
  /Sidebar.tsx
  /SidebarToggleButton.tsx
  /AnalyseButton.tsx
  /StockSummaryCard.tsx
  /StockChart.tsx
  /StockTable.tsx
/pages
  /Dashboard.tsx
/styles
  /theme.ts
```

## 9️⃣ Suggested Libraries
- `@mui/material` or `chakra-ui`
- `react-chartjs-2` or `recharts`
- `react-datepicker`
- `react-slider`

## 🚀 Copilot Notes
👉 Copilot prompt:  
“Create a React + TypeScript dashboard with a collapsible left sidebar for input (search, dropdowns, sliders) and a main content area showing analysis results (cards, charts, table). Style with Tailwind. Sidebar should toggle width. On clicking Analyse button, show loader then results.”
