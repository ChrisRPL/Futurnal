# Futurnal Phase I: Frontend Design & Business Logic

## 1. Visual Identity

### 1.1. Design Philosophy
**"Sophisticated, Minimalist, Dark-Mode First"**
The design should evoke the feeling of a high-end developer tool (like VS Code, Linear, or Raycast) combined with the elegance of a data visualization platform. It should feel "alive" but not cluttered.
- **Keywords**: Precision, Depth, Clarity, Sovereignty.
- **Theme**: Dark mode is the default and primary experience.

### 1.2. Color Palette
The palette uses a pure monochrome aesthetic - black backgrounds with white/gray text. No accent colors. This creates a timeless, sophisticated, and premium feel aligned with the Futurnal brand identity.

**Dark Mode (Primary)**
- **Background**: `#000000` (Pure black)
- **Text Primary**: `#FFFFFF` (Pure white)
- **Text Secondary**: `rgba(255, 255, 255, 0.7)` (70% white)
- **Text Tertiary**: `rgba(255, 255, 255, 0.5)` (50% white)
- **Text Muted**: `rgba(255, 255, 255, 0.3)` (30% white)
- **Border**: `rgba(255, 255, 255, 0.1)` (10% white)
- **Surface**: `rgba(255, 255, 255, 0.05)` (5% white - for inputs/cards)

**Light Mode (Secondary)**
- **Background**: `#FFFFFF` (Pure white)
- **Text Primary**: `#000000` (Pure black)
- **Text Secondary**: `rgba(0, 0, 0, 0.7)` (70% black)
- **Text Tertiary**: `rgba(0, 0, 0, 0.5)` (50% black)
- **Border**: `rgba(0, 0, 0, 0.1)` (10% black)

**Semantic Colors**
- **Error**: `#EF4444` (Red - used sparingly for error states only)
- **Success**: `#10B981` (Green - used sparingly for success states only)

### 1.3. Typography
Futurnal uses a refined serif typography system that evokes timeless elegance and sophistication.

- **Brand Headlines**: `Cinzel` (Google Fonts) - Elegant serif font with classical proportions. Used for page titles, hero text, and brand moments.
  - Weights: Regular (400), Medium (500), SemiBold (600), Bold (700)
  - Letter-spacing: `0.05em` (slightly expanded for elegance)

- **Taglines & Italic Text**: `Times New Roman` / `Georgia` (System serif) - Classic serif for taglines, quotes, and italic emphasis.
  - Style: Italic
  - Used for: Taglines, quotes, secondary brand text

**CSS Classes**:
```css
.font-brand {
  font-family: 'Cinzel', serif;
  letter-spacing: 0.05em;
}

.font-tagline {
  font-family: 'Times New Roman', 'Georgia', serif;
  font-style: italic;
}
```

**Usage Guidelines**:
- Page titles: `font-brand` + `text-3xl` or larger
- Button text: `font-brand` or system sans-serif
- Body text: System serif or `font-brand` at smaller sizes
- Taglines: `font-tagline` + italic styling

### 1.4. Logo Assets

Futurnal uses a geometric logo system with variants for different contexts and themes.

**Source Assets** (in `assets/` folder):
| Asset | Description |
|-------|-------------|
| `logo.png` | Small icon logo (geometric mark only) |
| `logo_big.png` | Large logo for hero sections |
| `logo_text_horizon.png` | Horizontal logo with "FUTURNAL" text and tagline |
| `logo_text_horizontal.png` | Alternative horizontal wordmark |

**Desktop App Assets** (in `desktop/public/` folder):
Each logo has a standard version (for light backgrounds) and a `_dark` version (inverted colors for dark backgrounds).

| Asset | Theme | Usage |
|-------|-------|-------|
| `logo.png` | Light mode | Header icon, favicon |
| `logo_dark.png` | Dark mode | Header icon on dark backgrounds |
| `logo_big.png` | Light mode | Welcome page hero (light theme) |
| `logo_big_dark.png` | Dark mode | Welcome page hero (dark theme) |
| `logo_text_horizon.png` | Light mode | Dashboard header (light theme) |
| `logo_text_horizon_dark.png` | Dark mode | Dashboard header (dark theme) |

**Usage by Page**:
| Page | Header Logo | Hero/Main Logo |
|------|-------------|----------------|
| Welcome | `logo_dark.png` | `logo_big_dark.png` |
| Login | `logo_dark.png` | - |
| Signup | `logo_dark.png` | - |
| ForgotPassword | `logo_dark.png` | - |
| Dashboard | `logo_text_horizon_dark.png` | - |

**Theme-Aware Implementation**:
```tsx
// When implementing light mode support:
const { resolvedTheme } = useTheme();
const logoSrc = resolvedTheme === 'dark' ? '/logo_dark.png' : '/logo.png';
```

---

## 2. UI Components

### 2.1. Buttons
Buttons follow the monochrome aesthetic - no colored backgrounds.

- **Primary Button** (Dark Mode):
    - Background: `#FFFFFF` (Pure white)
    - Text: `#000000` (Pure black)
    - Radius: `0px` (Sharp corners for modern aesthetic)
    - Hover: `rgba(255, 255, 255, 0.9)` (Slight dim)
    - Font: Medium weight, larger text (`text-lg`)

- **Primary Button** (Light Mode):
    - Background: `#000000` (Pure black)
    - Text: `#FFFFFF` (Pure white)
    - Hover: `rgba(0, 0, 0, 0.9)` (Slight dim)

- **Secondary Button** (Outline):
    - Background: Transparent
    - Border: `1px solid rgba(255, 255, 255, 0.3)` (dark mode)
    - Text: `rgba(255, 255, 255, 0.7)`
    - Hover: Border becomes `rgba(255, 255, 255, 0.6)`, text becomes white

- **Ghost Button** (Text only):
    - Background: Transparent
    - Text: `rgba(255, 255, 255, 0.6)`
    - Hover: Text becomes white

### 2.2. Text Boxes & Inputs
- **Style**: Minimalist, flat, monochrome.
- **Background**: `rgba(255, 255, 255, 0.05)` (5% white on dark mode)
- **Border**: `1px solid rgba(255, 255, 255, 0.1)`
- **Focus State**: Border becomes `rgba(255, 255, 255, 0.3)` (no colored glow)
- **Text**: `#FFFFFF` (Pure white)
- **Placeholder**: `rgba(255, 255, 255, 0.3)` (30% white)
- **Radius**: `0px` (Sharp corners to match buttons)

### 2.3. Shadows & Depth
Shadows are minimal in the monochrome design. The focus is on borders and opacity for depth.

- **Card Shadow**: None or very subtle `0 4px 6px -1px rgba(0, 0, 0, 0.3)`
- **Modal Shadow**: `0 20px 25px -5px rgba(0, 0, 0, 0.5)`
- **Depth Indication**: Use border opacity (`border-white/10` to `border-white/30`) rather than shadows

---

## 3. Layout & Structure (Phase I)

### 3.1. Main Layout
- **Sidebar (Left)**:
    - Width: ~240px, collapsible.
    - Content:
        - Top: `logo_text_horizon_dark.png` (or `logo_text_horizon.png` in light mode).
        - Middle: Navigation (Search, Graph View, Data Sources, Settings).
        - Bottom: User Profile & Plan Status.
- **Main Content Area**:
    - Fluid width.
    - Top Bar: Contextual Breadcrumbs + Global Search Input.
    - Central View: Dynamic content (Dashboard, Graph, etc.).

### 3.2. The Dashboard (Home View)
The "Command Center" for the user's Personal Knowledge Graph (PKG).
- **Header**: "Good evening, [User]. Your Ghost is active."
- **Stats Row**:
    - "Total Nodes": [Number] (e.g., 12,403)
    - "Sources Connected": [Number] (e.g., 4)
    - "Memory Usage": [Size] (e.g., 1.2 GB)
- **Central Visual**: A mini-view of the **Knowledge Graph** (using a library like `react-force-graph` or `cytoscape.js`). It should be interactive, slowly rotating or "breathing" to show liveness.
- **Recent Activity / Insights**:
    - A list of cards showing recently ingested files or "Emergent Insights" (Phase 2 preview).
    - Example Card: "Ingested 'Project_Alpha_Specs.pdf' - 14 new entities found."

---

## 4. Business Model & User Flows

### 4.1. Business Model Integration
Futurnal operates on a Freemium model with a focus on privacy and "Prosumer" features.
- **Free Tier ("The Archivist")**:
    - Core Ghost functionality.
    - Limited Data Sources (e.g., up to 3).
    - Local-only storage.
- **Pro Tier ("Futurnal Pro")**:
    - Unlimited Data Sources.
    - Advanced "Analyst" features (Phase 2).
    - Cloud backup (encrypted).
    - Priority Support.

### 4.2. User Flow: Launch & Onboarding (Phase I)

**Step 1: Landing / Welcome Screen**
- **Visual**: Pure black background (`#000000`), no images.
- **Header**: Small `logo_dark.png` in top-left.
- **Center**: Large `logo_big_dark.png` as hero element.
- **Value Proposition**: Clean typography explaining Futurnal's purpose.
- **Actions**: "Get Started" (primary) and "Sign In" (secondary) buttons.

**Step 2: Authentication (Sign Up / Login)**
- **Design**: Full-screen black background with centered form (no glassmorphism).
- **Header**: Small `logo_dark.png` + navigation link to alternate auth page.
- **Options**: Email/Password only (no OAuth providers).
- **Form Style**: Monochrome inputs with sharp corners, minimal borders.
- **Privacy Note**: "Your data remains yours. We only authenticate your identity."

**Step 3: Pricing Selection (The Gate)**
- Before accessing the app, the user selects their tier.
- **Layout**: Two large cards side-by-side.
    - **Card 1: Free**: "For personal exploration." (Selected by default).
    - **Card 2: Pro**: "For serious knowledge architects." ($XX/month).
- **Action**: "Continue with Free" or "Upgrade to Pro".

**Step 4: Onboarding (Grounding the Ghost)**
- **Goal**: Connect the first data source.
- **UI**: A list of integrations (Local Folder, GitHub, Google Drive, Notion).
- **Action**: User selects "Local Folder" -> System dialog opens -> User picks folder.
- **Feedback**: Progress bar showing "Ingesting... Extracting Entities... Building Graph...".

**Step 5: Redirect to Dashboard**
- Once initial ingestion is done, transition to the Main Dashboard.
- A subtle animation (fade in) reveals the Graph View.

---

## 5. Technical Stack

- **Framework**: React + Vite + Tauri (desktop app)
- **Styling**: TailwindCSS v4 with `@tailwindcss/vite` plugin
- **State Management**: Zustand + TanStack Query
- **Graph Visualization**: `react-force-graph-2d`
- **Icons**: Lucide React
- **Authentication**: Firebase Authentication (email/password)
- **Fonts**: Google Fonts (Cinzel) + System fonts (Times New Roman)

### CSS Architecture
```css
/* globals.css structure */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&display=swap');
@import "tailwindcss";

/* Custom utility classes */
.font-brand { font-family: 'Cinzel', serif; letter-spacing: 0.05em; }
.font-tagline { font-family: 'Times New Roman', 'Georgia', serif; font-style: italic; }
```

### Vite Configuration
```typescript
// vite.config.ts - Plugin order is critical!
export default defineConfig({
  plugins: [tailwindcss(), react()], // tailwindcss MUST be first
  // ...
});
```
