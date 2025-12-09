# Futurnal Phase I: Frontend Design & Business Logic

## 1. Visual Identity

### 1.1. Design Philosophy
**"Sophisticated, Minimalist, Dark-Mode First"**
The design should evoke the feeling of a high-end developer tool (like VS Code, Linear, or Raycast) combined with the elegance of a data visualization platform. It should feel "alive" but not cluttered.
- **Keywords**: Precision, Depth, Clarity, Sovereignty.
- **Theme**: Dark mode is the default and primary experience.

### 1.2. Color Palette
The palette uses deep grays for backgrounds to reduce eye strain during deep work, with high-contrast text and vibrant accents for data and interactions.

**Neutral / Backgrounds**
- **Background Deep**: `#0A0A0A` (Main app background)
- **Background Surface**: `#161616` (Cards, Sidebars, Modals)
- **Background Elevated**: `#222222` (Hover states, Inputs)
- **Border**: `#333333` (Subtle dividers)

**Typography**
- **Text Primary**: `#EDEDED` (High readability)
- **Text Secondary**: `#A0A0A0` (Metadata, labels)
- **Text Tertiary**: `#666666` (Placeholders, disabled)

**Brand & Accents**
- **Primary Brand**: `#3B82F6` (Electric Blue - representing "The Ghost" / Intelligence)
- **Secondary Brand**: `#10B981` (Emerald Green - representing "Growth" / "Animal" evolution)
- **Accent/Focus**: `#8B5CF6` (Violet - for "Insights" and special states)
- **Error**: `#EF4444`
- **Warning**: `#F59E0B`

### 1.3. Typography
- **Primary Font**: `Inter` (Google Fonts) - Clean, modern, highly legible at all sizes.
- **Monospace Font**: `JetBrains Mono` or `Fira Code` - For code snippets, data structures, and logs.
- **Weights**:
    - Regular (400): Body text.
    - Medium (500): Buttons, Navigation.
    - SemiBold (600): Headings, emphasized data.

### 1.4. Assets Usage
- **Logo**: `assets/logo.png` - Used as the app icon and in the login screen.
- **Wordmark**: `assets/logo_text_horizontal.png` - Used in the top-left of the Sidebar or Navigation bar.
- **Hero Image**: `assets/Gemini_Generated_Image_rcdxgarcdxgarcdx.png` - Used on the Landing/Login page background (with a dark overlay) to set the mood.

---

## 2. UI Components

### 2.1. Buttons
Buttons should feel tactile and precise.
- **Primary Button**:
    - Background: Primary Brand (`#3B82F6`) or White (`#EDEDED`) for high contrast.
    - Text: Dark (`#0A0A0A`) if white background, or White if blue background.
    - Radius: `6px` (Slightly rounded, not pill-shaped).
    - Hover: Slight brightness increase + subtle lift (transform-y).
    - Shadow: `0 4px 12px rgba(59, 130, 246, 0.3)` (Glow effect).
- **Secondary Button**:
    - Background: Transparent or Surface (`#222222`).
    - Border: 1px solid `#333333`.
    - Text: Secondary (`#A0A0A0`).
    - Hover: Border becomes lighter (`#666666`), text becomes Primary.

### 2.2. Text Boxes & Inputs
- **Style**: Minimalist, flat, immersive.
- **Background**: `#121212` or `#1A1A1A`.
- **Border**: 1px solid `#333333`.
- **Focus State**: Border color changes to Primary Brand (`#3B82F6`) with a subtle glow `box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2)`.
- **Text**: Primary (`#EDEDED`).
- **Placeholder**: Tertiary (`#666666`).
- **Radius**: `6px`.

### 2.3. Shadows & Depth
- **Card Shadow**: `0 4px 6px -1px rgba(0, 0, 0, 0.5), 0 2px 4px -1px rgba(0, 0, 0, 0.3)` (Deep, soft shadows to lift content off the dark background).
- **Modal Shadow**: `0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.5)`.
- **Glows**: Use subtle colored shadows behind key elements (like the "Insight" cards) to indicate activity/intelligence.

---

## 3. Layout & Structure (Phase I)

### 3.1. Main Layout
- **Sidebar (Left)**:
    - Width: ~240px, collapsible.
    - Content:
        - Top: `logo_text_horizontal.png`.
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
- **Visual**: Full-screen background using `Gemini_Generated_Image_rcdxgarcdxgarcdx.png` with a heavy dark overlay.
- **Center**: Large `logo.png` + "Futurnal".
- **Action**: "Enter the Void" (or "Get Started") button.

**Step 2: Authentication (Sign Up / Login)**
- **Design**: Centered card on glassmorphism background.
- **Options**:
    - "Continue with GitHub" (Target audience is devs).
    - "Continue with Google".
    - Email/Password.
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

## 5. Technical Stack Recommendations
- **Framework**: React (Vite) or Next.js (if server-side rendering is needed for SEO/Marketing pages, but for a local-first app, Vite + SPA is fine).
- **Styling**: TailwindCSS (for rapid, consistent styling matching the design system).
- **State Management**: Zustand or TanStack Query.
- **Graph Visualization**: `react-force-graph` or `cytoscape.js`.
- **Icons**: Lucide React or Heroicons.
