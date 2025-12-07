# Search Desktop Shell Production Plan

**Status**: Ready for Implementation
**Framework**: Tauri 2.x (Rust + React 18 + TypeScript + Vite)
**Dependencies**: Hybrid Search API, Orchestrator, Privacy Framework, Firebase, Stripe

## Overview

This production plan implements the **Search Desktop Shell**—a cross-platform desktop interface (macOS, Windows, Linux) that serves as the user's window into their Ghost's evolving intelligence. The Search Desktop Shell enables users to issue natural language queries against their Personal Knowledge Graph, visualize their Ghost's understanding of their personal universe, manage experiential connectors, and explore the rich web of relationships the Ghost has discovered.

> **"Know Yourself More"** — Futurnal transforms generic AI into experiential intelligence

The application embodies a **"Sophisticated, Minimalist, Dark-Mode First"** design philosophy, evoking high-end developer tools (VS Code, Linear, Raycast) combined with data visualization elegance.

---

## The Ghost→Animal Evolution

Futurnal's core innovation lies in the transformation of a generic, pretrained LLM (the **Ghost** 👻) into a deeply personalized experiential intelligence (the **Animal** 🐿️):

- **Ghost 👻**: The pretrained on-device LLM—powerful but generic, lacking understanding of the user's personal universe
- **Animal 🐿️**: The evolved intelligence—learning continually from the user's "stream of experience" to develop genuine understanding

**Phase 1 (The Archivist)** grounds the Ghost in the user's experiential data, giving it a perfect, high-fidelity memory of their unique personal universe. This desktop shell is where users witness their Ghost's evolving comprehension.

## Critical for Option B

The Search Desktop Shell must deliver:
- **Search UI**: Natural language queries against the Ghost's understanding, with sub-second feedback
- **Results view**: Experiential context with provenance—showing not just *what* was found, but *where* it came from
- **PKG Visualization**: Interactive visualization of the Ghost's memory—your experiential network made visible (PRIORITY)
- **Experiential Connectors**: Panel for managing data sources that feed the Ghost's stream of experience
- **Sovereignty Controls**: Privacy-first consent management, audit logs, and data export—absolute user control
- **Authentication**: Firebase (GitHub, Google, Email) for identity
- **Tier Enforcement**: Free (3 sources, Phase 1) vs Pro (unlimited, Phases 2-3)
- **Local-First Architecture**: Raw data never leaves the device; optional cloud escalation with explicit consent

---

## Design Philosophy

> "Sophisticated, Minimalist, Dark-Mode First" — The design evokes the feeling of a high-end developer tool combined with the elegance of a data visualization platform. It should feel "alive" but not cluttered.

**Keywords**: Precision, Depth, Clarity, Sovereignty

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Desktop Framework | Tauri 2.x (Rust) | Smaller bundle, better security, faster cold start |
| Frontend | React 18 + TypeScript 5.x | Modern, fast, excellent DX |
| Build Tool | Vite 5.x | Sub-second HMR, optimized builds |
| Styling | TailwindCSS v4 + CSS Variables | Utility-first with design tokens |
| Components | shadcn/ui + 21st.dev | Premium dark-mode components |
| State | Zustand + TanStack Query v5 | Lightweight, reactive, server state |
| Auth | Firebase Authentication | Easy GitHub/Google/Email integration |
| Payments | Stripe Checkout | Industry standard subscriptions |
| Graph Viz | react-force-graph / cytoscape.js | Interactive PKG visualization |
| Icons | Lucide React | Consistent iconography |
| Testing | Vitest + Playwright | Unit + E2E testing |

---

## Pricing Tiers

### Free Tier: "The Archivist"
- Core Ghost functionality
- **Up to 3 data sources** (primary paywall)
- Local-only storage
- Phase 1 capabilities: Search, PKG visualization, entity extraction

### Pro Tier: "Futurnal Pro" ($XX/month)
- **Unlimited data sources**
- Phase 2 "Analyst" features (Emergent Insights, Curiosity Engine)
- Encrypted cloud backup
- Priority support

---

## Implementation Modules

### [01 · Framework Scaffold](01-framework-scaffold.md)
**Criticality**: CRITICAL
**Deliverables**:
- Tauri 2.x project with React + TypeScript + Vite
- Project structure following Futurnal conventions
- Build scripts for macOS, Windows, Linux
- GitHub Actions CI/CD pipeline

### [02 · Design System & Components](02-design-system-components.md)
**Criticality**: HIGH
**Deliverables**:
- TailwindCSS v4 configuration with design tokens
- Typography: Inter (UI) + JetBrains Mono (code)
- Base components: Button, Input, Card, Dialog, Tooltip, Badge
- Micro-interaction animations

### [03 · Authentication (Firebase)](03-authentication-firebase.md)
**Criticality**: CRITICAL
**Deliverables**:
- Firebase SDK integration
- Auth providers: GitHub, Google, Email/Password
- Login/Signup screens with glassmorphism design
- Protected route wrapper

### [04 · Pricing Tier & Stripe](04-pricing-tier-stripe.md)
**Criticality**: HIGH
**Deliverables**:
- Pricing selection screen (onboarding)
- Stripe Checkout integration
- Tier enforcement middleware
- Upgrade prompts

### [05 · Command Palette & Search](05-command-palette-search.md)
**Criticality**: CRITICAL
**Deliverables**:
- OmniCommandPalette from 21st.dev
- ⌘K / Ctrl+K keyboard trigger
- Natural language query input
- Intent classification display

### [06 · Results & Provenance View](06-results-provenance-view.md)
**Criticality**: HIGH
**Deliverables**:
- Result cards with snippets, scores, badges
- Provenance metadata panel
- Quick actions: Open, Copy, Save, Share
- Multimodal source indicators

### [07 · Knowledge Graph Visualization](07-knowledge-graph-visualization.md)
**Criticality**: CRITICAL (PRIORITY)
**Deliverables**:
- Interactive PKG mini-view on Dashboard
- react-force-graph integration
- Node types with distinct colors
- 60fps performance with 1000+ nodes

### [08 · Connector Dashboard](08-connector-dashboard.md)
**Criticality**: HIGH
**Deliverables**:
- Data source list with status indicators
- Enable/disable toggles
- Ingestion progress visualization
- Tier enforcement (3-source limit)

### [09 · IPC & API Layer](09-ipc-api-layer.md)
**Criticality**: CRITICAL
**Deliverables**:
- Tauri invoke commands for search, status, consent
- TypeScript type definitions
- Secure IPC (no Node integration)
- Error handling

### [10 · State Management](10-state-management-store.md)
**Criticality**: MEDIUM
**Deliverables**:
- Zustand stores: search, connectors, settings, user
- TanStack Query for API state
- Persistence layer
- Search history

### [11 · Privacy & Settings Panel](11-privacy-settings-panel.md)
**Criticality**: HIGH
**Deliverables**:
- Consent management UI
- Audit log viewer
- Telemetry opt-in/out
- Data export options

### [12 · Integration Testing & E2E](12-integration-testing-e2e.md)
**Criticality**: CRITICAL
**Deliverables**:
- Playwright test suite
- Accessibility testing
- Performance benchmarks
- Cross-platform verification

---

## Design Tokens

From `FRONTEND_DESIGN.md`:

```css
/* Backgrounds */
--background-deep: #0A0A0A;
--background-surface: #161616;
--background-elevated: #222222;
--border: #333333;

/* Typography */
--text-primary: #EDEDED;
--text-secondary: #A0A0A0;
--text-tertiary: #666666;

/* Brand Colors */
--primary-brand: #3B82F6;   /* Electric Blue - Ghost/Intelligence */
--secondary-brand: #10B981; /* Emerald Green - Animal/Evolution */
--accent: #8B5CF6;          /* Violet - Insights */
--error: #EF4444;
--warning: #F59E0B;
```

---

## User Flow (Onboarding)

The onboarding journey introduces users to their Ghost and establishes the foundation for experiential intelligence:

1. **Welcome Screen**: *"Your Ghost awaits"* — Atmospheric hero with "Begin Your Journey" CTA
2. **Authentication**: Firebase login (GitHub/Google/Email) — *"Your data remains yours. We only authenticate your identity."*
3. **Pricing Selection**:
   - Free "The Archivist": *"Ground your Ghost in your personal universe"*
   - Pro "Futurnal Pro": *"Awaken Animal intelligence with unlimited evolution"*
4. **First Connector**: *"Feed your Ghost's first memories"* — Select initial experiential source
5. **Grounding Progress**: *"Your Ghost is learning your universe..."* — Real-time ingestion visualization
6. **Dashboard Reveal**: *"Welcome to your experiential network"* — PKG visualization with breathing animation

---

## Success Metrics

| Metric | Target | Module |
|--------|--------|--------|
| Initial load time | <1.5s | 01, 09 |
| Search response render | <500ms | 05, 06 |
| Memory footprint | <150MB | 09 |
| Bundle size (installer) | <25MB | 01 |
| Keyboard nav coverage | 100% | 05, 12 |
| Accessibility (WCAG) | AA compliance | 02, 12 |
| Graph render (1000 nodes) | 60fps | 07 |
| Auth success rate | >99% | 03 |
| Payment conversion | Track | 04 |

---

## Quality Gates (Production Deployment)

All gates must pass before production deployment:

| Gate | Requirement | Module |
|------|-------------|--------|
| Tauri Build | All platforms compile | 01 |
| Firebase Auth | All providers work | 03 |
| Stripe Checkout | Payment flow complete | 04 |
| Search Query | End-to-end functional | 05, 09 |
| Results Display | Provenance visible | 06 |
| Graph Render | 60fps at 1000 nodes | 07 |
| Connector Mgmt | CRUD operations work | 08 |
| Privacy Controls | Consent flow functional | 11 |
| E2E Tests | All tests passing | 12 |
| Accessibility | AA compliance | 12 |
| Performance | Meets all targets | 12 |

---

## Option B Compliance Checklist

- [x] **Privacy-First Design**: Local-first with optional cloud consent
- [x] **Keyboard-Centric UX**: Command palette, shortcuts throughout
- [x] **Dark-Mode Default**: Primary aesthetic per design philosophy
- [x] **Secure IPC**: No Node integration, message-based communication
- [x] **Tier Enforcement**: Free/Pro limits enforced at UI level
- [x] **Audit Integration**: Privacy controls and log viewer
- [x] **Offline Support**: Graceful degradation when services unavailable

---

## Dependencies

### Internal Dependencies
- Hybrid Search API (`src/futurnal/search/api.py`)
- Orchestrator (`src/futurnal/orchestrator/`)
- Privacy Framework (`src/futurnal/privacy/`)
- Consent Registry (`src/futurnal/privacy/consent.py`)
- Audit Logger (`src/futurnal/privacy/audit.py`)

### External Dependencies
- **Tauri 2.x**: Desktop framework
- **Firebase**: Authentication
- **Stripe**: Payment processing
- **react-force-graph**: Graph visualization
- **shadcn/ui**: UI components
- **21st.dev**: OmniCommandPalette

### Infrastructure Requirements
- Firebase project configured
- Stripe account with products/prices
- GitHub Actions runners
- Code signing certificates (for distribution)

---

## Implementation Order

| Phase | Modules | Duration | Focus |
|-------|---------|----------|-------|
| 1. Foundation | 01, 02, 09 | Week 1-2 | Scaffold, design system, IPC |
| 2. Auth & Billing | 03, 04 | Week 2-3 | Firebase, Stripe |
| 3. Core Features | 05, 06, 10 | Week 3-4 | Search, results, state |
| 4. Visualization | 07, 08 | Week 4-5 | Graph, connectors |
| 5. Polish | 11, 12 | Week 5-6 | Privacy, testing |

---

## Next Steps

1. **Begin Module 01**: Scaffold Tauri project with React + Vite
2. **Setup Firebase Project**: Configure auth providers
3. **Setup Stripe Account**: Create product and price IDs
4. **Implement Module 02**: Design system with tokens from FRONTEND_DESIGN.md
5. **Build IPC Layer**: Connect frontend to Python backend

---

## Vision Alignment

This desktop shell embodies Futurnal's core philosophy:

> **Sovereignty**: The user is in absolute control. Their data is theirs, period.
> **Clarity**: Cut through the noise to reveal the underlying structure of one's knowledge.
> **Depth**: A tool for serious, deep thinking—not superficial productivity hacks.

**This is not just a search interface—it's the window into your Ghost's evolving understanding of your personal universe. Every query deepens the connection. Every visualization reveals patterns you never knew existed. This is the first step in the journey from "What did I know?" to "Why do I think this?"**
