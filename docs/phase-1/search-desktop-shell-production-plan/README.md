# Search Desktop Shell Production Plan

**Status**: Ready for Implementation
**Framework**: Tauri 2.x (Rust + React 18 + TypeScript + Vite)
**Dependencies**: Hybrid Search API, Orchestrator, Privacy Framework, Firebase

## Overview

This production plan implements the **Search Desktop Shell**—a cross-platform desktop interface (macOS, Windows, Linux) that serves as the user's window into their evolving personal intelligence. The Search Desktop Shell enables users to issue natural language queries against their Personal Knowledge Graph, visualize their understanding of their personal universe, manage data connectors, and explore the rich web of relationships discovered within their data.

> **"Know Yourself More"** — Futurnal transforms generic AI into deeply personalized intelligence

The application embodies a **"Sophisticated, Minimalist, Dark-Mode First"** design philosophy, evoking high-end developer tools (VS Code, Linear, Raycast) combined with data visualization elegance.

---

## Futurnal's Core Innovation

Futurnal is the world's first **AI evolution platform** that transforms generic AI into deeply personalized intelligence through three phases:

### Phase 1: The Archivist
**Focus**: Building your personal knowledge foundation
- Ingests and indexes your personal data from multiple sources
- Constructs a Personal Knowledge Graph (PKG) of entities and relationships
- Provides hybrid search (semantic + graph traversal)
- Interactive PKG visualization
- Privacy-first with consent/audit logging

### Phase 2: The Analyst
**Focus**: Discovering patterns you never knew existed
- **Emergent Insights Engine**: Surfaces non-obvious correlations in your data
- **Curiosity Engine**: Identifies knowledge gaps and unexplored connections
- Intelligent ranking aligned with your interests
- Proactive notification system

### Phase 3: The Guide
**Focus**: Understanding the "why" behind your patterns
- **Conversational Exploration**: Dialogue-driven hypothesis investigation
- **Causal Inference Engine**: Understanding cause and effect in your life
- **Aspirational Self Integration**: Tracking progress toward your goals
- **Reward Signal Dashboard**: Visualization of goal alignment

---

## Critical Deliverables

The Search Desktop Shell must deliver:
- **Search UI**: Natural language queries with sub-second feedback
- **Results View**: Context with provenance—showing not just *what* was found, but *where* it came from
- **PKG Visualization**: Interactive visualization of your knowledge network (PRIORITY)
- **Data Connectors**: Panel for managing data sources
- **Sovereignty Controls**: Privacy-first consent management, audit logs, and data export
- **Authentication**: Firebase email/password for identity
- **Tier Verification**: Free vs Pro feature gating
- **Local-First Architecture**: Raw data never leaves the device

---

## Design Philosophy

> "Sophisticated, Minimalist, Dark-Mode First" — The design evokes the feeling of a high-end developer tool combined with the elegance of a data visualization platform. It should feel "alive" but not cluttered.

**Keywords**: Precision, Depth, Clarity, Sovereignty

**Typography**: Cinzel (brand headlines) + Times New Roman (taglines)

**Colors**: Pure monochrome (black #000, white #FFF, opacity-based grays)

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Desktop Framework | Tauri 2.x (Rust) | Smaller bundle, better security, faster cold start |
| Frontend | React 18 + TypeScript 5.x | Modern, fast, excellent DX |
| Build Tool | Vite 5.x | Sub-second HMR, optimized builds |
| Styling | TailwindCSS v4 | Utility-first with monochrome aesthetic |
| State | Zustand + TanStack Query v5 | Lightweight, reactive, server state |
| Auth | Firebase Authentication | Email/password integration |
| Graph Viz | react-force-graph-2d | Interactive PKG visualization |
| Icons | Lucide React | Consistent iconography |
| Testing | Vitest + Playwright | Unit + E2E testing |

---

## Pricing Architecture

### Desktop App Responsibility
- Login only (no signup in desktop)
- Tier verification via API call
- Feature gating based on tier
- "Manage Subscription" → opens browser to web portal

### Landing Page Responsibility (Separate Project)
- Marketing, features, pricing display
- Signup flow with tier selection
- Stripe Checkout integration
- Stripe Customer Portal for subscription management
- Download links for desktop app

### Free Tier: "Explorer"
- Core search functionality
- **Up to 3 data sources**
- Local-only storage
- Phase 1 capabilities

### Pro Tier: "Futurnal Pro" ($XX/month)
- **Unlimited data sources**
- Phase 2 + Phase 3 features
- Encrypted cloud backup
- Priority support

---

## Implementation Modules

### [01 · Framework Scaffold](01-framework-scaffold.md) ✅
**Status**: COMPLETED

### [02 · Design System & Components](02-design-system-components.md) ✅
**Status**: COMPLETED
- TailwindCSS v4 with monochrome aesthetic
- Typography: Cinzel + Times New Roman

### [03 · Authentication (Firebase)](03-authentication-firebase.md) ✅
**Status**: COMPLETED
- Email/password authentication only
- Login, Signup, ForgotPassword pages
- Protected route wrapper

### [04 · Tier Verification](04-pricing-tier-stripe.md)
**Status**: Needs Revision
- Desktop app verifies tier via API
- Feature gating based on subscription
- "Manage Subscription" opens web browser

### [05 · Command Palette & Search](05-command-palette-search.md)
**Criticality**: CRITICAL

### [06 · Results & Provenance View](06-results-provenance-view.md)
**Criticality**: HIGH

### [07 · Knowledge Graph Visualization](07-knowledge-graph-visualization.md)
**Criticality**: CRITICAL (PRIORITY)

### [08 · Connector Dashboard](08-connector-dashboard.md)
**Criticality**: HIGH

### [09 · IPC & API Layer](09-ipc-api-layer.md)
**Criticality**: CRITICAL

### [10 · State Management](10-state-management-store.md)
**Criticality**: MEDIUM

### [11 · Privacy & Settings Panel](11-privacy-settings-panel.md)
**Criticality**: HIGH

### [12 · Integration Testing & E2E](12-integration-testing-e2e.md)
**Criticality**: CRITICAL

### [13 · Light Mode Theme](13-light-mode-theme.md)
**Status**: Documented
- Theme toggle component
- System preference detection
- Theme-aware logo assets

---

## Design Tokens

```css
/* Monochrome Palette */
--background: #000000;
--text-primary: #FFFFFF;
--text-secondary: rgba(255, 255, 255, 0.7);
--text-tertiary: rgba(255, 255, 255, 0.5);
--border: rgba(255, 255, 255, 0.1);

/* Semantic (used sparingly) */
--error: #EF4444;
--success: #10B981;
```

---

## User Flow (Desktop App)

1. **Welcome Screen**: Brand introduction with "Get Started" and "Sign In" CTAs
2. **Authentication**: Login with email/password (signup redirects to landing page)
3. **Dashboard**: Home view with PKG visualization, stats, quick actions
4. **Onboarding**: Connect first data source (if new user)
5. **Core Experience**: Search, explore PKG, manage connectors

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

---

## Vision Alignment

This desktop shell embodies Futurnal's core philosophy:

> **Sovereignty**: The user is in absolute control. Their data is theirs, period.
> **Clarity**: Cut through the noise to reveal the underlying structure of one's knowledge.
> **Depth**: A tool for serious, deep thinking—not superficial productivity hacks.

**This is not just a search interface—it's the window into your evolving understanding of your personal universe. Every query deepens the insight. Every visualization reveals patterns you never knew existed. This is the first step in the journey from "What did I know?" to "Why do I think this?"**
