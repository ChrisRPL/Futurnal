# Futurnal Desktop Shell

Desktop application for Futurnal - a privacy-first personal knowledge and causal insight engine. Built with Tauri 2.x, React 18, TypeScript, and TailwindCSS v4.

## Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Desktop Framework | Tauri | 2.9.x |
| Frontend | React | 18.3.x |
| Build Tool | Vite | 6.x |
| Styling | TailwindCSS | 4.0.x |
| State Management | Zustand + TanStack Query | 4.5/5.x |
| Language | TypeScript | 5.x |

## Prerequisites

- **Node.js** 18+ and npm
- **Rust** (latest stable) - Install via [rustup](https://rustup.rs/)
- **Platform-specific dependencies** - See [Tauri Prerequisites](https://v2.tauri.app/start/prerequisites/)

### macOS
```bash
xcode-select --install
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

### Windows
- Install [Microsoft Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [WebView2](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)

## Getting Started

```bash
# Install dependencies
npm install

# Run in development mode
npm run tauri dev

# Run frontend only (for rapid UI development)
npm run dev
```

## Available Commands

### Development
| Command | Description |
|---------|-------------|
| `npm run dev` | Start Vite dev server (frontend only) |
| `npm run tauri dev` | Start full Tauri app in dev mode |
| `npm run build` | Build frontend for production |
| `npm run tauri build` | Build complete desktop app |

### Testing
| Command | Description |
|---------|-------------|
| `npm test` | Run unit tests with Vitest |
| `npm run test:ui` | Run tests with UI |
| `npm run test:coverage` | Run tests with coverage report |
| `npm run test:e2e` | Run Playwright E2E tests |

### Utilities
| Command | Description |
|---------|-------------|
| `npm run lint` | Lint TypeScript/React code |
| `npm run generate-icons` | Generate app icons from logo |

## Project Structure

```
desktop/
├── src/                    # Frontend source
│   ├── components/         # React components
│   │   └── ui/            # shadcn/ui components
│   ├── hooks/             # React hooks
│   ├── lib/               # Utilities and API client
│   ├── pages/             # Page components
│   ├── stores/            # Zustand stores
│   ├── styles/            # Global styles
│   └── types/             # TypeScript types
├── src-tauri/             # Tauri/Rust backend
│   ├── src/
│   │   ├── commands/      # IPC command handlers
│   │   ├── python.rs      # Python CLI integration
│   │   ├── lib.rs         # Tauri app setup
│   │   └── main.rs        # Entry point
│   ├── icons/             # App icons
│   └── capabilities/      # Permission definitions
├── tests/
│   ├── unit/              # Vitest unit tests
│   └── e2e/               # Playwright E2E tests
└── scripts/               # Build scripts
```

## IPC Commands

The desktop shell communicates with the Python backend via Tauri IPC commands:

| Command | Description | CLI Mapping |
|---------|-------------|-------------|
| `search_query` | Execute semantic search | `futurnal search --json` |
| `get_search_history` | Get recent searches | `futurnal search history --json` |
| `list_sources` | List data sources | `futurnal sources list --json` |
| `add_source` | Add new data source | `futurnal sources add --json` |
| `get_orchestrator_status` | Get system status | `futurnal orchestrator status --json` |
| `get_knowledge_graph` | Export knowledge graph | `futurnal graph export --json` |
| `get_consent` | Check consent status | `futurnal privacy consent status --json` |
| `get_audit_logs` | Get audit history | `futurnal privacy audit --json` |

## Design System

The app follows the Futurnal design system (dark-mode first):

### Colors
- **Background Deep**: `#0A0A0A` - Primary background
- **Background Surface**: `#161616` - Cards and panels
- **Primary (Ghost)**: `#3B82F6` - Intelligence, AI actions
- **Secondary (Animal)**: `#10B981` - Growth, evolution
- **Accent (Insight)**: `#8B5CF6` - Special states, insights

### Typography
- **Sans**: Inter, system-ui
- **Mono**: JetBrains Mono, Fira Code

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Firebase (optional, for cloud features)
VITE_FIREBASE_API_KEY=
VITE_FIREBASE_AUTH_DOMAIN=
VITE_FIREBASE_PROJECT_ID=

# Stripe (optional, for payments)
VITE_STRIPE_PUBLISHABLE_KEY=

# Tauri Updater (for releases)
TAURI_SIGNING_PRIVATE_KEY=
TAURI_SIGNING_PRIVATE_KEY_PASSWORD=
```

## Building for Production

```bash
# Build for current platform
npm run tauri build

# Build outputs location:
# macOS: src-tauri/target/release/bundle/dmg/
# Windows: src-tauri/target/release/bundle/msi/
# Linux: src-tauri/target/release/bundle/appimage/
```

## CI/CD

GitHub Actions workflows are configured for:

- **build.yml**: Runs on every push - builds and tests on macOS, Windows, Linux
- **release.yml**: Triggered by version tags - creates releases with signed binaries

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/)
- [Tauri Extension](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode)
- [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- [Tailwind CSS IntelliSense](https://marketplace.visualstudio.com/items?itemName=bradlc.vscode-tailwindcss)

## Related Documentation

- [Production Plan](../docs/phase-1/search-desktop-shell-production-plan/)
- [Frontend Design System](../FRONTEND_DESIGN.md)
- [System Architecture](../architecture/system-architecture.md)
