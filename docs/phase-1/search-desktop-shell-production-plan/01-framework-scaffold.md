Summary: Scaffold Tauri 2.x desktop application with React 18, TypeScript, and Vite for cross-platform builds.

# 01 · Framework Scaffold

## Purpose

Establish the foundational desktop application scaffold using Tauri 2.x with React 18, TypeScript, and Vite. This module creates the cross-platform build infrastructure for macOS, Windows, and Linux, following Futurnal conventions for project structure and CI/CD.

**Criticality**: CRITICAL - Foundation for entire desktop shell

## Scope

- Tauri 2.x project initialization with `npm create tauri-app`
- React 18 + TypeScript 5.x + Vite 5.x configuration
- Project structure aligned with Futurnal conventions
- Tauri configuration for window management, app icon, permissions
- Build scripts for macOS (.dmg), Windows (.msi/.exe), Linux (.AppImage/.deb)
- Environment variable handling (.env, Vite imports)
- GitHub Actions CI/CD pipeline for automated builds
- Development workflow documentation

## Requirements Alignment

- **Feature Requirement**: "Application shell scaffolding with window management, preferences storage, and update channel"
- **Architecture Principle**: Secure IPC between frontend and backend
- **Privacy-First**: No unnecessary Node APIs exposed
- **Performance**: Sub-1.5s initial load time

## Component Design

### Project Structure

```
futurnal-desktop/
├── src-tauri/                      # Rust backend
│   ├── Cargo.toml
│   ├── tauri.conf.json             # Tauri configuration
│   ├── capabilities/               # Permission capabilities
│   ├── icons/                      # App icons (all sizes)
│   └── src/
│       ├── main.rs                 # Tauri entry point
│       ├── lib.rs                  # Library exports
│       └── commands/               # IPC command handlers
│           ├── mod.rs
│           ├── search.rs
│           ├── connectors.rs
│           └── privacy.rs
├── src/                            # React frontend
│   ├── main.tsx                    # React entry point
│   ├── App.tsx                     # Root component
│   ├── vite-env.d.ts
│   ├── components/
│   │   ├── ui/                     # shadcn/ui components
│   │   ├── search/                 # Search components
│   │   ├── results/                # Results components
│   │   ├── graph/                  # Graph visualization
│   │   ├── connectors/             # Connector components
│   │   ├── settings/               # Settings components
│   │   └── auth/                   # Auth components
│   ├── pages/                      # Page components
│   ├── stores/                     # Zustand stores
│   ├── hooks/                      # Custom hooks
│   ├── lib/                        # Utilities
│   │   ├── api.ts                  # Tauri invoke wrappers
│   │   ├── firebase.ts             # Firebase config
│   │   └── utils.ts                # Helpers
│   ├── types/                      # TypeScript types
│   └── styles/
│       └── globals.css             # Global styles + Tailwind
├── public/                         # Static assets
├── tests/
│   ├── e2e/                        # Playwright tests
│   └── unit/                       # Vitest tests
├── .github/
│   └── workflows/
│       ├── build.yml               # Build pipeline
│       └── test.yml                # Test pipeline
├── package.json
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── README.md
```

### Tauri Configuration

```json
// src-tauri/tauri.conf.json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Futurnal",
  "identifier": "com.futurnal.desktop",
  "version": "0.1.0",
  "build": {
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:5173",
    "beforeBuildCommand": "npm run build",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Futurnal",
        "width": 1280,
        "height": 800,
        "minWidth": 900,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false,
        "decorations": true,
        "transparent": false,
        "center": true
      }
    ],
    "security": {
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://*.firebaseapp.com https://*.googleapis.com https://api.stripe.com"
    }
  },
  "bundle": {
    "active": true,
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "targets": "all",
    "macOS": {
      "minimumSystemVersion": "10.15",
      "signingIdentity": null,
      "providerShortName": null,
      "entitlements": null
    },
    "windows": {
      "certificateThumbprint": null,
      "timestampUrl": null,
      "webviewInstallMode": {
        "type": "downloadBootstrapper",
        "silent": true
      }
    }
  }
}
```

### Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ['VITE_'],
  build: {
    target: 'esnext',
    minify: 'esbuild',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'graph-vendor': ['react-force-graph-2d'],
        },
      },
    },
  },
});
```

### Package.json Dependencies

```json
{
  "name": "futurnal-desktop",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "tauri": "tauri",
    "test": "vitest",
    "test:e2e": "playwright test",
    "lint": "eslint . --ext ts,tsx"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.0.0",
    "@tauri-apps/plugin-store": "^2.0.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-router-dom": "^6.26.0",
    "zustand": "^4.5.0",
    "@tanstack/react-query": "^5.50.0",
    "firebase": "^10.12.0",
    "@stripe/stripe-js": "^4.1.0",
    "react-force-graph-2d": "^1.25.0",
    "lucide-react": "^0.400.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.4.0"
  },
  "devDependencies": {
    "@tauri-apps/cli": "^2.0.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "tailwindcss": "^4.0.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "vitest": "^2.0.0",
    "@playwright/test": "^1.45.0",
    "eslint": "^9.0.0"
  }
}
```

### Rust Main Entry Point

```rust
// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            commands::search::search_query,
            commands::search::get_search_history,
            commands::connectors::list_sources,
            commands::connectors::pause_source,
            commands::connectors::resume_source,
            commands::connectors::get_orchestrator_status,
            commands::privacy::get_consent,
            commands::privacy::grant_consent,
            commands::privacy::revoke_consent,
            commands::privacy::get_audit_logs,
        ])
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### React Entry Point

```tsx
// src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles/globals.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/build.yml
name: Build Desktop App

on:
  push:
    branches: [main, feat/p1-archivist]
  pull_request:
    branches: [main]

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        platform: [macos-latest, ubuntu-22.04, windows-latest]

    runs-on: ${{ matrix.platform }}

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install Rust stable
        uses: dtolnay/rust-action@stable

      - name: Install dependencies (ubuntu only)
        if: matrix.platform == 'ubuntu-22.04'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf

      - name: Install frontend dependencies
        run: npm ci

      - name: Build Tauri app
        uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tagName: v__VERSION__
          releaseName: 'Futurnal v__VERSION__'
          releaseBody: 'See the assets to download and install this version.'
          releaseDraft: true
          prerelease: false
```

## Implementation Details

### Week 1: Project Initialization

**Day 1-2: Scaffold**
1. Initialize Tauri project: `npm create tauri-app@latest futurnal-desktop`
2. Select React + TypeScript template
3. Configure Vite with path aliases
4. Setup TailwindCSS v4

**Day 3-4: Configuration**
1. Configure `tauri.conf.json` with app settings
2. Setup capabilities and permissions
3. Create icon assets (all required sizes)
4. Configure environment variables

**Day 5: CI/CD**
1. Create GitHub Actions workflow
2. Configure build matrix for all platforms
3. Setup artifact uploads
4. Test build pipeline

### Week 2: Integration Setup

**Day 1-2: Rust Commands**
1. Create command module structure
2. Implement placeholder commands
3. Test IPC communication
4. Setup error handling

**Day 3-4: Frontend Foundation**
1. Setup React Router
2. Create page components
3. Implement layout structure
4. Setup Zustand stores

**Day 5: Testing**
1. Configure Vitest
2. Configure Playwright
3. Write initial smoke tests

## Acceptance Criteria

- [ ] Tauri app compiles on macOS, Windows, Linux
- [ ] React frontend loads without errors
- [ ] Tauri invoke commands work (test with ping)
- [ ] Environment variables accessible in frontend
- [ ] Build artifacts generated for all platforms
- [ ] GitHub Actions pipeline passes
- [ ] Initial load time <2s (development mode)
- [ ] No console errors on startup
- [ ] Window management works (resize, minimize, maximize)
- [ ] DevTools accessible in development

## Test Plan

### Unit Tests
```typescript
// tests/unit/setup.test.ts
import { describe, it, expect } from 'vitest';

describe('App Setup', () => {
  it('should render without crashing', () => {
    // Test React app renders
  });

  it('should have correct environment variables', () => {
    expect(import.meta.env.VITE_APP_NAME).toBeDefined();
  });
});
```

### Integration Tests
```typescript
// tests/e2e/startup.spec.ts
import { test, expect } from '@playwright/test';

test('app starts successfully', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/Futurnal/);
});
```

## Dependencies

- Node.js 20+
- Rust stable (1.70+)
- Tauri CLI 2.x
- Platform-specific build tools (Xcode, Visual Studio, GTK)

## Open Questions

- Should we implement auto-update from the start or defer?
- Code signing certificates: self-signed for development or Apple/Microsoft?
- Should the app minimize to system tray?

## Next Steps

After framework scaffold complete:
1. Proceed to Module 02 (Design System & Components)
2. Begin Module 09 (IPC & API Layer) in parallel

**This module establishes the cross-platform foundation for the Futurnal desktop experience.**
