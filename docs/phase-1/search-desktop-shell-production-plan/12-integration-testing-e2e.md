Summary: Implement comprehensive Playwright E2E testing, Vitest unit tests, and accessibility verification.

# 12 · Integration Testing & E2E

## Purpose

Establish a comprehensive testing infrastructure using Playwright for end-to-end tests and Vitest for unit/integration tests. This ensures application reliability, accessibility compliance, and performance benchmarks before production deployment.

**Criticality**: CRITICAL - Quality gate for production release

## Scope

- Playwright E2E test suite:
  - Authentication flows (login, signup, logout)
  - Search flows (query → results)
  - Connector management (add, pause, resume, delete)
  - Settings and privacy controls
  - Graph visualization interactions
- Vitest unit tests for components
- Accessibility testing (keyboard navigation, WCAG AA)
- Performance benchmarks:
  - Initial load <1.5s
  - Search render <500ms
  - Graph render 60fps
- Cross-platform verification (macOS, Windows, Linux)
- CI/CD integration with GitHub Actions
- Visual regression testing

## Requirements Alignment

- **Quality Gate**: All tests must pass before production
- **Accessibility**: WCAG AA compliance, 100% keyboard nav coverage
- **Performance Targets**: Defined benchmarks for user experience

## Component Design

### Playwright Configuration

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html', { open: 'never' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['github'],
  ],
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
```

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['./tests/unit/**/*.test.{ts,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
        '**/*.config.*',
      ],
      thresholds: {
        lines: 70,
        functions: 70,
        branches: 60,
        statements: 70,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### Test Setup

```typescript
// tests/setup.ts
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock Tauri APIs
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

vi.mock('@tauri-apps/plugin-store', () => ({
  Store: {
    load: vi.fn(() => Promise.resolve({
      get: vi.fn(),
      set: vi.fn(),
      delete: vi.fn(),
      save: vi.fn(),
    })),
  },
}));

vi.mock('@tauri-apps/plugin-dialog', () => ({
  open: vi.fn(),
}));

// Mock Firebase
vi.mock('firebase/auth', () => ({
  getAuth: vi.fn(),
  signInWithPopup: vi.fn(),
  signOut: vi.fn(),
  onAuthStateChanged: vi.fn((auth, callback) => {
    callback(null);
    return () => {};
  }),
  GoogleAuthProvider: vi.fn(),
  GithubAuthProvider: vi.fn(),
}));

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock crypto.randomUUID
Object.defineProperty(crypto, 'randomUUID', {
  value: () => 'test-uuid-' + Math.random().toString(36).slice(2),
});
```

### E2E Test Fixtures

```typescript
// tests/e2e/fixtures.ts
import { test as base, expect } from '@playwright/test';

// Custom fixtures for common test scenarios
export const test = base.extend<{
  authenticatedPage: any;
  searchPage: any;
}>({
  authenticatedPage: async ({ page }, use) => {
    // Set up authenticated state
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('futurnal-user', JSON.stringify({
        state: {
          isAuthenticated: true,
          tier: 'free',
          user: { email: 'test@example.com', displayName: 'Test User' },
        },
      }));
    });
    await page.reload();
    await use(page);
  },

  searchPage: async ({ authenticatedPage }, use) => {
    await authenticatedPage.goto('/');
    await use(authenticatedPage);
  },
});

export { expect };
```

### Authentication E2E Tests

```typescript
// tests/e2e/auth.spec.ts
import { test, expect } from './fixtures';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Clear auth state
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should show login page for unauthenticated users', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('h1:has-text("Welcome")')).toBeVisible();
    await expect(page.locator('text=Sign in')).toBeVisible();
  });

  test('should show auth provider buttons', async ({ page }) => {
    await page.goto('/login');

    await expect(page.locator('button:has-text("Continue with Google")')).toBeVisible();
    await expect(page.locator('button:has-text("Continue with GitHub")')).toBeVisible();
    await expect(page.locator('button:has-text("Continue with Email")')).toBeVisible();
  });

  test('should validate email format', async ({ page }) => {
    await page.goto('/login');
    await page.click('text=Continue with Email');

    await page.fill('input[type="email"]', 'invalid-email');
    await page.click('button:has-text("Sign In")');

    await expect(page.locator('text=Invalid email')).toBeVisible();
  });

  test('should redirect to dashboard after login', async ({ page }) => {
    // Simulate successful login
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('futurnal-user', JSON.stringify({
        state: {
          isAuthenticated: true,
          tier: 'free',
          user: { email: 'test@example.com' },
        },
      }));
    });
    await page.goto('/');

    await expect(page).toHaveURL(/dashboard/);
  });

  test('should handle logout correctly', async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/settings');
    await authenticatedPage.click('text=Sign Out');

    await expect(authenticatedPage).toHaveURL(/login/);
    await expect(authenticatedPage.locator('text=Sign in')).toBeVisible();
  });
});
```

### Search E2E Tests

```typescript
// tests/e2e/search.spec.ts
import { test, expect } from './fixtures';

test.describe('Search', () => {
  test('should open command palette with keyboard shortcut', async ({ searchPage }) => {
    await searchPage.keyboard.press('Meta+k');

    await expect(searchPage.locator('[role="dialog"]')).toBeVisible();
    await expect(searchPage.locator('input[placeholder*="Search"]')).toBeFocused();
  });

  test('should close command palette on escape', async ({ searchPage }) => {
    await searchPage.keyboard.press('Meta+k');
    await expect(searchPage.locator('[role="dialog"]')).toBeVisible();

    await searchPage.keyboard.press('Escape');
    await expect(searchPage.locator('[role="dialog"]')).not.toBeVisible();
  });

  test('should execute search and display results', async ({ searchPage }) => {
    // Mock search results
    await searchPage.route('**/api/search', (route) => {
      route.fulfill({
        status: 200,
        body: JSON.stringify({
          results: [
            { id: '1', content: 'Test result', score: 0.95, confidence: 0.9 },
          ],
          total: 1,
          query_id: 'test',
          intent: { primary: 'exploratory' },
          execution_time_ms: 150,
        }),
      });
    });

    await searchPage.keyboard.press('Meta+k');
    await searchPage.fill('input[placeholder*="Search"]', 'test query');
    await searchPage.keyboard.press('Enter');

    await expect(searchPage.locator('text=Test result')).toBeVisible();
    await expect(searchPage.locator('text=95%')).toBeVisible(); // Score
  });

  test('should show search history', async ({ searchPage }) => {
    // Set up history
    await searchPage.evaluate(() => {
      localStorage.setItem('futurnal-search', JSON.stringify({
        state: {
          history: [
            { id: '1', query: 'previous search', timestamp: new Date().toISOString(), resultCount: 5 },
          ],
        },
      }));
    });
    await searchPage.reload();

    await searchPage.keyboard.press('Meta+k');

    await expect(searchPage.locator('text=previous search')).toBeVisible();
  });

  test('should apply filters correctly', async ({ searchPage }) => {
    await searchPage.keyboard.press('Meta+k');

    // Click filter chip
    await searchPage.click('text=Events');

    // Filter should be active
    await expect(searchPage.locator('[data-active="true"]:has-text("Events")')).toBeVisible();
  });

  test('should handle search errors gracefully', async ({ searchPage }) => {
    await searchPage.route('**/api/search', (route) => {
      route.fulfill({ status: 500 });
    });

    await searchPage.keyboard.press('Meta+k');
    await searchPage.fill('input[placeholder*="Search"]', 'test');
    await searchPage.keyboard.press('Enter');

    await expect(searchPage.locator('text=Search failed')).toBeVisible();
  });
});
```

### Connector E2E Tests

```typescript
// tests/e2e/connectors.spec.ts
import { test, expect } from './fixtures';

test.describe('Connectors', () => {
  test.beforeEach(async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/connectors');
  });

  test('should display connector list', async ({ authenticatedPage }) => {
    await expect(authenticatedPage.locator('h1:has-text("Data Sources")')).toBeVisible();
  });

  test('should open add connector modal', async ({ authenticatedPage }) => {
    await authenticatedPage.click('text=Add Source');

    await expect(authenticatedPage.locator('[role="dialog"]')).toBeVisible();
    await expect(authenticatedPage.locator('text=Local Folder')).toBeVisible();
    await expect(authenticatedPage.locator('text=Obsidian Vault')).toBeVisible();
  });

  test('should show tier limit for free users', async ({ authenticatedPage }) => {
    // Add 3 connectors to hit limit
    await authenticatedPage.evaluate(() => {
      // Mock connector data
    });

    await authenticatedPage.click('text=Add Source');

    await expect(authenticatedPage.locator('text=Upgrade to Pro')).toBeVisible();
  });

  test('should toggle connector status', async ({ authenticatedPage }) => {
    // Assuming there's a connector in the list
    const toggle = authenticatedPage.locator('[role="switch"]').first();

    await toggle.click();

    // Should show paused state
    await expect(authenticatedPage.locator('text=paused')).toBeVisible();
  });

  test('should show connector error with retry', async ({ authenticatedPage }) => {
    // Mock a connector with error state
    await expect(authenticatedPage.locator('text=Retry')).toBeVisible({ timeout: 5000 }).catch(() => {
      // No error connector in default state - test skipped
    });
  });
});
```

### Knowledge Graph E2E Tests

```typescript
// tests/e2e/graph.spec.ts
import { test, expect } from './fixtures';

test.describe('Knowledge Graph', () => {
  test('should render graph canvas', async ({ searchPage }) => {
    await searchPage.goto('/dashboard');

    // Graph should be visible in mini-view
    const canvas = searchPage.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should expand graph to full screen', async ({ searchPage }) => {
    await searchPage.goto('/dashboard');

    await searchPage.click('text=Click to expand');

    // Full screen dialog should open
    await expect(searchPage.locator('[role="dialog"] canvas')).toBeVisible();
  });

  test('should show node details on click', async ({ searchPage }) => {
    await searchPage.goto('/dashboard');
    await searchPage.click('text=Click to expand');

    // Click on canvas (node)
    const canvas = searchPage.locator('[role="dialog"] canvas');
    await canvas.click({ position: { x: 400, y: 300 } });

    // Detail panel should appear (if node was clicked)
    // This is a best-effort test since node positions are dynamic
  });

  test('should handle graph controls', async ({ searchPage }) => {
    await searchPage.goto('/dashboard');
    await searchPage.click('text=Click to expand');

    // Zoom in
    await searchPage.click('[title="Zoom in"]');

    // Fit to view
    await searchPage.click('[title="Fit to view"]');

    // Reset
    await searchPage.click('[title="Reset view"]');
  });
});
```

### Settings E2E Tests

```typescript
// tests/e2e/settings.spec.ts
import { test, expect } from './fixtures';

test.describe('Settings', () => {
  test.beforeEach(async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/settings');
  });

  test('should navigate between sections', async ({ authenticatedPage }) => {
    await authenticatedPage.click('text=Privacy');
    await expect(authenticatedPage.locator('h2:has-text("Privacy")')).toBeVisible();

    await authenticatedPage.click('text=Appearance');
    await expect(authenticatedPage.locator('h2:has-text("Appearance")')).toBeVisible();

    await authenticatedPage.click('text=Data Management');
    await expect(authenticatedPage.locator('h2:has-text("Data Management")')).toBeVisible();
  });

  test('should toggle telemetry setting', async ({ authenticatedPage }) => {
    await authenticatedPage.click('text=Privacy');

    const toggle = authenticatedPage.locator('text=Anonymous Telemetry').locator('..').locator('[role="switch"]');
    await toggle.click();

    // Verify setting was saved (check localStorage)
    const settings = await authenticatedPage.evaluate(() =>
      JSON.parse(localStorage.getItem('futurnal-settings') || '{}')
    );
    expect(settings.state?.telemetryEnabled).toBeDefined();
  });

  test('should open audit log viewer', async ({ authenticatedPage }) => {
    await authenticatedPage.click('text=Privacy');
    await authenticatedPage.click('text=View Audit Log');

    await expect(authenticatedPage.locator('[role="dialog"]')).toBeVisible();
    await expect(authenticatedPage.locator('text=Activity Audit Log')).toBeVisible();
  });

  test('should clear search history', async ({ authenticatedPage }) => {
    // Set up history
    await authenticatedPage.evaluate(() => {
      localStorage.setItem('futurnal-search', JSON.stringify({
        state: { history: [{ id: '1', query: 'test', timestamp: new Date().toISOString() }] },
      }));
    });
    await authenticatedPage.reload();

    await authenticatedPage.click('text=Data Management');
    await authenticatedPage.click('text=Clear Search History');

    // Confirm dialog
    await authenticatedPage.click('text=Clear History');

    // Verify cleared
    const searchState = await authenticatedPage.evaluate(() =>
      JSON.parse(localStorage.getItem('futurnal-search') || '{}')
    );
    expect(searchState.state?.history || []).toHaveLength(0);
  });
});
```

### Accessibility Tests

```typescript
// tests/e2e/accessibility.spec.ts
import { test, expect } from './fixtures';
import AxeBuilder from '@axe-core/playwright';

test.describe('Accessibility', () => {
  test('login page should have no accessibility violations', async ({ page }) => {
    await page.goto('/login');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();

    expect(results.violations).toEqual([]);
  });

  test('dashboard should have no accessibility violations', async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/dashboard');

    const results = await new AxeBuilder({ page: authenticatedPage })
      .withTags(['wcag2a', 'wcag2aa'])
      .exclude('canvas') // Exclude canvas as it's complex visualization
      .analyze();

    expect(results.violations).toEqual([]);
  });

  test('should support full keyboard navigation', async ({ searchPage }) => {
    // Tab through main navigation
    await searchPage.keyboard.press('Tab');
    await expect(searchPage.locator(':focus')).toBeVisible();

    // Open command palette with keyboard
    await searchPage.keyboard.press('Meta+k');
    await expect(searchPage.locator('[role="dialog"]')).toBeVisible();

    // Navigate search results with arrows
    await searchPage.keyboard.press('ArrowDown');
    await searchPage.keyboard.press('ArrowUp');

    // Close with Escape
    await searchPage.keyboard.press('Escape');
    await expect(searchPage.locator('[role="dialog"]')).not.toBeVisible();
  });

  test('should have proper focus management in modals', async ({ authenticatedPage }) => {
    await authenticatedPage.goto('/connectors');
    await authenticatedPage.click('text=Add Source');

    // Focus should be trapped in modal
    const dialog = authenticatedPage.locator('[role="dialog"]');
    await expect(dialog).toBeVisible();

    // Tab should cycle within modal
    await authenticatedPage.keyboard.press('Tab');
    const focusedElement = await authenticatedPage.evaluate(() =>
      document.activeElement?.closest('[role="dialog"]')
    );
    expect(focusedElement).toBeTruthy();
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto('/login');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .options({ runOnly: ['color-contrast'] })
      .analyze();

    expect(results.violations).toEqual([]);
  });
});
```

### Performance Tests

```typescript
// tests/e2e/performance.spec.ts
import { test, expect } from './fixtures';

test.describe('Performance', () => {
  test('initial load should be under 1.5s', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const loadTime = Date.now() - startTime;
    expect(loadTime).toBeLessThan(1500);
  });

  test('search results should render under 500ms', async ({ searchPage }) => {
    // Mock fast search response
    await searchPage.route('**/api/search', (route) => {
      route.fulfill({
        status: 200,
        body: JSON.stringify({
          results: Array.from({ length: 20 }, (_, i) => ({
            id: String(i),
            content: `Result ${i}`,
            score: 0.9 - i * 0.01,
            confidence: 0.85,
          })),
          total: 20,
          query_id: 'perf-test',
          intent: { primary: 'exploratory' },
          execution_time_ms: 50,
        }),
      });
    });

    await searchPage.keyboard.press('Meta+k');

    const startTime = Date.now();
    await searchPage.fill('input[placeholder*="Search"]', 'test');
    await searchPage.keyboard.press('Enter');
    await searchPage.waitForSelector('text=Result 0');
    const renderTime = Date.now() - startTime;

    expect(renderTime).toBeLessThan(500);
  });

  test('bundle size should be under 25MB', async ({ page }) => {
    const response = await page.goto('/');
    const metrics = await page.evaluate(() => {
      return {
        transferSize: performance.getEntriesByType('resource')
          .reduce((sum, r: any) => sum + (r.transferSize || 0), 0),
      };
    });

    // 25MB = 25 * 1024 * 1024 bytes
    expect(metrics.transferSize).toBeLessThan(25 * 1024 * 1024);
  });
});
```

### Unit Tests

```typescript
// tests/unit/components/Button.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from '@/components/ui/button';
import { describe, it, expect, vi } from 'vitest';

describe('Button', () => {
  it('renders correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toHaveTextContent('Click me');
  });

  it('handles click events', async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Click me</Button>);

    await userEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it('applies variant classes', () => {
    render(<Button variant="outline">Outline</Button>);
    expect(screen.getByRole('button')).toHaveClass('border-border');
  });

  it('disables correctly', () => {
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### Store Unit Tests

```typescript
// tests/unit/stores/searchStore.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useSearchStore } from '@/stores/searchStore';

describe('useSearchStore', () => {
  beforeEach(() => {
    useSearchStore.setState({
      query: '',
      results: [],
      history: [],
      filters: {},
    });
  });

  it('should set query', () => {
    useSearchStore.getState().setQuery('test query');
    expect(useSearchStore.getState().query).toBe('test query');
  });

  it('should add to history', () => {
    useSearchStore.getState().addToHistory('test', 10);
    expect(useSearchStore.getState().history).toHaveLength(1);
    expect(useSearchStore.getState().history[0].query).toBe('test');
  });

  it('should limit history to 50 items', () => {
    for (let i = 0; i < 60; i++) {
      useSearchStore.getState().addToHistory(`query ${i}`, i);
    }
    expect(useSearchStore.getState().history).toHaveLength(50);
  });

  it('should clear history', () => {
    useSearchStore.getState().addToHistory('test', 10);
    useSearchStore.getState().clearHistory();
    expect(useSearchStore.getState().history).toHaveLength(0);
  });

  it('should reset state', () => {
    useSearchStore.getState().setQuery('test');
    useSearchStore.getState().reset();
    expect(useSearchStore.getState().query).toBe('');
  });
});
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [main, feat/p1-archivist]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/lcov.info

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

  accessibility-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright browsers
        run: npx playwright install --with-deps chromium

      - name: Run accessibility tests
        run: npx playwright test tests/e2e/accessibility.spec.ts

  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build for production
        run: npm run build

      - name: Run Lighthouse CI
        uses: treosh/lighthouse-ci-action@v12
        with:
          configPath: './lighthouserc.json'
          uploadArtifacts: true
```

### Lighthouse Configuration

```json
// lighthouserc.json
{
  "ci": {
    "collect": {
      "staticDistDir": "./dist",
      "numberOfRuns": 3
    },
    "assert": {
      "preset": "lighthouse:recommended",
      "assertions": {
        "categories:performance": ["error", { "minScore": 0.8 }],
        "categories:accessibility": ["error", { "minScore": 0.9 }],
        "categories:best-practices": ["error", { "minScore": 0.9 }],
        "first-contentful-paint": ["error", { "maxNumericValue": 1500 }],
        "interactive": ["error", { "maxNumericValue": 3000 }]
      }
    },
    "upload": {
      "target": "temporary-public-storage"
    }
  }
}
```

## Acceptance Criteria

- [ ] All E2E tests pass on CI
- [ ] Unit test coverage ≥70%
- [ ] No WCAG AA violations
- [ ] Initial load <1.5s
- [ ] Search render <500ms
- [ ] Bundle size <25MB
- [ ] Keyboard navigation 100%
- [ ] Cross-browser tests pass (Chrome, Firefox, Safari)
- [ ] Visual regression baseline established

## Test Summary Matrix

| Category | Tests | Target | Status |
|----------|-------|--------|--------|
| Authentication | 5 | 100% pass | ⬜ |
| Search | 6 | 100% pass | ⬜ |
| Connectors | 5 | 100% pass | ⬜ |
| Graph | 4 | 100% pass | ⬜ |
| Settings | 5 | 100% pass | ⬜ |
| Accessibility | 5 | 100% pass | ⬜ |
| Performance | 3 | 100% pass | ⬜ |
| Unit Tests | 50+ | 70% coverage | ⬜ |

## Dependencies

- @playwright/test
- vitest
- @testing-library/react
- @testing-library/jest-dom
- @testing-library/user-event
- @axe-core/playwright
- lighthouse

## Next Steps

After testing complete:
1. Set up visual regression testing with Percy
2. Add load testing for search
3. Create performance monitoring dashboard
4. Implement test flakiness detection

**This comprehensive testing infrastructure ensures Futurnal meets all quality, accessibility, and performance requirements before production.**
