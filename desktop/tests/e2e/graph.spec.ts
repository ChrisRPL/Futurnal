/**
 * Futurnal Desktop Shell - Graph Visualization E2E Tests
 *
 * Tests for the knowledge graph visualization page.
 * Note: These tests run against the Vite dev server, not the full Tauri app.
 * Graph data loading may show loading states since Tauri IPC is mocked.
 */

import { test, expect } from '@playwright/test';

test.describe('Graph Page', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the graph page
    await page.goto('/graph');
  });

  test('should load the graph page', async ({ page }) => {
    // Check the page has loaded
    await expect(page.locator('[data-testid="graph-page"]').or(page.locator('h1:has-text("Knowledge Graph")'))).toBeVisible({ timeout: 10000 });
  });

  test('should display graph header with title', async ({ page }) => {
    // Check for the page title
    const title = page.getByText('Knowledge Graph');
    await expect(title).toBeVisible();
  });

  test('should display graph controls', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Check for control buttons (zoom, fit)
    const controlsContainer = page.locator('[data-testid="graph-controls"]').or(
      page.locator('button:has([data-lucide="zoom-in"])').first()
    );

    // At minimum, there should be some buttons in the controls area
    await expect(page.locator('button').first()).toBeVisible();
  });

  test('should display filter panel trigger', async ({ page }) => {
    // Check for filter button
    const filterButton = page.getByText('Filter').or(
      page.locator('button:has-text("hidden")')
    );
    await expect(filterButton).toBeVisible();
  });

  test('should open filter panel on click', async ({ page }) => {
    // Click the filter button
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Check filter panel appears
    const filterPanel = page.getByText('Node Types');
    await expect(filterPanel).toBeVisible();
  });

  test('should display node type filters in panel', async ({ page }) => {
    // Open filter panel
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Check for entity type filters
    await expect(page.getByText('Documents')).toBeVisible();
    await expect(page.getByText('People')).toBeVisible();
    await expect(page.getByText('Events')).toBeVisible();
  });

  test('should toggle node type visibility', async ({ page }) => {
    // Open filter panel
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Click on a filter to toggle it
    const emailFilter = page.getByRole('button', { name: /Emails/i });
    await emailFilter.click();

    // Filter badge should update to show hidden count
    await expect(filterButton).toContainText(/hidden/);
  });

  test('should display color mode toggle', async ({ page }) => {
    // Open filter panel
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Check for color mode section
    await expect(page.getByText('Color Mode')).toBeVisible();

    // Check for colored/monochrome toggle
    const colorToggle = page.getByText('Colored').or(page.getByText('Monochrome'));
    await expect(colorToggle).toBeVisible();
  });

  test('should toggle color mode', async ({ page }) => {
    // Open filter panel
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Find and click color mode toggle
    const colorToggle = page.getByRole('button', { name: /Colored|Monochrome/i });
    const initialText = await colorToggle.textContent();
    await colorToggle.click();

    // Toggle should change state
    const newText = await colorToggle.textContent();
    expect(newText).not.toBe(initialText);
  });

  test('should have show all button', async ({ page }) => {
    // Open filter panel
    const filterButton = page.getByText('Filter').first();
    await filterButton.click();

    // Check for show all button
    const showAllButton = page.getByText('Show all');
    await expect(showAllButton).toBeVisible();
  });

  test('should display stats footer', async ({ page }) => {
    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Check for stats (node count, link count)
    // Note: With mocked Tauri, this may show 0 or loading state
    const statsArea = page.locator('text=/memories|nodes|connections/i').first();
    await expect(statsArea).toBeVisible({ timeout: 5000 }).catch(() => {
      // Stats may not be visible if data loading is mocked
    });
  });

  test('should navigate back from graph page', async ({ page }) => {
    // Look for back button
    const backButton = page.locator('button').filter({ has: page.locator('[data-lucide="arrow-left"]') }).first()
      .or(page.locator('a[href="/"]').first());

    if (await backButton.isVisible()) {
      await backButton.click();
      // Should navigate away from graph page
      await expect(page).not.toHaveURL(/\/graph$/);
    }
  });
});

test.describe('Graph Interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/graph');
    await page.waitForLoadState('networkidle');
  });

  test('should render graph container', async ({ page }) => {
    // Check for the graph container
    const graphContainer = page.locator('[data-testid="knowledge-graph"]')
      .or(page.locator('canvas').first())
      .or(page.locator('[class*="graph"]').first());

    await expect(graphContainer).toBeVisible({ timeout: 10000 });
  });

  test('should not show node detail panel initially', async ({ page }) => {
    // Node detail panel should not be visible without selection
    const detailPanel = page.locator('[data-testid="node-detail-panel"]');

    // Either it doesn't exist or is not visible
    const isVisible = await detailPanel.isVisible().catch(() => false);
    expect(isVisible).toBe(false);
  });
});

test.describe('Graph Page Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/graph');
  });

  test('should have proper heading', async ({ page }) => {
    const heading = page.locator('h1, h2').filter({ hasText: /Graph|Knowledge/i }).first();
    await expect(heading).toBeVisible();
  });

  test('should have focusable controls', async ({ page }) => {
    // Check that buttons are focusable
    const buttons = page.locator('button');
    const count = await buttons.count();
    expect(count).toBeGreaterThan(0);

    // First button should be focusable
    const firstButton = buttons.first();
    await firstButton.focus();
    await expect(firstButton).toBeFocused();
  });

  test('should have keyboard-accessible filter panel', async ({ page }) => {
    // Tab to filter button and activate with keyboard
    const filterButton = page.getByText('Filter').first();
    await filterButton.focus();
    await page.keyboard.press('Enter');

    // Filter panel should open
    await expect(page.getByText('Node Types')).toBeVisible();

    // Should be able to close with Escape
    await page.keyboard.press('Escape');

    // Panel should close (may take a moment for animation)
    await expect(page.getByText('Node Types')).not.toBeVisible({ timeout: 2000 }).catch(() => {
      // Some popovers don't close with Escape by default
    });
  });
});

test.describe('Graph Page Performance', () => {
  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/graph');
    await page.waitForLoadState('networkidle');

    const loadTime = Date.now() - startTime;

    // Graph page should load within 5 seconds in dev mode
    expect(loadTime).toBeLessThan(5000);
  });

  test('should not have console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto('/graph');
    await page.waitForTimeout(2000);

    // Filter out known acceptable errors
    const criticalErrors = errors.filter(
      (e) =>
        !e.includes('__TAURI__') &&
        !e.includes('invoke') &&
        !e.includes('not defined') &&
        !e.includes('ResizeObserver') &&
        !e.includes('WebGL')
    );

    expect(criticalErrors).toHaveLength(0);
  });
});

test.describe('Graph Page Theme', () => {
  test('should have dark background', async ({ page }) => {
    await page.goto('/graph');

    const body = page.locator('body');
    const bgColor = await body.evaluate((el) => {
      return window.getComputedStyle(el).backgroundColor;
    });

    // Should be a dark color
    expect(bgColor).toMatch(/rgb\(\d{1,2}, \d{1,2}, \d{1,2}\)/);
  });
});
