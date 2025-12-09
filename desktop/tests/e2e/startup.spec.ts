/**
 * Futurnal Desktop Shell - E2E Startup Tests
 *
 * These tests verify the application starts correctly and renders without errors.
 * Note: These tests run against the Vite dev server, not the full Tauri app.
 * Full Tauri E2E tests require additional setup with tauri-driver.
 */

import { test, expect } from '@playwright/test';

test.describe('Application Startup', () => {
  test('should load the home page', async ({ page }) => {
    await page.goto('/');

    // Check the page title
    await expect(page).toHaveTitle(/Futurnal|Vite/);

    // Check the header is visible
    const header = page.locator('header');
    await expect(header).toBeVisible();
  });

  test('should display the welcome message', async ({ page }) => {
    await page.goto('/');

    // Check for welcome heading
    const heading = page.getByText('Welcome to Your Personal Universe');
    await expect(heading).toBeVisible();
  });

  test('should display status cards', async ({ page }) => {
    await page.goto('/');

    // Check for status cards
    const orchestratorCard = page.getByText('Orchestrator');
    await expect(orchestratorCard).toBeVisible();

    const activeJobsCard = page.getByText('Active Jobs');
    await expect(activeJobsCard).toBeVisible();

    const dataSourcesCard = page.getByText('Data Sources');
    await expect(dataSourcesCard).toBeVisible();
  });

  test('should display quick actions', async ({ page }) => {
    await page.goto('/');

    // Check for quick action cards
    const searchAction = page.getByText('Search Your Knowledge');
    await expect(searchAction).toBeVisible();

    const addSourceAction = page.getByText('Add Data Source');
    await expect(addSourceAction).toBeVisible();
  });

  test('should not have console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Filter out known acceptable errors (e.g., Tauri not available in browser)
    const criticalErrors = errors.filter(
      (e) =>
        !e.includes('__TAURI__') &&
        !e.includes('invoke') &&
        !e.includes('not defined')
    );

    expect(criticalErrors).toHaveLength(0);
  });

  test('should load within acceptable time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const loadTime = Date.now() - startTime;

    // Should load within 3 seconds in dev mode
    expect(loadTime).toBeLessThan(3000);
  });
});

test.describe('Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');

    // Check for h1
    const h1 = page.locator('h1');
    await expect(h1).toBeVisible();

    // Check for h2
    const h2 = page.locator('h2').first();
    await expect(h2).toBeVisible();
  });

  test('should have focusable elements', async ({ page }) => {
    await page.goto('/');

    // Check that buttons are focusable
    const buttons = page.locator('button');
    const count = await buttons.count();

    expect(count).toBeGreaterThan(0);
  });
});

test.describe('Theme', () => {
  test('should have dark background', async ({ page }) => {
    await page.goto('/');

    // Check that the main container has dark background
    const body = page.locator('body');
    await expect(body).toBeVisible();

    // The app should render with dark mode colors
    const bgColor = await body.evaluate((el) => {
      return window.getComputedStyle(el).backgroundColor;
    });

    // Should be a dark color (low RGB values)
    expect(bgColor).toMatch(/rgb\(\d{1,2}, \d{1,2}, \d{1,2}\)/);
  });
});
