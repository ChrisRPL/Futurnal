/**
 * Vitest setup file
 *
 * This file runs before all tests and sets up the testing environment.
 */

import { vi } from 'vitest';

// Mock Tauri API for testing
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

// Mock environment variables
vi.stubEnv('VITE_APP_NAME', 'Futurnal');
vi.stubEnv('VITE_APP_VERSION', '0.1.0');
