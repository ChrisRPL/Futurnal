/**
 * Subscription Management Utilities
 *
 * Opens browser to landing page for subscription-related actions.
 * All payment handling happens on the web - desktop app only verifies tier.
 */

import { open } from '@tauri-apps/plugin-shell';

const PORTAL_URL = import.meta.env.VITE_LANDING_URL || 'https://futurnal.com';

/**
 * Open subscription management portal (Stripe Customer Portal)
 * For Pro users to manage billing, cancel, view invoices
 */
export async function openSubscriptionPortal(): Promise<void> {
  await open(`${PORTAL_URL}/account`);
}

/**
 * Open pricing/upgrade page
 * For Free users hitting feature limits
 */
export async function openUpgradePage(): Promise<void> {
  await open(`${PORTAL_URL}/pricing`);
}

/**
 * Open signup page
 * For unauthenticated users to create account
 */
export async function openSignupPage(): Promise<void> {
  await open(`${PORTAL_URL}/signup`);
}
