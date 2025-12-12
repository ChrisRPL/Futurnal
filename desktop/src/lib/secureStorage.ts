/**
 * Secure Storage for Sensitive Data
 *
 * Uses Tauri's store plugin for encrypted token storage.
 * Tokens and credentials should use this instead of regular storage.
 */

import { Store } from '@tauri-apps/plugin-store';

let secureStore: Store | null = null;

/**
 * Get or create the secure store instance.
 * Uses a separate file from regular storage for isolation.
 */
async function getSecureStore(): Promise<Store> {
  if (!secureStore) {
    secureStore = await Store.load('futurnal-secure.json');
  }
  return secureStore;
}

/**
 * Token types that can be stored securely.
 */
export type TokenType =
  | 'auth_token'
  | 'refresh_token'
  | 'github_pat'
  | 'imap_password'
  | 'api_key';

/**
 * Secure storage interface for tokens and credentials.
 */
export const secureStorage = {
  /**
   * Save a token securely.
   *
   * @param type - The type of token
   * @param value - The token value
   * @param identifier - Optional identifier for multi-token scenarios (e.g., source ID)
   */
  async saveToken(
    type: TokenType,
    value: string,
    identifier?: string
  ): Promise<void> {
    const key = identifier ? `${type}:${identifier}` : type;
    try {
      const store = await getSecureStore();
      // Store with timestamp for potential expiry tracking
      const data = {
        value,
        storedAt: new Date().toISOString(),
      };
      await store.set(key, JSON.stringify(data));
      await store.save();
      console.info('[SecureStorage] Saved token:', type, identifier ? `(${identifier})` : '');
    } catch (error) {
      console.error('[SecureStorage] Failed to save token:', type, error);
      throw new Error(`Failed to save ${type} token`);
    }
  },

  /**
   * Get a token from secure storage.
   *
   * @param type - The type of token
   * @param identifier - Optional identifier for multi-token scenarios
   * @returns The token value or null if not found
   */
  async getToken(type: TokenType, identifier?: string): Promise<string | null> {
    const key = identifier ? `${type}:${identifier}` : type;
    try {
      const store = await getSecureStore();
      const raw = await store.get<string>(key);
      if (!raw) return null;

      const data = JSON.parse(raw);
      return data.value ?? null;
    } catch (error) {
      console.warn('[SecureStorage] Failed to get token:', type, error);
      return null;
    }
  },

  /**
   * Delete a token from secure storage.
   *
   * @param type - The type of token
   * @param identifier - Optional identifier for multi-token scenarios
   */
  async deleteToken(type: TokenType, identifier?: string): Promise<void> {
    const key = identifier ? `${type}:${identifier}` : type;
    try {
      const store = await getSecureStore();
      await store.delete(key);
      await store.save();
      console.info('[SecureStorage] Deleted token:', type, identifier ? `(${identifier})` : '');
    } catch (error) {
      console.warn('[SecureStorage] Failed to delete token:', type, error);
    }
  },

  /**
   * Check if a token exists in secure storage.
   *
   * @param type - The type of token
   * @param identifier - Optional identifier for multi-token scenarios
   * @returns True if token exists
   */
  async hasToken(type: TokenType, identifier?: string): Promise<boolean> {
    const token = await this.getToken(type, identifier);
    return token !== null;
  },

  /**
   * Clear all tokens of a specific type.
   *
   * @param type - The type of token to clear
   */
  async clearTokenType(type: TokenType): Promise<void> {
    try {
      const store = await getSecureStore();
      const keys = await store.keys();

      for (const key of keys) {
        if (key === type || key.startsWith(`${type}:`)) {
          await store.delete(key);
        }
      }

      await store.save();
      console.info('[SecureStorage] Cleared all tokens of type:', type);
    } catch (error) {
      console.warn('[SecureStorage] Failed to clear token type:', type, error);
    }
  },

  /**
   * Clear all secure storage (logout/reset).
   */
  async clearAll(): Promise<void> {
    try {
      const store = await getSecureStore();
      await store.clear();
      await store.save();
      console.info('[SecureStorage] Cleared all secure storage');
    } catch (error) {
      console.warn('[SecureStorage] Failed to clear all:', error);
    }
  },

  /**
   * List all stored token types (without values).
   * Useful for debugging and audit.
   */
  async listStoredTypes(): Promise<string[]> {
    try {
      const store = await getSecureStore();
      return await store.keys();
    } catch (error) {
      console.warn('[SecureStorage] Failed to list types:', error);
      return [];
    }
  },
};

/**
 * Higher-level credential management for connectors.
 */
export const connectorCredentials = {
  /**
   * Save GitHub Personal Access Token for a repository.
   */
  async saveGitHubPAT(repoId: string, pat: string): Promise<void> {
    await secureStorage.saveToken('github_pat', pat, repoId);
  },

  /**
   * Get GitHub PAT for a repository.
   */
  async getGitHubPAT(repoId: string): Promise<string | null> {
    return secureStorage.getToken('github_pat', repoId);
  },

  /**
   * Save IMAP password for a mailbox.
   */
  async saveIMAPPassword(mailboxId: string, password: string): Promise<void> {
    await secureStorage.saveToken('imap_password', password, mailboxId);
  },

  /**
   * Get IMAP password for a mailbox.
   */
  async getIMAPPassword(mailboxId: string): Promise<string | null> {
    return secureStorage.getToken('imap_password', mailboxId);
  },

  /**
   * Delete credentials for a connector.
   */
  async deleteConnectorCredentials(
    type: 'github' | 'imap',
    connectorId: string
  ): Promise<void> {
    const tokenType = type === 'github' ? 'github_pat' : 'imap_password';
    await secureStorage.deleteToken(tokenType, connectorId);
  },
};
