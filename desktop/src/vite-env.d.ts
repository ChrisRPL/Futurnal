/// <reference types="vite/client" />

// Legacy Electron API placeholder (not used in Tauri, but referenced in some code)
interface Window {
  electron?: {
    invoke: (channel: string, ...args: unknown[]) => Promise<unknown>;
  };
}
