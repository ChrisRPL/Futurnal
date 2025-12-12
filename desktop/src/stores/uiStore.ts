/**
 * UI Store - Zustand state management for ephemeral UI state
 *
 * Manages command palette, modals, notifications, and other transient UI state.
 * This store is NOT persisted - it resets on app restart.
 */

import { create } from 'zustand';

/**
 * Notification type for toast/alerts
 */
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message?: string;
  /** Duration in ms before auto-dismiss (0 = no auto-dismiss) */
  duration?: number;
}

interface UIState {
  // Command palette
  /** Whether command palette is open */
  isCommandPaletteOpen: boolean;
  /** Open the command palette */
  openCommandPalette: () => void;
  /** Close the command palette */
  closeCommandPalette: () => void;
  /** Toggle the command palette */
  toggleCommandPalette: () => void;

  // Sidebar
  /** Whether sidebar is collapsed */
  isSidebarCollapsed: boolean;
  /** Toggle sidebar collapsed state */
  toggleSidebar: () => void;
  /** Set sidebar collapsed state */
  setSidebarCollapsed: (collapsed: boolean) => void;

  // Modals
  /** Currently active modal ID */
  activeModal: string | null;
  /** Props passed to the active modal */
  modalProps: Record<string, unknown>;
  /** Open a modal by ID with optional props */
  openModal: (modalId: string, props?: Record<string, unknown>) => void;
  /** Close the active modal */
  closeModal: () => void;

  // Notifications
  /** Active notifications */
  notifications: Notification[];
  /** Add a notification */
  addNotification: (notification: Omit<Notification, 'id'>) => void;
  /** Remove a notification by ID */
  removeNotification: (id: string) => void;
  /** Clear all notifications */
  clearNotifications: () => void;

  // Detail panel
  /** Currently selected item ID for detail panel */
  selectedDetailId: string | null;
  /** Set selected detail item */
  setSelectedDetail: (id: string | null) => void;

  // Loading states
  /** Global loading overlay */
  isGlobalLoading: boolean;
  /** Loading message */
  loadingMessage: string | null;
  /** Show global loading overlay */
  showLoading: (message?: string) => void;
  /** Hide global loading overlay */
  hideLoading: () => void;
}

/**
 * Generate unique ID for notifications
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export const useUIStore = create<UIState>((set) => ({
  // Command palette - initial state
  isCommandPaletteOpen: false,
  openCommandPalette: () => set({ isCommandPaletteOpen: true }),
  closeCommandPalette: () => set({ isCommandPaletteOpen: false }),
  toggleCommandPalette: () =>
    set((state) => ({ isCommandPaletteOpen: !state.isCommandPaletteOpen })),

  // Sidebar - initial state
  isSidebarCollapsed: false,
  toggleSidebar: () =>
    set((state) => ({ isSidebarCollapsed: !state.isSidebarCollapsed })),
  setSidebarCollapsed: (collapsed) => set({ isSidebarCollapsed: collapsed }),

  // Modals - initial state
  activeModal: null,
  modalProps: {},
  openModal: (modalId, props = {}) =>
    set({ activeModal: modalId, modalProps: props }),
  closeModal: () => set({ activeModal: null, modalProps: {} }),

  // Notifications - initial state
  notifications: [],
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: generateId() },
      ],
    })),
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
  clearNotifications: () => set({ notifications: [] }),

  // Detail panel - initial state
  selectedDetailId: null,
  setSelectedDetail: (id) => set({ selectedDetailId: id }),

  // Loading states - initial state
  isGlobalLoading: false,
  loadingMessage: null,
  showLoading: (message) =>
    set({ isGlobalLoading: true, loadingMessage: message ?? null }),
  hideLoading: () => set({ isGlobalLoading: false, loadingMessage: null }),
}));

// Convenience hooks
export const useCommandPalette = () => ({
  isOpen: useUIStore((state) => state.isCommandPaletteOpen),
  open: useUIStore((state) => state.openCommandPalette),
  close: useUIStore((state) => state.closeCommandPalette),
  toggle: useUIStore((state) => state.toggleCommandPalette),
});

export const useSidebar = () => ({
  isCollapsed: useUIStore((state) => state.isSidebarCollapsed),
  toggle: useUIStore((state) => state.toggleSidebar),
  setCollapsed: useUIStore((state) => state.setSidebarCollapsed),
});

export const useModal = () => ({
  activeModal: useUIStore((state) => state.activeModal),
  modalProps: useUIStore((state) => state.modalProps),
  open: useUIStore((state) => state.openModal),
  close: useUIStore((state) => state.closeModal),
});

export const useNotifications = () => ({
  notifications: useUIStore((state) => state.notifications),
  add: useUIStore((state) => state.addNotification),
  remove: useUIStore((state) => state.removeNotification),
  clear: useUIStore((state) => state.clearNotifications),
});

export const useGlobalLoading = () => ({
  isLoading: useUIStore((state) => state.isGlobalLoading),
  message: useUIStore((state) => state.loadingMessage),
  show: useUIStore((state) => state.showLoading),
  hide: useUIStore((state) => state.hideLoading),
});
