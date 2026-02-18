//! Overlay window command handlers.
//!
//! Provides IPC commands for managing the global command palette overlay window.
//! On macOS, this uses NSPanel for Spotlight-like behavior (doesn't steal focus).
//! On other platforms, it uses a standard frameless, transparent window.

use std::sync::atomic::Ordering;
use tauri::{command, AppHandle, Manager};

use crate::MAIN_WINDOW_WAS_VISIBLE;

#[cfg(target_os = "macos")]
use tauri_nspanel::ManagerExt;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSApp, NSApplication, NSApplicationActivationPolicy};

/// Toggle the command palette overlay visibility.
///
/// Returns true if the window is now visible, false if hidden.
/// Tracks main window visibility to properly restore state on hide.
#[command]
pub async fn toggle_command_palette(app: AppHandle) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        match app.get_webview_panel("command-palette") {
            Ok(panel) => {
                if panel.is_visible() {
                    // Hiding - use same logic as hide_command_palette
                    panel.order_out(None);
                    let should_restore = MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst);

                    if should_restore {
                        unsafe {
                            let ns_app = NSApp();
                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
                        }
                        if let Some(main_window) = app.get_webview_window("main") {
                            let _ = main_window.show();
                            let _ = main_window.set_focus();
                        }
                        log::debug!("Command palette panel hidden, restored main window");
                    } else {
                        unsafe {
                            let ns_app = NSApp();
                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
                        }
                        log::debug!("Command palette panel hidden, keeping app in background");
                    }
                    Ok(false)
                } else {
                    // Showing - always force was_visible=true to ensure restoration
                    // (macOS reports misleading values when window is on different space)
                    if let Some(main_window) = app.get_webview_window("main") {
                        let is_visible = main_window.is_visible().unwrap_or(false);
                        let is_minimized = main_window.is_minimized().unwrap_or(false);
                        MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);

                        if is_visible && !is_minimized {
                            let _ = main_window.hide();
                            log::debug!("Hiding main window for Spotlight mode");
                        }
                    } else {
                        MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);
                    }

                    unsafe {
                        let ns_app = NSApp();
                        ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
                    }
                    panel.order_front_regardless();
                    log::debug!("Command palette panel shown with Accessory policy");
                    Ok(true)
                }
            }
            Err(e) => Err(format!("Command palette panel not found: {:?}", e))
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Some(window) = app.get_webview_window("command-palette") {
            let is_visible = window.is_visible().map_err(|e| e.to_string())?;
            if is_visible {
                window.hide().map_err(|e| e.to_string())?;
                // Restore main window if it was visible
                if MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst) {
                    if let Some(main_window) = app.get_webview_window("main") {
                        let _ = main_window.show();
                        let _ = main_window.set_focus();
                    }
                }
                log::debug!("Command palette overlay hidden");
                Ok(false)
            } else {
                // Track main window state before showing
                if let Some(main_window) = app.get_webview_window("main") {
                    let main_visible = main_window.is_visible().unwrap_or(false);
                    MAIN_WINDOW_WAS_VISIBLE.store(main_visible, Ordering::SeqCst);
                } else {
                    MAIN_WINDOW_WAS_VISIBLE.store(false, Ordering::SeqCst);
                }

                window.center().map_err(|e| e.to_string())?;
                window.show().map_err(|e| e.to_string())?;
                window.set_focus().map_err(|e| e.to_string())?;
                log::debug!("Command palette overlay shown");
                Ok(true)
            }
        } else {
            Err("Command palette window not found".to_string())
        }
    }
}

/// Hide the command palette overlay.
///
/// Uses MAIN_WINDOW_WAS_VISIBLE to determine whether to restore the main window
/// or keep the app in background mode. This prevents crashes when the palette
/// was triggered while the app was minimized or in background.
#[command]
pub async fn hide_command_palette(app: AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        // Hide the panel first
        if let Ok(panel) = app.get_webview_panel("command-palette") {
            panel.order_out(None);
        }

        // Check if we should restore main window based on its state before palette was shown
        let should_restore = MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst);

        if should_restore {
            // Main window was visible before - restore it
            unsafe {
                let ns_app = NSApp();
                ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
            }
            if let Some(main_window) = app.get_webview_window("main") {
                let _ = main_window.show();
                let _ = main_window.set_focus();
            }
            log::debug!("Command palette hidden via command, restored main window");
        } else {
            // App was minimized/hidden - keep it that way
            // We need to cycle through Regular to prevent macOS from thinking app should terminate,
            // then back to Accessory to keep the app hidden but alive
            unsafe {
                let ns_app = NSApp();
                ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
                ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
            }
            log::debug!("Command palette hidden via command, keeping app in background");
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Some(window) = app.get_webview_window("command-palette") {
            window.hide().map_err(|e| e.to_string())?;
        }
        // Only restore main window if it was visible before
        if MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst) {
            if let Some(main_window) = app.get_webview_window("main") {
                let _ = main_window.show();
                let _ = main_window.set_focus();
            }
        }
        log::debug!("Command palette overlay hidden via command");
    }
    Ok(())
}

/// Show the command palette overlay.
///
/// Tracks main window visibility before showing, so hide_command_palette knows
/// whether to restore it or keep the app in background.
#[command]
pub async fn show_command_palette(app: AppHandle) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(panel) = app.get_webview_panel("command-palette") {
            // Always force was_visible=true to ensure restoration after closing
            // (macOS reports misleading values when window is on different space)
            if let Some(main_window) = app.get_webview_window("main") {
                let is_visible = main_window.is_visible().unwrap_or(false);
                let is_minimized = main_window.is_minimized().unwrap_or(false);
                MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);
                log::debug!("show_command_palette: main window state - visible: {}, minimized: {} (forcing was_visible=true)",
                           is_visible, is_minimized);

                // Only hide if it was actually visible on current space
                if is_visible && !is_minimized {
                    let _ = main_window.hide();
                    log::debug!("Hiding main window for Spotlight mode");
                }
            } else {
                MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);
            }

            // Set Accessory policy BEFORE showing panel to prevent main window from appearing
            unsafe {
                let ns_app = NSApp();
                ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
            }
            // Use order_front_regardless to show WITHOUT activating the app
            panel.order_front_regardless();
            log::debug!("Command palette panel shown via command with Accessory policy");
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Track main window visibility before showing palette
        if let Some(main_window) = app.get_webview_window("main") {
            let is_visible = main_window.is_visible().unwrap_or(false);
            MAIN_WINDOW_WAS_VISIBLE.store(is_visible, Ordering::SeqCst);
        } else {
            MAIN_WINDOW_WAS_VISIBLE.store(false, Ordering::SeqCst);
        }

        if let Some(window) = app.get_webview_window("command-palette") {
            window.center().map_err(|e| e.to_string())?;
            window.show().map_err(|e| e.to_string())?;
            window.set_focus().map_err(|e| e.to_string())?;
            log::debug!("Command palette overlay shown via command");
        }
    }
    Ok(())
}

/// Check if the command palette overlay is currently visible.
#[command]
pub async fn is_command_palette_visible(app: AppHandle) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(panel) = app.get_webview_panel("command-palette") {
            Ok(panel.is_visible())
        } else {
            Ok(false)
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        if let Some(window) = app.get_webview_window("command-palette") {
            window.is_visible().map_err(|e| e.to_string())
        } else {
            Ok(false)
        }
    }
}
