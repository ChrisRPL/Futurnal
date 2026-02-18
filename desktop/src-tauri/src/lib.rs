//! Futurnal Desktop Shell - Tauri Application Library
//!
//! This is the main library for the Futurnal desktop application.
//! It provides the bridge between the React frontend and the Python backend.

mod commands;
mod python;

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// Track whether main window was visible before showing command palette.
/// This allows proper restoration when command palette is dismissed.
/// If app was minimized/hidden, we keep it that way after closing palette.
pub static MAIN_WINDOW_WAS_VISIBLE: AtomicBool = AtomicBool::new(true);

#[cfg(target_os = "macos")]
use tauri_nspanel::ManagerExt;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSApp, NSApplication, NSApplicationActivationPolicy};

#[cfg(target_os = "macos")]
use cocoa::appkit::NSWindowCollectionBehavior;

/// Run the Tauri application.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting Futurnal Desktop Shell v{}", env!("CARGO_PKG_VERSION"));

    let mut builder = tauri::Builder::default()
        // Plugins
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init());

    // NSPanel plugin for Spotlight-style overlay on macOS
    #[cfg(target_os = "macos")]
    {
        builder = builder.plugin(tauri_nspanel::init());
    }

    // Global shortcut plugin for command palette overlay
    builder = builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        log::info!("Global shortcut triggered: Cmd+Shift+Space");

                        // Toggle command palette panel on Cmd+Shift+Space
                        // Use Accessory activation policy so app won't activate and main window stays hidden
                        #[cfg(target_os = "macos")]
                        {
                            if let Ok(panel) = app.get_webview_panel("command-palette") {
                                if panel.is_visible() {
                                    // Hiding panel - delegate to hide_command_palette for consistent behavior
                                    panel.order_out(None);

                                    // Check if we should restore main window
                                    let should_restore = MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst);
                                    if should_restore {
                                        // Restore Regular activation policy and show main window
                                        unsafe {
                                            let ns_app = NSApp();
                                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
                                        }
                                        if let Some(main_window) = app.get_webview_window("main") {
                                            let _ = main_window.show();
                                            let _ = main_window.set_focus();
                                        }
                                        log::info!("Global shortcut: hiding panel, restored main window");
                                    } else {
                                        // App was minimized/hidden - keep it that way
                                        // Reset to Regular then immediately back to Accessory to prevent crash
                                        unsafe {
                                            let ns_app = NSApp();
                                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyRegular);
                                            ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
                                        }
                                        log::info!("Global shortcut: hiding panel, keeping app in background");
                                    }
                                } else {
                                    // Showing panel - track main window visibility first
                                    // NOTE: On macOS, when the window is on a different space,
                                    // is_visible() returns false and is_minimized() returns true,
                                    // which is misleading. To avoid app disappearing, we always
                                    // consider the window as "was visible" and restore it after
                                    // closing the palette. The only exception would be if we could
                                    // reliably detect dock-minimized state, but that's not possible
                                    // with current Tauri APIs.
                                    if let Some(main_window) = app.get_webview_window("main") {
                                        let is_visible = main_window.is_visible().unwrap_or(false);
                                        let is_minimized = main_window.is_minimized().unwrap_or(false);

                                        // Always consider as "was visible" to ensure we restore it
                                        // This prevents the app from disappearing when triggered
                                        // from a different space or background
                                        MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);
                                        log::info!("Global shortcut: main window state - visible: {}, minimized: {} (forcing was_visible=true)",
                                                   is_visible, is_minimized);

                                        // Only hide if it was actually visible on current space
                                        if is_visible && !is_minimized {
                                            let _ = main_window.hide();
                                            log::info!("Global shortcut: hiding main window for Spotlight mode");
                                        }
                                    } else {
                                        // No main window found, but still try to show it later
                                        MAIN_WINDOW_WAS_VISIBLE.store(true, Ordering::SeqCst);
                                    }

                                    // Set Accessory policy BEFORE showing panel to prevent main window from appearing
                                    unsafe {
                                        let ns_app = NSApp();
                                        ns_app.setActivationPolicy_(NSApplicationActivationPolicy::NSApplicationActivationPolicyAccessory);
                                    }
                                    // Show panel WITHOUT activating the app (Spotlight-like)
                                    panel.order_front_regardless();
                                    log::info!("Global shortcut: showing command palette panel with Accessory policy");
                                }
                            } else {
                                log::warn!("Global shortcut: command palette panel not found");
                            }
                        }
                        // Fallback for non-macOS platforms
                        #[cfg(not(target_os = "macos"))]
                        {
                            if let Some(window) = app.get_webview_window("command-palette") {
                                if window.is_visible().unwrap_or(false) {
                                    let _ = window.hide();
                                } else {
                                    let _ = window.center();
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                }
                            }
                        }
                    }
                })
                .build(),
        );

    // IPC Command Handlers
    builder = builder.invoke_handler(tauri::generate_handler![
            // Infrastructure commands (auto-start services)
            commands::infrastructure::get_infrastructure_status,
            commands::infrastructure::start_infrastructure,
            commands::infrastructure::stop_infrastructure,
            commands::infrastructure::ensure_infrastructure_running,
            // Search commands
            commands::search::search_query,
            commands::search::get_search_history,
            commands::search::search_with_answer,
            // Chat commands (Step 03: Conversational AI)
            commands::chat::send_chat_message,
            commands::chat::get_chat_history,
            commands::chat::list_chat_sessions,
            commands::chat::clear_chat_session,
            commands::chat::delete_chat_session,
            commands::chat::create_chat_session,
            // Connector commands
            commands::connectors::list_sources,
            commands::connectors::add_source,
            commands::connectors::pause_source,
            commands::connectors::resume_source,
            commands::connectors::delete_source,
            commands::connectors::retry_source,
            commands::connectors::pause_all_sources,
            commands::connectors::resume_all_sources,
            commands::connectors::sync_source,
            commands::connectors::sync_all_github,
            commands::connectors::authenticate_imap,
            commands::connectors::test_imap_connection,
            // Privacy commands
            commands::privacy::get_consent,
            commands::privacy::grant_consent,
            commands::privacy::revoke_consent,
            commands::privacy::get_audit_logs,
            // Cloud sync consent commands
            commands::cloud_sync::get_cloud_sync_consent,
            commands::cloud_sync::grant_cloud_sync_consent,
            commands::cloud_sync::revoke_cloud_sync_consent,
            commands::cloud_sync::get_cloud_sync_audit_logs,
            commands::cloud_sync::log_cloud_sync_audit,
            commands::cloud_sync::get_cloud_sync_scope_info,
            // Orchestrator commands
            commands::orchestrator::get_orchestrator_status,
            commands::orchestrator::start_orchestrator,
            commands::orchestrator::stop_orchestrator,
            commands::orchestrator::ensure_orchestrator_running,
            // Graph commands
            commands::graph::get_knowledge_graph,
            commands::graph::get_filtered_graph,
            commands::graph::get_node_neighbors,
            commands::graph::get_graph_stats,
            commands::graph::get_bookmarked_nodes,
            commands::graph::bookmark_node,
            commands::graph::unbookmark_node,
            commands::graph::open_file,
            // Ollama model management commands
            commands::ollama::list_ollama_models,
            commands::ollama::is_model_installed,
            commands::ollama::pull_ollama_model,
            // Multimodal commands (Step 08: Frontend Intelligence Integration)
            commands::multimodal::transcribe_voice,
            commands::multimodal::analyze_image,
            commands::multimodal::describe_image,
            commands::multimodal::process_document,
            commands::multimodal::get_multimodal_status,
            // Causal chain commands (Step 08: Causal Visualization)
            commands::causal::find_causes,
            commands::causal::find_effects,
            commands::causal::find_causal_path,
            // Activity stream commands (Step 08: Activity Stream)
            commands::activity::get_activity_log,
            commands::activity::get_recent_activities,
            // Schema evolution commands (Step 08: Schema Dashboard)
            commands::schema::get_schema_stats,
            // Learning progress commands (Step 08: Learning Progress)
            commands::learning::get_learning_progress,
            commands::learning::record_document_learning,
            // Insights commands (AGI Phase 8: Frontend Integration)
            commands::insights::get_insights,
            commands::insights::mark_insight_read,
            commands::insights::dismiss_insight,
            commands::insights::get_knowledge_gaps,
            commands::insights::mark_gap_addressed,
            commands::insights::get_pending_verifications,
            commands::insights::submit_causal_verification,
            commands::insights::get_insight_stats,
            commands::insights::trigger_insight_scan,
            commands::insights::detect_patterns,
            commands::insights::save_user_insight,
            // Phase 2C: User Feedback Integration
            commands::insights::submit_insight_feedback,
            commands::insights::get_feedback_stats,
            // Phase 2D: Notification System
            commands::notifications::get_notification_preferences,
            commands::notifications::set_notification_frequency,
            commands::notifications::set_notification_dnd,
            commands::notifications::get_notification_history,
            commands::notifications::mark_notification_read,
            commands::notifications::clear_notifications,
            commands::notifications::get_notification_status,
            commands::notifications::deliver_notifications,
            // Paper search commands (Phase D: Academic Paper Agent)
            commands::papers::search_papers,
            commands::papers::download_paper,
            commands::papers::get_paper_recommendations,
            commands::papers::get_paper_details,
            commands::papers::agentic_search_papers,
            commands::papers::ingest_papers,
            commands::papers::get_paper_status,
            commands::papers::get_all_papers_status,
            // Research commands (Web Search & Deep Research)
            commands::research::web_search,
            commands::research::deep_research,
            commands::research::quick_search,
            commands::research::get_research_status,
            // Phase 2E: AgentFlow Architecture
            commands::agents::get_memory_stats,
            commands::agents::get_memory_recent,
            commands::agents::search_memory,
            commands::agents::clear_memory,
            commands::agents::get_hypotheses,
            commands::agents::generate_hypotheses,
            commands::agents::investigate_hypothesis,
            commands::agents::verify_hypothesis,
            commands::agents::get_verification_history,
            commands::agents::get_agentflow_status,
            commands::agents::export_token_priors,
            // Overlay commands (Global Command Palette)
            commands::overlay::toggle_command_palette,
            commands::overlay::hide_command_palette,
            commands::overlay::show_command_palette,
            commands::overlay::is_command_palette_visible,
        ]);

    builder.setup(|app| {
            log::info!("Futurnal Desktop Shell initialized");

            // Create command palette as NSPanel on macOS (Spotlight-style, doesn't steal focus)
            #[cfg(target_os = "macos")]
            {
                use tauri_nspanel::WebviewWindowExt;

                // Convert the command-palette window to an NSPanel
                if let Some(window) = app.get_webview_window("command-palette") {
                    match window.to_panel() {
                        Ok(panel) => {
                            // Configure panel for Spotlight-like behavior
                            // Floating level above most windows
                            panel.set_level(101);
                            panel.set_floating_panel(true);

                            // CRITICAL: NSWindowStyleMaskNonActivatingPanel = 1 << 7 = 128
                            // This prevents the app from activating when the panel is clicked/shown
                            // Source: tauri-plugin-spotlight working implementation
                            panel.set_style_mask(128);

                            // Don't hide when app deactivates - panel stays visible
                            panel.set_hides_on_deactivate(false);

                            // Becomes key only when explicitly needed
                            panel.set_becomes_key_only_if_needed(true);

                            // Receive keyboard events even when modal
                            panel.set_works_when_modal(true);

                            // CRITICAL: Set collection behavior so panel appears on current space
                            // instead of switching to another space
                            // NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0 = 1
                            // NSWindowCollectionBehaviorMoveToActiveSpace = 1 << 1 = 2
                            // NSWindowCollectionBehaviorFullScreenAuxiliary = 1 << 8 = 256
                            // NSWindowCollectionBehaviorTransient = 1 << 3 = 8
                            // Combined for Spotlight-like behavior: CanJoinAllSpaces | Transient | FullScreenAuxiliary
                            let behavior = NSWindowCollectionBehavior::NSWindowCollectionBehaviorCanJoinAllSpaces
                                | NSWindowCollectionBehavior::NSWindowCollectionBehaviorTransient
                                | NSWindowCollectionBehavior::NSWindowCollectionBehaviorFullScreenAuxiliary;
                            panel.set_collection_behaviour(behavior);

                            log::info!("Command palette NSPanel configured with NonActivatingPanel style and CanJoinAllSpaces+Transient behavior");
                        }
                        Err(e) => {
                            log::error!("Failed to convert window to panel: {:?}", e);
                        }
                    }
                }
            }

            // Register global shortcut for command palette (Cmd+Shift+Space)
            #[cfg(desktop)]
            {
                let shortcut = Shortcut::new(
                    Some(Modifiers::SUPER | Modifiers::SHIFT),
                    Code::Space
                );

                match app.global_shortcut().register(shortcut) {
                    Ok(_) => log::info!("Global shortcut registered: Cmd+Shift+Space"),
                    Err(e) => log::error!("Failed to register global shortcut: {:?}", e),
                }
            }

            // Show main window on normal app launch
            // The window starts hidden (visible: false in config) so shortcuts don't un-minimize it
            if let Some(main_window) = app.get_webview_window("main") {
                let _ = main_window.show();
                let _ = main_window.set_focus();
                log::info!("Main window shown on app launch");
            }

            // Open DevTools in development mode
            #[cfg(debug_assertions)]
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.open_devtools();
                    log::debug!("DevTools opened in development mode");
                }
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            match event {
                tauri::RunEvent::ExitRequested { api, .. } => {
                    // Smart exit prevention: only prevent if we're in Spotlight mode
                    // (command palette was triggered while app was minimized/hidden)
                    let main_window_was_visible = MAIN_WINDOW_WAS_VISIBLE.load(Ordering::SeqCst);

                    #[cfg(target_os = "macos")]
                    {
                        // Check if command palette is visible
                        if let Ok(panel) = app.get_webview_panel("command-palette") {
                            if panel.is_visible() {
                                // Panel is visible, prevent exit
                                api.prevent_exit();
                                log::debug!("Prevented app exit - command palette visible");
                                return;
                            }
                        }

                        // If main window wasn't visible before showing palette,
                        // we're in background Spotlight mode - prevent exit
                        if !main_window_was_visible {
                            api.prevent_exit();
                            log::debug!("Prevented app exit - app in background Spotlight mode");
                            return;
                        }
                    }

                    // For other cases, allow exit (but still prevent for Spotlight mode reliability)
                    // Keep app alive for global shortcut
                    api.prevent_exit();
                    log::debug!("Prevented app exit - keeping alive for global shortcut");
                }
                tauri::RunEvent::WindowEvent { label, event: tauri::WindowEvent::CloseRequested { api, .. }, .. } => {
                    if label == "main" {
                        // Hide main window instead of closing for Spotlight mode
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.hide();
                        }
                        api.prevent_close();
                        log::debug!("Main window close requested - hiding instead for Spotlight mode");
                    }
                }
                _ => {}
            }
        });
}
