//! Futurnal Desktop Shell - Tauri Application Library
//!
//! This is the main library for the Futurnal desktop application.
//! It provides the bridge between the React frontend and the Python backend.

mod commands;
mod python;

use tauri::Manager;

/// Run the Tauri application.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting Futurnal Desktop Shell v{}", env!("CARGO_PKG_VERSION"));

    tauri::Builder::default()
        // Plugins
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        // IPC Command Handlers
        .invoke_handler(tauri::generate_handler![
            // Search commands
            commands::search::search_query,
            commands::search::get_search_history,
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
        ])
        .setup(|app| {
            log::info!("Futurnal Desktop Shell initialized");

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
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
