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
            commands::insights::save_user_insight,
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
