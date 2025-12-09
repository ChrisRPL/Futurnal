//! Futurnal Desktop Shell - Main Entry Point
//!
//! This is the entry point for the Futurnal desktop application.
//! The actual application logic is in lib.rs.

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    futurnal_desktop_lib::run()
}
