mod daemon_manager;

use std::sync::Once;

fn request_speech_authorization() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        #[cfg(target_os = "macos")]
        {
            // For now, just print a message about speech authorization
            // The sidecar will handle authorization checks on its own
            println!("🎤 macOS Speech Recognition available - sidecar will handle authorization");
        }
    });
}

fn main() {
    // Request speech authorization early in main thread
    request_speech_authorization();

    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                // Cleanup any stale processes before starting
                if let Err(e) = daemon_manager::cleanup_stale_server_processes() {
                    println!("⚠️  Cleanup warning: {}", e);
                }

                if let Err(e) = daemon_manager::start_server(&handle) {
                    println!("⚠️  Server start: {}", e);
                }
                if let Err(e) = daemon_manager::start_daemon(&handle) {
                    println!("⚠️  TTS daemon: {}", e);
                }
            });
            Ok(())
        })
        .on_window_event(|_win, event| {
            // Stop all processes when main window is closed
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                daemon_manager::stop_server();
                daemon_manager::stop_daemon();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running LocalCat app");
}
