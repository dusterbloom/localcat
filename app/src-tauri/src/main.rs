mod daemon_manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
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
            // Stop daemon when main window is closed
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                daemon_manager::stop_daemon();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running LocalCat app");
}
