use std::fs;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tauri::{AppHandle, Manager};

const DAEMON_PORT: u16 = 8770;
const PID_FILE: &str = "/tmp/localcat-tts-daemon.pid";
const SERVER_PID_FILE: &str = "/tmp/localcat-server.pid";
const MAX_STARTUP_ATTEMPTS: u32 = 40; // 20 seconds max
const HEALTH_CHECK_INTERVAL_MS: u64 = 500;

/// Check if the daemon is currently running by verifying PID file and health endpoint
pub fn is_daemon_running() -> bool {
    // Check if PID file exists
    let pid_file = Path::new(PID_FILE);
    if !pid_file.exists() {
        return false;
    }

    // Read PID and verify process is alive
    if let Ok(pid_str) = fs::read_to_string(pid_file) {
        if let Ok(pid) = pid_str.trim().parse::<i32>() {
            // Check if process exists (macOS)
            let check = Command::new("kill")
                .args(&["-0", &pid.to_string()])
                .output();

            if let Ok(output) = check {
                if output.status.success() {
                    // Process exists, verify health endpoint
                    return check_health_endpoint();
                }
            }
        }
    }

    // PID file exists but process is dead - clean up
    let _ = fs::remove_file(pid_file);
    false
}

/// Check if the daemon's /health endpoint responds
fn check_health_endpoint() -> bool {
    let url = format!("http://127.0.0.1:{}/health", DAEMON_PORT);

    match ureq::get(&url).timeout(Duration::from_secs(2)).call() {
        Ok(response) => response.status() == 200,
        Err(_) => false,
    }
}

/// Wait for daemon to become ready (polling /health endpoint)
pub fn wait_for_daemon_ready() -> Result<(), String> {
    println!("⏳ Waiting for TTS daemon to be ready...");

    for attempt in 1..=MAX_STARTUP_ATTEMPTS {
        if check_health_endpoint() {
            println!("✅ TTS daemon is ready and responding");
            return Ok(());
        }

        if attempt % 4 == 0 {
            println!(
                "   Still waiting for daemon... ({}/{})",
                attempt, MAX_STARTUP_ATTEMPTS
            );
        }

        std::thread::sleep(Duration::from_millis(HEALTH_CHECK_INTERVAL_MS));
    }

    Err(format!(
        "Daemon did not become ready after {} attempts",
        MAX_STARTUP_ATTEMPTS
    ))
}

/// Start the TTS daemon process
pub fn start_daemon(app: &AppHandle) -> Result<(), String> {
    if std::env::var("DISABLE_TTS_DAEMON").ok().as_deref() == Some("1") {
        println!("⚙️  TTS daemon disabled via DISABLE_TTS_DAEMON=1");
        return Ok(());
    }
    println!("🚀 Starting TTS daemon on port {}", DAEMON_PORT);

    // Get paths (dev vs production)
    let (python, server_dir) = get_daemon_paths(app);

    // If .env indicates Siri streaming TTS, skip spawning ONNX daemon
    let env_path = server_dir.join(".env");
    if let Ok(env_str) = fs::read_to_string(&env_path) {
        for line in env_str.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() { continue; }
            if let Some(rest) = line.strip_prefix("VOICE_AGENT_TTS_ENGINE=") {
                let val = rest.split('#').next().unwrap_or("").trim();
                if val.eq_ignore_ascii_case("siri_streaming") {
                    println!("🟡 VOICE_AGENT_TTS_ENGINE=siri_streaming → skipping ONNX TTS daemon");
                    return Ok(());
                }
            }
        }
    }

    // If Siri sidecar binary exists in resources, prefer it (macOS) and skip ONNX daemon
    if let Ok(resource_dir) = app.path().resource_dir() {
        let siri_bin = resource_dir.join("sidecar/siri-tts/siri-tts");
        if siri_bin.exists() {
            println!("🟡 Siri sidecar detected at {:?} → skipping ONNX TTS daemon", siri_bin);
            return Ok(());
        }
    }

    // Use ONNX hardened sidecar for production bundles
    // MLX is incompatible with macOS app bundle sandboxing (hangs during Metal init)
    let sidecar_script = server_dir.join("sidecars/tts_sidecar_onnx_hardened.py");
    if !sidecar_script.exists() {
        return Err(format!("TTS sidecar script not found at {:?}", sidecar_script));
    }

    // Check if port is available
    if !check_port_available(DAEMON_PORT) {
        println!(
            "⚠️  Port {} is already in use, attempting to kill existing process...",
            DAEMON_PORT
        );
        kill_process_on_port(DAEMON_PORT);

        // Verify port is now available
        std::thread::sleep(Duration::from_secs(1));
        if !check_port_available(DAEMON_PORT) {
            return Err(format!("Port {} still in use after cleanup attempt", DAEMON_PORT));
        }
    }

    // Setup environment
    let hf_home = server_dir.join("models/hf_cache");
    let hf_hub_cache = hf_home.join("hub");
    let voice = std::env::var("VOICE_AGENT_TTS_VOICE").unwrap_or_else(|_| "af_heart".to_string());
    let speed = std::env::var("VOICE_AGENT_TTS_SPEED").unwrap_or_else(|_| "1.0".to_string());

    println!("🔧 Daemon environment:");
    println!("   Python: {:?}", python);
    println!("   Script: {:?}", sidecar_script);
    println!("   Voice: {}", voice);
    println!("   HF_HOME: {:?}", hf_home);

    // Spawn daemon process (detached, survives parent)
    let mut cmd = Command::new(&python);
    cmd.arg(&sidecar_script)
        .current_dir(&server_dir)
        .env("LOCALCAT_TTS_PORT", DAEMON_PORT.to_string())
        .env("LOCALCAT_TTS_VOICE", voice)
        .env("LOCALCAT_TTS_SPEED", speed)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HOME", &hf_home)
        .env("HUGGINGFACE_HUB_CACHE", &hf_hub_cache)
        .stdin(Stdio::null());

    // Sidecar logs are written by loguru to ~/Library/Logs/LocalCat/tts-daemon-hardened.log
    // but also pipe stdout/stderr to a file for additional visibility
    if let Ok(home) = std::env::var("HOME") {
        let logs_dir = Path::new(&home).join("Library/Logs/LocalCat");
        let _ = std::fs::create_dir_all(&logs_dir);
        let daemon_log_path = logs_dir.join("daemon.log");
        if let Ok(file) = OpenOptions::new().create(true).append(true).open(&daemon_log_path) {
            if let Ok(file_err) = OpenOptions::new().create(true).append(true).open(&daemon_log_path) {
                cmd.stdout(Stdio::from(file));
                cmd.stderr(Stdio::from(file_err));
            }
        }
    }

    // Provide TAURI_RESOURCE_DIR so Python can locate bundled sidecars (e.g., siri-tts)
    if let Ok(resource_dir) = app.path().resource_dir() {
        cmd.env("TAURI_RESOURCE_DIR", &resource_dir);
        println!("   TAURI_RESOURCE_DIR set to: {:?}", resource_dir);
    }

    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id();
            println!("✅ Daemon spawned with PID: {}", pid);

            // Don't hold onto the child handle - let it run independently
            std::mem::forget(child);

            // Wait for daemon to be ready
            wait_for_daemon_ready()
        }
        Err(e) => Err(format!("Failed to spawn daemon: {}", e)),
    }
}

/// Start the main Python server (FastAPI) inside the bundle so the UI can connect.
pub fn start_server(app: &AppHandle) -> Result<(), String> {
    // Resolve paths (dev vs production)
    let (python, server_dir) = get_daemon_paths(app);

    // Determine resource dir for sidecars
    let resource_dir = app.path().resource_dir().ok();

    // Build command: python bot.py --host 127.0.0.1 --port 7860
    let mut cmd = Command::new(&python);
    cmd.arg("bot.py")
        .arg("--host").arg("127.0.0.1")
        .arg("--port").arg("7860")
        .current_dir(&server_dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")  // Force Python to not use bytecode cache
        .stdin(Stdio::null());

    // Pipe server stdout/stderr to persistent log for debugging
    if let Ok(home) = std::env::var("HOME") {
        let logs_dir = Path::new(&home).join("Library/Logs/LocalCat");
        let _ = std::fs::create_dir_all(&logs_dir);
        let server_log_path = logs_dir.join("server.log");
        if let Ok(file) = OpenOptions::new().create(true).append(true).open(&server_log_path) {
            if let Ok(file_err) = OpenOptions::new().create(true).append(true).open(&server_log_path) {
                cmd.stdout(Stdio::from(file));
                cmd.stderr(Stdio::from(file_err));
            }
        }
    }

    // Provide TAURI_RESOURCE_DIR so Siri sidecar can be resolved during runtime
    if let Some(res_dir) = &resource_dir {
        cmd.env("TAURI_RESOURCE_DIR", res_dir);
        println!("   TAURI_RESOURCE_DIR for server: {:?}", res_dir);
    }

    // Offline defaults consistent with bundle
    let hf_home = server_dir.join("models/hf_cache");
    let hf_hub_cache = hf_home.join("hub");
    cmd.env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HOME", &hf_home)
        .env("HUGGINGFACE_HUB_CACHE", &hf_hub_cache);

    // In production mode, override data paths to use user-writable locations
    // This keeps .env unchanged for local development while ensuring bundle app works correctly
    if resource_dir.is_some() {
        if let Ok(home) = std::env::var("HOME") {
            let data_dir = Path::new(&home).join("Library/Application Support/LocalCat/data");

            // Ensure data directory exists
            let _ = std::fs::create_dir_all(&data_dir);
            println!("📁 Using data directory: {:?}", data_dir);

            // Override all .env data paths with production locations
            cmd.env("SESSION_DB_PATH", data_dir.join("sessions.db"))
                .env("MEMORY_SEMANTIC_DIR", data_dir.join("semantic_index"))
                .env("MEMORY_SQLITE_PATH", data_dir.join("memory.db"))
                .env("MEMORY_LMDB_PATH", data_dir.join("graph.lmdb"))
                .env("YAML_GRAPH_JUDGE_MODEL", data_dir.join("graph_judge.json"))
                .env("YAML_GRAPH_JUDGE_GRAYZONE_LOG", data_dir.join("judge_grayzone.jsonl"))
                .env("YAML_GRAPH_JUDGE_SCHEMA", data_dir.join("graph_judge_schema.json"))
                .env("SPEAKER_PROFILE_DIR", data_dir.join("speaker_profiles"));

            println!("   All data paths configured to: {:?}", data_dir);
        }
    }

    // Spawn server
    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id();
            println!("✅ Server spawned with PID: {}", pid);
            let _ = fs::write(SERVER_PID_FILE, pid.to_string());
            std::mem::forget(child);

            // Poll health: try GET /openapi.json until reachable
            for _ in 0..40u32 {
                if ureq::get("http://127.0.0.1:7860/openapi.json")
                    .timeout(Duration::from_millis(500))
                    .call()
                    .ok()
                    .is_some()
                {
                    println!("✅ LocalCat server ready on http://127.0.0.1:7860");
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err("Server did not become ready".to_string())
        }
        Err(e) => Err(format!("Failed to spawn server: {}", e)),
    }
}

/// Stop the TTS daemon gracefully
pub fn stop_daemon() {
    println!("🛑 Stopping TTS daemon...");

    // Try graceful shutdown via /shutdown endpoint first
    let url = format!("http://127.0.0.1:{}/shutdown", DAEMON_PORT);
    match ureq::post(&url).timeout(Duration::from_secs(2)).call() {
        Ok(_) => {
            println!("✅ Sent shutdown signal to daemon");
            // Give it time to shut down
            std::thread::sleep(Duration::from_millis(1000));
        }
        Err(e) => {
            println!("⚠️  Failed to send shutdown signal: {}", e);
        }
    }

    // If daemon is still running, force kill via PID file
    if let Ok(pid_str) = fs::read_to_string(PID_FILE) {
        if let Ok(pid) = pid_str.trim().parse::<i32>() {
            println!("🔪 Force killing daemon PID: {}", pid);
            let _ = Command::new("kill")
                .args(&["-9", &pid.to_string()])
                .spawn();
        }
    }

    // Clean up PID file
    let _ = fs::remove_file(PID_FILE);
    println!("✅ Daemon stopped");
}

/// Get paths for daemon (development vs production)
fn get_daemon_paths(app: &AppHandle) -> (PathBuf, PathBuf) {
    // Check if we're in production (bundled app)
    if let Ok(resource_path) = app.path().resource_dir() {
        let server_dir = resource_path.join("_up_/_up_/server");
        if server_dir.exists() {
            // Production mode
            let python = server_dir.join(".venv/bin/python3");
            return (python, server_dir);
        }
    }

    // Development mode - fallback
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    let manifest_path = Path::new(&manifest_dir);

    // Safely navigate up the directory tree
    let server_dir = manifest_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("server"))
        .unwrap_or_else(|| PathBuf::from("server"));

    let python = server_dir.join(".venv/bin/python3");

    (python, server_dir)
}

/// Check if a port is available (not in use)
fn check_port_available(port: u16) -> bool {
    use std::net::TcpListener;
    TcpListener::bind(format!("127.0.0.1:{}", port)).is_ok()
}

/// Kill process using a specific port
fn kill_process_on_port(port: u16) {
    let _ = Command::new("lsof")
        .args(&["-ti", &format!(":{}", port)])
        .output()
        .and_then(|output| {
            if let Ok(pid) = String::from_utf8(output.stdout) {
                if let Ok(pid_num) = pid.trim().parse::<i32>() {
                    println!("🔪 Killing process {} on port {}", pid_num, port);
                    let _ = Command::new("kill")
                        .args(&["-9", &pid_num.to_string()])
                        .spawn();
                }
            }
            Ok(())
        });
}
