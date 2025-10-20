# Plan to Fix Kokoro TTS Worker Communication in Tauri Production Build

## Current Issue Analysis
From the logs, I can see that:
1. The worker process starts successfully (pid=1263) 
2. The command and paths are correct
3. Worker script exists and is accessible
4. But the worker does NOT respond to the init command, timing out after ~3.5 seconds

## Root Cause Identification
The issue appears to be that the Python worker subprocess is not entering its main command loop properly, likely due to:

1. **Missing ML Dependencies**: The worker may be failing to import MLX/numpy when launched in the Tauri subprocess context
2. **Environment Variables**: Insufficient environment setup for the worker subprocess 
3. **Stdin/Stdout Buffering**: Communication channel issues between parent and child process
4. **MLX Model Loading**: Model loading hanging in the production environment

## Implementation Plan

### Phase 1: Enhanced Worker Diagnostics
1. **Add Worker Startup Logging**: 
   - Log all import attempts in the worker
   - Add explicit error handling around MLX model loading
   - Log when worker enters the main command loop

2. **Subprocess Environment Debug**:
   - Add comprehensive environment variable logging
   - Verify Python path and working directory
   - Add stderr capture and logging

### Phase 2: Worker Robustness Improvements  
1. **Graceful Fallback System**:
   - Add timeout handling for model loading
   - Implement basic TTS fallback if Kokoro fails
   - Add worker health checks and restart logic

2. **Communication Protocol Enhancement**:
   - Add heartbeat/ping-pong mechanism
   - Implement JSON validation for all messages
   - Add connection state tracking

### Phase 3: Tauri Integration Improvements
1. **Sidecar Architecture**:
   - Convert worker to proper Tauri sidecar
   - Use Tauri's built-in subprocess management
   - Implement proper process lifecycle management

2. **Bundle Optimization**:
   - Ensure all Python dependencies are properly bundled
   - Verify MLX libraries are accessible in production
   - Add production-specific environment configuration

### Phase 4: Alternative Approaches (if needed)
1. **Direct MLX Integration**: Remove worker subprocess entirely
2. **HTTP API**: Convert worker to HTTP server for communication
3. **Precompiled Binary**: Package worker as standalone executable

## Key Files to Modify
- `server/core/tts/kokoro_worker_optimized.py` - Enhanced diagnostics and robustness
- `server/core/tts/tts_mlx_ultra_low_latency.py` - Better error handling and fallbacks
- `app/src-tauri/tauri.conf.json` - Sidecar configuration
- `app/src-tauri/src/main.rs` - Process management improvements

## Success Criteria
1. Worker responds to init commands reliably in production
2. TTS functionality works consistently in bundled app
3. Graceful error handling when models fail to load
4. Production build parity with development behavior

This plan addresses the immediate communication issue while implementing a robust, production-ready TTS integration following Tauri best practices.