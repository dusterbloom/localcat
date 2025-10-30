# TODO: Three-Tier Build System for LocalCat Pre-Release

**Status**: Planning Phase
**Created**: 2025-10-30
**Target**: v1.0.0 Release
**Estimated Effort**: 14-20 hours

---

## Executive Summary

Transform LocalCat into a modular build system with three distribution tiers optimized for different use cases:
- **Zero Tier** (~813MB): Native macOS STT/TTS + LM Studio integration
- **Light Tier** (~3.2GB): Offline-ready with essential models
- **Full Tier** (~5.8GB): Complete offline experience with all features

---

## Tier Specifications

### Zero Tier (~813MB)
**Target Users**: Demos, CI/CD, first-time users, metered connections

**Bundled Components**:
```
- Tauri + Next.js frontend (~50MB)
- Server Python code (~10MB)
- Smart-turn-v2 model (~362MB) - REQUIRED for turn detection
- Kokoro ONNX TTS (~337MB) - fallback if not using Siri
- Native sidecars: siri-tts, macos-stt (~4MB)
- Minimal Python runtime stub (~50MB)
Total: ~813MB
```

**First Launch Experience**:
```
┌──────────────────────────────────────────┐
│  Welcome to LocalCat                     │
├──────────────────────────────────────────┤
│  Choose your setup:                      │
│                                           │
│  ○ Quick Start (Recommended)             │
│    macOS Speech + LM Studio              │
│    No downloads, works immediately       │
│    → Requires LM Studio running locally  │
│                                           │
│  ○ Download Models (Full Offline)        │
│    Download ~1.9GB of AI models          │
│    Choose from preset or custom URL      │
│                                           │
│  [Continue]                              │
└──────────────────────────────────────────┘
```

**If User Chooses "Download Models"**:
- Show model picker with presets or custom HuggingFace URL
- Download: Whisper-small (~200MB) + LFM2-1.2B-4bit (~1.2GB)
- Progress UI with pause/cancel support
- Store in: `~/Library/Application Support/LocalCat/models/`

---

### Light Tier (~3.2GB) - RECOMMENDED
**Target Users**: General users wanting good offline experience

**Bundled Components**:
```
Everything from Zero +
- Python venv (slimmed with SLIM_VENV=1, ~1GB)
- LFM2-1.2B-4bit LLM (~1.2GB)
- Whisper-small.en-mlx-q4 STT (~200MB)
Total: ~3.2GB
```

**User Experience**:
- Works offline immediately
- All core features functional
- Optional: Download Parakeet STT or speaker recognition from settings

---

### Full Tier (~5.8GB)
**Target Users**: Power users, enterprise, completely offline environments

**Bundled Components**:
```
Everything from Light +
- Full Python venv (~1.8GB instead of slimmed 1GB)
- Parakeet STT (~2.3GB)
- SpeechBrain speaker recognition (~85MB)
- Multiple TTS variants (~800MB total)
- Emotion recognition (~100MB, optional)
Total: ~5.8GB
```

**User Experience**:
- Zero configuration required
- All features work offline immediately
- Voice enrollment, emotion detection, high-quality STT

---

## Implementation Plan

### Phase 1: Build Script Refactoring (2-3 hours)

#### 1.1 Update build-production.sh
**File**: `app/build-production.sh`

**Changes**:
- [ ] Extend BUILD_PROFILE to support `zero|light|full` (currently only has `light|full`)
- [ ] Add Zero tier bundling logic:
  ```bash
  case "$BUILD_PROFILE" in
    zero)
      BUNDLE_VENV=false
      BUNDLE_LLM=false
      BUNDLE_STT=false
      BUNDLE_PARAKEET=false
      BUNDLE_SPEAKER_REC=false
      ;;
    light)
      # Existing light profile
      ;;
    full)
      # Existing full profile
      ;;
  esac
  ```
- [ ] Add bundle size calculation and reporting
- [ ] Update DMG naming:
  - `LocalCat-Zero-1.0.0-aarch64.dmg`
  - `LocalCat-Light-1.0.0-aarch64.dmg`
  - `LocalCat-Full-1.0.0-aarch64.dmg`

#### 1.2 Create tier configuration files
**New Files**:
- [ ] `app/build-configs/zero.env` - Zero tier model manifest
- [ ] `app/build-configs/light.env` - Light tier model manifest
- [ ] `app/build-configs/full.env` - Full tier model manifest

**Format** (example for zero.env):
```bash
# Zero Tier Configuration
BUNDLE_VENV=false
BUNDLE_LLM=false
BUNDLE_STT=false
BUNDLE_MODELS=(
  "pipecat-ai/smart-turn-v2:required"
  "kokoro-onnx:required"
)
REQUIRED_DOWNLOADS=(
  "mlx-community/whisper-small.en-mlx-q4:200MB"
  "mlx-community/LFM2-1.2B-4bit:1200MB"
)
```

#### 1.3 Add bundle verification
- [ ] Print actual bundle sizes after build
- [ ] Verify tier requirements are met
- [ ] Generate build manifest (what's included)
- [ ] Exit with error if required components missing

---

### Phase 2: Model Download Infrastructure (4-5 hours)

#### 2.1 Create server/core/model_downloader.py
**New File**: `server/core/model_downloader.py`

**Features**:
- [ ] Async HuggingFace downloader with resume support
- [ ] Progress tracking (bytes downloaded, speed, ETA)
- [ ] File integrity validation (checksum, size verification)
- [ ] Error handling with exponential backoff retry
- [ ] Support for custom HuggingFace URLs

**API**:
```python
class ModelDownloader:
    async def download_model(
        model_id: str,
        progress_callback: Callable[[DownloadProgress], None] = None
    ) -> Path

    async def cancel_download(task_id: str) -> None

    def get_download_progress(task_id: str) -> DownloadProgress

    def validate_model(model_path: Path) -> bool
```

#### 2.2 Add Tauri IPC commands
**File**: `app/src-tauri/src/daemon_manager.rs` (or new `model_manager.rs`)

**New Commands**:
```rust
#[tauri::command]
async fn download_model(model_id: String) -> Result<DownloadTask, String>

#[tauri::command]
async fn get_download_progress(task_id: String) -> Result<DownloadProgress, String>

#[tauri::command]
async fn cancel_download(task_id: String) -> Result<(), String>

#[tauri::command]
async fn list_available_models() -> Vec<ModelInfo>

#[tauri::command]
async fn validate_lm_studio_connection() -> Result<bool, String>
```

#### 2.3 Create model registry
**New File**: `server/models/registry.json`

```json
{
  "presets": {
    "essential": {
      "name": "Essential Models",
      "description": "Minimum for offline voice agent",
      "total_size": 1400000000,
      "models": [
        {
          "id": "mlx-community/whisper-small.en-mlx-q4",
          "name": "Whisper Small (STT)",
          "size": 200000000,
          "required": true
        },
        {
          "id": "mlx-community/LFM2-1.2B-4bit",
          "name": "LFM2 1.2B (LLM)",
          "size": 1200000000,
          "required": true
        }
      ]
    },
    "quality": {
      "name": "High Quality Models",
      "description": "Best offline experience",
      "total_size": 3500000000,
      "models": [
        {
          "id": "mlx-community/parakeet-tdt-0.6b-v3",
          "name": "Parakeet (High-Quality STT)",
          "size": 2300000000,
          "required": false
        },
        {
          "id": "mlx-community/LFM2-1.2B-4bit",
          "name": "LFM2 1.2B (LLM)",
          "size": 1200000000,
          "required": true
        }
      ]
    }
  },
  "custom_url_supported": true
}
```

---

### Phase 3: First-Run Setup UI (3-4 hours)

#### 3.1 Create React setup wizard
**New Files**:
- [ ] `client/src/components/SetupWizard.tsx` - Main wizard component
- [ ] `client/src/components/ModelPicker.tsx` - Choose models to download
- [ ] `client/src/components/DownloadProgress.tsx` - Progress UI with live updates
- [ ] `client/src/components/LMStudioGuide.tsx` - Connection helper

**SetupWizard Flow**:
1. **Welcome Screen**: Choose Quick Start or Download Models
2. **Quick Start Path**:
   - Check if LM Studio running (localhost:1234)
   - Show connection instructions if not
   - Test connection before proceeding
   - Configure .env to use macos_native STT and siri_streaming TTS
3. **Download Path**:
   - Show preset model packs (Essential, Quality)
   - Allow custom HuggingFace URL input
   - Confirm disk space available
   - Show download progress with pause/cancel
   - Validate downloaded models
   - Configure .env for offline mode

#### 3.2 Update Tauri app initialization
**File**: `app/src-tauri/src/main.rs`

**Changes**:
- [ ] Add first-run detection (check if models exist)
- [ ] Show SetupWizard window instead of main UI if first run
- [ ] Store setup completion flag in AppSupport
- [ ] Handle setup cancellation gracefully

#### 3.3 LM Studio integration
- [ ] Auto-detect LM Studio on port 1234
- [ ] Show clear instructions if not running
- [ ] Test connection with simple prompt
- [ ] Configure .env with LM_BACKEND=http

---

### Phase 4: Developer Experience (1-2 hours)

#### 4.1 Create build wrapper scripts
**New Files**:
- [ ] `app/build-zero.sh` → `BUILD_PROFILE=zero ./build-production.sh "$@"`
- [ ] `app/build-light.sh` → `BUILD_PROFILE=light ./build-production.sh "$@"`
- [ ] `app/build-full.sh` → `BUILD_PROFILE=full ./build-production.sh "$@"`
- [ ] `app/build-all.sh` → Build all three tiers sequentially

**build-all.sh** (for CI/CD):
```bash
#!/bin/bash
set -e

echo "Building all LocalCat tiers..."

./build-zero.sh
echo "✅ Zero tier complete"

./build-light.sh
echo "✅ Light tier complete"

./build-full.sh
echo "✅ Full tier complete"

echo ""
echo "🎉 All tiers built successfully!"
ls -lh src-tauri/target/aarch64-apple-darwin/release/bundle/dmg/LocalCat-*.dmg
```

#### 4.2 Add build verification
- [ ] Print bundle size comparison table
- [ ] Verify all required files present
- [ ] Generate `build-manifest.json` with included models
- [ ] Add `--verify-only` flag to check bundle without rebuilding

#### 4.3 Update documentation
- [ ] Update `HOW_TO_INSTALL_AND_USE.md` with tier comparison
- [ ] Create `DEVELOPER_BUILD_GUIDE.md` for building specific tiers
- [ ] Add CI/CD examples for GitHub Actions
- [ ] Document model registry format

---

### Phase 5: Runtime Intelligence (2-3 hours)

#### 5.1 Smart tier detection
**New File**: `app/src-tauri/src/tier_detector.rs`

```rust
#[derive(Debug, Clone, Serialize)]
pub enum BundleTier {
    Zero,
    Light,
    Full,
}

pub fn detect_bundle_tier(resource_dir: &Path) -> BundleTier {
    let server_dir = resource_dir.join("_up_/_up_/server");

    // Check for Full tier markers
    if server_dir.join("models/hf_cache/hub/models--mlx-community--parakeet-tdt-0.6b-v3").exists()
        && server_dir.join("models/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb").exists() {
        return BundleTier::Full;
    }

    // Check for Light tier markers
    if server_dir.join(".venv").exists()
        && server_dir.join("models/hf_cache/hub/models--mlx-community--LFM2-1.2B-4bit").exists() {
        return BundleTier::Light;
    }

    // Default to Zero
    BundleTier::Zero
}
```

#### 5.2 Model resolver enhancements
**File**: `server/core/factories/utils/model_resolver.py`

**Changes**:
- [ ] Add search path priority:
  1. Bundle (Resources/_up_/_up_/server/models/)
  2. User downloads (~/Library/Application Support/LocalCat/models/)
  3. System cache (~/.cache/huggingface/)
- [ ] Return None if model not found (instead of crashing)
- [ ] Log search paths for debugging
- [ ] Cache resolved paths for performance

#### 5.3 Graceful degradation logic
**File**: `server/core/factories/service_factory.py`

**Enhancements**:
- [ ] If LLM missing in Zero tier: Return error with "Start LM Studio or download models"
- [ ] If Whisper missing: Fall back to macos_native STT
- [ ] If Kokoro missing: Fall back to siri_streaming TTS
- [ ] Show user-friendly error messages in UI

---

### Phase 6: Testing & Polish (2-3 hours)

#### 6.1 Test matrix
- [ ] **Zero Tier Tests**:
  - [ ] Build on clean machine
  - [ ] Verify bundle size ~813MB
  - [ ] Test Quick Start path with LM Studio
  - [ ] Test Download path with preset models
  - [ ] Test Download path with custom HuggingFace URL
  - [ ] Test download cancellation
  - [ ] Test network interruption handling

- [ ] **Light Tier Tests**:
  - [ ] Build on clean machine
  - [ ] Verify bundle size ~3.2GB
  - [ ] Test offline functionality (HF_HUB_OFFLINE=1)
  - [ ] Test all core features work
  - [ ] Test optional model downloads from settings

- [ ] **Full Tier Tests**:
  - [ ] Build on clean machine
  - [ ] Verify bundle size ~5.8GB
  - [ ] Test all features offline (voice enrollment, emotion detection)
  - [ ] Test Parakeet STT quality
  - [ ] Test multiple TTS variants

#### 6.2 Error handling scenarios
- [ ] Download fails due to network
- [ ] Download fails due to invalid URL
- [ ] Download fails due to insufficient disk space
- [ ] Corrupted model file (checksum mismatch)
- [ ] LM Studio not running (Zero tier)
- [ ] Model incompatibility (wrong MLX version)

#### 6.3 Performance validation
- [ ] Measure startup times for each tier
- [ ] Verify <800ms voice latency maintained
- [ ] Check memory usage doesn't balloon
- [ ] Profile model loading times
- [ ] Test with limited RAM (8GB, 16GB, 32GB)

---

## File Structure After Implementation

```
app/
├── build-production.sh          (Enhanced with 3 tiers)
├── build-zero.sh               (NEW - Quick wrapper)
├── build-light.sh              (NEW - Quick wrapper)
├── build-full.sh               (NEW - Quick wrapper)
├── build-all.sh                (NEW - CI/CD helper)
├── build-configs/              (NEW)
│   ├── zero.env                (Zero tier manifest)
│   ├── light.env               (Light tier manifest)
│   └── full.env                (Full tier manifest)
├── src-tauri/
│   ├── src/
│   │   ├── daemon_manager.rs   (Enhanced with download commands)
│   │   ├── model_manager.rs    (NEW - Model download IPC)
│   │   ├── tier_detector.rs    (NEW - Runtime tier detection)
│   │   └── main.rs             (Updated for first-run setup)
│   └── tauri.conf.json         (Updated for tier-aware resources)
└── docs/                       (NEW)
    ├── TIER_COMPARISON.md      (User-facing tier guide)
    └── DEVELOPER_BUILD_GUIDE.md (Dev docs)

server/
├── core/
│   ├── model_downloader.py     (NEW - Async HF downloader)
│   └── factories/
│       ├── service_factory.py  (Enhanced with better degradation)
│       └── utils/
│           └── model_resolver.py  (Enhanced with AppSupport paths)
└── models/
    └── registry.json           (NEW - Available models catalog)

client/
└── src/
    └── components/
        ├── SetupWizard.tsx     (NEW - First-run setup)
        ├── ModelPicker.tsx     (NEW - Choose models to download)
        ├── DownloadProgress.tsx (NEW - Progress UI)
        └── LMStudioGuide.tsx   (NEW - Connection helper)
```

---

## Developer Workflows

### Building a specific tier
```bash
cd app/

# Method 1: Direct flag
BUILD_PROFILE=zero ./build-production.sh

# Method 2: Wrapper script
./build-zero.sh

# Method 3: Build all for release
./build-all.sh
```

### Testing locally without bundle
```bash
# Server
cd server/
source .venv/bin/activate
python bot.py

# Client (separate terminal)
cd client/
npm run dev
```

### CI/CD example (GitHub Actions)
```yaml
name: Build Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-tiers:
    runs-on: macos-latest
    strategy:
      matrix:
        tier: [zero, light, full]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Build ${{ matrix.tier }} tier
        run: |
          cd app
          BUILD_PROFILE=${{ matrix.tier }} ./build-production.sh

      - name: Upload DMG
        uses: actions/upload-artifact@v3
        with:
          name: LocalCat-${{ matrix.tier }}-dmg
          path: app/src-tauri/target/aarch64-apple-darwin/release/bundle/dmg/LocalCat-*.dmg
```

---

## User Experience Flows

### Zero Tier - Quick Start Path
```
1. User downloads LocalCat-Zero-1.0.0.dmg (813MB)
2. Drags to Applications, opens app
3. Setup wizard appears:
   "Welcome to LocalCat
    Choose your setup:

    ○ Quick Start (Recommended)
      Use macOS Speech + LM Studio
      No downloads, works immediately

    ○ Download Models
      Download AI models for offline use"
4. User selects "Quick Start"
5. App checks if LM Studio running on localhost:1234
6. If not running, shows: "Please start LM Studio first:
   1. Download from lmstudio.ai
   2. Load any model (e.g., Qwen2.5-3B)
   3. Start server from Developer tab
   [Retry Connection]"
7. Once connected, configures .env and starts voice agent
8. User can chat immediately using macOS STT + Siri TTS + LM Studio LLM
```

### Zero Tier - Download Models Path
```
1-3. (Same as above)
4. User selects "Download Models"
5. Model picker appears:
   "Choose models to download:

    ○ Essential Pack (Recommended) - 1.4GB
      Whisper Small STT + LFM2 1.2B LLM

    ○ Quality Pack - 3.5GB
      Parakeet STT + LFM2 1.2B LLM

    ○ Custom URL
      Enter HuggingFace model URL

    [Continue]"
6. User selects "Essential Pack"
7. Progress screen:
   "Downloading models... 45%

    ✓ Whisper Small (200MB) - Complete
    ↓ LFM2 1.2B (540MB / 1.2GB) - 2min remaining

    [Pause] [Cancel]"
8. After completion, validates models and starts app
9. User can chat fully offline
```

### Light Tier
```
1. User downloads LocalCat-Light-1.0.0.dmg (3.2GB)
2. Drags to Applications, opens app
3. No setup wizard - app starts immediately
4. Voice agent ready with full offline capabilities
5. Can optionally download Parakeet STT from settings for higher quality
```

### Full Tier
```
1. User downloads LocalCat-Full-1.0.0.dmg (5.8GB)
2. Drags to Applications, opens app
3. No setup wizard - all features work immediately
4. Voice enrollment, emotion detection, high-quality STT all ready
5. Complete offline experience
```

---

## Website Download Page Design

```html
<h1>Download LocalCat</h1>
<p>Choose the edition that fits your needs:</p>

<div class="tier-card">
  <h2>Zero Edition</h2>
  <span class="size">813 MB</span>
  <p>Best for: Trying LocalCat, demos, using with LM Studio</p>
  <ul>
    <li>✓ Native macOS Speech Recognition & Siri TTS</li>
    <li>✓ Works immediately with LM Studio</li>
    <li>⚠ Requires internet OR LM Studio for first use</li>
  </ul>
  <button>Download LocalCat-Zero-1.0.0.dmg</button>
</div>

<div class="tier-card recommended">
  <span class="badge">RECOMMENDED</span>
  <h2>Light Edition</h2>
  <span class="size">3.2 GB</span>
  <p>Best for: Most users wanting offline voice AI</p>
  <ul>
    <li>✓ Works completely offline</li>
    <li>✓ Includes LFM2 LLM & Whisper STT</li>
    <li>✓ Fast startup, low latency</li>
  </ul>
  <button class="primary">Download LocalCat-Light-1.0.0.dmg</button>
</div>

<div class="tier-card">
  <h2>Full Edition</h2>
  <span class="size">5.8 GB</span>
  <p>Best for: Power users, enterprise, complete offline</p>
  <ul>
    <li>✓ All features included</li>
    <li>✓ Voice enrollment & speaker recognition</li>
    <li>✓ High-quality Parakeet STT</li>
    <li>✓ Emotion detection</li>
  </ul>
  <button>Download LocalCat-Full-1.0.0.dmg</button>
</div>

<h3>System Requirements</h3>
<ul>
  <li>macOS 13.0+ (Ventura or later)</li>
  <li>Apple Silicon (M1/M2/M3/M4)</li>
  <li>8GB RAM minimum (16GB recommended for Full edition)</li>
</ul>
```

---

## Success Metrics

### Developer Ergonomics ✅
- [x] Build any tier with single command: `./build-zero.sh`
- [x] Clear naming (zero/light/full, not profile1/profile2)
- [x] Fast iteration (no rebuilding unchanged tiers)
- [x] CI-friendly (parallel tier builds via matrix strategy)

### User Experience ✅
- [x] Choice appropriate to needs (size vs features table)
- [x] Zero tier works immediately with Quick Start (LM Studio)
- [x] Download progress is clear, pausable, and cancellable
- [x] No surprise downloads after installing 6GB bundle
- [x] First-run setup is intuitive and helpful

### Maintainability ✅
- [x] Single build script handles all tiers (no copy-paste)
- [x] Tier configs are declarative (env/JSON files)
- [x] Model registry is centralized and extensible
- [x] Easy to add/remove models from tiers
- [x] Runtime tier detection works automatically

---

## Timeline Estimate

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Build script refactoring | 2-3 hours |
| 2 | Model download infrastructure | 4-5 hours |
| 3 | First-run setup UI | 3-4 hours |
| 4 | Developer experience | 1-2 hours |
| 5 | Runtime intelligence | 2-3 hours |
| 6 | Testing & polish | 2-3 hours |
| **Total** | | **14-20 hours** |

**Recommended Approach**: Tackle phases sequentially, testing thoroughly after each phase before moving to the next. Create separate feature branches or PRs for easier review.

---

## Risk Mitigation

### High-Risk Items
1. **Download failures**: Implement robust retry logic with exponential backoff
2. **Corrupted models**: Always validate checksums and file sizes
3. **LM Studio detection**: Gracefully handle when not running, provide clear instructions
4. **Disk space**: Check available space before starting downloads

### Medium-Risk Items
1. **Version mismatches**: Pin MLX versions in requirements.txt
2. **Platform differences**: Test on Intel Macs if possible (Rosetta)
3. **Complex UI flows**: User test the setup wizard with 3-5 people

### Low-Risk Items
1. **Build time increase**: Acceptable for release builds
2. **Documentation debt**: Address incrementally

---

## Future Enhancements (Post-v1.0)

### Phase 7: Advanced Features (Future)
- [ ] Smart differential downloads (only download when feature enabled)
- [ ] Model compression experiments (test more quantized variants)
- [ ] CDN-hosted model repository for faster downloads
- [ ] In-app model management UI (delete, re-download, update)
- [ ] Automatic model updates (with user consent)
- [ ] Model A/B testing framework
- [ ] Telemetry for most popular tier (opt-in)

### Phase 8: Cross-Platform (Future)
- [ ] Windows build support with tier system
- [ ] Linux build support with tier system
- [ ] Platform-specific model optimization
- [ ] Windows TTS fallbacks (no Siri available)

---

## Notes & Decisions

### Key Design Decisions
1. **Separate DMGs**: Easier to understand than in-app installer, clearer download expectations
2. **CLI flags only**: Matches existing BUILD_PROFILE pattern, simpler than menu system
3. **LFM2 for all tiers**: Proven to work well for voice, smaller than Qwen3-VL
4. **Quick Start in Zero**: Honors "local" philosophy while being practical for first-timers
5. **Model registry in JSON**: Easy to extend, machine-readable, can generate UI from it

### Open Questions
- [ ] Should we support Intel Macs or Apple Silicon only?
- [ ] Should Full tier include even more models (e.g., translation models)?
- [ ] Should we bundle Python 3.12 in Zero tier or require system Python?
- [ ] Should download progress be shown in app window or macOS notification?

---

## References

### Relevant Files (Current State)
- `app/build-production.sh` - Current build script with light/full profiles
- `app/src-tauri/tauri.conf.json` - Tauri bundling configuration
- `app/src-tauri/src/daemon_manager.rs` - Process lifecycle management
- `server/core/factories/service_factory.py` - Service creation with fallbacks
- `server/core/factories/utils/model_resolver.py` - Model path resolution
- `server/.env` - Runtime configuration
- `server/models/registry.json` - (TO CREATE) Model catalog

### Recent Commits
- `cecb96c` - First working bundle (Oct 19)
- `a1285a7` - Production bundle with code signing (Oct 20)
- `e9bdea2` - ServiceFactory and build profiles (Oct 27)

---

**Last Updated**: 2025-10-30
**Next Review**: After Phase 1 completion
**Owner**: Development Team
