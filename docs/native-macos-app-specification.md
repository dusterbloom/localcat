# LocalCat Native macOS Application Specification

## Executive Summary

This document provides a comprehensive specification for developing a native macOS application that bundles the entire LocalCat voice agent system into a single, user-friendly application. The goal is to eliminate technical setup requirements and provide a one-click installation and launch experience while maintaining the sub-800ms latency performance of the current system.

## Current System Analysis

### Architecture Overview
LocalCat currently consists of:
- **Server**: Python FastAPI application with Pipecat framework
- **Client**: Next.js React web application using Pipecat Voice UI Kit
- **External Dependencies**: Local LLM server (Ollama/LM Studio), MLX models
- **Communication**: Serverless WebRTC transport

### Technical Stack Analysis

#### Server Components
- **Language**: Python 3.12+
- **Framework**: FastAPI with Uvicorn
- **Core Dependencies**:
  - Pipecat AI framework (voice pipeline)
  - MLX libraries (mlx-lm, mlx-audio) for Apple Silicon optimization
  - OpenCV, NLTK, spaCy for NLP
  - HotMem memory system with LMDB storage
  - Audio Intelligence with SpeechBrain

#### Client Components
- **Framework**: Next.js 15.4.3 with React 19
- **Dependencies**:
  - @pipecat-ai/client-js and client-react
  - @pipecat-ai/voice-ui-kit
  - WebRTC transport libraries

#### External Dependencies
- **LLM Server**: Ollama (default) or LM Studio
- **Models**: Gemma3n 4B, MLX Whisper, Kokoro TTS, Silero VAD
- **Storage**: Local file system for models and memory databases

### Current User Setup Process & Pain Points

**Required Steps:**
1. Install Python 3.12+ and Node.js 18+
2. Install UV package manager
3. Install Ollama or LM Studio
4. Clone repository and setup virtual environment
5. Install Python dependencies (~100MB)
6. Install Node.js dependencies (~200MB)
7. Configure environment variables
8. Pre-cache ML models (optional but recommended)
9. Start Python server
10. Start Node.js client
11. Start LLM server
12. Navigate to web interface

**Major Pain Points:**
- 12+ manual setup steps
- Multiple terminal windows required
- Technical knowledge prerequisites
- Model download delays on first run
- Environment configuration complexity
- Port management and conflicts
- Service dependency management

## Native macOS Application Specification

### 1. Technical Approach Decision

**Recommended Approach: Electron with Native Extensions**

**Rationale:**
- **Web Leverage**: Reuse existing React client with minimal changes
- **Native Integration**: Access macOS APIs for system services
- **Development Speed**: Faster than full native rewrite
- **Maintenance**: Single codebase for UI
- **Performance**: Sufficient for UI requirements; core processing remains Python

**Alternative: Native Swift/SwiftUI**
- **Pros**: Best performance, truly native feel
- **Cons**: Complete rewrite of client, longer development time, need to reimplement WebRTC integration

### 2. Architecture Design

#### High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│                LocalCat Native App                       │
├─────────────────────────────────────────────────────────┤
│  Electron Main Process                                   │
│  ├── Service Orchestrator                               │
│  ├── Process Manager                                     │
│  ├── Configuration Manager                              │
│  └── Auto-updater                                       │
├─────────────────────────────────────────────────────────┤
│  Renderer Process (React UI)                            │
│  ├── Voice UI Kit Components                            │
│  ├── Status Dashboard                                   │
│  ├── Settings Panel                                     │
│  └── Setup Wizard                                       │
├─────────────────────────────────────────────────────────┤
│  Embedded Services                                      │
│  ├── Python Server (bundled)                           │
│  ├── Ollama Engine (embedded)                          │
│  ├── Model Cache Manager                                │
│  └── Health Monitor                                     │
└─────────────────────────────────────────────────────────┘
```

#### Service Orchestration
```
Service Orchestrator
├── LLM Service Manager
│   ├── Embedded Ollama
│   ├── Model Download Manager
│   └── Health Monitoring
├── Python Server Manager
│   ├── Virtual Environment
│   ├── Dependency Management
│   └── Process Lifecycle
├── Model Cache Manager
│   ├── Initial Download Queue
│   ├── Cache Validation
│   └── Storage Optimization
└── Configuration Manager
    ├── User Preferences
    ├── Environment Variables
    └── Service Discovery
```

### 3. Application Bundle Structure

#### Package Organization
```
LocalCat.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   ├── LocalCat (Electron main)
│   │   ├── python/ (Embedded Python runtime)
│   │   ├── ollama/ (Embedded Ollama)
│   │   └── resources/
│   │       ├── server/ (Python server code)
│   │       ├── models/ (Pre-bundled or cached models)
│   │       └── client/ (React build output)
│   ├── Resources/
│   │   ├── app.icns
│   │   ├── assets/
│   │   └── defaults/
│   │       ├── config.json
│   │       └── models.json
│   └── Frameworks/
│       ├── Electron Framework.framework
│       └── required native libraries
```

#### Model Management Strategy
```
Model Categories:
├── Core Models (bundled)
│   ├── Silero VAD (~50MB)
│   └── Smart Turn Models (~20MB)
├── Optional Models (download on demand)
│   ├── MLX Whisper (~150MB)
│   ├── Kokoro TTS (~200MB)
│   └── Gemma3n 4B (~2.5GB)
└── Cache Management
    ├── Progressive Download
    ├── Compression
    └── Delta Updates
```

### 4. User Experience Flow

#### First-Time Setup Experience
```
1. Application Download
   ├── DMG installer (~200MB base)
   ├── Drag-and-drop installation
   └── Security bypass prompt

2. Initial Launch
   ├── Welcome screen with progress indicators
   ├── Model download manager
   │   ├── Essential models (VAD, Smart Turn)
   │   ├── Optional models (with size indicators)
   │   └── Background downloading
   ├── Microphone permission request
   └── System preferences integration

3. Quick Setup Wizard
   ├── Voice calibration
   ├── Model selection (speed vs quality)
   ├── Privacy settings
   └── Performance tuning

4. Ready State
   ├── System tray integration
   ├── Global hotkey setup
   └── Voice activation ready
```

#### Ongoing User Experience
```
Main Interface:
├── Voice Interaction Panel
│   ├── Visual feedback
│   ├── Transcript display
│   └── Memory context viewer
├── Status Dashboard
│   ├── Service health indicators
│   ├── Latency metrics
│   └── Resource usage
├── Settings & Configuration
│   ├── Model management
│   ├── Performance tuning
│   ├── Privacy controls
│   └── Advanced options
└── Memory Management
    ├── Conversation history
    ├── Memory search
    └── Data export/import
```

### 5. Installation and Packaging Strategy

#### Distribution Method
**Primary: Mac App Store**
- Advantages: Automatic updates, trusted distribution, easier installation
- Challenges: App Store review process, sandboxing limitations

**Secondary: Direct Distribution**
- DMG with notarization
- Auto-updater integration
- Developer certificate signing

#### Bundle Size Optimization
```
Target Sizes:
├── Base Application: ~200MB
│   ├── Electron runtime: ~100MB
│   ├── Python runtime: ~50MB
│   ├── Ollama binary: ~30MB
│   └── UI assets: ~20MB
├── Essential Models: ~70MB
│   ├── Silero VAD: ~50MB
│   └── Smart Turn: ~20MB
└── Optional Models: ~2.8GB
    ├── Whisper: ~150MB
    ├── Kokoro TTS: ~200MB
    └── Gemma3n 4B: ~2.5GB

Compression Strategy:
├── LZMA compression for models
├── Delta updates for model versions
└── Lazy loading of optional components
```

### 6. Service Orchestration Approach

#### Process Management
```
Service Lifecycle:
├── Application Startup
│   ├── Health checks
│   ├── Service discovery
│   ├── Port allocation
│   └── Dependency resolution
├── Service Orchestration
│   ├── Ollama server (embedded)
│   │   ├── Model loading
│   │   ├── API endpoint management
│   │   └── Resource monitoring
│   ├── Python server (embedded)
│   │   ├── Virtual environment setup
│   │   ├── Configuration injection
│   │   └── Process monitoring
│   └── Health monitoring
│       ├── Service health checks
│       ├── Performance metrics
│       └── Automatic recovery
└── Graceful Shutdown
    ├── Service coordination
    ├── State persistence
    └── Resource cleanup
```

#### Configuration Management
```
Configuration Layers:
├── Default Configuration (embedded)
├── User Preferences (~/Library/Application Support/LocalCat)
├── Environment Variables (runtime injection)
└── Model-Specific Settings

Key Features:
├── Automatic port allocation
├── Conflict resolution
├── Migration handling
└── Validation and error recovery
```

### 7. Error Handling and Troubleshooting

#### Error Categories
```
System Errors:
├── Insufficient resources
├── Permission denied
├── Port conflicts
└── Service failures

Model Errors:
├── Download failures
├── Corrupted models
├── Incompatible versions
└── Storage space issues

Performance Issues:
├── High latency
├── Memory usage
├── CPU throttling
└── Thermal management
```

#### Recovery Strategies
```
Automated Recovery:
├── Service restart on failure
├── Model re-download on corruption
├── Configuration reset on invalid state
└── Fallback to minimal configuration

User Guidance:
├── Clear error messages
├── Suggested solutions
├── Diagnostic tools
└── Support integration
```

### 8. Performance Considerations

#### Latency Optimization
```
Target: <800ms end-to-end response

Optimization Strategies:
├── Model pre-loading and caching
├── Service warm-up on launch
├── Memory pre-allocation
├── CPU affinity management
└── Thermal throttling awareness
```

#### Resource Management
```
Memory Usage:
├── Model loading optimization
├── Garbage collection tuning
├── Memory pressure monitoring
└── Swap file management

CPU Usage:
├── Process priority management
├── Core utilization optimization
├── Background task scheduling
└── Power efficiency settings
```

### 9. Security and Privacy

#### Security Measures
```
Application Security:
├── Code signing and notarization
├── Runtime application signing
├── Sandboxing (where possible)
├── Network security (localhost only)
└── Input validation and sanitization

Data Protection:
├── Local encryption of sensitive data
├── Secure key storage
├── Memory sanitization
└── Secure model distribution
```

#### Privacy Features
```
User Privacy:
├── Local-only processing
├── No telemetry without consent
├── Data retention controls
├── Anonymous mode options
└── Data export/deletion tools
```

### 10. Development Roadmap

#### Phase 1: Foundation (4-6 weeks)
```
Week 1-2: Basic Electron Application
├── Electron setup and configuration
├── Basic UI integration (React client)
├── Service orchestration framework
└── Configuration management system

Week 3-4: Python Integration
├── Embedded Python runtime
├── Server process management
├── Basic communication layer
└── Error handling framework

Week 5-6: Core Services
├── Health monitoring system
├── Basic model management
├── Service lifecycle management
└── Initial testing framework
```

#### Phase 2: Service Integration (6-8 weeks)
```
Week 7-8: LLM Service Integration
├── Embedded Ollama setup
├── Model download manager
├── Configuration injection
└── API endpoint management

Week 9-10: Model Management System
├── Progressive model downloading
├── Cache validation and cleanup
├── Model version management
└── Storage optimization

Week 11-12: Advanced Features
├── Auto-updater integration
├── System tray integration
├── Global hotkeys
└── Performance monitoring

Week 13-14: Polish and Testing
├── UI/UX refinements
├── Performance optimization
├── Error handling improvements
└── Comprehensive testing
```

#### Phase 3: Production Readiness (4-6 weeks)
```
Week 15-16: Distribution Preparation
├── App Store submission preparation
├── Code signing and notarization
├── Installation testing
└── Documentation creation

Week 17-18: Final Testing
├── Beta testing program
├── Performance validation
├── Security audit
└── Compatibility testing

Week 19-20: Launch Preparation
├── Marketing materials
├── Support documentation
├── Release finalization
└── Distribution setup
```

### 11. Technical Specifications

#### System Requirements
```
Minimum Requirements:
├── macOS 13.0 (Ventura) or later
├── Apple Silicon M1 or later
├── 8GB RAM (16GB recommended)
├── 10GB available storage
└── Internet connection for model downloads

Recommended Configuration:
├── macOS 14.0 (Sonoma) or later
├── Apple Silicon M2 or later
├── 16GB RAM or more
├── 20GB available storage
└── Broadband internet connection
```

#### Performance Targets
```
Latency Goals:
├── Cold start: <30 seconds
├── Warm start: <5 seconds
├── First response: <800ms
├── Subsequent responses: <600ms
└── Model switching: <2 seconds

Resource Usage:
├── Memory: 2-4GB baseline
├── CPU: 20-30% during conversation
├── Storage: 5-10GB with all models
└── Network: Only for model downloads
```

### 12. Testing Strategy

#### Test Categories
```
Unit Testing:
├── Service orchestration
├── Configuration management
├── Error handling
└── Model management

Integration Testing:
├── Service communication
├── Model loading
├── Performance validation
└── Error recovery

User Acceptance Testing:
├── Installation flow
├── Setup wizard
├── Voice interaction
└── Settings management

Performance Testing:
├── Latency measurement
├── Resource usage
├── Stress testing
└── Long-term stability
```

## Conclusion

This specification provides a comprehensive roadmap for transforming LocalCat from a developer-focused system into a user-friendly native macOS application. The key innovations include:

1. **Service Orchestration**: Automated management of all system components
2. **Model Management**: Progressive downloading and intelligent caching
3. **User Experience**: One-click installation and setup wizard
4. **Performance**: Maintaining sub-800ms latency targets
5. **Reliability**: Robust error handling and recovery mechanisms

The phased development approach ensures manageable increments while delivering value to users quickly. The Electron-based approach balances development speed with native integration capabilities.

The resulting application will democratize access to local voice AI technology, making it accessible to non-technical users while preserving the power and flexibility of the underlying LocalCat system.
