# LocalCat Documentation

Welcome to the LocalCat documentation! This is your central hub for all project documentation, guides, and references.

## 📚 Documentation Structure

### [01. Getting Started](./01-getting-started/)
- **[Installation](./01-getting-started/installation.md)** - Setup requirements and installation guide
- **[Quick Start](./01-getting-started/quick-start.md)** - Get up and running in minutes
- **[Configuration](./01-getting-started/configuration.md)** - Environment variables and settings

### [02. Architecture](./02-architecture/)
- **[System Overview](./02-architecture/system-overview.md)** - High-level architecture and design
- **[Memory System](./02-architecture/memory-system.md)** - HotMem and memory processing architecture
- **[HotMem Design](./02-architecture/hotmem-design.md)** - Ultra-fast memory layer design
- **[Components](./02-architecture/components/)** - Deep dives into system components
  - TTS Engine
  - STT Engine
  - Memory Processor

### [03. Development](./03-development/)
- **[Setup](./03-development/setup.md)** - Developer environment setup
- **[Testing](./03-development/testing.md)** - Testing guidelines and practices
- **[Debugging](./03-development/debugging.md)** - Debugging tools and techniques
- **[Code Style](./03-development/code-style.md)** - Coding standards and conventions
- **[Contributing](./03-development/contributing.md)** - How to contribute to LocalCat

### [04. Tasks](./04-tasks/)
- **[Active Tasks](./04-tasks/active/)** - Current sprint and active development
- **[Backlog](./04-tasks/backlog/)** - Upcoming features and improvements
- **[Completed](./04-tasks/completed/)** - Completed tasks and their documentation

### [05. Agents](./05-agents/)
- **[Overview](./05-agents/README.md)** - AI agent system documentation
- **[Claude Agents](./05-agents/claude-agents.md)** - Claude-specific agent configurations
- **[Factory Droids](./05-agents/factory-droids/)** - Automated development agents
- **[Memory Architect](./05-agents/memory-architect.md)** - Memory system design agent

### [06. API](./06-api/)
- **[REST API](./06-api/rest-api.md)** - HTTP API endpoints
- **[WebSocket](./06-api/websocket.md)** - Real-time WebSocket protocols
- **[Pipecat Integration](./06-api/pipecat-integration.md)** - Pipecat framework integration

### [07. Guides](./07-guides/)
- **[Memory Usage](./07-guides/memory-usage.md)** - HotMem usage and best practices
- **[Intent Classification](./07-guides/intent-classification.md)** - DIET intent classification guide
- **[Coreference Resolution](./07-guides/coreference.md)** - Coreference integration guide
- **[Optimization](./07-guides/optimization.md)** - Performance optimization techniques

### [08. Roadmap](./08-roadmap/)
- **[Overview](./08-roadmap/README.md)** - Product vision and roadmap
- **[Backlog](./08-roadmap/backlog.md)** - Feature backlog and priorities
- **[Completed](./08-roadmap/completed.md)** - Delivered features and milestones
- **[Drafts](./08-roadmap/drafts/)** - Planning documents and proposals

### [09. Reports](./09-reports/)
- **[Performance](./09-reports/performance/)** - Performance analysis and benchmarks
- **[Investigations](./09-reports/investigations/)** - Technical investigations
- **[Decisions](./09-reports/decisions/)** - Architectural decision records

### [10. Reference](./10-reference/)
- **[Environment Variables](./10-reference/environment-vars.md)** - Complete env var reference
- **[Dependencies](./10-reference/dependencies.md)** - Package dependencies and versions
- **[Troubleshooting](./10-reference/troubleshooting.md)** - Common issues and solutions
- **[Glossary](./10-reference/glossary.md)** - Terms and concepts

### [Archive](./archive/)
Historical and deprecated documentation organized by date.

---

## 🔍 Quick Links

### For New Users
- Start with [Quick Start](./01-getting-started/quick-start.md)
- Review [System Overview](./02-architecture/system-overview.md)
- Check [Configuration](./01-getting-started/configuration.md)

### For Developers
- [Development Setup](./03-development/setup.md)
- [Testing Guide](./03-development/testing.md)
- [Active Tasks](./04-tasks/active/)

### For Contributors
- [Contributing Guide](./03-development/contributing.md)
- [Code Style](./03-development/code-style.md)
- [Roadmap](./08-roadmap/)

---

## 📝 Documentation Standards

### File Naming
- Use lowercase with hyphens: `memory-system.md`
- Be descriptive but concise
- Include dates for time-sensitive docs: `2025-10-14-release-notes.md`

### Document Structure
- Start with a clear title and description
- Use hierarchical headings (H1 for title, H2 for sections, etc.)
- Include a table of contents for long documents
- Add cross-references to related documentation

### Maintenance
- Keep documentation up-to-date with code changes
- Archive outdated documents to `/archive/`
- Review and update quarterly

---

## 🤝 Contributing to Documentation

To contribute to the documentation:

1. Follow the structure outlined above
2. Write clear, concise content
3. Include examples and diagrams where helpful
4. Cross-reference related documentation
5. Submit a PR with your changes

For questions or suggestions about documentation, please open an issue or contact the maintainers.

---

---

## 🎯 Current Project Status (October 2025)

### Latest Architecture (v2.0 - Factory Pattern Refactor)

**Major Components:**
- **Unified Configuration**: VoiceAgentConfig with centralized settings management
- **Service Factory**: Factory pattern for all service creation (STT, TTS, LLM, Memory, Transport)
- **Token-Aware Context**: Prevents LLM degradation with intelligent pruning (3000 tokens, 70% threshold)
- **HotMem Service**: Modular memory system with prosody-enhanced retrieval
- **Vision Processing**: Keyword-filtered vision with deduplication
- **Audio Intelligence**: Speaker recognition with automatic enrollment

**Current Model Pipeline:**
```
┌─ Voice Pipeline ──────────────────────────────────────────┐
│  Silero VAD → Smart Turn → Parakeet STT → LLM (MiniCPM-V) │
│                                                  ↓          │
│  Kokoro TTS ← Token Pruning ← Memory Injection ←┘         │
└───────────────────────────────────────────────────────────┘

┌─ Memory System (HotMem Service) ─────────────────────────┐
│  Background Summarizer → Context Injector → Frame Processor │
│         ↓                       ↓                  ↓        │
│  Session Manager ← Quality Filter ← Entity Resolver        │
│         ↓                                                  │
│  Multi-source Retrieval (convo, graph, summary, semantic)  │
└───────────────────────────────────────────────────────────┘
```

### Recent Milestones (September-October 2025)

**October 16, 2025 - Config Unification & Factory Decomposition**
- Centralized configuration with VoiceAgentConfig
- Factory pattern for service creation
- Reduced coupling and improved testability
- Comprehensive test coverage for configuration

**October 16, 2025 - Token-Aware Context Management**
- Prevents LLM degradation in long conversations
- Token counting with tiktoken
- Intelligent context pruning at 70% capacity
- Maintains minimum conversation coherence (4 turns)

**October 15-16, 2025 - Vision/TTS/STT Performance Optimizations**
- Keyword-filtered vision processing (saves tokens)
- Image deduplication
- Parakeet STT hallucination filtering
- Kokoro TTS ultra-low latency improvements

**October 15, 2025 - Prosody-Aware Retrieval**
- Audio intelligence integration with memory
- Prosody confidence scoring
- Video frame processing
- Enhanced retrieval with multi-signal scoring

**October 14, 2025 - Complete HotMem Modularization**
- Broke up 1,100-line God-object into focused components
- 66% code reduction (1,100 → 373 lines)
- SOLID compliance across all components
- Comprehensive test suite (8 unit + 1 integration test)

---

*Last Updated: October 16, 2025*