# Using the LocalCat Team in Factory CLI

## Quick Start

The LocalCat voice agent team is now configured and ready to use in your Factory CLI. Here's how to activate and work with each specialist droid.

## Available Droids

### 🎭 Manager (Orchestrator)
**When to use**: Strategic planning, requirements analysis, backlog management
```bash
task-cli manager "Analyze current voice agent performance and create optimization plan"
task-cli manager "Review memory systems implementation and suggest improvements"
task-cli manager "Plan next development sprint for voice latency optimization"
```

### 🧠 Memory Systems Specialist
**When to use**: Memory optimization, context retrieval, hotmem service issues
```bash
task-cli memory-systems "Optimize memory retrieval for <800ms latency target"
task-cli memory-systems "Implement enhanced FTS with BM25 + expansion"
task-cli memory-systems "Fix context injection to show only 2 relevant bullets"
task-cli memory-systems "Debug hotmem service performance bottlenecks"
```

### 🎙️ Python Voice Specialist
**When to use**: Pipecat pipeline, TTS/STT optimization, Metal threading issues
```bash
task-cli python-voice-specialist "Optimize TTS process isolation for Metal conflicts"
task-cli python-voice-specialist "Configure MLX Whisper for faster speech-to-text"
task-cli python-voice-specialist "Fix WebRTC audio buffer management"
task-cli python-voice-specialist "Integrate new TTS model (Kokoro/Marvis)"
```

### 🎨 React UI Developer
**When to use**: Voice interface, WebRTC integration, client-side optimization
```bash
task-cli react-ui-developer "Implement voice activity detection visualization"
task-cli react-ui-developer "Fix WebRTC connection issues in voice UI"
task-cli react-ui-developer "Add real-time audio level meters"
task-cli react-ui-developer "Optimize client-side performance for low latency"
```

### ⚡ Performance Analyzer
**When to use**: Latency profiling, bottleneck identification, performance validation
```bash
task-cli performance-analyzer "Profile end-to-end voice latency pipeline"
task-cli performance-analyzer "Identify bottlenecks in memory retrieval path"
task-cli performance-analyzer "Validate <800ms latency target compliance"
task-cli performance-analyzer "Optimize Metal framework performance"
```

### 🤖 ML Model Optimizer
**When to use**: MLX optimization, model quantization, inference acceleration
```bash
task-cli ml-model-optimizer "Optimize Gemma3n 4B for sub-300ms inference"
task-cli ml-model-optimizer "Quantize TTS models for faster startup"
task-cli ml-model-optimizer "Configure MLX for Apple Silicon M-series chips"
task-cli ml-model-optimizer "Implement model caching with HF_HUB_OFFLINE"
```

### 🧪 Testing Automation
**When to use**: Voice testing, performance validation, CI/CD setup
```bash
task-cli testing-automation "Create latency tests for <800ms validation"
task-cli testing-automation "Set up automated voice interaction testing"
task-cli testing-automation "Implement memory context accuracy tests"
task-cli testing-automation "Configure CI/CD for voice agent deployment"
```

### 🔧 DevOps Automation
**When to use**: Environment setup, dependency management, workflow optimization
```bash
task-cli devops-automation "Set up local development environment"
task-cli devops-automation "Configure model preparation scripts"
task-cli devops-automation "Optimize build process for voice development"
task-cli devops-automation "Set up process isolation for Metal threading"
```

## Example Workflows

### Workflow 1: Performance Optimization
```bash
# Step 1: Manager analyzes and creates plan
task-cli manager "Analyze current voice latency and create optimization spec"

# Step 2: Performance Analyzer identifies bottlenecks
task-cli performance-analyzer "Profile current voice pipeline and identify bottlenecks"

# Step 3: Specialists implement fixes
task-cli python-voice-specialist "Optimize TTS process isolation for Metal conflicts"
task-cli ml-model-optimizer "Optimize MLX inference for faster response times"
task-cli memory-systems "Optimize memory retrieval path performance"

# Step 4: Testing validates improvements
task-cli testing-automation "Run latency tests to validate <800ms target"
```

### Workflow 2: New Feature Development
```bash
# Step 1: Manager plans feature
task-cli manager "Plan speaker enrollment feature for voice agent"

# Step 2: Specialists implement components
task-cli python-voice-specialist "Implement speaker enrollment in Pipecat pipeline"
task-cli react-ui-developer "Create enrollment UI with voice feedback"
task-cli memory-systems "Add speaker profile storage to memory system"

# Step 3: Testing validates functionality
task-cli testing-automation "Create tests for speaker enrollment workflow"

# Step 4: Performance validates latency impact
task-cli performance-analyzer "Validate enrollment doesn't break <800ms latency"
```

### Workflow 3: Bug Investigation
```bash
# Step 1: Performance Analyzer investigates
task-cli performance-analyzer "Investigate memory retrieval slowdown in conversations"

# Step 2: Memory Systems fixes issues
task-cli memory-systems "Fix enhanced FTS indexing performance issue"

# Step 3: Testing validates fix
task-cli testing-automation "Test memory path performance after optimization"
```

## Best Practices

### 1. Start with Manager
Always begin complex tasks with the Manager droid for planning and delegation:
```bash
task-cli manager "Plan the implementation of [feature/optimization]"
```

### 2. Delegate to Appropriate Specialists
Use the right droid for each domain:
- **Performance issues** → `performance-analyzer`
- **Memory/context problems** → `memory-systems`  
- **Audio pipeline** → `python-voice-specialist`
- **UI/UX issues** → `react-ui-developer`
- **ML optimization** → `ml-model-optimizer`
- **Testing** → `testing-automation`
- **Dev/ops** → `devops-automation`

### 3. Validate with Testing
Always validate changes with testing droid:
```bash
task-cli testing-automation "Test [feature] implementation"
```

### 4. Performance Gate
Ensure all changes meet performance requirements:
```bash
task-cli performance-analyzer "Validate <800ms latency after [changes]"
```

## Context Tips

### Provide Specific Context
When calling droids, provide specific information:
```bash
# Instead of:
task-cli performance-analyzer "Fix performance issues"

# Use:
task-cli performance-analyzer "Profile server/bot.py pipeline, TTS response times are 900ms, need to get under 800ms"
```

### Reference Files
Mention specific files or components:
```bash
task-cli memory-systems "Optimize context injection in server/core/memory/context.py for faster retrieval"
task-cli python-voice-specialist "Fix Metal threading in server/tts_mlx_isolated.py"
```

### Include Error Messages
If debugging, include error messages:
```bash
task-cli python-voice-specialist "Fix this Metal threading error: 'Metal device assertion failed' in TTS service"
```

## Droid Autonomy Settings

### Low Autonomy (--auto low)
Use for documentation, analysis, planning:
```bash
task-cli manager --auto low "Create documentation for voice agent architecture"
task-cli performance-analyzer --auto low "Analyze current performance bottlenecks"
```

### Medium Autonomy (--auto medium)
Use for implementation, testing, execution:
```bash
task-cli python-voice-specialist --auto medium "Implement TTS optimization"
task-cli testing-automation --auto medium "Run performance tests"
task-cli memory-systems --auto medium "Deploy enhanced FTS indexing"
```

## Common Commands Reference

### File Operations
```bash
task-cli [droid] --auto low "Create new file at [path] with [content]"
task-cli [droid] --auto low "Edit [file] to fix [issue]"
task-cli [droid] --auto low "Read and analyze [file]"
```

### Command Execution
```bash
task-cli [droid] --auto medium "Run pytest to validate changes"
task-cli [droid] --auto medium "Install new dependencies with uv/pip"
task-cli [droid] --auto medium "Start voice agent server for testing"
task-cli [droid] --auto medium "Commit changes to git repository"
```

### Analysis and Planning
```bash
task-cli manager "Analyze current state and create implementation plan"
task-cli performance-analyzer "Profile performance and identify bottlenecks"
task-cli memory-systems "Review memory implementation and suggest optimizations"
```

## Integration with Git Workflow

### Feature Development Flow
```bash
# 1. Manager creates spec
task-cli manager "Create spec for latency optimization in tasks/latency_optimization.md"

# 2. Specialists implement
task-cli performance-analyzer --auto medium "Implement latency optimizations from spec"
task-cli python-voice-specialist --auto medium "Apply TTS optimizations"

# 3. Testing validates
task-cli testing-automation --auto medium "Run all tests and validate performance"

# 4. Commit changes
git add .
git commit -m "perf: optimize voice latency to <800ms target"
```

### Debugging Flow
```bash
# 1. Analyze issue
task-cli performance-analyzer "Investigate performance regression in voice pipeline"

# 2. Fix identified issues
task-cli [specialist] --auto medium "Implement fixes for identified bottlenecks"

# 3. Validate fixes
task-cli testing-automation --auto medium "Run regression tests"
```

## Tips for Success

### 1. Be Specific
Provide detailed context, file paths, and expected outcomes.

### 2. Use the Right Droid
Each droid has specialized expertise - use the appropriate one.

### 3. Validate Performance
Always validate that changes meet the <800ms latency requirement.

### 4. Test Thoroughly
Use the testing droid to validate functionality and performance.

### 5. Follow the Workflow
Manager → Specialists → Testing → Performance validation → Commit

### 6. Iterate as Needed
Don't hesitate to refine and iterate on solutions.

---

## Quick Reference Commands

```bash
# Planning
task-cli manager "Plan [task]"

# Analysis  
task-cli performance-analyzer "Analyze [component]"
task-cli memory-systems "Review [memory feature]"

# Implementation
task-cli python-voice-specialist --auto medium "Implement [audio feature]"
task-cli react-ui-developer --auto medium "Create [UI component]"
task-cli ml-model-optimizer --auto medium "Optimize [ML model]"

# Testing
task-cli testing-automation --auto medium "Test [feature]"
task-cli performance-analyzer --auto medium "Validate [performance]"

# DevOps
task-cli devops-automation --auto medium "Setup [environment]"
```

The LocalCat team is now ready to help you build and optimize your voice agent with sub-800ms latency performance!
