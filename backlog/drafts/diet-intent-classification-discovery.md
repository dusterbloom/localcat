# DIET Intent Classification Discovery Report

**Date**: September 19, 2025
**Project**: LocalCat Voice Agent
**Research Focus**: DIET (Dual Intent and Entity Transformer) Integration for Intent Recognition

## Executive Summary

DIET (Dual Intent and Entity Transformer) represents a breakthrough lightweight architecture that could significantly enhance our voice agent's understanding capabilities while maintaining our <200ms latency requirements. This report documents our research findings and presents a strategic integration plan.

## Current State Analysis

### Existing Architecture Limitations
Our current voice agent pipeline processes all user utterances uniformly through:
1. **STT** (Kyutai/Whisper) → Raw transcription
2. **HotPathMemoryProcessor** → Universal memory extraction using spaCy UD patterns
3. **LLM** → Response generation

**Key Issues Identified**:
- No intent differentiation: "Remember my birthday" vs "What's the weather?" processed identically
- Inefficient memory operations on non-memory utterances
- Missing conversational flow understanding
- Latency overhead for unnecessary memory processing

### Current Performance Baseline
- Memory processing: <200ms p95 target
- Total pipeline latency: ~800ms end-to-end
- Processing all utterances through full memory extraction pipeline

## DIET Architecture Research

### Core Capabilities
DIET is a **multi-task transformer architecture** that simultaneously handles:
1. **Intent Classification**: Understanding user intentions
2. **Entity Recognition**: Extracting structured information

### Performance Characteristics
- **6x faster training** than BERT-based approaches
- **Lightweight inference**: ~10-20ms on modern hardware
- **State-of-the-art accuracy** without massive pre-training requirements
- **Modular design**: Can be configured for intent-only classification

### Technical Architecture
```
Input Text → Tokenization → Transformer Layers → Dual Output Heads
                                                 ├── Intent Classification
                                                 └── Entity Recognition
```

**Key Technical Advantages**:
- Shared transformer backbone for both tasks
- Plug-and-play embedding support (BERT, GloVe, ConveRT)
- Configurable model size and complexity
- No GPU requirements for training on small datasets

## Integration Strategy

### Proposed Pipeline Enhancement
```
STT → [NEW] DIET Intent Classifier → Smart Memory Processor → LLM
```

### Intent Categories for Voice Agent

#### Memory Operations
- `remember_fact`: "Remember that I like coffee", "Save this information"
- `recall_query`: "What did I tell you about my job?", "Remind me about my meeting"
- `forget_request`: "Forget what I said about that", "Delete that information"
- `memory_check`: "Do you remember when I...?", "What do you know about..."

#### Conversational Flow
- `general_chat`: "How are you?", "Tell me a joke", "What's your favorite color?"
- `clarification`: "What do you mean?", "Can you explain?", "I don't understand"
- `correction`: "No, that's wrong", "Actually it's different", "Let me correct that"
- `continuation`: "Go on", "Tell me more", "What else?"

#### System Operations
- `capability_query`: "What can you do?", "Help me understand your features"
- `greeting`: "Hello", "Hi there", "Good morning"
- `goodbye`: "Goodbye", "See you later", "Bye"
- `affirmation`: "Yes", "That's right", "Correct"
- `negation`: "No", "That's wrong", "Incorrect"

### Integration Points

#### 1. Pre-Memory Processing
**Location**: Between STT and HotPathMemoryProcessor
**Function**: Intent-based routing decisions

```python
# Pseudo-code integration
if intent in ["remember_fact", "recall_query", "forget_request"]:
    await memory_processor.process_with_intent(text, intent)
elif intent == "general_chat":
    skip_memory_processing()  # Save 200ms!
else:
    await memory_processor.process_lightweight(text)
```

#### 2. Memory Retrieval Optimization
**Current**: Universal bullet retrieval for all utterances
**Enhanced**: Intent-aware retrieval strategies

```python
retrieval_strategies = {
    "recall_query": priority_retrieval_with_semantic_search,
    "remember_fact": minimal_retrieval_focus_storage,
    "general_chat": contextual_retrieval_only,
    "correction": recent_memory_focus
}
```

#### 3. Context Injection Intelligence
**Current**: Fixed 3-bullet injection
**Enhanced**: Intent-adaptive context sizing

```python
context_sizing = {
    "recall_query": 5,      # More context for explicit queries
    "remember_fact": 1,     # Minimal context for storage operations
    "general_chat": 2,      # Moderate context for conversation
    "clarification": 3      # Current conversation context
}
```

## Performance Impact Analysis

### Latency Budget
- **DIET Inference**: ~10-20ms (well within budget)
- **Memory Processing Savings**: Up to 200ms for non-memory intents
- **Net Performance Gain**: Significant latency reduction for conversational utterances

### Training Requirements
- **Dataset Size**: 50-100 examples per intent (minimal)
- **Training Time**: Minutes on Apple Silicon (6x faster than BERT)
- **Iteration Cycle**: Rapid experimentation and improvement

### Resource Utilization
- **Memory Footprint**: Lightweight compared to large language models
- **CPU Usage**: Minimal inference overhead
- **Compatibility**: Seamless integration with existing MLX stack

## Implementation Approach

### Phase 1: Proof of Concept
1. **Minimal Training Set**: 10 intents with 20 examples each
2. **Basic Integration**: Intent classification before memory processing
3. **Performance Validation**: Latency impact measurement

### Phase 2: Production Integration
1. **Expanded Training Set**: LLM-generated diverse examples
2. **Smart Routing**: Full intent-based pipeline optimization
3. **Context Intelligence**: Adaptive memory retrieval

### Phase 3: Advanced Features
1. **Confidence Thresholding**: Fallback strategies for uncertain classifications
2. **Online Learning**: Continuous improvement from user interactions
3. **Multi-turn Context**: Intent history for conversation flow

## Technical Requirements

### Dependencies
```bash
pip install rasa[transformers]  # Core DIET implementation
pip install openai              # Training data generation
pip install google-cloud-ai    # Alternative data generation
```

### Training Infrastructure
- **Local Training**: Apple Silicon optimized
- **Cloud Alternative**: Google Colab for experimentation
- **Data Generation**: LLM-powered synthetic training data

### Integration Touchpoints
- `hotpath_processor.py`: Intent-aware memory processing
- `bot.py`: Pipeline integration
- New: `intent_classifier.py`: DIET wrapper and utilities

## Risk Assessment

### Technical Risks
- **Model Accuracy**: Low confidence classifications could degrade experience
- **Latency Regression**: Poorly optimized integration could exceed budget
- **Training Data Quality**: Synthetic data may not capture real usage patterns

### Mitigation Strategies
- **Fallback Pipeline**: Route uncertain classifications through existing pipeline
- **Performance Monitoring**: Real-time latency tracking and alerting
- **Iterative Training**: Start with real user utterances, expand with synthetic data

## Success Metrics

### Performance Metrics
- **Intent Classification Accuracy**: >90% on validation set
- **Latency Impact**: <20ms additional processing time
- **Memory Processing Efficiency**: >50% reduction in unnecessary memory operations

### User Experience Metrics
- **Response Relevance**: Improved contextual accuracy
- **Conversation Flow**: More natural interaction patterns
- **Error Recovery**: Better handling of clarifications and corrections

## Conclusion

DIET integration represents a strategic enhancement that addresses current pipeline inefficiencies while maintaining our performance requirements. The lightweight architecture, rapid training capabilities, and proven performance make it an ideal fit for our voice agent's intent understanding needs.

**Recommended Next Steps**:
1. Implement proof-of-concept with minimal training set
2. Develop automated training data generation pipeline
3. Create comprehensive integration guide for team deployment

## References

- [DIET: Lightweight Language Understanding for Dialogue Systems](https://arxiv.org/abs/2004.09936)
- [Rasa DIET Implementation](https://rasa.com/blog/introducing-dual-intent-and-entity-transformer-diet-state-of-the-art-performance-on-a-lightweight-architecture/)
- [Building DIET from Scratch with PyTorch](https://medium.com/botisan-ai/building-rasas-diet-classifier-from-scratch-using-pytorch-part-1-a5f2a71982ac)

---

**Document Status**: Draft Research Report
**Next Action**: Create implementation guide and training data generation tools