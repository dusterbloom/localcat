# Enhanced Intent Classification System Design

## Current System Analysis

### Strengths
- **Language-agnostic**: Uses Universal Dependencies (UD) patterns
- **Integrated**: Already plugged into HotMemory pipeline
- **Efficient**: Fast rule-based classification
- **Comprehensive intents**: 8 intent types with specific handling

### Limitations
- **Structural-only**: No contextual understanding
- **Always retrieves**: Inefficient - retrieves then filters
- **Rule-based**: Limited to pattern recognition
- **No confidence scoring**: Binary decisions only
- **No adaptation**: Static rules don't learn from user

## SOTA 2025 Enhancement Strategy

### Phase 1: Transformer-Based Intent Classification

#### 1.1 Hybrid Intent Classifier Architecture
```
Current UD Rule-based ──┐
                       ├─→ Hybrid Classifier ──→ Enhanced Intent Decision
DSPy Transformer    ──┘
```

**Components:**
- **UD Rules**: Maintain existing fast path (20% weight)
- **DSPy Transformer**: Contextual understanding (80% weight)
- **Confidence Fusion**: Weighted decision based on confidence scores
- **Fallback Logic**: Graceful degradation to UD only

#### 1.2 MiniLM Intent Classification Integration
```python
class MiniLMIntentClassifier:
    def __init__(self):
        # Use kousik-2310/intent-classifier-minilm - 22.7M params, fast inference
        self.model_name = "kousik-2310/intent-classifier-minilm"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Optimize for Apple Silicon using existing MLX stack
        if mlx_available:
            self.model = mlx.convert(self.model)  # Convert to MLX for Metal acceleration
    
    def classify(self, text: str) -> IntentClassificationResult:
        # Fast inference using MiniLM architecture
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
        # Get top intent and confidence score
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return IntentClassificationResult(
            intent_type=self.id_to_intent(predicted_class),
            confidence=confidence,
            all_probabilities=probabilities[0].tolist()
        )
```

#### 1.3 Hybrid DSPy + MiniLM Classifier
```python
class HybridIntentClassifier:
    def __init__(self):
        self.minilm_classifier = MiniLMIntentClassifier()
        self.ud_classifier = IntentClassifier()  # Existing UD classifier
        self.dspy_modules = {
            "contextual_analysis": dspy.Predict("conversation_history, question -> contextual_intent"),
            "confidence_refinement": dspy.Predict("intent, context, confidence -> refined_confidence")
        }
    
    def classify_with_context(self, text: str, conversation_history: List[str]) -> EnhancedIntentAnalysis:
        # Fast MiniLM classification (80% weight)
        minilm_result = self.minilm_classifier.classify(text)
        
        # UD rule-based classification (20% weight)  
        ud_result = self.ud_classifier.analyze(text)
        
        # DSPy contextual refinement
        if conversation_history and minilm_result.confidence < 0.8:
            dspy_refined = self.dspy_modules["contextual_analysis"](
                question=text,
                conversation_history=conversation_history[-3:]  # Last 3 turns
            )
            # Combine results with contextual understanding
            return self._combine_results(minilm_result, ud_result, dspy_refined)
        
        return self._combine_results(minilm_result, ud_result)
```

#### 1.3 Enhanced Intent Types
```python
class EnhancedIntentType(Enum):
    # Existing intents
    FACT_STATEMENT = "fact_statement"
    QUESTION_WITH_FACT = "question_with_fact"
    PURE_QUESTION = "pure_question"
    REACTION = "reaction"
    HYPOTHETICAL = "hypothetical"
    CORRECTION = "correction"
    TEMPORAL_FACT = "temporal_fact"
    MULTIPLE_FACTS = "multiple_facts"
    
    # New SOTA intents
    CONVERSATION_CONTINUATION = "conversation_continuation"  # "Yeah, and..."
    CLARIFICATION_REQUEST = "clarification_request"      # "What do you mean?"
    META_CONVERSATION = "meta_conversation"              # "How do you work?"
    PERSONAL_QUERY = "personal_query"                      # "Remember when I..."
    ACTION_REQUEST = "action_request"                      # "Set a reminder for..."
    SENTIMENT_EXPRESSION = "sentiment_expression"          # "I love this!"
    CONTEXT_SWITCH = "context_switch"                      # "Let's talk about..."
```

### Phase 2: Confidence-Based Retrieval Decisions

#### 2.1 Intelligent Retrieval Router
```python
class RetrievalRouter:
    def __init__(self):
        self.retrieval_thresholds = {
            "high_confidence_memory_need": 0.8,
            "moderate_confidence": 0.5,
            "low_confidence_fast_path": 0.3
        }
    
    def should_retrieve(self, intent_analysis: EnhancedIntentAnalysis) -> RetrievalDecision:
        if intent_analysis.confidence >= self.retrieval_thresholds["high_confidence_memory_need"]:
            return RetrievalDecision.COMPREHENSIVE_RETRIEVAL
        elif intent_analysis.confidence >= self.retrieval_thresholds["moderate_confidence"]:
            return RetrievalDecision.QUICK_RETRIEVAL
        else:
            return RetrievalDecision.NO_RETRIEVAL
```

#### 2.2 Dynamic Retrieval Strategies
- **NO_RETRIEVAL**: Direct LLM response (greetings, reactions, simple questions)
- **QUICK_RETRIEVAL**: Entity-only lookup (specific factual queries)
- **COMPREHENSIVE_RETRIEVAL**: Full memory search (complex queries, corrections)

### Phase 3: Multi-Modal Intent Recognition

#### 3.1 Audio-Text Fusion
```python
class MultiModalIntentClassifier:
    def __init__(self):
        self.audio_features_extractor = AudioProsodyExtractor()
        self.text_classifier = DSPyIntentClassifier()
        self.fusion_model = LateFusionNetwork()
    
    def classify_with_audio(self, text: str, audio_features: Dict) -> MultiModalIntentAnalysis:
        text_intent = self.text_classifier.classify(text)
        audio_intent = self.audio_features_extractor.extract_intent(audio_features)
        return self.fusion_model.fuse(text_intent, audio_intent)
```

#### 3.2 Real-time Adaptation
```python
class AdaptiveIntentLearner:
    def __init__(self):
        self.user_preference_model = UserPreferenceModel()
        self.feedback_collector = ImplicitFeedbackCollector()
    
    def adapt_to_user(self, interaction_history: List[Interaction]) -> None:
        # Use Unsloth for efficient fine-tuning
        # Adapt intent thresholds based on user patterns
        pass
```

## Technical Implementation Plan

### File Structure
```
server/components/intent/
├── enhanced_intent_classifier.py      # Main hybrid classifier
├── dspy_intent_modules.py             # DSPy intent modules
├── retrieval_router.py                # Intelligent retrieval decisions
├── multimodal_intent.py               # Audio-text fusion (future)
├── adaptive_intent_learner.py         # User adaptation (future)
└── intent_config.py                   # Configuration and thresholds

server/components/intent/signatures/
├── intent_classification.py           # DSPy signatures
└── intent_confidence.py              # Confidence scoring

tests/intent/
├── test_enhanced_classifier.py       # Unit tests
├── test_retrieval_router.py           # Router tests
└── test_multimodal_intent.py          # Multi-modal tests
```

### Integration Points

#### 1. HotMemoryFacade Integration
```python
# In hotmemory_facade.py
class HotMemoryFacade:
    def __init__(self):
        # Replace current intent classifier
        self.intent_classifier = EnhancedIntentClassifier()
        self.retrieval_router = RetrievalRouter()
    
    def process_transcription(self, text: str, turn_id: int, lang: str = "en"):
        # Enhanced intent analysis
        intent_analysis = self.intent_classifier.analyze_with_context(text, self.conversation_history)
        
        # Intelligent retrieval decision
        retrieval_decision = self.retrieval_router.should_retrieve(intent_analysis)
        
        if retrieval_decision == RetrievalDecision.NO_RETRIEVAL:
            return self._generate_direct_response(text, intent_analysis)
        elif retrieval_decision == RetrievalDecision.QUICK_RETRIEVAL:
            return self._quick_retrieve_and_respond(text, intent_analysis)
        else:
            return self._comprehensive_memory_processing(text, intent_analysis)
```

#### 2. Performance Optimizations
- **Model Caching**: Cache DSPy model outputs
- **Fast Path**: UD-only for simple queries
- **Async Processing**: Parallel intent and audio analysis
- **Model Quantization**: Use CoreML for on-device inference

### Expected Performance Improvements

#### Latency Reduction
- **Simple queries**: 40-60% faster (no memory retrieval)
- **Complex queries**: 20-30% faster (optimized retrieval)
- **Average improvement**: 35% latency reduction

#### Accuracy Improvements
- **Intent classification**: 85% → 95%+ accuracy
- **Retrieval decisions**: 90%+ optimal routing
- **User satisfaction**: Natural conversation flow

#### Resource Efficiency
- **Memory access**: 50% reduction in unnecessary retrievals
- **CPU usage**: Lower average load with intelligent routing
- **Model efficiency**: Leverage existing DSPy/Unsloth stack

## Implementation Timeline

### Week 1: Foundation
- [ ] Implement DSPy intent classification modules
- [ ] Create hybrid classifier (UD + DSPy)
- [ ] Add enhanced intent types
- [ ] Unit tests for new classifier

### Week 2: Intelligent Routing
- [ ] Implement retrieval router
- [ ] Add confidence-based decision logic
- [ ] Integrate with HotMemoryFacade
- [ ] Performance testing and optimization

### Week 3: Advanced Features
- [ ] Multi-modal intent classification (audio-text)
- [ ] User adaptation and personalization
- [ ] Real-time learning capabilities
- [ ] End-to-end system testing

## Risk Mitigation

1. **Backward Compatibility**: Maintain existing UD classifier as fallback
2. **Performance**: Fast path for simple queries using existing UD rules
3. **Reliability**: Graceful degradation to rule-based system
4. **Testing**: Comprehensive test suite covering all scenarios

## Success Metrics

1. **Intent Accuracy**: ≥95% classification accuracy
2. **Latency**: ≥35% average reduction in response time
3. **User Experience**: Natural conversation flow with appropriate memory usage
4. **Resource Efficiency**: ≥50% reduction in unnecessary memory retrievals

This design leverages the existing 2025 AI stack (DSPy, Unsloth, GEPA) to create a state-of-the-art intent classification system that dramatically improves both performance and user experience.