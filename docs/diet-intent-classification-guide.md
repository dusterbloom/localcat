# DIET Intent Classification Implementation Guide

**Complete Guide for Integrating DIET Intent Classification into LocalCat Voice Agent**

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Training Data Generation](#training-data-generation)
4. [Model Training](#model-training)
5. [Integration into Voice Agent](#integration-into-voice-agent)
6. [Configuration & Deployment](#configuration--deployment)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

## Overview

This guide provides step-by-step instructions for integrating DIET (Dual Intent and Entity Transformer) intent classification into your LocalCat voice agent. DIET will enable intent-aware processing, reducing latency for non-memory operations while improving conversational understanding.

### Architecture Enhancement
```
Before: STT → Memory Processing → LLM
After:  STT → DIET Intent → Smart Memory Processing → LLM
```

### Expected Benefits
- **Latency Reduction**: Skip heavy memory processing for casual conversation (~200ms savings)
- **Better Context**: Intent-aware memory retrieval and context injection
- **Conversation Flow**: Understanding user intentions for more natural interactions

## Prerequisites

### System Requirements
- Python 3.8+
- Apple Silicon Mac (for optimal MLX performance)
- 8GB+ RAM available
- Existing LocalCat voice agent setup

### Dependencies Installation
```bash
# Core DIET dependencies
pip install rasa[transformers]==3.6.13
pip install torch torchvision torchaudio

# Training data generation (choose one or both)
pip install openai                    # For OpenAI via OpenRouter
pip install google-cloud-aiplatform  # For Google APIs

# Optional: For model optimization
pip install onnx onnxruntime-coreml   # Apple Silicon optimization
```

### Environment Setup
Add to your `.env` file:
```bash
# Intent Classification Configuration
DIET_INTENT_ENABLED=true
DIET_MODEL_PATH=../models/diet_intent_classifier
DIET_CONFIDENCE_THRESHOLD=0.7
DIET_FALLBACK_TO_MEMORY=true

# Training Data Generation
OPENROUTER_API_KEY=your_key_here
GOOGLE_AI_API_KEY=your_key_here
```

## Training Data Generation

### Method 1: Automated with LLMs (Recommended)

Use the provided Colab notebook: `diet_training_data_generator.ipynb`

**Key Features**:
- Generates diverse examples for each intent
- Uses OpenAI GPT-4 or Google Gemini
- Exports directly to Rasa training format
- Includes data validation and quality checks

### Method 2: Manual Data Creation

Create `data/nlu.yml` with your training examples:

```yaml
version: "3.1"

nlu:
- intent: remember_fact
  examples: |
    - Remember that I like coffee
    - Save this information please
    - Store that my birthday is in March
    - Keep in mind I'm allergic to peanuts
    - Don't forget I work at Google
    - Note that I live in San Francisco
    - Remember my wife's name is Sarah
    - Save that I prefer morning meetings

- intent: recall_query
  examples: |
    - What did I tell you about my job?
    - Remind me about my meeting
    - Do you remember my favorite food?
    - What do you know about my family?
    - Tell me what I said about vacation
    - What information do you have about me?
    - Recall my preferences
    - What do you remember about my schedule?

- intent: general_chat
  examples: |
    - How are you doing today?
    - Tell me a joke
    - What's your favorite color?
    - How's the weather?
    - That's interesting
    - You're funny
    - I'm having a good day
    - What do you think about that?

- intent: forget_request
  examples: |
    - Forget what I said about that
    - Delete that information
    - Don't remember that anymore
    - Remove that from your memory
    - I don't want you to save that
    - Erase what I just told you

- intent: clarification
  examples: |
    - What do you mean?
    - Can you explain that?
    - I don't understand
    - Could you clarify?
    - What are you talking about?
    - Can you be more specific?

- intent: correction
  examples: |
    - No, that's wrong
    - Actually, it's different
    - Let me correct that
    - That's not right
    - I misspoke earlier
    - I need to fix what I said

- intent: greeting
  examples: |
    - Hello
    - Hi there
    - Good morning
    - Hey
    - What's up
    - How's it going

- intent: goodbye
  examples: |
    - Goodbye
    - See you later
    - Bye
    - Talk to you soon
    - Have a good day
    - Catch you later

- intent: affirmation
  examples: |
    - Yes
    - That's right
    - Correct
    - Exactly
    - You got it
    - Absolutely

- intent: negation
  examples: |
    - No
    - That's wrong
    - Incorrect
    - Not really
    - I disagree
    - Nope
```

### Intent Categories Explained

| Intent | Purpose | Memory Processing |
|--------|---------|------------------|
| `remember_fact` | Store new information | Full extraction + storage |
| `recall_query` | Retrieve stored info | Enhanced retrieval focus |
| `general_chat` | Casual conversation | Skip memory processing |
| `forget_request` | Delete information | Targeted deletion |
| `clarification` | Ask for explanation | Context-aware retrieval |
| `correction` | Fix previous statement | Update existing facts |
| `greeting` | Start conversation | Minimal processing |
| `goodbye` | End conversation | Session cleanup |
| `affirmation` | Confirm/agree | Light processing |
| `negation` | Deny/disagree | Conflict resolution |

## Model Training

### Step 1: Create Rasa Project Structure
```bash
mkdir diet_intent_model
cd diet_intent_model

# Create required directories
mkdir data models

# Create configuration file
cat > config.yml << EOF
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
    model_confidence: softmax
    entity_recognition: false  # Intent classification only
  - name: FallbackClassifier
    threshold: 0.7
    ambiguity_threshold: 0.1

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
EOF

# Create domain file
cat > domain.yml << EOF
version: "3.1"
intents:
  - remember_fact
  - recall_query
  - general_chat
  - forget_request
  - clarification
  - correction
  - greeting
  - goodbye
  - affirmation
  - negation

responses:
  utter_default:
  - text: "I understand you said: {user_message}"
EOF
```

### Step 2: Train the Model
```bash
# Copy your training data
cp /path/to/your/nlu.yml data/

# Train the model
rasa train nlu --config config.yml --nlu data/nlu.yml --out models/

# Test the model
rasa shell nlu
```

### Step 3: Export for Integration
```bash
# The trained model will be in models/ directory
# Copy to your LocalCat project
cp -r models/nlu-* /path/to/localcat/models/diet_intent_classifier/
```

## Integration into Voice Agent

### Step 1: Create Intent Classifier Wrapper

Create `server/intent_classifier.py`:

```python
"""
DIET Intent Classifier Wrapper for LocalCat Voice Agent
Provides lightweight intent classification for smart memory processing
"""

import os
import time
from typing import Optional, Dict, Any, List
from loguru import logger
import asyncio
from pathlib import Path

try:
    from rasa.nlu.model import Interpreter
    RASA_AVAILABLE = True
except ImportError:
    RASA_AVAILABLE = False
    logger.warning("Rasa not available - intent classification disabled")

class IntentClassifier:
    """Lightweight DIET-based intent classifier"""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7):
        self.model_path = model_path or os.getenv("DIET_MODEL_PATH", "../models/diet_intent_classifier")
        self.confidence_threshold = confidence_threshold
        self.interpreter = None
        self.enabled = RASA_AVAILABLE and os.getenv("DIET_INTENT_ENABLED", "true").lower() == "true"

        if self.enabled:
            self._load_model()

    def _load_model(self):
        """Load the trained DIET model"""
        try:
            if not Path(self.model_path).exists():
                logger.warning(f"DIET model not found at {self.model_path}")
                self.enabled = False
                return

            self.interpreter = Interpreter.load(self.model_path)
            logger.info(f"DIET intent classifier loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load DIET model: {e}")
            self.enabled = False

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify intent for given text

        Returns:
            {
                'intent': str,
                'confidence': float,
                'fallback': bool
            }
        """
        if not self.enabled or not self.interpreter:
            return {
                'intent': 'general_chat',
                'confidence': 0.0,
                'fallback': True
            }

        try:
            start_time = time.perf_counter()

            # Run intent classification
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.interpreter.parse, text
            )

            intent_name = result.get('intent', {}).get('name', 'general_chat')
            confidence = result.get('intent', {}).get('confidence', 0.0)

            # Check confidence threshold
            fallback = confidence < self.confidence_threshold
            if fallback:
                intent_name = 'general_chat'  # Safe fallback

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Intent classification: {intent_name} ({confidence:.2f}) in {elapsed_ms:.1f}ms")

            return {
                'intent': intent_name,
                'confidence': confidence,
                'fallback': fallback,
                'processing_time_ms': elapsed_ms
            }

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                'intent': 'general_chat',
                'confidence': 0.0,
                'fallback': True
            }

    def get_intent_categories(self) -> Dict[str, List[str]]:
        """Get intent categories for routing decisions"""
        return {
            'memory_operations': ['remember_fact', 'recall_query', 'forget_request'],
            'conversational': ['general_chat', 'greeting', 'goodbye', 'affirmation', 'negation'],
            'clarification': ['clarification', 'correction'],
            'skip_memory': ['general_chat', 'greeting', 'goodbye']
        }

# Singleton instance
_intent_classifier = None

def get_intent_classifier() -> IntentClassifier:
    """Get or create intent classifier singleton"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
```

### Step 2: Enhance HotPath Processor

Update `server/hotpath_processor.py` to include intent-aware processing:

```python
# Add to imports
from intent_classifier import get_intent_classifier

# Add to HotPathMemoryProcessor.__init__
self.intent_classifier = get_intent_classifier()
self._intent_aware_processing = os.getenv("DIET_INTENT_ENABLED", "true").lower() == "true"

# Modify _process_transcription method
async def _process_transcription(self, frame: TranscriptionFrame, direction: FrameDirection):
    """Process final user transcription with intent awareness"""
    if not getattr(self, "_enabled", True):
        return

    text = frame.text or ""
    if not text.strip():
        return

    start = time.perf_counter()

    # Intent classification for smart processing
    intent_result = None
    if self._intent_aware_processing:
        try:
            intent_result = await self.intent_classifier.classify_intent(text)
            logger.info(f"[HotMem] Intent classified: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")
        except Exception as e:
            logger.warning(f"[HotMem] Intent classification failed: {e}")

    # Smart processing based on intent
    if intent_result:
        intent_name = intent_result['intent']

        # Skip memory processing for casual conversation
        if intent_name in ['general_chat', 'greeting', 'goodbye', 'affirmation', 'negation']:
            logger.info(f"[HotMem] Skipping memory processing for intent: {intent_name}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_turn_metrics(elapsed_ms)
            return

        # Enhanced processing for memory operations
        if intent_name in ['remember_fact']:
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='storage')
        elif intent_name in ['recall_query']:
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='retrieval')
        elif intent_name in ['forget_request']:
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id, focus='deletion')
        else:
            # Standard processing for other intents
            bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id)
    else:
        # Fallback to standard processing
        bullets, triples = self.hot.process_turn(text, self._session_id, self._turn_id)

    # Rest of the method remains the same...
    # [Continue with existing logic]
```

### Step 3: Update Pipeline Configuration

Modify `server/bot.py` to include intent classification:

```python
# Add import
from intent_classifier import get_intent_classifier

# In run_bot function, after memory initialization:
# Initialize intent classifier
intent_classifier = get_intent_classifier()

# Optional: Add intent classification frame for downstream processors
class IntentFrame(Frame):
    def __init__(self, intent: str, confidence: float):
        super().__init__()
        self.intent = intent
        self.confidence = confidence
```

## Configuration & Deployment

### Environment Variables

Add these configurations to your `.env` file:

```bash
# DIET Intent Classification
DIET_INTENT_ENABLED=true
DIET_MODEL_PATH=../models/diet_intent_classifier
DIET_CONFIDENCE_THRESHOLD=0.7
DIET_FALLBACK_TO_MEMORY=true

# Intent-specific memory processing
INTENT_SKIP_MEMORY_FOR=general_chat,greeting,goodbye,affirmation,negation
INTENT_ENHANCED_RETRIEVAL_FOR=recall_query
INTENT_ENHANCED_STORAGE_FOR=remember_fact

# Performance monitoring
INTENT_LOG_CLASSIFICATION_TIME=true
INTENT_LOG_ROUTING_DECISIONS=true
```

### Model Deployment

```bash
# Create models directory
mkdir -p /path/to/localcat/models/diet_intent_classifier

# Copy trained model
cp -r your_trained_model/* /path/to/localcat/models/diet_intent_classifier/

# Verify model structure
ls -la /path/to/localcat/models/diet_intent_classifier/
# Should contain: model files, config, domain, etc.
```

### Testing Integration

```python
# Test script: test_intent_integration.py
import asyncio
from intent_classifier import get_intent_classifier

async def test_integration():
    classifier = get_intent_classifier()

    test_cases = [
        "Remember that I like coffee",
        "What did I tell you about my job?",
        "How are you doing today?",
        "Forget what I said about that",
        "Hello there"
    ]

    for text in test_cases:
        result = await classifier.classify_intent(text)
        print(f"Text: {text}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Fallback: {result['fallback']}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_integration())
```

## Performance Optimization

### Latency Optimization

1. **Model Quantization** (Optional):
```python
# For production deployment, consider model quantization
import torch
from torch.quantization import quantize_dynamic

# Quantize the model for faster inference
model_quantized = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **Batch Processing** (Future enhancement):
```python
# For handling multiple simultaneous requests
async def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
    results = await asyncio.gather(*[
        self.classify_intent(text) for text in texts
    ])
    return results
```

3. **Caching** (Session-based):
```python
# Cache recent classifications for similar utterances
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_classify(self, text_hash: str, text: str):
    return self.interpreter.parse(text)
```

### Memory Usage Optimization

```python
# Lazy model loading
def _load_model_lazy(self):
    if not hasattr(self, '_interpreter_loaded'):
        self._load_model()
        self._interpreter_loaded = True
```

### Monitoring & Metrics

Add performance tracking:

```python
# In intent_classifier.py
class IntentMetrics:
    def __init__(self):
        self.classifications = 0
        self.avg_latency = 0.0
        self.confidence_distribution = {}
        self.fallback_rate = 0.0

    def record_classification(self, latency_ms: float, confidence: float, fallback: bool):
        self.classifications += 1
        self.avg_latency = (self.avg_latency * (self.classifications - 1) + latency_ms) / self.classifications

        if fallback:
            self.fallback_rate = (self.fallback_rate * (self.classifications - 1) + 1) / self.classifications
```

## Troubleshooting

### Common Issues

1. **Model Not Loading**
```bash
# Check model path
ls -la /path/to/models/diet_intent_classifier/
# Verify Rasa installation
python -c "import rasa; print(rasa.__version__)"
```

2. **High Latency**
```python
# Profile intent classification
import cProfile
cProfile.run('classifier.classify_intent("test")')
```

3. **Low Confidence Scores**
```yaml
# Retrain with more diverse data or adjust threshold
DIET_CONFIDENCE_THRESHOLD=0.5  # Lower threshold
```

4. **Memory Issues**
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Debug Logging

Enable detailed logging:

```python
# In your .env
INTENT_LOG_LEVEL=DEBUG
INTENT_TRACE_CLASSIFICATIONS=true

# In intent_classifier.py
if os.getenv("INTENT_TRACE_CLASSIFICATIONS", "false").lower() == "true":
    logger.debug(f"Classification result: {result}")
    logger.debug(f"All intents: {result.get('intent_ranking', [])}")
```

### Performance Validation

```bash
# Benchmark intent classification
python -c "
import time
import asyncio
from intent_classifier import get_intent_classifier

async def benchmark():
    classifier = get_intent_classifier()
    test_text = 'Remember that I like coffee'

    start = time.perf_counter()
    for _ in range(100):
        await classifier.classify_intent(test_text)
    elapsed = time.perf_counter() - start

    print(f'Average latency: {elapsed * 10:.1f}ms per classification')

asyncio.run(benchmark())
"
```

## Next Steps

1. **Generate Training Data**: Use the provided Colab notebook
2. **Train Initial Model**: Start with basic intent set
3. **Integrate & Test**: Deploy in development environment
4. **Iterate & Improve**: Expand intents based on usage patterns
5. **Production Deploy**: Full integration with monitoring

## Support & Resources

- **DIET Paper**: [arXiv:2004.09936](https://arxiv.org/abs/2004.09936)
- **Rasa Documentation**: [rasa.com/docs](https://rasa.com/docs)
- **LocalCat Integration**: See discovery report in `backlog/drafts/`

---

**Implementation Status**: Ready for deployment
**Last Updated**: September 19, 2025
**Version**: 1.0.0