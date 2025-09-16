# Changelog

## [v7.1.0] - 2025-09-16

### 🐛 Critical Fixes

**Intent Classification System Overhaul**
- **Fixed:** V2 Enhanced Rule Classifier returning `None` for facts and questions
- **Root Cause:** Indentation errors preventing proper code execution flow
- **Impact:** Eliminated memory retrieval on every turn regardless of intent

**Changes:**
- `components/memory/enhanced_rule_classifier_v2.py:218-287`: Fixed indentation structure
  - Request detection logic (lines 222-229)
  - Question detection logic (lines 233-252)
  - Fact detection logic (lines 256-287)
- `components/memory/memory_intent.py:42-43`: Extended `IntentAnalysis` schema
  - Added `requires_memory: bool = False`
  - Added `requires_retrieval: bool = False`
- Updated adapter initialization in both DistilBERT and V2 adapters

**Test Results:**
- ✅ 0 `None` returns (was 5/7 before fix)
- ✅ 100% working classifications with proper attributes
- ✅ Memory gating now functional based on intent types
- ✅ <1ms classification latency maintained

### 🚀 Performance Improvements

**Memory System Efficiency**
- **Questions:** No longer trigger unnecessary fact storage
- **Facts:** No longer trigger unnecessary context retrieval
- **Reactions:** Properly bypass both storage and retrieval
- **Corrections:** Properly trigger both storage and retrieval

### 🧪 Testing Infrastructure

**Added:**
- `test_intent_v2_fixed.py`: V2 classifier validation test
- `test_intent_final_validation.py`: Complete system validation
- Comprehensive test coverage for all intent types

**Results:**
```
Test Coverage: 7/7 intent types
Success Rate: 100% (was 28% before fix)
Performance: <1ms per classification
Memory Gating: Functional
```

---

## [v7.0.0] - 2025-09-15

### 🎯 Enhanced Level3 Default with Transformer Alias
- Set `ENHANCED_LEVEL3_SPACY_MODEL=en_core_web_rtf` (alias to transformer)
- Enabled lite coreference resolution with fusion
- Confidence-threaded persistence system
- Admin tools and regression scripts

### 🔧 Technical Infrastructure
- Centralized spaCy cache with prewarming
- Dual Graph + Unified Optimizer direction (DSPy/GEPA/TreeSearch)
- Edge metadata schema definition
- TTL/archival job automation

---

## [v6.x] - Previous Releases
See git history for detailed changes in previous versions.