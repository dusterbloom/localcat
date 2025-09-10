# 🚀 HotMem v3 Deployment Guide

## Complete Guide from Trained Model to Production Deployment

This guide will walk you through deploying your trained HotMem v3 model from Google Colab to your localcat voice agent on Mac.

## 📋 Prerequisites Checklist

- ✅ Colab training completed
- ✅ Trained model files downloaded from Colab
- ✅ Mac with Apple Silicon (M1/M2/M3)
- ✅ Python 3.12+ with virtual environment
- ✅ Existing localcat voice agent setup

---

## 🎯 Phase 1: Download & Prepare Model (Colab → Mac)

### Step 1.1: Download from Google Colab

When your Colab training completes, you'll have these files:
```
hotmem_v3_qwen/           # 16-bit merged model
hotmem_v3-*.gguf         # GGUF quantized versions
hotmem_v3_model.zip      # Download archive
```

**Download Actions:**
1. **Download the ZIP file**: Click the download link in Colab
2. **Alternative download**: Use Google Drive sync if available
3. **Verify files**: Ensure you have both `.safetensors` and config files

### Step 1.2: Prepare Model Directory

```bash
# Create model directory on your Mac
mkdir -p ~/localcat/models/hotmem_v3

# Extract downloaded model
cd ~/Downloads
unzip hotmem_v3_model.zip -d ~/localcat/models/hotmem_v3

# Verify structure
ls -la ~/localcat/models/hotmem_v3/
# Should contain: config.json, model.safetensors, tokenizer.json, etc.
```

### Step 1.3: Optimize Model for Mac

Run the model optimizer to create Mac-optimized versions:

```bash
cd ~/localcat/server

# Run model optimization
python hotmem_v3_model_optimizer.py --model_path ~/localcat/models/hotmem_v3
```

This will create:
- `hotmem_v3_int4/` - 4-bit quantized (memory efficient)
- `hotmem_v3_int8/` - 8-bit quantized (balanced)
- `hotmem_v3_coreml.mlpackage/` - Neural Engine optimized
- `hotmem_v3_gguf/` - llama.cpp compatible

---

## 🎯 Phase 2: Install Dependencies

### Step 2.1: Update Requirements

Add these dependencies to your `requirements.txt`:

```txt
# HotMem v3 Dependencies
torch>=2.0.0
transformers>=4.35.0
numpy>=1.21.0
networkx>=3.0
psutil>=5.9.0

# Optional - CoreML for Neural Engine acceleration
coremltools>=7.0

# Optional - MLX for Apple Silicon optimization
mlx>=0.16.0
mlx-lm>=0.4.0

# Optional - llama.cpp integration
llama-cpp-python>=0.2.0
```

### Step 2.2: Install Dependencies

```bash
cd ~/localcat/server

# Activate virtual environment
source venv/bin/activate

# Install updated requirements
pip install -r requirements.txt

# Install HotMem v3 specific packages
pip install coremltools mlx mlx-lm
```

---

## 🎯 Phase 3: Integrate with LocalCat

### Step 3.1: Update Bot Configuration

Modify your `bot.py` to include HotMem v3 integration:

```python
# Add to imports at top
from hotmem_v3_production_integration import LocalCatHotMemIntegration

# In your LocalCatBot class __init__:
def __init__(self):
    # ... existing initialization ...
    
    # Initialize HotMem v3
    self.hotmem_integration = LocalCatHotMemIntegration(
        localcat_bot=self,
        hotmem_integration=None  # Will be created below
    )
    
    # Initialize HotMem with optimized model
    model_path = "~/localcat/models/hotmem_v3/hotmem_v3_int4"
    try:
        self.hotmem_integration.hotmem = HotMemIntegration(
            model_path=model_path,
            enable_real_time=True,
            confidence_threshold=0.7
        )
        print("✅ HotMem v3 integration initialized")
    except Exception as e:
        print(f"⚠️ HotMem v3 initialization failed: {e}")
```

### Step 3.2: Integrate with Voice Pipeline

Add HotMem processing to your voice transcription pipeline:

```python
# In your transcription handling code:
async def handle_transcription(self, text: str, is_final: bool = False):
    """Handle voice transcription with HotMem integration"""
    
    # Process through HotMem v3
    if hasattr(self, 'hotmem_integration'):
        self.hotmem_integration.process_user_input(text, is_final)
    
    # Get enhanced context for LLM
    if hasattr(self, 'hotmem_integration'):
        enhanced_context = self.hotmem_integration.get_enhanced_context(text)
        # Use enhanced_context for LLM processing
    else:
        enhanced_context = text
    
    # Continue with existing voice processing...
```

### Step 3.3: Add Memory UI Integration

Create a simple memory visualization for your UI:

```python
# Add to your UI components
def get_memory_summary(self):
    """Get HotMem memory summary for UI"""
    if hasattr(self, 'hotmem_integration'):
        graph = self.hotmem_integration.get_knowledge_graph()
        return {
            'entities': len(graph['entities']),
            'relations': len(graph['relations']),
            'recent_extractions': self.hotmem_integration.get_memory_updates(5)
        }
    return {'entities': 0, 'relations': 0, 'recent_extractions': []}
```

---

## 🎯 Phase 4: Testing & Validation

### Step 4.1: Run Basic Tests

```bash
cd ~/localcat/server

# Test individual components
python hotmem_v3_streaming_extraction.py
python hotmem_v3_dual_graph_architecture.py
python hotmem_v3_active_learning.py
```

### Step 4.2: Run Full Validation Suite

```bash
# Run comprehensive validation
python hotmem_v3_end_to_end_validation.py
```

This will generate a detailed report showing:
- Integration test results
- Performance benchmarks
- Memory usage statistics
- Accuracy validation

### Step 4.3: Test Voice Integration

```bash
# Start the voice agent with HotMem integration
python bot.py

# Test with voice commands like:
# "Hi, I'm Sarah and I work at Google"
# "Remember that I'm a software engineer"
# "What do you know about me?"
```

---

## 🎯 Phase 5: Production Deployment

### Step 5.1: Choose Model Variant

Select the best model variant for your needs:

| Variant | Size | Speed | Accuracy | Best For |
|---------|------|-------|----------|----------|
| INT4 | ~300MB | ⚡ Fastest | Good | Memory-constrained systems |
| INT8 | ~600MB | ⚡ Fast | Better | Balanced performance |
| CoreML | ~400MB | 🚀 Fastest | Best | Apple Silicon with Neural Engine |
| GGUF | ~350MB | Fast | Good | C++ integration |

### Step 5.2: Configure for Production

Create `hotmem_v3_config.json`:

```json
{
  "model_path": "~/localcat/models/hotmem_v3/hotmem_v3_int4",
  "enable_real_time": true,
  "confidence_threshold": 0.7,
  "max_working_memory_entities": 100,
  "max_long_term_memory_entities": 10000,
  "active_learning_enabled": true,
  "promotion_interval": 1800,
  "validation_enabled": true
}
```

### Step 5.3: Production Startup Script

Create `start_hotmem_v3.sh`:

```bash
#!/bin/bash

# HotMem v3 Production Startup Script

echo "🚀 Starting HotMem v3 Voice Agent..."

# Activate environment
source ~/localcat/server/venv/bin/activate

# Set environment variables
export HOTMEM_MODEL_PATH="$HOME/localcat/models/hotmem_v3/hotmem_v3_int4"
export HOTMEM_CONFIG="$HOME/localcat/server/hotmem_v3_config.json"

# Start the voice agent
cd ~/localcat/server
python bot.py

echo "✅ HotMem v3 Voice Agent Started"
```

Make it executable:
```bash
chmod +x start_hotmem_v3.sh
```

---

## 🎯 Phase 6: Monitoring & Maintenance

### Step 6.1: Health Monitoring

Add health checks to your application:

```python
def get_hotmem_health(self):
    """Get HotMem v3 system health"""
    if not hasattr(self, 'hotmem_integration'):
        return {"status": "not_initialized"}
    
    try:
        stats = self.hotmem_integration.get_performance_stats()
        return {
            "status": "healthy",
            "extractions_processed": stats.get('extraction_count', 0),
            "memory_usage": {
                "working_memory": stats.get('knowledge_graph_size', {}).get('entities', 0),
                "long_term_memory": stats.get('memory_updates_count', 0)
            },
            "last_update": time.time()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### Step 6.2: Performance Monitoring

Monitor key metrics:
- **Extraction latency**: Should be <100ms
- **Memory usage**: Should be <1GB
- **Accuracy**: Should be >90%
- **Learning rate**: Should improve over time

### Step 6.3: Model Updates

When you retrain your model, follow this process:

```bash
# 1. Stop the service
pkill -f "python bot.py"

# 2. Backup current model
cp -r ~/localcat/models/hotmem_v3 ~/localcat/models/hotmem_v3_backup

# 3. Copy new model
cp -r /path/to/new/model ~/localcat/models/hotmem_v3

# 4. Re-optimize for Mac
python hotmem_v3_model_optimizer.py --model_path ~/localcat/models/hotmem_v3

# 5. Restart service
./start_hotmem_v3.sh
```

---

## 🎯 Phase 7: Advanced Features

### Step 7.1: Enable Active Learning

The active learning system will automatically:
- Detect patterns in extraction errors
- Generate training examples from user interactions
- Improve model accuracy over time

```python
# Enable active learning callbacks
def setup_active_learning(self):
    """Setup active learning callbacks"""
    
    def on_extraction_complete(event):
        """Handle extraction events"""
        # Log for learning
        self.hotmem_integration.hotmem.add_extraction_result(
            text=event.data.get('text', ''),
            extraction=event.data,
            confidence=event.data.get('confidence', 0.5),
            is_correct=True  # Assume correct unless corrected
        )
    
    def on_user_correction(correction):
        """Handle user corrections"""
        self.hotmem_integration.hotmem.add_user_correction(
            original_text=correction['original'],
            original_extraction=correction['original'],
            corrected_extraction=correction['corrected'],
            confidence=correction['confidence'],
            error_type='user_correction'
        )
    
    # Register callbacks
    self.hotmem_integration.add_event_callback('extraction_complete', on_extraction_complete)
```

### Step 7.2: Dual Graph Monitoring

Monitor the dual graph architecture:

```python
def get_dual_graph_stats(self):
    """Get dual graph statistics"""
    if hasattr(self, 'hotmem_integration'):
        stats = self.hotmem_integration.hotmem.get_system_stats()
        return {
            "working_memory": {
                "entities": stats['working_memory']['entity_count'],
                "relationships": stats['working_memory']['relationship_count']
            },
            "long_term_memory": {
                "entities": stats['long_term_memory']['entity_count'],
                "relationships": stats['long_term_memory']['relationship_count']
            },
            "promotions": stats.get('total_promotions', 0),
            "active_sessions": stats.get('active_sessions', 0)
        }
    return {}
```

### Step 7.3: Export/Import Memory

Save and load knowledge graphs:

```python
# Export memory
def export_hotmem_memory(self, filepath):
    """Export HotMem memory to file"""
    if hasattr(self, 'hotmem_integration'):
        self.hotmem_integration.hotmem.export_knowledge_graph(filepath)

# Import memory
def import_hotmem_memory(self, filepath):
    """Import HotMem memory from file"""
    if hasattr(self, 'hotmem_integration'):
        self.hotmem_integration.hotmem.import_knowledge_graph(filepath)
```

---

## 🎯 Troubleshooting

### Common Issues

**Issue: Model not loading**
```bash
# Check model path
ls -la ~/localcat/models/hotmem_v3/

# Check file permissions
chmod 644 ~/localcat/models/hotmem_v3/*

# Check disk space
df -h ~/localcat/models/
```

**Issue: High memory usage**
```bash
# Use INT4 quantized model
export HOTMEM_MODEL_PATH="$HOME/localcat/models/hotmem_v3/hotmem_v3_int4"

# Reduce memory limits
echo '{"max_working_memory_entities": 50, "max_long_term_memory_entities": 5000}' > hotmem_v3_config.json
```

**Issue: Slow extraction**
```bash
# Enable CoreML acceleration
pip install coremltools

# Use CoreML model
export HOTMEM_MODEL_PATH="$HOME/localcat/models/hotmem_v3/hotmem_v3_coreml.mlpackage"
```

**Issue: Poor accuracy**
```bash
# Run validation to identify issues
python hotmem_v3_end_to_end_validation.py

# Check training data quality
# Consider retraining with more diverse datasets
```

---

## 🎯 Performance Optimization

### Best Practices

1. **Use appropriate model variant** for your hardware
2. **Enable CoreML acceleration** on Apple Silicon
3. **Monitor memory usage** and adjust limits
4. **Regular validation** to maintain accuracy
5. **Backup knowledge graphs** regularly

### Expected Performance

| Metric | Target | Excellent |
|--------|---------|-----------|
| Extraction latency | <100ms | <50ms |
| Memory usage | <1GB | <500MB |
| Accuracy | >90% | >95% |
| Learning rate | 1%/day | 5%/day |

---

## 🎯 Success Criteria

Your HotMem v3 deployment is successful when:

- ✅ **Real-time extraction** works during voice conversations
- ✅ **Knowledge graphs** build as you speak
- ✅ **Memory persists** between sessions
- ✅ **Accuracy improves** over time
- ✅ **Performance** meets targets above
- ✅ **Integration** is seamless with existing voice agent

---

## 🎯 Next Steps

After deployment:

1. **Monitor performance** for first week
2. **Collect user feedback** on accuracy
3. **Retrain model** with accumulated data
4. **Experiment with features** like active learning
5. **Share results** with community

---

**🎉 Congratulations! You now have a revolutionary self-improving AI voice agent with HotMem v3!**

For support and questions, refer to the validation logs and health monitoring system.