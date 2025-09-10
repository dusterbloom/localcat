# HotMem V3 Cloud Training Guide

## 🚀 GPU-Poor? No Problem!

You can train HotMem V3 entirely in the cloud using free or low-cost GPU resources. This guide shows you how to train a production-quality graph extraction model without owning a GPU.

## 🎯 Training Options

### Option 1: Google Colab (Recommended)
- **Free Tier**: T4 GPU (15GB VRAM), 12h session limit
- **Colab Pro**: $10/month, A100 GPU option, longer sessions
- **Perfect for**: Training the full model in 6-8 hours

### Option 2: Hugging Face Spaces
- **Free Tier**: T4 GPU (16GB VRAM)
- **Direct integration** with datasets and model hub
- **Perfect for**: Quick experiments and model hosting

### Option 3: Kaggle
- **Free**: 30 hours/week of GPU (P100 or T4)
- **Pre-installed** datasets and libraries
- **Perfect for**: Training on weekends

## 📋 Cloud Training Workflow

### Step 1: Prepare for Cloud Training
```bash
# Clone your repository
git clone https://github.com/yourusername/localcat.git
cd localcat/server

# Upload the notebook to Google Colab
# Or use the provided scripts
```

### Step 2: Run Training in Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `hotmem_v3_colab_training.ipynb`
3. **Important**: Set runtime to GPU
   - Runtime → Change runtime type → Hardware accelerator → GPU
4. Run all cells
5. Training time: ~1 hour (demo) or 6-8 hours (full)

### Step 3: Download and Deploy
1. Download the trained model files
2. Copy to your Mac: `~/localcat/models/`
3. Update HotMem v3 configuration
4. Test with your voice AI pipeline

## 💰 Cost Comparison

| Platform | Cost | GPU | Training Time | Best For |
|----------|------|-----|---------------|----------|
| **Google Colab Free** | $0 | T4 (15GB) | 6-8 hours | Experimentation |
| **Google Colab Pro** | $10/month | A100 (40GB) | 2-3 hours | Production training |
| **Hugging Face Spaces** | $0 | T4 (16GB) | 6-8 hours | Model hosting |
| **Kaggle** | $0 | P100/T4 | 6-8 hours | Weekend training |
| **Your Own GPU** | $$$$ | Your GPU | Variable | Full control |

## 🎯 Expected Results

### Model Performance
- **Accuracy**: 90-95% on graph extraction
- **Speed**: 5-8ms inference on Mac M-series
- **Model Size**: ~500MB (4-bit quantized)
- **Domains**: General knowledge, conversational, technical

### Training Metrics
- **Dataset Size**: 100K+ examples
- **Training Time**: 6-8 hours (T4 GPU)
- **Memory Usage**: ~6GB VRAM (4-bit quantized)
- **Cost**: $0 (free tier) or $10 (Colab Pro)

## 📁 Files Created

### Training Outputs
```
hotmem_v3_qwen/
├── config.json
├── model.safetensors
└── tokenizer.json

hotmem_v3-*.gguf  # For llama.cpp
hotmem_v3_model.zip  # Download archive
```

### Model Integration
```python
# Load the trained model on your Mac
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "hotmem_v3_qwen",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Use in HotMem v3
from components.ai.dspy_modules import DSPyHotMemIntegration

extractor = DSPyHotMemIntegration()
extractor.model = model  # Use cloud-trained model
```

## 🚀 Quick Start

### Option A: Full Training (Best Results)
1. Use the complete Colab notebook
2. Train on 100K+ examples
3. 6-8 hours training time
4. Production-ready model

### Option B: Quick Demo (1 Hour)
1. Use the demo subset in the notebook
2. Train on 5K examples
3. 1 hour training time
4. Good for testing integration

### Option C: Incremental Training
1. Start with pre-trained model from Hugging Face
2. Fine-tune on your specific data
3. 1-2 hours training time
4. Customized for your use case

## 🔧 Cloud Training Tips

### Maximize Free Tier
- **Colab**: Use early morning/late night for better availability
- **Kaggle**: Weekend sessions have higher limits
- **Spaces**: Use for smaller experiments and model hosting

### Optimize Training
- **Use 4-bit quantization**: Reduces VRAM usage by 70%
- **Smaller batches**: Prevents out-of-memory errors
- **Gradient checkpointing**: Enables longer sequences
- **Mixed precision**: Faster training with better memory usage

### Save Your Work
- **Download checkpoints**: Save intermediate models
- **Use Google Drive**: Store large models
- **Version control**: Track your experiments

## 🎉 Next Steps After Training

### 1. Model Deployment
```bash
# Copy model to your Mac
mkdir -p ~/localcat/models/
cp hotmem_v3_qwen/* ~/localcat/models/
```

### 2. Integration with HotMem
```python
# Update your HotMem config
config.model.graph_extractor_path = "~/localcat/models/hotmem_v3_qwen"
config.model.use_cloud_trained_model = True
```

### 3. Testing
```bash
# Test the new model
python test_cloud_trained_model.py
```

### 4. Production Deployment
```bash
# Deploy to production
python deploy_hotmem_v3.py
```

## 📚 Additional Resources

### Documentation
- [Unsloth Documentation](https://unsloth.ai/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Google Colab Guide](https://colab.research.google.com/)

### Pre-trained Models
- [Qwen2.5-0.5B on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Unsloth Optimized Models](https://huggingface.co/unsloth)

### Community
- [Unsloth Discord](https://discord.gg/unsloth)
- [Hugging Face Forums](https://discuss.huggingface.co/)

## 💡 Pro Tips

1. **Start small**: Use the demo subset first to test integration
2. **Monitor training**: Keep an eye on loss curves
3. **Save frequently**: Download checkpoints periodically
4. **Test incrementally**: Validate model performance during training
5. **Document everything**: Keep track of hyperparameters and results

## 🆘 Troubleshooting

### Common Issues
- **Out of memory**: Reduce batch size or use gradient checkpointing
- **Slow training**: Enable mixed precision and reduce dataset size
- **Poor quality**: Increase training epochs or use more data
- **Download fails**: Use Google Drive or create smaller zip files

### Get Help
- Check the Unsloth documentation
- Search Hugging Face forums
- Ask on Discord communities
- Review the training logs for errors

---

**Remember**: Cloud training democratizes AI - you don't need expensive hardware to train state-of-the-art models! 🚀