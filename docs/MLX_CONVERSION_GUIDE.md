# MLX Model Conversion Guide for LM Studio

## HotMem Relation Classifier MLX Conversion Process

This document details the complete process used to convert the HotMem Relation Classifier model to MLX format for use in LM Studio on Apple Silicon.

### Prerequisites

1. **Python virtual environment with MLX**
   ```bash
   python3 -m venv mlx_venv
   source mlx_venv/bin/activate
   pip install mlx mlx-lm transformers safetensors
   ```

2. **Source model** - A Hugging Face format model (safetensors + config.json)

### Conversion Process

#### Step 1: Initial MLX Conversion

We created a conversion script to transform the PyTorch model to MLX format:

```python
#!/usr/bin/env python3
import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

def load_pytorch_weights(model_path):
    weights = {}
    with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            weights[key] = tensor.float().numpy()
    return weights

def convert_to_mlx(weights):
    mlx_weights = {}
    for key, value in weights.items():
        mlx_weights[key] = mx.array(value)
    return mlx_weights
```

This created a full-precision model (1.8GB) that worked with MLX but was too large.

#### Step 2: 4-bit Quantization

The key to getting a model that works well in LM Studio is **4-bit quantization**. This reduces model size by ~85% while maintaining good performance.

```bash
source mlx_venv/bin/activate
mlx_lm.convert \
  --hf-path /path/to/source/model \
  --mlx-path /path/to/output/model-MLX-4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

This produces a model with:
- **265MB model.safetensors** (vs 1.8GB unquantized)
- Quantization config in the model's config.json

#### Step 3: LM Studio Configuration

For LM Studio to recognize the model, ensure:

1. **Proper naming convention**: Use format `model-name-MLX-4bit`
2. **Config.json must include**:
   ```json
   {
     "architectures": ["Qwen2ForCausalLM"],
     "model_type": "qwen2",
     "quantization": {
       "group_size": 64,
       "bits": 4,
       "mode": "affine"
     },
     "quantization_config": {
       "group_size": 64,
       "bits": 4,
       "mode": "affine"
     }
   }
   ```

3. **Chat template in tokenizer_config.json**:
   ```json
   "chat_template": "{% set system_message = 'You are a helpful assistant.' %}..."
   ```

4. **Remove index files**: Single-file models shouldn't have `model.safetensors.index.json`

5. **Location**: Place in `~/.cache/lm-studio/models/lmstudio-community/`

### File Structure

Final model directory should contain:
```
hotmem-relation-classifier-MLX-4bit/
├── model.safetensors       # 265MB quantized weights
├── config.json             # Model config with quantization
├── tokenizer.json          # Main tokenizer file
├── tokenizer_config.json   # With chat_template
├── special_tokens_map.json
├── vocab.json
├── merges.txt
├── added_tokens.json
└── generation_config.json
```

### Testing the Model

Verify the model loads correctly:
```python
from mlx_lm import load, generate

model, tokenizer = load("/path/to/model")
response = generate(model, tokenizer, prompt="Test prompt", max_tokens=50)
print(response)
```

### Common Issues and Solutions

1. **Model not appearing in LM Studio**
   - Check naming convention (must end with -MLX or -MLX-4bit)
   - Verify it's in the correct directory
   - Ensure config.json has proper architecture and quantization fields

2. **Model too large**
   - Always use 4-bit quantization for models over 1GB
   - Group size of 64 provides good balance

3. **Generation errors**
   - Ensure tokenizer_config.json has chat_template
   - Check special tokens are properly configured

### Summary

The key steps for successful MLX conversion for LM Studio:
1. Convert PyTorch model to MLX format
2. Apply 4-bit quantization (crucial for size and compatibility)
3. Configure metadata properly (config.json, tokenizer files)
4. Place in LM Studio's model directory
5. Test loading and generation

This process converted a 1.8GB model to a 265MB quantized version that runs efficiently on Apple Silicon via LM Studio.