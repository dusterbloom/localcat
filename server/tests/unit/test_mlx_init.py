#!/usr/bin/env python3
"""
Simple test to understand MLX Kokoro model initialization.
"""

import sys
from pathlib import Path

# Add server directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_mlx_init():
    """Test different ways to initialize MLX Kokoro."""

    print("🧪 Testing MLX Kokoro initialization methods...\n")

    # Method 1: Try using Model.generate directly (it might auto-initialize)
    print("📊 Method 1: Direct Model.generate() call")
    try:
        from mlx_audio.tts.models.kokoro import Model
        print(f"  Model REPO_ID: {Model.REPO_ID}")

        # Try creating with empty config first
        print("  Testing auto-initialization...")
        model = Model(None)  # See if it auto-initializes
        result = model.generate("Hello world", voice="af_bella", speed=1.0)
        print(f"  ✅ Success! Generated audio")
        return model

    except Exception as e:
        print(f"  ❌ Failed: {e}")

    print()

    # Method 2: Try loading model config from repo
    print("📊 Method 2: Load config from HuggingFace repo")
    try:
        from mlx_audio.tts.models.kokoro import Model, ModelConfig
        from huggingface_hub import hf_hub_download
        import json

        # Download config from HF
        config_path = hf_hub_download(Model.REPO_ID, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)
        model = Model(config, repo_id=Model.REPO_ID)

        result = model.generate("Hello world", voice="af_bella", speed=1.0)
        print(f"  ✅ Success! Generated audio")
        return model

    except Exception as e:
        print(f"  ❌ Failed: {e}")

    print()

    # Method 3: Try KokoroPipeline approach
    print("📊 Method 3: KokoroPipeline approach")
    try:
        from mlx_audio.tts.models.kokoro import Model, ModelConfig, KokoroPipeline
        from huggingface_hub import hf_hub_download
        import json

        # Load config and create model
        config_path = hf_hub_download(Model.REPO_ID, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)
        model = Model(config, repo_id=Model.REPO_ID)

        # Create pipeline with model
        pipeline = KokoroPipeline(lang_code="en", model=model, repo_id=Model.REPO_ID)

        results = list(pipeline("Hello world", voice="af_bella", speed=1.0))
        print(f"  ✅ Success! Pipeline generated {len(results)} results")
        return pipeline

    except Exception as e:
        print(f"  ❌ Failed: {e}")

    print()
    print("❌ All initialization methods failed")
    return None

if __name__ == "__main__":
    test_mlx_init()