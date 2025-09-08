#!/usr/bin/env python3
"""
Pre-cache TTS models to avoid first-run timeouts.
Run this script before starting the bot to ensure models are downloaded.
"""

import sys
import os
import subprocess
import asyncio
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("❌ MLX Audio not available. Install with: pip install mlx-audio")
    sys.exit(1)

async def cache_model(model_name, voice="af_heart"):
    """Pre-cache a TTS model."""
    print(f"🔄 Caching {model_name}...")
    
    try:
        # Load the model (this will download it if not cached)
        model = load_model(model_name)
        print(f"✅ Model {model_name} loaded successfully")
        
        # Test generation to ensure everything works
        print("🔄 Testing model generation...")
        audio_chunks = list(model.generate(text="Hello, this is a test.", voice=voice, speed=1.0))
        
        if audio_chunks:
            print(f"✅ Model {model_name} test generation successful")
            return True
        else:
            print(f"❌ Model {model_name} test generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to cache {model_name}: {e}")
        return False

async def main():
    """Pre-cache all TTS models."""
    print("🚀 Starting TTS model pre-caching...")
    
    models_to_cache = [
        "mlx-community/Kokoro-82M-bf16",
        # Add more models here if needed
    ]
    
    success_count = 0
    total_count = len(models_to_cache)
    
    for model_name in models_to_cache:
        if await cache_model(model_name):
            success_count += 1
        print()  # Add spacing
    
    print(f"📊 Results: {success_count}/{total_count} models cached successfully")
    
    if success_count == total_count:
        print("🎉 All models cached! The bot should start quickly now.")
        return True
    else:
        print("⚠️  Some models failed to cache. The bot may still work but could be slower.")
        return False

if __name__ == "__main__":
    # Check if we're in the right directory
    if not (server_dir / "tts").exists():
        print("❌ Please run this script from the server directory")
        sys.exit(1)
    
    # Run the caching
    result = asyncio.run(main())
    sys.exit(0 if result else 1)