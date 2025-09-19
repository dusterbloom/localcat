#!/usr/bin/env python3
"""
Global Model Manager for caching and pre-warming ML models.
Ensures models are loaded once and shared across the application.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from loguru import logger
import os

# Singleton model cache
_model_cache: Dict[str, Any] = {}
_loading_locks: Dict[str, asyncio.Lock] = {}


class ModelManager:
    """Singleton manager for ML models with caching and pre-warming."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._model_cache = _model_cache
            self._loading_locks = _loading_locks
            self._load_times = {}
            self._initialized = True
            logger.info("ModelManager initialized")

    async def get_model(self, model_id: str, loader_func=None, **loader_kwargs):
        """
        Get a cached model or load it if not cached.

        Args:
            model_id: Unique identifier for the model
            loader_func: Function to load the model if not cached
            **loader_kwargs: Arguments to pass to loader_func
        """
        # Check cache first
        if model_id in self._model_cache:
            logger.debug(f"Model {model_id} found in cache")
            return self._model_cache[model_id]

        # Create lock for this model if needed
        if model_id not in self._loading_locks:
            self._loading_locks[model_id] = asyncio.Lock()

        # Load model with lock to prevent duplicate loading
        async with self._loading_locks[model_id]:
            # Double-check cache after acquiring lock
            if model_id in self._model_cache:
                return self._model_cache[model_id]

            # Load the model
            if loader_func is None:
                raise ValueError(f"No loader function provided for {model_id}")

            logger.info(f"Loading model {model_id}...")
            start_time = time.time()

            # Run loader in executor if it's not async
            if asyncio.iscoroutinefunction(loader_func):
                model = await loader_func(**loader_kwargs)
            else:
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(None, loader_func, **loader_kwargs)

            load_time = (time.time() - start_time) * 1000
            self._load_times[model_id] = load_time
            logger.info(f"Model {model_id} loaded in {load_time:.1f}ms")

            # Cache the model
            self._model_cache[model_id] = model

            return model

    def get_model_sync(self, model_id: str, loader_func=None, **loader_kwargs):
        """Synchronous version of get_model for non-async contexts."""
        # Check cache first
        if model_id in self._model_cache:
            logger.debug(f"Model {model_id} found in cache")
            return self._model_cache[model_id]

        # Load the model
        if loader_func is None:
            raise ValueError(f"No loader function provided for {model_id}")

        logger.info(f"Loading model {model_id} (sync)...")
        start_time = time.time()

        model = loader_func(**loader_kwargs)

        load_time = (time.time() - start_time) * 1000
        self._load_times[model_id] = load_time
        logger.info(f"Model {model_id} loaded in {load_time:.1f}ms")

        # Cache the model
        self._model_cache[model_id] = model

        return model

    async def prewarm_models(self):
        """Pre-warm commonly used models at startup."""
        logger.info("Pre-warming models...")

        tasks = []

        # Pre-warm Kokoro TTS
        if os.getenv("TTS_ULTRA_LOW_LATENCY", "true").lower() in ("1", "true"):
            tasks.append(self._prewarm_kokoro())

        # Pre-warm Kyutai STT if enabled
        if os.getenv("USE_STREAMING_STT", "true").lower() in ("1", "true"):
            tasks.append(self._prewarm_kyutai())

        # Pre-warm punctuation model
        tasks.append(self._prewarm_punctuation())

        # Run all pre-warming tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Pre-warming task {i} failed: {result}")

        logger.info(f"Pre-warming complete. Loaded {len(self._model_cache)} models")
        for model_id, load_time in self._load_times.items():
            logger.info(f"  {model_id}: {load_time:.1f}ms")

    async def _prewarm_kokoro(self):
        """Pre-warm Kokoro TTS model."""
        try:
            from mlx_audio.tts.utils import load_model

            model_name = "mlx-community/Kokoro-82M-bf16"
            model = await self.get_model(
                f"kokoro_tts_{model_name}",
                loader_func=load_model,
                model_id=model_name
            )

            # Warm up with test generation
            logger.info("Warming up Kokoro TTS...")
            test_result = list(model.generate(text="test", voice="af_heart", speed=1.0))
            logger.info(f"Kokoro TTS warmed up, generated {len(test_result)} chunks")

        except Exception as e:
            logger.error(f"Failed to pre-warm Kokoro: {e}")

    async def _prewarm_kyutai(self):
        """Pre-warm Kyutai STT model."""
        try:
            import json
            from huggingface_hub import hf_hub_download
            import mlx.core as mx
            from moshi_mlx import models

            hf_repo = "kyutai/stt-1b-en_fr-mlx"

            # Download model components (cached by HF)
            config_path = hf_hub_download(hf_repo, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            def load_kyutai():
                mimi_weights = hf_hub_download(hf_repo, config_dict["mimi_name"])
                moshi_weights = hf_hub_download(hf_repo, config_dict.get("moshi_name", "model.safetensors"))
                tokenizer_path = hf_hub_download(hf_repo, config_dict["tokenizer_name"])

                lm_config = models.LmConfig.from_config_dict(config_dict)
                model = models.Lm(lm_config)
                model.set_dtype(mx.bfloat16)

                # Load weights
                weights_dict = mx.load(moshi_weights)
                model.text_lm.update({"model.": v for k, v in weights_dict.items()})

                # Load tokenizers
                from sentencepiece import SentencePieceProcessor
                text_tokenizer = SentencePieceProcessor(tokenizer_path)

                from moshi_mlx.models import MimiModel, MimiConfig
                audio_config = MimiConfig.from_config_dict(config_dict)
                audio_tokenizer = MimiModel(audio_config)
                audio_tokenizer.load_weights(mimi_weights)

                return {
                    "model": model,
                    "text_tokenizer": text_tokenizer,
                    "audio_tokenizer": audio_tokenizer,
                    "config": lm_config
                }

            kyutai_models = await self.get_model(
                f"kyutai_stt_{hf_repo}",
                loader_func=load_kyutai
            )

            logger.info("Kyutai STT models cached and ready")

        except Exception as e:
            logger.error(f"Failed to pre-warm Kyutai: {e}")

    async def _prewarm_punctuation(self):
        """Pre-warm punctuation restoration model."""
        try:
            from transformers import pipeline

            def load_punctuation():
                return pipeline(
                    "token-classification",
                    model="1-800-BAD-CODE/punct_restore_en",
                    device="mps" if os.path.exists("/System/Library/Frameworks/Metal.framework") else "cpu",
                    aggregation_strategy="simple",
                    grouped_entities=False
                )

            model = await self.get_model(
                "punctuation_restorer",
                loader_func=load_punctuation
            )

            # Warm up
            test_result = model("hello world how are you")
            logger.info(f"Punctuation model warmed up")

        except Exception as e:
            logger.error(f"Failed to pre-warm punctuation: {e}")

    def clear_cache(self, model_id: Optional[str] = None):
        """Clear cached models."""
        if model_id:
            if model_id in self._model_cache:
                del self._model_cache[model_id]
                logger.info(f"Cleared cache for model {model_id}")
        else:
            self._model_cache.clear()
            logger.info("Cleared all model cache")

    def get_stats(self):
        """Get model manager statistics."""
        return {
            "cached_models": len(self._model_cache),
            "models": list(self._model_cache.keys()),
            "load_times": self._load_times
        }


# Global singleton instance
model_manager = ModelManager()


async def initialize_models():
    """Initialize and pre-warm all models at application startup."""
    await model_manager.prewarm_models()