import os
from typing import Any, Dict
from loguru import logger

from config import VoiceAgentConfig
from core.llm.openai_context_logger import OpenAIContextLoggerService as OpenAILLMService


class LLMServiceBuilder:
    def __init__(self, config: VoiceAgentConfig, preloaded_models=None):
        self.config = config
        self.preloaded_models = preloaded_models

    def build(self) -> OpenAILLMService:
        llm_config: Dict[str, Any] = self.config.get_component_config("llm")
        use_llm_streaming = (str(llm_config.get("use_streaming", True)).lower() == "true")
        use_direct_mlx = str(
            __import__("os").environ.get("LLM_USE_DIRECT_MLX", "false")
        ).lower() in ("true", "1", "yes")

        if use_direct_mlx:
            from core.llm.direct_mlx_llm_with_tools import DirectMLXLLMServiceWithTools
            logger.info("🚀 Using Direct MLX-LM with Tools (zero HTTP overhead + tool calling)")

            # Extract preloaded model/tokenizer if available
            preloaded_model = None
            preloaded_tokenizer = None
            if self.preloaded_models:
                preloaded_model = getattr(self.preloaded_models, 'mlx_llm_model', None)
                preloaded_tokenizer = getattr(self.preloaded_models, 'mlx_llm_tokenizer', None)
                if preloaded_model and preloaded_tokenizer:
                    logger.debug("🎯 LLMServiceBuilder: Passing preloaded MLX model to service")

            return DirectMLXLLMServiceWithTools(
                model=llm_config["model"],
                max_tokens=llm_config.get("max_tokens", 256),
                temperature=llm_config.get("temperature", 0.7),
                preloaded_model=preloaded_model,
                preloaded_tokenizer=preloaded_tokenizer,
            )

        return OpenAILLMService(
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            max_tokens=llm_config["max_tokens"],
            stream=use_llm_streaming,
            debug=False,
            extra_body={
                "think": False,  # Performance optimization: thinking mode disabled for faster responses
                "stream": use_llm_streaming,
                "options": {
                    "num_predict": 768,
                    "temperature": llm_config["temperature"],
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_ctx": 4096,
                    "num_batch": 64,
                    "use_mlock": True,
                    "f16_kv": True,
                    "keep_alive": "15m",
                },
            },
        )

