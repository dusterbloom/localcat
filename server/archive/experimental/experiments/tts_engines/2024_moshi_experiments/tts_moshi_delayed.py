"""
Moshi TTS with Delayed Streams Modeling for ultra-low latency.
Based on Kyutai's delayed streams modeling approach for real-time TTS.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional, List
import numpy as np
from loguru import logger

# Moshi MLX imports
import mlx.core as mx
import mlx.nn as nn
import sentencepiece
from moshi_mlx import models
from moshi_mlx.models import LmGen
from moshi_mlx.models.tts import (
    Entry,
    TTSModel,
    script_to_entries,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    ConditionAttributes,
    ConditionTensor,
    dropout_all_conditions,
)
from moshi_mlx.utils import Sampler
from huggingface_hub import hf_hub_download

# Pipecat imports
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# Import global MLX lock to prevent Metal threading conflicts
from utils.mlx_lock import MLX_GLOBAL_LOCK


class TTSGen:
    """Generator for streaming TTS with delayed streams modeling."""

    def __init__(
        self,
        tts_model: TTSModel,
        attributes: List[ConditionAttributes],
    ):
        self.tts_model = tts_model
        self.attributes = attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        # Handle CFG if needed
        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = dropout_all_conditions(attributes)
            attributes = list(attributes) + nulled

        # Setup conditioning
        self.ct = None
        self.cross_attention_src = None

        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if self.ct is None:
                    self.ct = _ct
                else:
                    self.ct = ConditionTensor(self.ct.tensor + _ct.tensor)

            for _key, _value in _attr.tensor.items():
                _conditioner = tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if self.cross_attention_src is None:
                    self.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        # Setup hooks for delayed stream modeling
        def _on_audio_hook(audio_tokens):
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        # Initialize generator
        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=tts_model.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            cfg_coef=tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

        self.audio_frames = []

    def process(self):
        """Process streaming generation while maintaining proper lookahead."""
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def process_last(self):
        """Process final generation for remaining text."""
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()

        # Generate additional steps for proper ending
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def _step(self):
        """Execute a single generation step."""
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )

        self.lm_gen.step(
            input_tokens,
            ct=self.ct,
            cross_attention_src=self.cross_attention_src
        )

        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1

        if frame is not None:
            self.audio_frames.append(frame)

    def append_entry(self, entry):
        """Add new text entry to process."""
        self.state.entries.append(entry)

    def get_audio_frames(self):
        """Get and clear accumulated audio frames."""
        frames = self.audio_frames
        self.audio_frames = []
        return frames


class MoshiDelayedTTS(TTSService):
    """
    Moshi TTS with Delayed Streams Modeling for ultra-low latency.
    Provides true streaming TTS with ~500ms delay.
    """

    def __init__(
        self,
        *,
        voice: str = "expressive_1",
        sample_rate: int = 24000,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=True,  # Use sentence aggregation
            **kwargs
        )

        self._voice = voice
        self._sample_rate = sample_rate
        self._hf_repo = hf_repo
        self._voice_repo = voice_repo
        self._temperature = temperature

        # Initialize model
        self._tts_model = None
        self._mimi = None
        self._initialize_model()

        logger.info(f"✅ Moshi Delayed TTS initialized with voice: {self._voice}")

    def _initialize_model(self):
        """Initialize the Moshi TTS model with delayed streams."""
        try:
            logger.info("🔄 Initializing Moshi TTS with Delayed Streams...")

            with MLX_GLOBAL_LOCK:
                # Load configuration
                config_path = hf_hub_download(self._hf_repo, "config.json")
                with open(config_path, "r") as f:
                    import json
                    config_dict = json.load(f)

                # Load tokenizer
                tokenizer_path = hf_hub_download(self._hf_repo, "tokenizer_spm_8k_en_fr_audio.model")
                tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

                # Load model
                model_path = hf_hub_download(self._hf_repo, "dsm_tts_1e68beda@240.safetensors")
                lm_config = models.LmConfig.from_config_dict(config_dict)
                lm_model = models.Lm(lm_config)
                lm_model.load_weights(model_path, strict=False)  # DSM TTS has extra depformer layers

                # Load Mimi audio encoder
                mimi_name = config_dict.get("mimi_name", "tokenizer-e351c8d8-checkpoint125.safetensors")
                mimi_path = hf_hub_download(self._hf_repo, mimi_name)
                generated_codebooks = lm_config.generated_codebooks
                audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
                audio_tokenizer.load_pytorch_weights(str(mimi_path), strict=True)

                # Create TTS model
                self._tts_model = TTSModel(
                    lm_model,
                    audio_tokenizer,
                    tokenizer,
                    voice_repo=self._voice_repo,
                    temp=self._temperature,
                    cfg_coef=1.0,
                    max_padding=8,
                    initial_padding=2,
                    final_padding=2,
                    padding_bonus=0,
                    raw_config=config_dict,  # Pass dict not string
                )

                self._mimi = audio_tokenizer

                # Warmup
                logger.debug("🔥 Warming up Moshi TTS...")
                warmup_start = time.time()

                # Create test entries for warmup
                test_text = "Hello, this is a test."
                entries = script_to_entries(
                    self._tts_model.tokenizer,
                    self._tts_model.machine.token_ids,
                    self._mimi.frame_rate,
                    [test_text],
                    multi_speaker=False,
                    padding_between=1,
                )

                # Get voice conditioning
                attributes = self._tts_model.make_condition_attributes(
                    self._voice,
                    entries,
                )

                # Run warmup generation
                gen = TTSGen(self._tts_model, attributes)
                for entry in entries:
                    gen.append_entry(entry)
                gen.process()
                gen.process_last()

                warmup_time = (time.time() - warmup_start) * 1000
                logger.debug(f"🔥 Moshi TTS warmup completed in {warmup_time:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Moshi TTS: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS using Moshi with delayed streams."""

        logger.info(f"🎤 Moshi TTS RECEIVED: '{text}' ({len(text)} chars)")

        if not text.strip():
            logger.debug("🔇 Skipping TTS for empty text")
            yield TTSStartedFrame()
            yield TTSStoppedFrame()
            return

        yield TTSStartedFrame()

        try:
            overall_start_time = time.time()
            first_audio_sent = False

            with MLX_GLOBAL_LOCK:
                # Prepare text entries
                entries = script_to_entries(
                    self._tts_model.tokenizer,
                    self._tts_model.machine.token_ids,
                    self._mimi.frame_rate,
                    [text],
                    multi_speaker=False,
                    padding_between=1,
                )

                # Get voice conditioning
                attributes = self._tts_model.make_condition_attributes(
                    self._voice,
                    entries,
                )

                # Create generator
                gen = TTSGen(self._tts_model, attributes)

                # Add all entries
                for entry in entries:
                    gen.append_entry(entry)

                # Process with streaming
                while len(gen.state.entries) > 0:
                    gen.process()

                    # Get audio frames
                    frames = gen.get_audio_frames()

                    for frame in frames:
                        # Decode audio frame
                        audio = self._mimi.decode_step(frame[None])

                        if audio is not None:
                            # Convert to numpy
                            audio_np = np.array(audio[0, 0], copy=True).astype(np.float32)

                            # Convert to int16 PCM
                            audio_int16 = (audio_np * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()

                            if not first_audio_sent:
                                ttfb = (time.time() - overall_start_time) * 1000
                                logger.debug(f"🚀 MOSHI DSM TTFB: {ttfb:.1f}ms")
                                first_audio_sent = True

                            # Stream the audio chunk
                            yield TTSAudioRawFrame(
                                audio=audio_bytes,
                                sample_rate=self._sample_rate,
                                num_channels=1
                            )

                            # Small delay for natural flow
                            await asyncio.sleep(0.01)

                # Process final frames
                gen.process_last()
                frames = gen.get_audio_frames()

                for frame in frames:
                    audio = self._mimi.decode_step(frame[None])
                    if audio is not None:
                        audio_np = np.array(audio[0, 0], copy=True).astype(np.float32)
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        yield TTSAudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self._sample_rate,
                            num_channels=1
                        )

                        await asyncio.sleep(0.01)

            total_time = (time.time() - overall_start_time) * 1000
            logger.debug(f"✅ Moshi TTS completed in {total_time:.1f}ms")

        except asyncio.CancelledError:
            logger.debug("Moshi TTS generation cancelled")
            raise
        except Exception as e:
            logger.error(f"Moshi TTS error: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            yield TTSStoppedFrame()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass