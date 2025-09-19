"""
Kyutai Streaming STT Service for Pipecat
Provides ultra-low latency streaming transcription using Kyutai's delayed streams modeling
"""

import array
import asyncio
import json
import numpy as np
import os
import queue
import threading
import time
from typing import AsyncGenerator, Optional
from loguru import logger

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
from huggingface_hub import hf_hub_download

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    ErrorFrame,
    AudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame
)
from pipecat.services.ai_services import STTService

# Import fast punctuation restoration
try:
    from fast_punctuation import FastPunctuationRestorer
    PUNCTUATION_AVAILABLE = True
except ImportError:
    logger.warning("FastPunctuationRestorer not available, final transcription will be without punctuation")
    PUNCTUATION_AVAILABLE = False

try:
    from moshi_mlx import models, utils
    KYUTAI_AVAILABLE = True
except ImportError:
    logger.warning("Kyutai/moshi_mlx not available. Install with: pip install moshi_mlx")
    KYUTAI_AVAILABLE = False


class KyutaiStreamingSTT(STTService):
    """
    Streaming STT using Kyutai's delayed streams modeling.
    Provides true real-time transcription with ~80ms latency.
    """

    def __init__(
        self,
        *,
        hf_repo: str = "kyutai/stt-1b-en_fr-candle",
        max_steps: int = 4096,
        enable_vad: bool = False,
        sample_rate: int = 24000,  # Target sample rate for Kyutai (24 kHz)
        **kwargs
    ):
        super().__init__(**kwargs)

        if not KYUTAI_AVAILABLE:
            raise ImportError("Kyutai/moshi_mlx is required but not installed")

        self._hf_repo = hf_repo
        self._max_steps = max_steps
        self._enable_vad = enable_vad
        # Important: STTService also has a _sample_rate field that is set from
        # StartFrame (input capture rate). To avoid clobbering, keep a separate
        # target sample rate for the Kyutai model.
        self._target_sample_rate = sample_rate
        self._block_size = 1920  # 80ms at 24kHz

        # Audio processing
        self._audio_queue = queue.Queue()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._processing_lock = asyncio.Lock()
        self._running = False

        # Text buffering for proper transcription frames
        self._text_buffer = []
        self._last_speech_time = 0
        self._consecutive_eos_count = 0
        self._consecutive_pad_count = 0  # Track PAD tokens separately

        # Initialize punctuation restorer if available
        self._punctuation_restorer = None
        if PUNCTUATION_AVAILABLE:
            try:
                self._punctuation_restorer = FastPunctuationRestorer()
                logger.info("Fast punctuation restoration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize punctuation restorer: {e}")
                self._punctuation_restorer = None
        self._audio_threshold = 0.001  # Very low threshold to catch all audio
        self._last_reset_time = 0
        self._reset_threshold = 8  # Reset sooner to prevent degeneracy
        self._max_eos_before_skip = 3  # Skip processing after fewer EOS tokens

        # Kyutai components (will be initialized)
        self._model = None
        self._gen = None
        self._text_tokenizer = None
        self._audio_tokenizer = None
        self._lm_config = None

        # Initialize the model
        self._init_kyutai_model()

        logger.info(f"Kyutai STT initialized: {hf_repo} with VAD={enable_vad}")

    def _add_punctuation(self, text: str) -> str:
        """Add punctuation to text if punctuation restorer is available"""
        if self._punctuation_restorer and text.strip():
            try:
                punctuated = self._punctuation_restorer.restore_punctuation(text)
                logger.debug(f"Punctuation: '{text}' -> '{punctuated}'")
                return punctuated
            except Exception as e:
                logger.warning(f"Punctuation restoration failed: {e}")
        return text

    def _init_kyutai_model(self):
        """Initialize the Kyutai streaming model."""
        try:
            logger.info(f"Loading Kyutai streaming model: {self._hf_repo}")

            # Download model components
            config_path = hf_hub_download(self._hf_repo, "config.json")
            logger.info(f"Config loaded from: {config_path}")
            with open(config_path, "r") as fobj:
                config_dict = json.load(fobj)

            mimi_weights = hf_hub_download(self._hf_repo, config_dict["mimi_name"])
            logger.info(f"Mimi weights loaded from: {mimi_weights}")
            moshi_name = config_dict.get("moshi_name", "model.safetensors")
            moshi_weights = hf_hub_download(self._hf_repo, moshi_name)
            logger.info(f"Moshi weights loaded from: {moshi_weights}")
            tokenizer_path = hf_hub_download(self._hf_repo, config_dict["tokenizer_name"])
            logger.info(f"Tokenizer loaded from: {tokenizer_path}")

            # Initialize model
            self._lm_config = models.LmConfig.from_config_dict(config_dict)
            self._model = models.Lm(self._lm_config)
            self._model.set_dtype(mx.bfloat16)

            # Apply quantization if model is quantized
            if moshi_weights.endswith(".q4.safetensors"):
                nn.quantize(self._model, bits=4, group_size=32)
            elif moshi_weights.endswith(".q8.safetensors"):
                nn.quantize(self._model, bits=8, group_size=64)

            # Load weights
            logger.info(f"Loading model weights from {moshi_weights}")
            if self._hf_repo.endswith("-candle"):
                self._model.load_pytorch_weights(moshi_weights, self._lm_config, strict=True)
            else:
                self._model.load_weights(moshi_weights, strict=True)

            # Initialize tokenizers
            logger.info(f"Loading text tokenizer from {tokenizer_path}")
            self._text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

            # Debug tokenizer configuration
            logger.info(f"Tokenizer vocab size: {self._text_tokenizer.get_piece_size()}")
            logger.info(f"Tokenizer EOS token ID: {self._text_tokenizer.eos_id()}")
            logger.info(f"Tokenizer BOS token ID: {self._text_tokenizer.bos_id()}")
            logger.info(f"Tokenizer PAD token ID: {self._text_tokenizer.pad_id()}")

            # Check what token 3 actually is
            if self._text_tokenizer.get_piece_size() > 3:
                token_3_piece = self._text_tokenizer.id_to_piece(3)
                logger.info(f"Token 3 represents: '{token_3_piece}'")

            # Store proper token IDs based on tokenizer configuration
            self._eos_token_id = self._text_tokenizer.eos_id()  # Official EOS
            self._pad_token_id = self._text_tokenizer.pad_id()  # PAD token
            self._bos_token_id = self._text_tokenizer.bos_id()  # BOS token

            # Only treat actual EOS tokens as EOS, not PAD tokens
            self._eos_token_ids = {self._eos_token_id}

            # Common fallback EOS token IDs from research (exclude PAD token ID)
            fallback_eos_ids = {2, 32000} - {self._pad_token_id}
            self._eos_token_ids.update(fallback_eos_ids)

            logger.info(f"EOS token IDs: {self._eos_token_ids}")
            logger.info(f"PAD token ID: {self._pad_token_id}")
            logger.info(f"BOS token ID: {self._bos_token_id}")

            logger.info(f"Loading audio tokenizer from {mimi_weights}")
            generated_codebooks = self._lm_config.generated_codebooks
            other_codebooks = self._lm_config.other_codebooks
            mimi_codebooks = max(generated_codebooks, other_codebooks)
            if self._hf_repo.endswith("-candle"):
                # Candle variant uses RustyMimi tokenizer
                self._audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)
                logger.info("Audio tokenizer backend: RustyMimi (Candle)")
            else:
                # MLX variant uses moshi_mlx Mimi implementation
                self._audio_tokenizer = models.mimi.Mimi(models.mimi_202407(mimi_codebooks))
                # Load PyTorch Mimi weights (even on MLX backend)
                self._audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)
                logger.info("Audio tokenizer backend: moshi_mlx Mimi (MLX)")

            # Warm up the model
            logger.info("Warming up the model...")
            self._model.warmup()

            # Initialize generator - use config defaults when available
            lmgen_cfg = config_dict.get("lm_gen_config", {})
            top_k_text = int(lmgen_cfg.get("top_k_text", 25))
            temp_text = float(lmgen_cfg.get("temp_text", 0.0))
            top_k_audio = int(lmgen_cfg.get("top_k", 250))
            temp_audio = float(lmgen_cfg.get("temp", 0.8))

            # Allow env overrides for quick experiments
            env_top_k_text = os.getenv("KYUTAI_TEXT_TOP_K")
            env_temp_text = os.getenv("KYUTAI_TEXT_TEMP")
            env_top_k_audio = os.getenv("KYUTAI_AUDIO_TOP_K")
            env_temp_audio = os.getenv("KYUTAI_AUDIO_TEMP")
            if env_top_k_text:
                top_k_text = int(env_top_k_text)
            if env_temp_text:
                temp_text = float(env_temp_text)
            if env_top_k_audio:
                top_k_audio = int(env_top_k_audio)
            if env_temp_audio:
                temp_audio = float(env_temp_audio)

            self._gen = models.LmGen(
                model=self._model,
                max_steps=self._max_steps,
                text_sampler=utils.Sampler(top_k=top_k_text, temp=temp_text),
                audio_sampler=utils.Sampler(top_k=top_k_audio, temp=temp_audio),
                check=False,
            )

            logger.info("Kyutai model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kyutai model: {e}")
            raise

    def _reset_model_state(self):
        """Reset the model generator state to recover from degeneracy."""
        try:
            logger.warning("Resetting Kyutai model state due to EOS degeneracy")

            # Reinitialize generator to reset state - try with different sampling params
            self._gen = models.LmGen(
                model=self._model,
                max_steps=self._max_steps,
                text_sampler=utils.Sampler(top_k=25, temp=0),  # Match original working script configuration
                audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                check=False,
            )

            # Clear audio buffer to start fresh
            self._audio_buffer = np.array([], dtype=np.float32)

            # Reset counters
            self._consecutive_eos_count = 0
            self._consecutive_pad_count = 0
            self._text_buffer = []
            self._last_reset_time = time.time()

            logger.info("Model state reset completed with fresh audio buffer")

        except Exception as e:
            logger.error(f"Failed to reset model state: {e}")

    def _resample_audio(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample audio to 24kHz if needed."""
        if source_rate == self._target_sample_rate:
            return audio

        # Simple linear interpolation resampling
        ratio = self._target_sample_rate / source_rate
        length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, length)
        return np.interp(indices, np.arange(len(audio)), audio)

    async def run_stt(self, audio: bytes, sample_rate: Optional[int] = None) -> AsyncGenerator[Frame, None]:
        """
        Process audio in streaming chunks using Kyutai.

        Args:
            audio: Raw audio bytes (16-bit PCM)

        Yields:
            TranscriptionFrame or InterimTranscriptionFrame
        """
        try:
            # Convert bytes to numpy array (assume 16-bit PCM)
            audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            if audio_np.size:
                if not hasattr(self, "_debug_input_count"):
                    self._debug_input_count = 0
                if self._debug_input_count < 5:
                    peak_in = float(np.max(np.abs(audio_np)))
                    rms_in = float(np.sqrt(np.mean(audio_np * audio_np)))
                    logger.debug(f"Input chunk: peak={peak_in:.4f} rms={rms_in:.4f}")
                    self._debug_input_count += 1

            # Resample to 24kHz if needed (Kyutai expects 24kHz)
            # Use provided sample_rate if available, otherwise fall back to 16 kHz assumption
            # Prefer explicit sample_rate; otherwise use STTService-provided capture sample_rate
            # (set during start), with a safe fallback to 16 kHz.
            source_rate = sample_rate if sample_rate is not None else (self.sample_rate or 16000)
            if not hasattr(self, "_debug_resample_count"):
                self._debug_resample_count = 0
            if self._debug_resample_count < 3:
                logger.debug(f"Resampling from {source_rate} Hz to {self._target_sample_rate} Hz; input bytes={len(audio)}")
                self._debug_resample_count += 1
            audio_np = self._resample_audio(audio_np, source_rate)

            # Add to buffer
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_np])

            # Process in blocks of 1920 samples (80ms at 24kHz)
            async with self._processing_lock:
                while len(self._audio_buffer) >= self._block_size:
                    # Extract block
                    block = self._audio_buffer[:self._block_size]
                    self._audio_buffer = self._audio_buffer[self._block_size:]

                    try:
                        # Process block with Kyutai - match original script format exactly
                        # Original script format analysis:
                        # sounddevice provides: (1920, 1) float32
                        # block[None, :, 0] -> (1, 1920)
                        # encode_step(block[None, 0:1]) -> (1, 1, 1920)

                        # Start with our audio block (1920 samples)
                        block_f32 = block.astype(np.float32)

                        # Simulate sounddevice format: (blocksize, channels)
                        block_sd_format = block_f32.reshape(-1, 1)  # (1920, 1)

                        # Apply original script transformations
                        block_transformed = block_sd_format[None, :, 0]  # (1, 1920)

                        # Final format for encode_step: (1, 1, 1920)
                        block_final = block_transformed[None, 0:1]  # (1, 1, 1920)

                        # Check audio levels
                        audio_max = float(np.max(np.abs(block_final)))
                        audio_mean = float(np.mean(np.abs(block_final)))
                        # Log audio stats periodically
                        if not hasattr(self, "_debug_block_count"):
                            self._debug_block_count = 0
                        self._debug_block_count += 1
                        if self._debug_block_count <= 10 or self._debug_block_count % 50 == 0:
                            logger.info(f"🎤 Audio block #{self._debug_block_count}: shape={block_final.shape}, max={audio_max:.4f}, mean={audio_mean:.4f}")
                            if audio_max < 0.001:
                                logger.warning(f"⚠️ Audio is silent or nearly silent!")
                            elif audio_max > 0.01:
                                logger.info(f"✅ Good audio level detected")

                        # Encode audio depending on tokenizer backend
                        if isinstance(self._audio_tokenizer, rustymimi.Tokenizer):
                            other_audio_tokens = self._audio_tokenizer.encode_step(block_final)
                        else:
                            # moshi_mlx Mimi expects MLX arrays
                            other_audio_tokens = self._audio_tokenizer.encode_step(mx.array(block_final))
                        other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                            :, :, :self._lm_config.other_codebooks
                        ]

                        # # Debug: Log audio token shape for first few blocks
                        # if self._debug_block_count <= 5:
                        #     # logger.debug(f"Audio tokens shape: {other_audio_tokens.shape}, other_codebooks: {self._lm_config.other_codebooks}")

                        # Generate text token
                        if self._enable_vad:
                            text_token, vad_heads = self._gen.step_with_extra_heads(other_audio_tokens[0])
                            # VAD end of turn detection - only log for now, don't spam frames
                            # if vad_heads:
                            #     pr_vad = vad_heads[2][0, 0, 0].item()
                            #     if pr_vad > 0.5:
                            #         # logger.debug(f"VAD detected end of turn (confidence: {pr_vad:.3f})")
                                    # Note: We rely on EOS tokens for actual turn detection, not VAD
                        else:
                            text_token = self._gen.step(other_audio_tokens[0])

                        # Debug: Log token generation
                        if not hasattr(self, '_debug_token_count'):
                            self._debug_token_count = 0
                        self._debug_token_count += 1

                        if self._debug_token_count <= 20 or self._debug_token_count % 100 == 0:
                            if hasattr(text_token, 'shape'):
                                logger.info(f"🔤 Token #{self._debug_token_count}: shape={text_token.shape}, value={text_token}")
                            else:
                                logger.info(f"🔤 Token #{self._debug_token_count}: {text_token}")

                        # Convert token to text
                        text_token = text_token[0].item()
                        # Only log non-PAD tokens to reduce noise
                        if text_token != self._pad_token_id:
                            logger.debug(f"Generated text token: {text_token} (consecutive EOS: {self._consecutive_eos_count})")

                        # Check token types
                        is_eos_token = text_token in self._eos_token_ids
                        is_pad_token = text_token == self._pad_token_id
                        is_bos_token = text_token == self._bos_token_id

                        # Filter out UNK tokens (token ID 0) and special tokens
                        is_unk_token = (text_token == 0)

                        if is_unk_token:
                            logger.debug(f"Ignoring UNK token: {text_token}")

                        if not is_eos_token and not is_pad_token and not is_bos_token and not is_unk_token:
                            # Reset counters on real speech tokens
                            self._consecutive_eos_count = 0
                            self._consecutive_pad_count = 0

                            text = self._text_tokenizer.id_to_piece(text_token)
                            text = text.replace("▁", " ")
                            logger.info(f"Kyutai STT: '{text}' (token: {text_token})")

                            if text.strip():
                                # Add to text buffer
                                self._text_buffer.append(text)
                                self._last_speech_time = len(block) / self._target_sample_rate
                                logger.debug(f"Added to text buffer: '{text}', buffer length: {len(self._text_buffer)}")

                                # Yield interim transcription with accumulated text every few tokens
                                if len(self._text_buffer) % 3 == 0:  # Every 3 tokens
                                    accumulated_text = "".join(self._text_buffer).strip()
                                    logger.info(f"📝 Yielding InterimTranscriptionFrame: '{accumulated_text}'")
                                    if accumulated_text:
                                        yield InterimTranscriptionFrame(
                                            text=accumulated_text,
                                            user_id=os.getenv("USER_ID", "user"),
                                            timestamp=str(self._last_speech_time)  # Convert to string for RTVI
                                        )
                        elif is_pad_token:
                            # Track PAD tokens but do not reset; throttle logging
                            self._consecutive_pad_count += 1
                            if self._consecutive_pad_count <= 5 or (self._consecutive_pad_count % 50 == 0):
                                logger.debug(f"PAD token detected (count: {self._consecutive_pad_count})")

                        elif is_eos_token:  # Any EOS token - end of speech
                            self._consecutive_eos_count += 1

                            # Only finalize if we have accumulated text and see an EOS
                            if self._text_buffer and self._consecutive_eos_count >= 1:
                                logger.debug(f"EOS token detected with text in buffer - finalizing")
                                # Emit final transcription with punctuation
                                final_text = "".join(self._text_buffer).strip()
                                if final_text:
                                    # Add punctuation restoration
                                    punctuated_text = self._add_punctuation(final_text)
                                    yield TranscriptionFrame(
                                        text=punctuated_text,
                                        user_id=os.getenv("USER_ID", "user"),
                                        timestamp=str(self._last_speech_time)  # Convert to string for RTVI
                                    )
                                # Clear buffer after sending
                                self._text_buffer = []
                                self._consecutive_eos_count = 0
                        # Note: PAD token handling is already done above (lines 387-403)
                        elif is_bos_token:
                            logger.debug(f"Ignoring BOS token: {text_token}")
                            continue
                        else:
                            logger.debug(f"Skipped unknown special token: {text_token}")

                    except Exception as e:
                        logger.error(f"Error processing audio block: {e}")
                        yield ErrorFrame(error=str(e))

            # Small yield to prevent blocking
            await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Kyutai processing error: {e}")
            yield ErrorFrame(error=str(e))

    # Intercept VAD stop to finalize; leave audio handling to STTService
    async def process_frame(self, frame: Frame, direction=None):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Clear buffer when user starts speaking to prevent carryover
            logger.info("🎤 VAD detected user started speaking - clearing text buffer")
            self._text_buffer = []
            self._consecutive_eos_count = 0
            self._consecutive_pad_count = 0

        if isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("🛑 VAD detected user stopped speaking - finalizing transcription")
            await self._finalize_transcription(direction)

    async def flush(self) -> AsyncGenerator[Frame, None]:
        """
        Flush any remaining audio in the buffer and finalize pending transcription.
        """
        try:
            # Finalize any pending transcription
            if self._text_buffer:
                final_text = "".join(self._text_buffer).strip()
                if final_text:
                    # Add punctuation restoration
                    punctuated_text = self._add_punctuation(final_text)
                    logger.info(f"Finalizing buffered transcription: '{final_text}' -> '{punctuated_text}'")
                    yield TranscriptionFrame(
                        text=punctuated_text,
                        user_id=os.getenv("USER_ID", "user"),
                        timestamp=str(self._last_speech_time)  # Convert to string for RTVI
                    )
                # Clear buffer
                self._text_buffer = []

            # Process remaining audio if we have at least partial block
            if len(self._audio_buffer) > 0:
                if len(self._audio_buffer) >= self._block_size // 2:
                    # Pad to block size
                    padding_needed = self._block_size - (len(self._audio_buffer) % self._block_size)
                    if padding_needed < self._block_size:
                        self._audio_buffer = np.concatenate([
                            self._audio_buffer,
                            np.zeros(padding_needed, dtype=np.float32)
                        ])

                    # Process final blocks
                    async for result_frame in self.run_stt(b''):  # Empty bytes to trigger processing
                        yield result_frame

                # Clear buffer
                self._audio_buffer = np.array([], dtype=np.float32)

        except Exception as e:
            logger.error(f"Error flushing Kyutai buffer: {e}")
            yield ErrorFrame(error=str(e))

    async def cancel(self):
        """Cancel any ongoing processing."""
        self._running = False
        self._audio_buffer = np.array([], dtype=np.float32)
        if self._gen:
            # Reset generator state if possible
            try:
                # Reinitialize generator to reset state
                self._gen = models.LmGen(
                    model=self._model,
                    max_steps=self._max_steps,
                    text_sampler=utils.Sampler(top_k=25, temp=0),
                    audio_sampler=utils.Sampler(top_k=250, temp=0.8),
                    check=False,
                )
            except Exception as e:
                logger.warning(f"Failed to reset Kyutai generator: {e}")

    # Remove duplicate process_frame definition (merged above)

    async def _finalize_transcription(self, direction=None):
        """
        Finalize any buffered transcription and send final TranscriptionFrame.
        """
        if self._text_buffer:
            final_text = "".join(self._text_buffer).strip()
            if final_text:
                # Add punctuation restoration
                punctuated_text = self._add_punctuation(final_text)
                logger.info(f"📝 Finalizing transcription: '{final_text}' -> '{punctuated_text}'")
                final_frame = TranscriptionFrame(
                    text=punctuated_text,
                    user_id=os.getenv("USER_ID", "user"),
                    timestamp=str(self._last_speech_time) if hasattr(self, '_last_speech_time') else str(time.time())
                )
                await self.push_frame(final_frame, direction)
            # Clear buffer for next utterance
            self._text_buffer = []
            logger.debug("Text buffer cleared after finalization")

    def __del__(self):
        """Cleanup when service is destroyed."""
        self._running = False
        if hasattr(self, '_gen') and self._gen:
            try:
                del self._gen
            except:
                pass
        if hasattr(self, '_model') and self._model:
            try:
                del self._model
            except:
                pass
