#!/usr/bin/env python3
"""
Parakeet STT Worker - Process-isolated transcription service.
Runs in separate process to avoid Metal threading conflicts with TTS.

Communication: JSON over stdin/stdout
Commands:
  - {"cmd": "init", "model_path": "...", "streaming": true/false}
  - {"cmd": "transcribe", "audio": "<base64>"}
  - {"cmd": "reset"}  # Reset streaming context
  - {"cmd": "config"}  # Get configuration
"""

import os
import sys
import json
import base64
import traceback
import numpy as np
from typing import Optional

# Disable MLX lock in worker process - we own the entire Metal context
os.environ["MLX_DISABLE_LOCK"] = "1"

try:
    from parakeet_mlx import from_pretrained
    import mlx.core as mx
    PARAKEET_AVAILABLE = True
    PARAKEET_OLD_FORMAT = False
except ImportError:
    try:
        # Fallback to old mlx_audio
        from mlx_audio.stt.utils import load_model
        PARAKEET_AVAILABLE = True
        PARAKEET_OLD_FORMAT = True
    except ImportError:
        PARAKEET_AVAILABLE = False
        PARAKEET_OLD_FORMAT = False


class ParakeetWorker:
    """Isolated Parakeet STT worker - owns its own Metal context."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._transcriber = None
        self._transcriber_context = None
        self._sample_rate = 16000  # Parakeet expects 16kHz
        self._streaming_mode = False

        print(json.dumps({
            "status": "Parakeet worker initialized",
            "parakeet_available": PARAKEET_AVAILABLE,
            "old_format": PARAKEET_OLD_FORMAT
        }), flush=True)

    def initialize(self, model_path: str, streaming: bool = True,
                   context_size: tuple = (256, 256), depth: int = 3,
                   beam_width: int = 8, temperature: float = 0.0):
        """Initialize Parakeet model in isolated process."""
        if not PARAKEET_AVAILABLE:
            return {"error": "Parakeet MLX not available"}

        try:
            print(json.dumps({"status": f"Loading model: {model_path}"}), flush=True)

            self._streaming_mode = streaming
            self._beam_width = beam_width
            self._temperature = temperature

            if PARAKEET_OLD_FORMAT:
                # Legacy mlx_audio format - batch only
                self._model = load_model(model_path)
                self._processor = None
                self._streaming_mode = False
                print(json.dumps({"status": "Loaded (legacy batch mode)"}), flush=True)
            else:
                # New parakeet_mlx format
                result = from_pretrained(model_path)
                if isinstance(result, tuple):
                    if len(result) >= 2:
                        self._model, self._processor = result[0], result[1]
                    elif len(result) == 1:
                        self._model = result[0]
                        self._processor = None
                    else:
                        raise ValueError(f"Unexpected return: {result}")
                else:
                    self._model = result
                    self._processor = None

                # Initialize streaming context if requested
                if streaming and hasattr(self._model, 'transcribe_stream'):
                    # Configure decoding based on model type and beam width
                    from parakeet_mlx.parakeet import DecodingConfig

                    # Check if this is a TDT model (which only supports greedy decoding)
                    is_tdt_model = 'ParakeetTDT' in str(type(self._model))

                    if is_tdt_model:
                        # TDT models only support greedy decoding
                        decoding_config = DecodingConfig(decoding='greedy')
                        if beam_width > 1:
                            print(json.dumps({"warning": f"TDT models only support greedy decoding (requested beam_width={beam_width}), using greedy"}), flush=True)
                        print(json.dumps({"status": "Greedy decoding enabled (TDT model)"}), flush=True)
                    else:
                        # Non-TDT models can support beam search
                        if beam_width > 1:
                            # Try beam search if beam_width > 1
                            try:
                                decoding_config = DecodingConfig(decoding='beam')
                                self._transcriber_context = self._model.transcribe_stream(
                                    context_size=context_size,
                                    depth=depth,
                                    keep_original_attention=False,
                                    decoding_config=decoding_config
                                )
                                self._transcriber = self._transcriber_context.__enter__()
                                print(json.dumps({"status": f"Beam search enabled (width={beam_width})"}), flush=True)
                                print(json.dumps({"status": "Streaming mode enabled"}), flush=True)
                            except Exception as beam_error:
                                # Fallback to greedy if beam search not supported
                                print(json.dumps({"warning": f"Beam search not supported: {beam_error}, falling back to greedy"}), flush=True)
                                decoding_config = DecodingConfig(decoding='greedy')
                                self._transcriber_context = self._model.transcribe_stream(
                                    context_size=context_size,
                                    depth=depth,
                                    keep_original_attention=False,
                                    decoding_config=decoding_config
                                )
                                self._transcriber = self._transcriber_context.__enter__()
                                print(json.dumps({"status": "Greedy decoding enabled (fallback)"}), flush=True)
                                print(json.dumps({"status": "Streaming mode enabled"}), flush=True)
                        else:
                            # Use greedy decoding for beam_width = 1
                            decoding_config = DecodingConfig(decoding='greedy')
                            self._transcriber_context = self._model.transcribe_stream(
                                context_size=context_size,
                                depth=depth,
                                keep_original_attention=False,
                                decoding_config=decoding_config
                            )
                            self._transcriber = self._transcriber_context.__enter__()
                            print(json.dumps({"status": "Greedy decoding enabled"}), flush=True)
                            print(json.dumps({"status": "Streaming mode enabled"}), flush=True)
                else:
                    self._streaming_mode = False
                    print(json.dumps({"status": "Batch mode enabled"}), flush=True)

            print(json.dumps({
                "success": True,
                "config": {
                    "sample_rate": self._sample_rate,
                    "streaming": self._streaming_mode,
                    "beam_width": beam_width,
                    "temperature": temperature,
                    "context_size": context_size,
                    "depth": depth
                }
            }), flush=True)

            return {"success": True}

        except Exception as e:
            error_msg = f"Failed to initialize: {str(e)}"
            print(json.dumps({"error": error_msg}), flush=True)
            traceback.print_exc(file=sys.stderr)
            return {"error": error_msg}

    def transcribe(self, audio_b64: str):
        """Transcribe audio chunk (base64 encoded PCM16)."""
        if not self._model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Transcribe using appropriate mode
            if self._streaming_mode and self._transcriber:
                # Streaming mode - incremental results using Parakeet streaming API
                try:
                    # Import MLX for audio array conversion
                    import mlx.core as mx

                    # Convert numpy audio to MLX array
                    audio_mlx = mx.array(audio_float32)

                    # Add audio to streaming transcriber
                    self._transcriber.add_audio(audio_mlx)

                    # Get current transcription result
                    result = self._transcriber.result

                    if result and hasattr(result, 'text'):
                        text = result.text.strip()
                        if text:
                            print(json.dumps({
                                "text": text,
                                "is_final": False  # Streaming results are interim
                            }), flush=True)
                        else:
                            print(json.dumps({"text": "", "is_final": False}), flush=True)
                    else:
                        print(json.dumps({"text": "", "is_final": False}), flush=True)
                except Exception as e:
                    print(json.dumps({"error": f"Streaming transcription error: {str(e)}"}), flush=True)
                    traceback.print_exc(file=sys.stderr)
            else:
                # For compatibility with new parakeet-mlx API, use streaming mode even for batch processing
                # This avoids the need to save audio to temporary files
                try:
                    # Import MLX for audio array conversion
                    import mlx.core as mx

                    # Create a temporary streaming context for this batch
                    from parakeet_mlx.parakeet import DecodingConfig

                    # Check if this is a TDT model (which only supports greedy decoding)
                    is_tdt_model = 'ParakeetTDT' in str(type(self._model))

                    if is_tdt_model:
                        decoding_config = DecodingConfig(decoding='greedy')
                    else:
                        decoding_config = DecodingConfig(decoding='greedy')

                    # Create temporary streaming context
                    with self._model.transcribe_stream(
                        context_size=(256, 256),
                        depth=1,
                        keep_original_attention=False,
                        decoding_config=decoding_config
                    ) as temp_transcriber:
                        # Convert numpy audio to MLX array
                        audio_mlx = mx.array(audio_float32)

                        # Add audio to transcriber
                        temp_transcriber.add_audio(audio_mlx)

                        # Get result
                        result = temp_transcriber.result

                        if result and hasattr(result, 'text'):
                            text = result.text.strip()
                            if text:
                                print(json.dumps({"text": text, "is_final": True}), flush=True)
                            else:
                                print(json.dumps({"text": "", "is_final": True}), flush=True)
                        else:
                            print(json.dumps({"text": "", "is_final": True}), flush=True)

                except Exception as e:
                    print(json.dumps({"error": f"Batch transcription error: {str(e)}"}), flush=True)
                    traceback.print_exc(file=sys.stderr)

        except Exception as e:
            print(json.dumps({"error": f"Transcription failed: {str(e)}"}), flush=True)
            traceback.print_exc(file=sys.stderr)

    def reset(self):
        """Reset streaming context (if streaming mode is active)."""
        if self._streaming_mode:
            try:
                # Try using the transcriber's reset method if available
                if self._transcriber and hasattr(self._transcriber, 'reset'):
                    self._transcriber.reset()
                    print(json.dumps({"status": "Transcriber reset successful"}), flush=True)
                else:
                    # Fallback: re-create the streaming context
                    if self._transcriber_context:
                        try:
                            self._transcriber_context.__exit__(None, None, None)
                            self._transcriber = None
                            self._transcriber_context = None
                        except Exception as e:
                            print(json.dumps({"warning": f"Context exit error: {e}"}), flush=True)

                    # Re-enter new context
                    from parakeet_mlx.parakeet import DecodingConfig

                    try:
                        # Check if this is a TDT model (which only supports greedy decoding)
                        is_tdt_model = 'ParakeetTDT' in str(type(self._model))

                        if is_tdt_model:
                            # TDT models only support greedy decoding
                            decoding_config = DecodingConfig(decoding='greedy')
                            if self._beam_width > 1:
                                print(json.dumps({"warning": f"TDT models only support greedy decoding (requested beam_width={self._beam_width}), using greedy"}), flush=True)
                            print(json.dumps({"status": "Greedy decoding enabled (TDT model)"}), flush=True)
                        else:
                            # Non-TDT models can support beam search
                            if self._beam_width > 1:
                                # Try beam search if beam_width > 1
                                try:
                                    decoding_config = DecodingConfig(decoding='beam')
                                except Exception as beam_error:
                                    # Fallback to greedy if beam search not supported
                                    print(json.dumps({"warning": f"Beam search not supported: {beam_error}, falling back to greedy"}), flush=True)
                                    decoding_config = DecodingConfig(decoding='greedy')
                            else:
                                # Use greedy decoding for beam_width = 1
                                decoding_config = DecodingConfig(decoding='greedy')

                        self._transcriber_context = self._model.transcribe_stream(
                            context_size=(256, 256),
                            depth=3,
                            keep_original_attention=False,
                            decoding_config=decoding_config
                        )
                        self._transcriber = self._transcriber_context.__enter__()
                        print(json.dumps({"status": "Streaming context reset"}), flush=True)
                    except Exception as reset_error:
                        # Fallback to greedy if beam search not supported during reset
                        print(json.dumps({"warning": f"Beam search reset failed: {reset_error}, using greedy"}), flush=True)
                        try:
                            decoding_config = DecodingConfig(decoding='greedy')
                            self._transcriber_context = self._model.transcribe_stream(
                                context_size=(256, 256),
                                depth=3,
                                keep_original_attention=False,
                                decoding_config=decoding_config
                            )
                            self._transcriber = self._transcriber_context.__enter__()
                            print(json.dumps({"status": "Streaming context reset (greedy fallback)"}), flush=True)
                        except Exception as fallback_error:
                            print(json.dumps({"error": f"Complete reset failure: {fallback_error}"}), flush=True)
                            self._transcriber = None
                            self._transcriber_context = None
            except Exception as e:
                print(json.dumps({"error": f"Reset failed: {str(e)}"}), flush=True)
                traceback.print_exc(file=sys.stderr)
                self._transcriber = None
                self._transcriber_context = None
        else:
            print(json.dumps({"status": "No streaming context to reset"}), flush=True)

    def finalize(self):
        """Get the final transcription result from the streaming context."""
        if not self._model:
            print(json.dumps({"error": "Not initialized"}), flush=True)
            return

        if self._streaming_mode and self._transcriber:
            try:
                draft_text_before = ""
                if hasattr(self._transcriber, 'result') and self._transcriber.result and hasattr(self._transcriber.result, 'text'):
                    draft_text_before = self._transcriber.result.text.strip()
                print(json.dumps({"status": f"Before finalize call. Current draft text: '{draft_text_before}'"}), flush=True)

                if hasattr(self._transcriber, 'finalize'):
                    self._transcriber.finalize()
                    print(json.dumps({"status": "Called transcriber.finalize()"}), flush=True)

                result = self._transcriber.result
                text = ""
                if result and hasattr(result, 'text'):
                    text = result.text.strip()

                print(json.dumps({"status": f"After finalize call. Final text: '{text}'"}), flush=True)

                print(json.dumps({
                    "text": text,
                    "is_final": True
                }), flush=True)

            except Exception as e:
                print(json.dumps({"error": f"Finalization error: {str(e)}"}), flush=True)
                traceback.print_exc(file=sys.stderr)
        else:
            # In batch mode, there's no ongoing stream to finalize.
            # The last `transcribe` result was final. Return empty.
            print(json.dumps({"text": "", "is_final": True}), flush=True)

    def get_config(self):
        """Return current configuration."""
        print(json.dumps({
            "sample_rate": self._sample_rate,
            "streaming": self._streaming_mode,
            "initialized": self._model is not None
        }), flush=True)


def main():
    """Main worker loop - read commands from stdin, write results to stdout."""
    print(json.dumps({"status": "Parakeet worker starting..."}), flush=True)
    worker = ParakeetWorker()

    for line in sys.stdin:
        try:
            if not line.strip():
                continue

            req = json.loads(line.strip())
            cmd = req.get("cmd")

            if cmd == "init":
                worker.initialize(
                    model_path=req.get("model_path", "mlx-community/parakeet-tdt-0.6b-v3"),
                    streaming=req.get("streaming", True),
                    context_size=tuple(req.get("context_size", [256, 256])),
                    depth=req.get("depth", 3),
                    beam_width=req.get("beam_width", 8),
                    temperature=req.get("temperature", 0.0)
                )

            elif cmd == "transcribe":
                worker.transcribe(req["audio"])

            elif cmd == "reset":
                worker.reset()

            elif cmd == "finalize":
                worker.finalize()

            elif cmd == "config":
                worker.get_config()

            else:
                print(json.dumps({"error": f"Unknown command: {cmd}"}), flush=True)

        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        except KeyError as e:
            print(json.dumps({"error": f"Missing parameter: {e}"}), flush=True)
        except Exception as e:
            print(json.dumps({"error": f"Worker error: {str(e)}"}), flush=True)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
