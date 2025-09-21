            # Minimal warmup to prevent cold start issues (reduced from 5 to 2 samples)
            logger.debug("🔥 Warming up Kokoro ONNX model...")
            warmup_texts = [
                "Hello",  # Short warmup
                "This is a test of the text to speech system.",  # Medium warmup
            ]

            warmup_start = time.time()
            for i, warmup_text in enumerate(warmup_texts):
                try:
                    if i == 0:
                        # First run - test voice compatibility
                        test_audio, test_sr = self._pipeline.create(warmup_text, voice=self._voice, speed=self._speed)
                        logger.debug(f"✅ Voice '{self._voice}' verified - generated {len(test_audio)} samples at {test_sr}Hz")
                    else:
                        # Subsequent runs - just warmup
                        self._pipeline.create(warmup_text, voice=self._voice, speed=self._speed)
                        logger.debug(f"🔥 Warmup {i+1}/{len(warmup_texts)}: {len(warmup_text)} chars")
                except Exception as voice_error:
                    if i == 0:
                        logger.error(f"❌ Voice '{self._voice}' failed: {voice_error}")
                        # Try safe fallback voice
                        try:
                            test_audio, test_sr = self._pipeline.create(warmup_text, voice="af_bella", speed=self._speed)
                            logger.debug(f"✅ Fallback to af_bella - generated {len(test_audio)} samples")
                            self._voice = "af_bella"
                        except:
                            # Last resort: try voice 0 as string
                            test_audio, test_sr = self._pipeline.create(warmup_text, voice="0", speed=self._speed)
                            logger.debug(f"✅ Ultimate fallback to voice 0 - generated {len(test_audio)} samples")
                            self._voice = "0"
                    else:
                        logger.warning(f"Warmup {i+1} failed: {voice_error}")

            warmup_time = (time.time() - warmup_start) * 1000
            logger.debug(f"🔥 Kokoro warmup completed in {warmup_time:.1f}ms")
