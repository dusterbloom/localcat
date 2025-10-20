#!/usr/bin/env python3
"""Quick test to call the MLX TTS sidecar and save audio.

Posts a short text to http://127.0.0.1:8770/tts and writes both raw PCM and
WAV outputs to /tmp.
"""
import sys
import json
import wave
import httpx

URL = "http://127.0.0.1:8770/tts"
TEXT = "Hello from LocalCat. This is a sidecar test."
VOICE = "af_heart"
SPEED = 1.0
LANG = "en"
SR = 24000

def main():
    out_pcm = "/tmp/localcat_sidecar_test.pcm"
    out_wav = "/tmp/localcat_sidecar_test.wav"
    payload = {"text": TEXT, "voice": VOICE, "speed": SPEED, "lang": LANG}

    total = 0
    with httpx.stream("POST", URL, json=payload, timeout=httpx.Timeout(None)) as r:
        r.raise_for_status()
        with open(out_pcm, "wb") as f:
            for chunk in r.iter_bytes():
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)

    # Convert raw PCM to WAV
    with open(out_pcm, "rb") as f:
        pcm = f.read()
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SR)
        wf.writeframes(pcm)

    print(json.dumps({
        "pcm_bytes": total,
        "pcm_path": out_pcm,
        "wav_path": out_wav
    }))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

