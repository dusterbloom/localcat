Directory structure:
└── shubhdotai-offline-voice-ai/
    ├── README.md
    ├── audio_buffer.py
    ├── config.py
    ├── correlator.js
    ├── index.html
    ├── llm_handler.py
    ├── requirements.txt
    ├── server.py
    ├── transcriber.py
    ├── tts_handler.py
    ├── vad_detector.py
    └── docs/
        └── TECHNICAL_DEEP_DIVE.md


Files Content:

================================================
FILE: README.md
================================================
# Real-Time MLX Voice Agent

Ultra-responsive, full-duplex voice assistant tuned for Apple Silicon + MLX. End-to-end round trip (speech → LLM → TTS) is consistently under **1 second**, even while handling barge-in.

## Click below for demo
[![Watch the demo](https://img.youtube.com/vi/6IEK2fXB_ok/0.jpg)](https://www.youtube.com/watch?v=6IEK2fXB_ok)

## Highlights
- On-device VAD, STT (Whisper), LLM, and Kokoro TTS sharing a single MLX runtime
- Sentence-streaming LLM responses with immediate, cancellable TTS playback
- Client-side AudioWorklet correlator for robust echo suppression and barge-in
- Rolling audio buffer to preserve context around interruptions

## Quick Start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```
Then open `http://localhost:8000` and hit **Start Listening**.

## Configuration
- `config.py`: audio rate, VAD thresholds, queue sizes, model choices
- `index.html`: `INTERRUPTION` object for interruption sensitivity
- `requirements.txt`: pin runtime dependencies for MLX + Kokoro

## Architecture (at a glance)
```
Mic (16 kHz) → Correlator → VAD State Machine → Segment Queue
      ↓                                         ↓
  WebSocket ↔ Browser UI                 MLX Whisper STT
                                               ↓
                                    MLX LLM → Kokoro TTS
                                               ↓
                                        Playback + barge-in feedback
```

## Repository Map
- `server.py` – FastAPI WebSocket server + pipeline orchestration
- `audio_buffer.py`, `vad_detector.py` – speech segmentation utilities
- `transcriber.py`, `llm_handler.py`, `tts_handler.py` – model wrappers
- `index.html`, `correlator.js` – interactive UI with barge-in logic

## Credits
Silero VAD, Whisper, MLX, Kokoro TTS, and Pipecat SmartTurn for EoU detection. MIT licensed.



================================================
FILE: audio_buffer.py
================================================
# audio_buffer.py
"""Simplified audio buffer for speech segments"""
import numpy as np
import wave
from typing import Optional, List
from enum import Enum
from config import CHUNK_SIZE, SAMPLE_RATE, SAFETY_CHUNKS_BEFORE


class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


class AudioBuffer:
    """Manages audio buffering with safety margins"""
    
    def __init__(self):
        self.pre_buffer: List[np.ndarray] = []
        self.active_segment: List[np.ndarray] = []
        self.is_capturing = False
    
    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        """Add chunk based on state"""
        chunk = chunk.copy()
        
        if state == SpeechState.QUIET:
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                self.is_capturing = True
                self.active_segment = self.pre_buffer.copy()
            self.active_segment.append(chunk)
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.SPEAKING:
            self.active_segment.append(chunk)
        
        elif state == SpeechState.STOPPING:
            self.active_segment.append(chunk)
    
    def get_segment(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        segment = np.concatenate(self.active_segment)
        self.active_segment = []
        self.is_capturing = False
        return segment


def save_audio_to_wav(audio: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE):
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())


def split_audio_into_chunks(audio: np.ndarray, chunk_size: int = CHUNK_SIZE) -> List[np.ndarray]:
    num_chunks = len(audio) // chunk_size
    return [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]



================================================
FILE: config.py
================================================
# config.py
"""Configuration constants for the voice agent"""

# Audio Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz
CHANNELS = 1

# VAD Configuration
VAD_ALPHA = 0.1
VAD_START_THRESHOLD = 0.3
VAD_SPEAKING_THRESHOLD = 0.5
VAD_STOP_THRESHOLD = 0.3
VAD_QUIET_THRESHOLD = 0.05
VAD_STATE_SHAPE = (2, 1, 128)
VAD_CONTEXT_SIZE = 64

# Speech Segmentation
SAFETY_CHUNKS_BEFORE = 4

# End-of-Utterance Detection
EOU_MIN_SAMPLES = 4 * SAMPLE_RATE
EOU_OPTIMAL_SAMPLES = 8 * SAMPLE_RATE
EOU_CONFIDENCE_THRESHOLD = 0.9

# Processing Configuration
MAX_TRANSCRIPTION_QUEUE_SIZE = 256
MIN_SEGMENT_DURATION = 0.3

# LLM Streaming Configuration
LLM_MIN_TOKENS_FOR_TTS = 3
LLM_SENTENCE_DELIMITERS = ".!?"
LLM_MAX_TOKENS = 256

# TTS Configuration
TTS_SAMPLE_RATE = 24000

# Feature Flags
ENABLE_RECORDING = False
ENABLE_TRANSCRIPTION = True

# Model Paths
VAD_MODEL_PATH = "models/silero_vad.onnx"
EOU_MODEL_PATH = "models/smart_turn_v3.onnx"
WHISPER_MODEL = "mlx-community/whisper-small.en-mlx-q4"
LLM_MODEL = "mlx-community/LFM2-1.2B-4bit"
# LLM_MODEL = "mlx-community/Qwen3-0.6B-8bit"
TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0



================================================
FILE: correlator.js
================================================
// correlator.js - AudioWorklet for echo cancellation via reference correlation
class Correlator extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = 480; // 10ms @48k (or proportionally less at 16k)
  }

  _rms(buf) {
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  }

  _corr(x, y) {
    // Cosine similarity = normalized correlation
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i < x.length; i++) {
      const a = x[i], b = y[i];
      xy += a * b;
      xx += a * a;
      yy += b * b;
    }
    return xy / (Math.sqrt(xx * yy) + 1e-9);
  }

  process(inputs, outputs) {
    const mic = inputs[0][0];       // Microphone input
    const ref = inputs[1]?.[0];     // Reference (TTS output)
    const out = outputs[0][0];      // Pass-through output

    if (!mic) return true;

    // Pass-through mic
    if (out) {
      for (let i = 0; i < mic.length; i++) out[i] = mic[i];
    }

    // Calculate correlation and RMS levels
    if (ref && ref.length === mic.length) {
      const corr = this._corr(mic, ref);   // -1..1
      const micRms = this._rms(mic);
      const refRms = this._rms(ref);
      this.port.postMessage({ 
        corr: Math.max(0, corr), 
        micRms, 
        refRms 
      });
    } else {
      const micRms = this._rms(mic);
      this.port.postMessage({ corr: 0, micRms, refRms: 0 });
    }
    return true;
  }
}

registerProcessor('correlator', Correlator);


================================================
FILE: index.html
================================================
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 20px;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            min-width: 450px;
            max-width: 700px;
            width: 100%;
        }
        h1 { color: #333; margin-bottom: 30px; }
        .state {
            font-size: 2.5em;
            font-weight: bold;
            margin: 20px 0;
            padding: 20px;
            border-radius: 15px;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }
        .state.quiet { background: #f5f5f5; color: #666; border: 2px solid #ddd; }
        .state.starting { background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; animation: pulse 1.5s ease-in-out infinite; }
        .state.speaking { background: #d4edda; color: #155724; border: 2px solid #4CAF50; box-shadow: 0 0 20px rgba(76, 175, 80, 0.3); }
        .state.stopping { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; animation: fadeOut 1s ease-in-out infinite alternate; }
        .state.processing { background: #e3f2fd; color: #0d47a1; border: 2px solid #2196F3; animation: pulse 1.5s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.05); opacity: 1; } }
        @keyframes fadeOut { 0% { opacity: 1; } 100% { opacity: 0.6; } }
        @keyframes listeningBlink { 0% { opacity: 1; } 100% { opacity: 0.5; } }
        .transcript-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            min-height: 120px;
            text-align: left;
            border: 2px solid #e0e0e0;
            max-height: 300px;
            overflow-y: auto;
        }
        .transcript-label { font-size: 14px; color: #666; font-weight: bold; margin-bottom: 10px; text-transform: uppercase; }
        .transcript-text { font-size: 18px; line-height: 1.6; word-wrap: break-word; color: #333; white-space: pre-wrap; }
        .message { margin: 10px 0; padding: 10px 15px; border-radius: 10px; }
        .message.user { background: #e3f2fd; text-align: left; }
        .message.assistant { background: #f1f8e9; text-align: left; }
        .message-label { font-weight: bold; font-size: 14px; margin-bottom: 5px; }
        .message.user .message-label { color: #1976d2; }
        .message.assistant .message-label { color: #558b2f; }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric { background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; text-align: center; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px; font-weight: 600; }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .status-line { font-size: 16px; color: #333; margin-top: 10px; }
        .controls { display: flex; gap: 15px; justify-content: center; margin: 30px 0; flex-wrap: wrap; }
        button {
            padding: 15px 25px;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s;
            min-width: 120px;
        }
        button:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .start { background: #4CAF50; color: white; }
        .stop { background: #f44336; color: white; }
        .connection, .listening-indicator {
            position: fixed;
            top: 20px;
            padding: 8px 16px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 14px;
            z-index: 1000;
        }
        .connection { right: 20px; }
        .listening-indicator { left: 20px; background: #f44336; color: white; display: none; }
        .listening-indicator.active { display: block; animation: listeningBlink 1s ease-in-out infinite alternate; }
        .connected { background: #4CAF50; color: white; }
        .disconnected { background: #f44336; color: white; }
    </style>
</head>
<body>
    <div class="connection disconnected" id="connection">Disconnected</div>
    <div class="listening-indicator" id="listeningIndicator">🎤 LISTENING</div>
    <div class="container">
        <h1>🤖 Voice Agent</h1>
        <div class="state quiet" id="stateDisplay">🔇 QUIET</div>
        <div class="transcript-container">
            <div class="transcript-label">💬 Conversation</div>
            <div class="transcript-text" id="transcriptDisplay">
                <em style="color: #999;">Start speaking to begin conversation...</em>
            </div>
        </div>
        <div class="metrics">
            <div class="metric"><div class="metric-label">VAD Probability</div><div class="metric-value" id="vadProb">0.000</div></div>
            <div class="metric"><div class="metric-label">STT Latency</div><div class="metric-value" id="sttLatency">--</div></div>
            <div class="metric"><div class="metric-label">LLM First Token</div><div class="metric-value" id="llmLatency">--</div></div>
            <div class="metric"><div class="metric-label">TTS First Audio</div><div class="metric-value" id="ttsLatency">--</div></div>
        </div>
        <div class="status-line">Status: <span id="statusValue">Ready</span></div>
        <div class="controls"><button class="start" id="startBtn">Start Listening</button></div>
    </div>
    <audio id="audioPlayer" style="display: none;"></audio>
    <script>
        const EVENTS = {MEDIA: 'media', TEXT: 'text', START: 'start', STOP: 'stop', STATE: 'state', METRICS: 'metrics', INTERRUPT: 'interrupt'};
        const STATE_LABELS = {quiet: '🔇 QUIET', started: '🎤 STARTED', speaking: '🗣️ SPEAKING', stop: '✋ STOPPING', processing: '⏳ PROCESSING'};
        const STATE_CLASSES = {quiet: 'quiet', started: 'starting', speaking: 'speaking', stop: 'stopping', processing: 'processing'};
        const INTERRUPTION = {RMS: 0.012, EXTRA: 0.008, RATIO: 1.20, ABS: 0.035, MIN_FRAMES: 2, REQUIRED_FRAMES: 2};

        const dom = {
            connection: document.getElementById('connection'),
            listeningIndicator: document.getElementById('listeningIndicator'),
            stateDisplay: document.getElementById('stateDisplay'),
            startBtn: document.getElementById('startBtn'),
            transcript: document.getElementById('transcriptDisplay'),
            status: document.getElementById('statusValue'),
            vad: document.getElementById('vadProb'),
            stt: document.getElementById('sttLatency'),
            llm: document.getElementById('llmLatency'),
            tts: document.getElementById('ttsLatency'),
            audio: document.getElementById('audioPlayer')
        };

        let ws = null, audioContext = null, processorNode = null, mediaStream = null, correlatorNode = null, ttsSourceNode = null;
        let isListening = false, micSuppressed = false, botSpeaking = false, interruptionInProgress = false;
        let conversation = [];
        let echoState = { corr: 0, micRms: 0, refRms: 0, isEcho: false };
        
        // Rolling buffer to capture audio before interruption detection
        const audioBuffer = {
            frames: [],
            maxFrames: 30, // ~1 second at 512 samples/frame @ 16kHz
            add(frame) {
                this.frames.push(Float32Array.from(frame));
                if (this.frames.length > this.maxFrames) {
                    this.frames.shift();
                }
            },
            getRecent(numFrames) {
                const start = Math.max(0, this.frames.length - numFrames);
                return this.frames.slice(start);
            },
            clear() {
                this.frames = [];
            }
        };

        const interruption = {
            frames: 0, pending: null, avg: 0, peak: 0, observed: 0,
            reset() { this.frames = 0; this.pending = null; this.avg = 0; this.peak = 0; this.observed = 0; },
            analyse(frame) {
                const value = rms(frame);
                this.observed += 1;
                this.avg = this.avg === 0 ? value : (0.9 * this.avg + 0.1 * value);
                this.peak = Math.max(this.peak * 0.95, value);
                if (this.observed < INTERRUPTION.MIN_FRAMES) return null;
                const baseline = Math.max(this.avg, this.peak);
                const threshold = Math.max(INTERRUPTION.RMS, this.avg + INTERRUPTION.EXTRA, baseline * INTERRUPTION.RATIO, INTERRUPTION.ABS);
                if (value > threshold) {
                    this.frames += 1;
                    if (this.frames === 1) this.pending = Float32Array.from(frame);
                    if (this.frames >= INTERRUPTION.REQUIRED_FRAMES) {
                        const segments = this.pending ? [this.pending, Float32Array.from(frame)] : [Float32Array.from(frame)];
                        this.reset();
                        return segments;
                    }
                } else {
                    this.frames = 0;
                    this.pending = null;
                }
                return null;
            }
        };

        const playback = {
            queue: [], active: false, url: null, finalizer: null,
            enqueue(buffer, mime = 'audio/wav') {
                if (!(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) return;
                this.queue.push({buffer, mime});
                if (!this.active) this._playNext();
            },
            stop({notifyServer = false, updateUi = true} = {}) {
                console.log('[playback] Stopping - active:', this.active, 'queued:', this.queue.length);
                const hadAudio = this.active || this.queue.length > 0;
                
                // CRITICAL: Clear queue first to prevent new items from playing
                this.queue.length = 0;
                
                // CRITICAL: Immediately stop audio element SYNCHRONOUSLY
                if (dom.audio && !dom.audio.paused) {
                    dom.audio.pause();
                    dom.audio.currentTime = 0;
                    dom.audio.onended = null;
                    console.log('[playback] Audio element stopped immediately');
                }
                
                // Set flags BEFORE finalizer
                this.active = false;
                botSpeaking = false;
                interruptionInProgress = false;
                
                if (this.finalizer) {
                    const finalize = this.finalizer;
                    this.finalizer = null;
                    finalize('interrupted', {updateUi});
                } else if (hadAudio) {
                    this._finalize('interrupted', {updateUi});
                } else if (updateUi) {
                    setIdleState();
                }
                if (notifyServer) sendJson({event: EVENTS.STOP, target: 'playback'});
                return hadAudio;
            },
            _playNext() {
                const next = this.queue.shift();
                if (!next) {
                    this.active = false;
                    botSpeaking = false;
                    interruptionInProgress = false;
                    if (ttsSourceNode) {
                        ttsSourceNode.disconnect();
                        ttsSourceNode = null;
                    }
                    setMicSuppressed(false);
                    setIdleState();
                    return;
                }
                this.active = true;
                botSpeaking = true;
                interruptionInProgress = false;
                setMicSuppressed(true);
                updateStateDisplay('speaking');
                updateStatusForState('speaking');
                if (this.url) URL.revokeObjectURL(this.url);
                this.url = URL.createObjectURL(new Blob([next.buffer], {type: next.mime}));
                dom.audio.src = this.url;
                
                // Connect TTS audio as reference signal to correlator
                if (audioContext && correlatorNode && dom.audio.captureStream) {
                    try {
                        ttsSourceNode = audioContext.createMediaStreamSource(dom.audio.captureStream());
                        ttsSourceNode.connect(correlatorNode, 0, 1); // Connect to input 1 (reference)
                    } catch (e) {
                        console.warn('Could not capture TTS stream:', e);
                    }
                }
                
                this.finalizer = (reason = 'natural', options) => this._finalize(reason, options);
                dom.audio.onended = () => this.finalizer('natural');
                dom.audio.play().catch(error => {
                    console.error('Error playing audio:', error);
                    this.finalizer('interrupted');
                });
            },
            _finalize(reason = 'natural', {updateUi = true} = {}) {
                dom.audio.onended = null;
                dom.audio.pause();
                dom.audio.currentTime = 0;
                dom.audio.src = '';
                if (this.url) {
                    URL.revokeObjectURL(this.url);
                    this.url = null;
                }
                if (ttsSourceNode) {
                    ttsSourceNode.disconnect();
                    ttsSourceNode = null;
                }
                this.active = false;
                botSpeaking = false;
                this.finalizer = null;
                setMicSuppressed(false);
                if (reason === 'interrupted') this.queue.length = 0;
                if (this.queue.length > 0 && reason !== 'interrupted') {
                    this._playNext();
                } else if (updateUi) {
                    setIdleState();
                }
            }
        };

        function formatLatency(s) {
            return (typeof s !== 'number' || !Number.isFinite(s) || s < 0) ? '--' : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(2)} s`;
        }
        function setLatency(el, s) { if (el) el.textContent = s === null ? '--' : formatLatency(s); }
        function resetLatencies() { setLatency(dom.stt, null); setLatency(dom.llm, null); setLatency(dom.tts, null); }
        function updateStateDisplay(state) {
            dom.stateDisplay.className = `state ${STATE_CLASSES[state] || 'quiet'}`;
            dom.stateDisplay.textContent = STATE_LABELS[state] || state.toUpperCase();
        }
        function updateStatusForState(state, responding = false) {
            if (responding) {
                dom.status.textContent = dom.audio.paused ? 'Processing' : 'Speaking';
            } else {
                const statusByState = {started: 'Listening', speaking: 'Listening', stop: 'Processing', quiet: isListening ? 'Ready' : 'Stopped'};
                dom.status.textContent = statusByState[state] || dom.status.textContent;
            }
        }
        function setIdleState() {
            const state = isListening ? 'started' : 'quiet';
            updateStateDisplay(state);
            updateStatusForState(state);
        }
        function rms(frame) {
            let energy = 0;
            for (let i = 0; i < frame.length; i++) energy += frame[i] * frame[i];
            return Math.sqrt(energy / frame.length);
        }
        function setMicSuppressed(flag) {
            if (micSuppressed === flag) return;
            micSuppressed = flag;
            if (!flag) interruption.reset();
        }
        function float32ToBase64(float32Array) {
            const bytes = new Uint8Array(float32Array.buffer.slice(0));
            let binary = '';
            for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
            return btoa(binary);
        }
        function base64ToArrayBuffer(base64) {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return bytes.buffer;
        }
        function addMessage(role, text, overwrite = false) {
            const trimmed = (text || '').trim();
            if (!trimmed) return;
            const last = conversation[conversation.length - 1];
            if (last && last.role === role) {
                last.text = overwrite ? trimmed : `${last.text} ${trimmed}`.trim();
                const node = dom.transcript.lastElementChild;
                if (node && node.lastChild) node.lastChild.textContent = last.text;
                dom.transcript.scrollTop = dom.transcript.scrollHeight;
                return;
            }
            if (conversation.length === 0) dom.transcript.textContent = '';
            conversation.push({role, text: trimmed});
            const wrapper = document.createElement('div');
            wrapper.className = `message ${role}`;
            const label = document.createElement('div');
            label.className = 'message-label';
            label.textContent = role === 'assistant' ? 'Assistant' : 'You';
            const body = document.createElement('div');
            body.textContent = trimmed;
            wrapper.append(label, body);
            dom.transcript.appendChild(wrapper);
            dom.transcript.scrollTop = dom.transcript.scrollHeight;
        }
        function sendJson(payload) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload)); }
        async function handleMediaEvent(data) {
            if (!data || !data.audio) return;
            try { playback.enqueue(base64ToArrayBuffer(data.audio), data.mime || 'audio/wav'); }
            catch (error) { console.error('Failed to handle media event:', error); }
        }
        function handleTextEvent(data) { addMessage(data.role === 'assistant' ? 'assistant' : 'user', data.text, Boolean(data.complete)); }
        function handleStateEvent(data) {
            const state = data.state || 'quiet';
            if (state === 'started' && (playback.active || playback.queue.length > 0)) {
                console.log('[state] User started speaking, stopping playback');
                playback.stop({notifyServer: true, updateUi: false});
            }
            
            // Reset interruption flag when bot finishes responding
            if (data.responding === false) {
                interruptionInProgress = false;
            }
            
            updateStateDisplay(state);
            updateStatusForState(state, Boolean(data.responding));
            if (typeof data.vad === 'number') dom.vad.textContent = data.vad.toFixed(3);
            dom.listeningIndicator.className = data.listening ? 'listening-indicator active' : 'listening-indicator';
        }
        function handleMetricsEvent(data) {
            if (!data || !data.metrics) return;
            const {stt, llm, tts} = data.metrics;
            if (stt) dom.stt.textContent = stt.status === 'running' ? '...' : formatLatency(stt.latency ?? null);
            if (llm) setLatency(dom.llm, llm.first_token ?? null);
            if (tts) setLatency(dom.tts, tts.first_audio ?? null);
        }
        function handleInterruptEvent() { playback.stop({updateUi: true}); setMicSuppressed(false); }
        function teardownAudio() {
            if (correlatorNode) {
                correlatorNode.port.onmessage = null;
                correlatorNode.disconnect();
                correlatorNode = null;
            }
            if (ttsSourceNode) {
                ttsSourceNode.disconnect();
                ttsSourceNode = null;
            }
            if (processorNode) {
                processorNode.disconnect();
                processorNode.onaudioprocess = null;
                processorNode = null;
            }
            if (audioContext) {
                audioContext.close().catch(() => {});
                audioContext = null;
            }
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
        }
        async function setupAudioProcessing(stream) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
            mediaStream = stream;
            if (audioContext.state === 'suspended') {
                await audioContext.resume().catch(error => console.error('AudioContext resume failed:', error));
            }

            // Load correlator worklet for echo cancellation
            try {
                await audioContext.audioWorklet.addModule('correlator.js');
                
                // Create correlator node with 2 inputs (mic + reference)
                correlatorNode = new AudioWorkletNode(audioContext, 'correlator', {
                    numberOfInputs: 2,
                    numberOfOutputs: 1,
                    outputChannelCount: [1]
                });

                // Listen to correlation metrics
                correlatorNode.port.onmessage = (e) => {
                    const { corr, micRms, refRms } = e.data;
                    echoState = {
                        corr,
                        micRms,
                        refRms,
                        // Echo if high correlation AND reference is playing AND mic is quiet
                        isEcho: corr > 0.30 && refRms > 0.01 && micRms < 0.05
                    };
                };

                // Connect mic to correlator input 0
                const micSource = audioContext.createMediaStreamSource(stream);
                micSource.connect(correlatorNode, 0, 0);

                // Setup TTS audio element source (will be connected when playing)
                const audioEl = dom.audio;
                if (!audioEl.captureStream) {
                    console.warn('captureStream not supported, using fallback');
                }

                // Create processor for sending audio to server
                processorNode = audioContext.createScriptProcessor(512, 1, 1);
                correlatorNode.connect(processorNode);
                processorNode.connect(audioContext.destination);

                processorNode.onaudioprocess = (event) => {
                    if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
                    
                    const frame = Float32Array.from(event.inputBuffer.getChannelData(0));
                    
                    // Always buffer recent frames for potential interruption recovery
                    audioBuffer.add(frame);
                    
                    // Check if this is echo using correlation
                    if (botSpeaking) {
                        // Check for interruption even during echo
                        const burst = interruption.analyse(frame);
                        if (burst) {
                            console.log('[interrupt] User interrupting - micRms:', echoState.micRms, 'corr:', echoState.corr);
                            
                            // IMMEDIATELY stop playback
                            botSpeaking = false;
                            playback.stop({notifyServer: true, updateUi: false});
                            
                            // Send buffered frames before interruption (last ~500ms)
                            const preBuffer = audioBuffer.getRecent(15); // ~500ms of context
                            console.log('[interrupt] Sending', preBuffer.length, 'buffered frames +', burst.length, 'burst frames');
                            preBuffer.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                            burst.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                            audioBuffer.clear();
                        }
                        
                        // Don't send echo frames to server (only send on interruption above)
                        if (echoState.isEcho) return;
                    }

                    // Normal flow: send audio to server
                    if (micSuppressed && !botSpeaking) {
                        const burst = interruption.analyse(frame);
                        if (burst) {
                            setMicSuppressed(false);
                            playback.stop({notifyServer: true});
                            const preBuffer = audioBuffer.getRecent(15);
                            preBuffer.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                            burst.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                            audioBuffer.clear();
                        }
                        return;
                    }

                    // Send clean audio (not echo) to server
                    if (!echoState.isEcho) {
                        sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(frame)});
                    }
                };

                console.log('Audio processing with echo cancellation initialized');
            } catch (error) {
                console.error('Failed to load correlator worklet, using fallback:', error);
                // Fallback to simple processing without correlator
                setupSimpleAudioProcessing(stream);
            }
        }

        function setupSimpleAudioProcessing(stream) {
            const source = audioContext.createMediaStreamSource(stream);
            processorNode = audioContext.createScriptProcessor(512, 1, 1);
            processorNode.onaudioprocess = (event) => {
                if (!isListening || !ws || ws.readyState !== WebSocket.OPEN) return;
                const frame = Float32Array.from(event.inputBuffer.getChannelData(0));
                
                if (interruptionInProgress) {
                    sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(frame)});
                    audioBuffer.add(frame);
                    return;
                }
                
                audioBuffer.add(frame);
                
                if (botSpeaking) {
                    const burst = interruption.analyse(frame);
                    if (burst) {
                        console.log('[interrupt] User interrupting bot');
                        interruptionInProgress = true;
                        botSpeaking = false;
                        playback.stop({notifyServer: true, updateUi: false});
                        const preBuffer = audioBuffer.getRecent(15);
                        preBuffer.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                        burst.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                        audioBuffer.clear();
                    }
                    return;
                }
                if (micSuppressed) {
                    const burst = interruption.analyse(frame);
                    if (burst) {
                        setMicSuppressed(false);
                        playback.stop({notifyServer: true, updateUi: false});
                        const preBuffer = audioBuffer.getRecent(15);
                        preBuffer.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                        burst.forEach(chunk => sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(chunk)}));
                        audioBuffer.clear();
                    }
                    return;
                }
                sendJson({event: EVENTS.MEDIA, audio: float32ToBase64(frame)});
            };
            source.connect(processorNode);
            processorNode.connect(audioContext.destination);
        }
        async function startListening() {
            if (isListening) return;
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: {sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true}
                });
                setupAudioProcessing(stream);
                isListening = true;
                resetLatencies();
                sendJson({event: EVENTS.START});
                dom.listeningIndicator.className = 'listening-indicator active';
                updateStateDisplay('quiet');
                updateStatusForState('quiet');
                dom.startBtn.textContent = 'Stop Listening';
                dom.startBtn.className = 'stop';
                dom.startBtn.onclick = stopListening;
            } catch (error) {
                alert('Microphone access required');
                console.error('Microphone error:', error);
            }
        }
        function stopListening() {
            if (!isListening) return;
            isListening = false;
            botSpeaking = false;
            interruptionInProgress = false;
            setMicSuppressed(false);
            teardownAudio();
            playback.stop({updateUi: true});
            sendJson({event: EVENTS.STOP});
            dom.listeningIndicator.className = 'listening-indicator';
            updateStateDisplay('quiet');
            updateStatusForState('quiet');
            dom.startBtn.textContent = 'Start Listening';
            dom.startBtn.className = 'start';
            dom.startBtn.onclick = startListening;
        }
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            ws.onopen = () => { dom.connection.textContent = 'Connected'; dom.connection.className = 'connection connected'; };
            ws.onclose = () => {
                dom.connection.textContent = 'Disconnected';
                dom.connection.className = 'connection disconnected';
                playback.stop({updateUi: true});
                stopListening();
                setTimeout(connect, 2000);
            };
            ws.onerror = error => console.error('WebSocket error:', error);
            ws.onmessage = async (event) => {
                let data;
                try { data = JSON.parse(event.data); } catch (error) { console.error('Invalid socket message:', error); return; }
                if (!data || !data.event) return;
                switch (data.event) {
                    case EVENTS.TEXT: handleTextEvent(data); break;
                    case EVENTS.MEDIA: await handleMediaEvent(data); break;
                    case EVENTS.INTERRUPT: 
                        console.log('[interrupt] Bot interrupted by user');
                        handleInterruptEvent(); 
                        break;
                    case EVENTS.STATE: handleStateEvent(data); break;
                    case EVENTS.METRICS: handleMetricsEvent(data); break;
                    case EVENTS.STOP:
                        if (data.target === 'playback') handleInterruptEvent();
                        else if (data.target === 'listening') stopListening();
                        if (data.state) { updateStateDisplay(data.state); updateStatusForState(data.state, Boolean(data.responding)); }
                        break;
                    case EVENTS.START:
                        if (data.target === 'listening') { updateStateDisplay('quiet'); updateStatusForState('quiet'); }
                        if (data.state) { updateStateDisplay(data.state); updateStatusForState(data.state, Boolean(data.responding)); }
                        break;
                }
            };
        }
        dom.startBtn.onclick = startListening;
        dom.vad.textContent = '0.000';
        updateStateDisplay('quiet');
        dom.status.textContent = 'Ready';
        resetLatencies();
        connect();
    </script>
</body>
</html>


================================================
FILE: llm_handler.py
================================================
# llm_handler.py
"""LLM handler with true MLX streaming support"""
from typing import List, Dict, Iterator
from mlx_lm import load, generate, stream_generate
import threading
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_SENTENCE_DELIMITERS, LLM_MIN_TOKENS_FOR_TTS


class LLMHandler:
    """Handles LLM inference with streaming support"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    ):
        print(f"Loading LLM: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = system_prompt
        # CRITICAL: Add lock to prevent concurrent access to MLX model
        self._generation_lock = threading.Lock()
        print("LLM loaded successfully")
    
    def stream_response(self, conversation_history: List[Dict[str, str]]) -> Iterator[str]:
        """
        Stream LLM response with true token-by-token generation from MLX.
        Yields complete sentences as they're formed.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            prompt = self._format_prompt(messages)
            print("Generating LLM response (streaming)...")
            
            # Buffer for accumulating tokens into sentences
            buffer = ""
            
            # Run generation under lock to avoid concurrent model access
            with self._generation_lock:
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=LLM_MAX_TOKENS
                ):
                    # response.text contains the newly generated token(s)
                    token_text = response.text
                    buffer += token_text
                    
                    # Check if we've hit a sentence delimiter
                    if any(delimiter in token_text for delimiter in LLM_SENTENCE_DELIMITERS):
                        # Extract complete sentences from buffer
                        sentences = self._extract_complete_sentences(buffer)
                        
                        for sentence in sentences['complete']:
                            if sentence:
                                print(f"[LLM] Sentence: {sentence}")
                                yield sentence
                        
                        # Keep incomplete part in buffer
                        buffer = sentences['remaining']
            
            # Yield any remaining text in buffer
            if buffer.strip():
                print(f"[LLM] Final: {buffer.strip()}")
                yield buffer.strip()
        
        except Exception as e:
            print(f"LLM streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield "I apologize, but I encountered an error."
    
    def stream_response_batched(self, conversation_history: List[Dict[str, str]]) -> Iterator[str]:
        """
        Legacy batched response (generates all tokens first, then streams sentences).
        Kept for backwards compatibility or fallback.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            prompt = self._format_prompt(messages)
            print("Generating LLM response (batched)...")
            
            # Run generation under lock to avoid concurrent model access
            with self._generation_lock:
                full_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=LLM_MAX_TOKENS
                )
            
            full_text = (full_text or "").strip()
            if not full_text:
                print("[LLM] Empty generation")
                yield "I'm sorry, I couldn't think of anything to say."
                return
            
            for sentence in self._split_into_sentences(full_text):
                if sentence:
                    print(f"[LLM] Sentence: {sentence}")
                    yield sentence
        
        except Exception as e:
            print(f"LLM streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield "I apologize, but I encountered an error."
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        
        # Fallback formatting
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted
    
    def _extract_complete_sentences(self, buffer: str) -> Dict[str, any]:
        """
        Extract complete sentences from buffer while preserving incomplete text.
        Returns dict with 'complete' (list of sentences) and 'remaining' (buffer).
        """
        complete_sentences = []
        remaining = buffer
        
        # Find all sentence delimiters
        last_delimiter_pos = -1
        for i, char in enumerate(buffer):
            if char in LLM_SENTENCE_DELIMITERS:
                last_delimiter_pos = i
        
        # If we found at least one delimiter, split there
        if last_delimiter_pos >= 0:
            complete_part = buffer[:last_delimiter_pos + 1]
            remaining = buffer[last_delimiter_pos + 1:]
            
            # Split complete part into sentences
            sentences = self._split_into_sentences(complete_part)
            
            # Apply minimum token threshold to avoid tiny chunks
            for sentence in sentences:
                if not complete_sentences:
                    complete_sentences.append(sentence)
                else:
                    word_count = len(sentence.split())
                    if word_count < LLM_MIN_TOKENS_FOR_TTS:
                        # Merge with previous sentence if too short
                        complete_sentences[-1] = (complete_sentences[-1] + " " + sentence).strip()
                    else:
                        complete_sentences.append(sentence)
        
        return {
            'complete': complete_sentences,
            'remaining': remaining
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split generated text into sensible sentence chunks"""
        raw_sentences: List[str] = []
        buffer = ""
        
        for char in text:
            buffer += char
            if char in LLM_SENTENCE_DELIMITERS:
                chunk = buffer.strip()
                if chunk:
                    raw_sentences.append(chunk)
                buffer = ""
        
        if buffer.strip():
            raw_sentences.append(buffer.strip())
        
        if not raw_sentences:
            return [text]
        
        merged: List[str] = []
        for sentence in raw_sentences:
            if not merged:
                merged.append(sentence)
                continue
            
            word_count = len(sentence.split())
            if word_count < LLM_MIN_TOKENS_FOR_TTS:
                merged[-1] = (merged[-1] + " " + sentence).strip()
            else:
                merged.append(sentence)
        
        return merged


================================================
FILE: requirements.txt
================================================
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
websockets==12.0

# Machine Learning & Audio Processing
numpy==1.26.3
onnxruntime==1.16.3
transformers==4.37.2

# MLX Whisper (Apple Silicon optimized)
mlx-whisper==0.3.1
mlx==0.15.0


================================================
FILE: server.py
================================================
# server.py
"""Optimized voice agent server with MLX concurrency protection"""
import json
import base64
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from config import *
from audio_buffer import AudioBuffer, SpeechState, save_audio_to_wav, split_audio_into_chunks
from vad_detector import VADDetector, EndOfUtteranceDetector
from llm_handler import LLMHandler
from transcriber import RealtimeTranscriber
from tts_handler import TTSHandler


# =============================================================================
# Utilities
# =============================================================================

def encode_audio(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_float32_audio(data: str) -> Optional[np.ndarray]:
    try:
        audio_bytes = base64.b64decode(data.encode("ascii"))
        return np.frombuffer(audio_bytes, dtype=np.float32) if len(audio_bytes) % 4 == 0 else None
    except:
        return None


# =============================================================================
# Events
# =============================================================================

class PipelineEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIBE = "transcribe"
    RESPOND = "respond"


# =============================================================================
# Resources (Singleton)
# =============================================================================

class PipelineResources:
    """Global resources initialized once"""
    
    def __init__(self):
        print("Initializing pipeline...")
        
        self.transcriber = RealtimeTranscriber() if ENABLE_TRANSCRIPTION else None
        self.llm_handler = LLMHandler()
        self.tts_handler = TTSHandler()
        
        # CRITICAL: Global lock for MLX operations (Whisper + LLM share MLX runtime)
        # This prevents heap corruption from concurrent MLX access
        self.mlx_lock = asyncio.Lock()
        
        # Warm up transcriber
        if self.transcriber:
            dummy_audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.001
            try:
                self.transcriber.transcribe(dummy_audio)
                print("Transcriber warmed up")
            except Exception as e:
                print(f"Warmup warning: {e}")
        
        print("Pipeline ready\n")


# =============================================================================
# Speech Detector
# =============================================================================

class SpeechDetector:
    """Manages VAD state machine and audio segmentation"""
    
    def __init__(self):
        self.vad = VADDetector()
        self.eou = EndOfUtteranceDetector() if ENABLE_TRANSCRIPTION else None
        self.buffer = AudioBuffer()
        
        self.state = SpeechState.QUIET
        self.is_listening = False
        self.is_responding = False
        self.user_speaking = False  # Tracks if user is mid-utterance (across pauses)
        
        self.recording: List[np.ndarray] = []
        self.segment_count = 0
        self.current_segment: Optional[np.ndarray] = None
    
    def start_listening(self):
        self.is_listening = True
        self.recording = []
        self.segment_count = 0
        self.user_speaking = False
        print("[detector] Started")
    
    def stop_listening(self):
        self.is_listening = False
        self.user_speaking = False
        
        if ENABLE_RECORDING and self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_audio_to_wav(np.concatenate(self.recording), f"recording_{timestamp}.wav")
            print(f"[detector] Saved recording")
        
        print(f"[detector] Stopped (segments: {self.segment_count})\n")
    
    def process_chunk(self, chunk: np.ndarray) -> List[PipelineEvent]:
        if not self.is_listening:
            return []
        
        if ENABLE_RECORDING:
            self.recording.append(chunk.copy())
        
        vad_prob = self.vad.process_chunk(chunk)
        self.buffer.add_chunk(chunk, self.state)
        
        # Always feed EOU detector so it has continuous context
        if self.eou:
            self.eou.add_audio(chunk)
        
        return self._update_state(vad_prob)
    
    def _update_state(self, vad_prob: float) -> List[PipelineEvent]:
        events = []
        prev_state = self.state
        
        if self.state == SpeechState.QUIET:
            if vad_prob >= VAD_START_THRESHOLD:
                self.state = SpeechState.STARTING
                self.user_speaking = True  # User started speaking
                events.append(PipelineEvent.SPEECH_START)
        
        elif self.state == SpeechState.STARTING:
            if vad_prob >= VAD_SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
            elif vad_prob < VAD_QUIET_THRESHOLD:
                self.state = SpeechState.QUIET
                self.user_speaking = False  # False start
        
        elif self.state == SpeechState.SPEAKING:
            if vad_prob < VAD_STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                # Capture segment for transcription
                self.current_segment = self.buffer.get_segment()
                if self.current_segment is not None:
                    print(f"[detector] Segment ({len(self.current_segment)/SAMPLE_RATE:.2f}s)")
                    events.append(PipelineEvent.TRANSCRIBE)
        
        elif self.state == SpeechState.STOPPING:
            vad_quiet = vad_prob < VAD_QUIET_THRESHOLD
            
            eou_confirms = not self.eou
            if self.eou and vad_quiet and self.eou.has_enough_audio():
                result = self.eou.detect()
                eou_confirms = result['ended'] and result['confidence'] > EOU_CONFIDENCE_THRESHOLD
                if eou_confirms:
                    print(f"[detector] EOU (conf: {result['confidence']:.2f})")
            
            if vad_quiet and eou_confirms:
                self.state = SpeechState.QUIET
                events.append(PipelineEvent.SPEECH_END)
                
                if self.user_speaking:
                    events.append(PipelineEvent.RESPOND)
                    self.user_speaking = False
                
                if self.eou:
                    self.eou.reset()
                self.current_segment = None
            
            # User resumed speaking (just a pause)
            elif vad_prob > VAD_SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                self.current_segment = None
                # user_speaking stays True
        
        if prev_state != self.state:
            print(f"[state] {prev_state.value} → {self.state.value} (vad: {vad_prob:.3f})")
        
        return events
    
    def get_state(self) -> dict:
        vad_value = float(self.vad.smoothed_prob)
        return {
            'state': self.state.value,
            # Provide both legacy and UI-friendly keys
            'vad': vad_value,
            'vad_prob': vad_value,
            'segments': self.segment_count,
            'listening': self.is_listening,
            'responding': self.is_responding
        }


# =============================================================================
# Voice Pipeline Orchestrator
# =============================================================================

class VoicePipeline:
    """Orchestrates the complete voice interaction pipeline"""
    
    def __init__(self, ws: WebSocket, resources: PipelineResources):
        self.ws = ws
        self.resources = resources
        self.detector = SpeechDetector()
        
        self.conversation: List[Dict[str, str]] = []
        self.transcription_queue = asyncio.Queue(maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE)
        self.accumulated_text = ""
        self.is_accumulating = False
        
        self._transcription_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None
        self._response_lock = asyncio.Lock()
        self._response_cancel_event: Optional[asyncio.Event] = None
    
    async def start(self):
        self._transcription_task = asyncio.create_task(self._transcription_worker())
        await self._send_state()
        print("[pipeline] Started\n")
    
    async def shutdown(self):
        if self._response_cancel_event and not self._response_cancel_event.is_set():
            self._response_cancel_event.set()
        
        await self._cancel_response_task()
        if self._response_cancel_event and self._response_cancel_event.is_set():
            self._response_cancel_event = None
        
        if self._transcription_task:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass
            self._transcription_task = None
        
        print("[pipeline] Shutdown\n")
    
    async def handle_message(self, payload: dict):
        event = payload.get("event")
        
        if event == "start":
            self.detector.start_listening()
            await self._send_state()
        
        elif event == "stop":
            if payload.get("target") == "playback":
                await self._interrupt_response(notify_client=True)
            else:
                self.detector.stop_listening()
            await self._send_state()
        
        elif event == "media":
            await self._handle_audio(payload.get("audio"))
        
        elif event == "interrupt":
            await self._interrupt_response(notify_client=True)
            await self._send_state()
    
    async def _handle_audio(self, audio_data: str):
        if not audio_data:
            return
        
        audio = decode_float32_audio(audio_data)
        if audio is None or len(audio) == 0:
            return
        
        for chunk in split_audio_into_chunks(audio, CHUNK_SIZE):
            events = self.detector.process_chunk(chunk)
            if events:
                await self._handle_events(events)
        
        await self._send_state()
    
    async def _handle_events(self, events: List[PipelineEvent]):
        if PipelineEvent.SPEECH_START in events:
            self.is_accumulating = True
            self.accumulated_text = ""
            print("[pipeline] Accumulating started")
            
            # Interrupt any ongoing response
            if self.detector.is_responding or (self._response_task and not self._response_task.done()):
                await self._interrupt_response(notify_client=True)
        
        for event in events:
            if event == PipelineEvent.TRANSCRIBE:
                await self._queue_transcription()
            
            elif event == PipelineEvent.RESPOND:
                await self._finalize_and_respond()
            
            elif event in [PipelineEvent.SPEECH_START, PipelineEvent.SPEECH_END]:
                await self.ws.send_text(json.dumps({"event": event.value}))
        
        await self._send_state()
    
    async def _queue_transcription(self):
        if not self.resources.transcriber or self.detector.current_segment is None:
            return
        
        segment = self.detector.current_segment
        if len(segment) < SAMPLE_RATE * MIN_SEGMENT_DURATION:
            return
        
        self.detector.segment_count += 1
        segment_id = self.detector.segment_count
        
        try:
            self.transcription_queue.put_nowait((segment_id, segment))
            print(f"[transcribe] Queued #{segment_id}")
        except asyncio.QueueFull:
            print(f"[transcribe] Queue full, dropped #{segment_id}")
    
    async def _transcription_worker(self):
        """Background worker for transcription with MLX lock protection"""
        if not self.resources.transcriber:
            return
        
        print("[transcribe] Worker started")
        
        while True:
            segment_id, audio = await self.transcription_queue.get()
            
            try:
                # Send STT status
                await self._send_metrics(stt={"status": "running", "segment_id": segment_id})
                
                # CRITICAL: Acquire MLX lock before transcription
                # This prevents concurrent Whisper + LLM access to MLX runtime
                stt_start = time.monotonic()
                async with self.resources.mlx_lock:
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.resources.transcriber.transcribe,
                        audio
                    )
                stt_latency = time.monotonic() - stt_start
                
                # Send STT completion
                await self._send_metrics(stt={"status": "completed", "segment_id": segment_id, "latency": stt_latency})
                
                if transcript and transcript.strip():
                    print(f"[transcribe] #{segment_id}: {transcript} ({stt_latency:.2f}s)")
                    
                    if self.is_accumulating:
                        # Accumulate transcript across pauses
                        if self.accumulated_text:
                            self.accumulated_text += " " + transcript.strip()
                        else:
                            self.accumulated_text = transcript.strip()
                        
                        print(f"[transcribe] Accumulated: {self.accumulated_text}")
                        
                        # Send partial transcript to client
                        await self.ws.send_text(json.dumps({
                            "event": "text",
                            "role": "user",
                            "text": transcript.strip(),
                            "segment_id": segment_id,
                            "partial": True
                        }))
            
            except Exception as e:
                print(f"[transcribe] Error: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                self.transcription_queue.task_done()
    
    async def _finalize_and_respond(self):
        """Finalize accumulated transcript and generate response"""
        # Wait for all pending transcriptions to complete
        if self.transcription_queue.qsize() > 0:
            print(f"[pipeline] Waiting for {self.transcription_queue.qsize()} transcriptions")
            await self.transcription_queue.join()
        
        self.is_accumulating = False
        
        if not self.accumulated_text:
            print("[pipeline] No text to respond to")
            return
        
        final_text = self.accumulated_text.strip()
        print(f"[pipeline] Final: {final_text}")
        
        # Add to conversation history (only once!)
        self.conversation.append({"role": "user", "content": final_text})
        
        # Send complete transcript to client
        await self.ws.send_text(json.dumps({
            "event": "text",
            "role": "user",
            "text": final_text,
            "complete": True
        }))
        
        self.accumulated_text = ""
        
        # Generate response
        await self._start_response()
    
    async def _start_response(self):
        async with self._response_lock:
            # Cancel any existing response task before starting a new one
            await self._cancel_response_task()
            
            # Fresh cancellation event for the new response
            self._response_cancel_event = asyncio.Event()
            self._response_task = asyncio.create_task(
                self._generate_response(self._response_cancel_event)
            )

    async def _cancel_response_task(self) -> bool:
        """Cancel the active response task if it exists."""
        task = self._response_task
        if not task:
            return False
        
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._response_task = None
        return True
    
    async def _generate_response(self, cancel_event: asyncio.Event):
        """Generate LLM response with streaming TTS.
        
        cancel_event: signals when downstream components should abandon work.
        """
        try:
            user_msgs = [m for m in self.conversation if m["role"] == "user"]
            if not user_msgs:
                return
            
            if cancel_event.is_set():
                return
            
            self.detector.is_responding = True
            await self._send_state()
            
            print(f"[response] Generating for: {user_msgs[-1]['content']}")
            
            full_response = []
            llm_start = time.monotonic()
            tts_start = None
            first_llm_time = None
            first_tts_time = None
            idx = 0
            
            # CRITICAL: Acquire MLX lock for entire LLM generation
            # This prevents concurrent Whisper transcription during LLM use
            async with self.resources.mlx_lock:
                # for sentence in self.resources.llm_handler.stream_response_batched(self.conversation):
                for sentence in self.resources.llm_handler.stream_response(self.conversation):
                    if cancel_event.is_set() or not self.detector.is_responding:
                        print("[response] Interrupted")
                        break
                    
                    full_response.append(sentence)
                    
                    if first_llm_time is None:
                        first_llm_time = time.monotonic() - llm_start
                        await self._send_metrics(llm={"first_token": first_llm_time})
                    
                    # Send sentence to client
                    await self.ws.send_text(json.dumps({
                        "event": "text",
                        "role": "assistant",
                        "text": sentence
                    }))
                    
                    # Start TTS timing
                    if tts_start is None:
                        tts_start = time.monotonic()
                    
                    # Generate TTS and await it (sequential for proper order)
                    await self._send_tts(sentence, idx, cancel_event)
                    
                    # Record first TTS latency
                    if first_tts_time is None and tts_start is not None:
                        first_tts_time = time.monotonic() - tts_start
                        await self._send_metrics(tts={"first_audio": first_tts_time})
                    
                    idx += 1
            
            # Add complete response to history
            complete = " ".join(full_response).strip()
            if complete and self.detector.is_responding and not cancel_event.is_set():
                self.conversation.append({"role": "assistant", "content": complete})
                
                await self.ws.send_text(json.dumps({
                    "event": "text",
                    "role": "assistant",
                    "text": complete,
                    "complete": True
                }))
                
                print(f"[response] Complete ({first_llm_time:.2f}s to first)")
        
        except asyncio.CancelledError:
            print("[response] Cancelled")
            raise
        
        except Exception as e:
            print(f"[response] Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.detector.is_responding = False
            await self._send_state()
            if self._response_cancel_event is cancel_event:
                self._response_cancel_event = None
    
    async def _send_tts(self, text: str, index: int, cancel_event: asyncio.Event):
        """Generate and send TTS audio"""
        try:
            if cancel_event.is_set():
                return
            
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None,
                self.resources.tts_handler.generate_speech,
                text
            )
            
            if cancel_event.is_set() or not self.detector.is_responding:
                return
            
            if audio_bytes:
                await self.ws.send_text(json.dumps({
                    "event": "media",
                    "mime": "audio/wav",
                    "audio": encode_audio(audio_bytes),
                    "index": index
                }))
                print(f"[tts] Sent {len(audio_bytes)} bytes (#{index})")
        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[tts] Error: {e}")
    
    async def _send_metrics(self, stt: Optional[Dict] = None, llm: Optional[Dict] = None, tts: Optional[Dict] = None):
        """Send performance metrics to client"""
        payload = {}
        if stt is not None:
            payload["stt"] = stt
        if llm is not None:
            payload["llm"] = llm
        if tts is not None:
            payload["tts"] = tts
        
        if payload:
            await self.ws.send_text(json.dumps({
                "event": "metrics",
                "metrics": payload
            }))
    
    async def _interrupt_response(self, notify_client: bool = False) -> bool:
        """Interrupt ongoing response generation and optionally notify the client."""
        interrupted = False
        
        cancel_event = self._response_cancel_event
        if cancel_event and not cancel_event.is_set():
            cancel_event.set()
            interrupted = True
        
        async with self._response_lock:
            task_cancelled = await self._cancel_response_task()
            if self._response_cancel_event and self._response_cancel_event.is_set() and not self._response_task:
                self._response_cancel_event = None
        
        if task_cancelled:
            interrupted = True
        
        if self.detector.is_responding:
            self.detector.is_responding = False
            interrupted = True
        
        if interrupted:
            if self.conversation and self.conversation[-1]["role"] == "assistant":
                self.conversation.pop()
                print("[interrupt] Removed incomplete response")
            
            if notify_client:
                await self.ws.send_text(json.dumps({"event": "interrupt"}))
        
        return interrupted
    
    async def _send_state(self):
        """Send current state to client"""
        await self.ws.send_text(json.dumps({
            "event": "state",
            **self.detector.get_state()
        }))


# =============================================================================
# FastAPI Application
# =============================================================================

RESOURCES = PipelineResources()
app = FastAPI()


@app.get("/")
async def get_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/correlator.js")
async def get_correlator():
    with open("correlator.js", "r") as f:
        return Response(content=f.read(), media_type="application/javascript")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] Client connected")
    
    pipeline = VoicePipeline(websocket, RESOURCES)
    await pipeline.start()
    
    try:
        while True:
            message = await websocket.receive_text()
            await pipeline.handle_message(json.loads(message))
    
    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    
    except Exception as e:
        print(f"[ws] Error: {e}")
    
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    import uvicorn
    print("Starting voice agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)



================================================
FILE: transcriber.py
================================================
# transcriber.py
"""Clean transcription handler using MLX Whisper"""
import numpy as np
import mlx_whisper
import time
from config import WHISPER_MODEL, SAMPLE_RATE, MIN_SEGMENT_DURATION


class RealtimeTranscriber:
    """Handles speech-to-text transcription"""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        print(f"Loading Whisper: {model_name}")
        self.model = model_name
        self.sample_rate = SAMPLE_RATE
        print("Whisper loaded")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text"""
        # Validate audio length
        duration = len(audio) / self.sample_rate
        if duration < MIN_SEGMENT_DURATION:
            print(f"Audio too short: {duration:.2f}s")
            return ""
        
        try:
            start_time = time.monotonic()
            
            # Run Whisper transcription
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
                verbose=False,
                language="en",
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            
            transcript = result["text"].strip()
            elapsed = time.monotonic() - start_time
            
            print(f"Transcribed in {elapsed:.2f}s (audio: {duration:.2f}s)")
            
            return transcript
        
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""


================================================
FILE: tts_handler.py
================================================
# tts_handler.py
"""Clean TTS handler using Kokoro"""
import torch
import numpy as np
import io
import wave
from kokoro import KPipeline
from config import TTS_MODEL, TTS_VOICE, TTS_SPEED, TTS_SAMPLE_RATE


class TTSHandler:
    """Handles text-to-speech generation"""
    
    def __init__(
        self,
        repo_id: str = TTS_MODEL,
        voice: str = TTS_VOICE,
        speed: float = TTS_SPEED,
        sample_rate: int = TTS_SAMPLE_RATE
    ):
        print(f"Loading TTS: {repo_id}")
        self.pipeline = KPipeline(lang_code='a', repo_id=repo_id)
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        print("TTS loaded")
    
    def generate_speech(self, text: str) -> bytes:
        """Generate speech audio from text"""
        if not text or not text.strip():
            return b''
        
        try:
            # Generate audio chunks
            audio_chunks = []
            for result in self.pipeline(text, voice=self.voice, speed=self.speed):
                if result.audio is not None:
                    audio_chunks.append(result.audio)
            
            if not audio_chunks:
                print("No audio generated")
                return b''
            
            # Concatenate and convert to WAV
            full_audio = torch.cat(audio_chunks, dim=0)
            audio_array = full_audio.numpy()
            wav_bytes = self._to_wav_bytes(audio_array)
            
            duration = len(audio_array) / self.sample_rate
            print(f"Generated {duration:.2f}s audio")
            
            return wav_bytes
        
        except Exception as e:
            print(f"TTS error: {e}")
            return b''
    
    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes"""
        # Clip and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()


================================================
FILE: vad_detector.py
================================================
# vad_detector.py
"""Voice Activity Detection and End-of-Utterance detection"""
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from config import (
    VAD_MODEL_PATH, EOU_MODEL_PATH, SAMPLE_RATE,
    VAD_ALPHA, VAD_STATE_SHAPE, VAD_CONTEXT_SIZE,
    EOU_MIN_SAMPLES, EOU_OPTIMAL_SAMPLES, EOU_CONFIDENCE_THRESHOLD
)


class VADDetector:
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(self, model_path: str = VAD_MODEL_PATH):
        print(f"Loading VAD: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0
        print("VAD loaded")
    
    def process_chunk(self, chunk: np.ndarray) -> float:
        """Process audio chunk and return smoothed VAD probability"""
        # Prepare input with context
        audio_input = np.concatenate([self.context, chunk.reshape(1, -1)], axis=1)
        
        # Run VAD inference
        output, self.state = self.session.run(
            None,
            {
                'input': audio_input,
                'state': self.state,
                'sr': np.array([SAMPLE_RATE], dtype=np.int64)
            }
        )
        
        # Update context for next chunk
        self.context = audio_input[:, -VAD_CONTEXT_SIZE:]
        
        # Apply exponential smoothing
        raw_prob = float(output[0][0])
        self.smoothed_prob = VAD_ALPHA * raw_prob + (1.0 - VAD_ALPHA) * self.smoothed_prob
        
        return self.smoothed_prob
    
    def reset(self):
        """Reset VAD state"""
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0


class EndOfUtteranceDetector:
    """Detect end of user utterance using ML model"""
    
    def __init__(self, model_path: str = EOU_MODEL_PATH):
        print(f"Loading EOU: {model_path}")
        
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        
        # Optimize ONNX session
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=options)
        self.audio_buffer = np.array([], dtype=np.float32)
        print("EOU loaded")
    
    def add_audio(self, chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        
        # Keep only recent audio (8 seconds max)
        if len(self.audio_buffer) > EOU_OPTIMAL_SAMPLES:
            self.audio_buffer = self.audio_buffer[-EOU_OPTIMAL_SAMPLES:]
    
    def has_enough_audio(self) -> bool:
        """Check if buffer has minimum required audio"""
        return len(self.audio_buffer) >= EOU_MIN_SAMPLES
    
    def detect(self) -> dict:
        """Detect if utterance has ended"""
        if not self.has_enough_audio():
            return {'ended': False, 'confidence': 0.0}
        
        try:
            # Use up to optimal amount of audio
            audio_length = min(len(self.audio_buffer), EOU_OPTIMAL_SAMPLES)
            audio = self.audio_buffer[-audio_length:]
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=EOU_OPTIMAL_SAMPLES,
                truncation=True,
                do_normalize=True
            )
            
            # Run inference
            features = np.expand_dims(
                inputs.input_features.squeeze(0).astype(np.float32), 
                axis=0
            )
            outputs = self.session.run(None, {"input_features": features})
            confidence = float(outputs[0][0].item())
            
            return {
                'ended': confidence > EOU_CONFIDENCE_THRESHOLD,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"EOU detection error: {e}")
            return {'ended': False, 'confidence': 0.0}
    
    def reset(self):
        """Clear audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)


================================================
FILE: docs/TECHNICAL_DEEP_DIVE.md
================================================
# Technical Overview

Fast snapshot of how the <1 s MLX voice agent stays responsive.

## Click below for demo
[![Watch the demo](https://img.youtube.com/vi/6IEK2fXB_ok/0.jpg)](https://www.youtube.com/watch?v=6IEK2fXB_ok)

## Pipeline
1. **Browser loop**
   - `correlator.js` AudioWorklet fuses mic + TTS reference; flags echo vs speech.
   - Rolling buffer keeps ~500 ms of context. Interruption bursts instantly trigger `stop` → server.
2. **Server orchestration (`VoicePipeline`)**
   - `SpeechDetector` runs Silero VAD + optional SmartTurn EoU; emits `TRANSCRIBE` / `RESPOND`.
   - Shared `asyncio.Lock` guards MLX so Whisper and the LLM never overlap.
   - A per-response `asyncio.Event` cancels LLM streaming + Kokoro generation the moment barge-in occurs.
3. **Models**
   - Whisper (MLX) transcribes queued segments in a background worker.
   - LLM streams sentence chunks; each chunk feeds Kokoro synchronously to keep audio ordered.
   - Completed turns update conversation history; partial turns are dropped on cancellation.

## Key Numbers
- Audio chunks: 512 samples (32 ms @ 16 kHz)
- Interruption detection: ~64 ms (2 frames) including playback halt
- Whisper latency: 0.3–0.7 s for typical user turns
- First LLM token: ~250 ms; first TTS audio: ~200 ms
- End-to-end (speech end → bot audio): **<1 s** steady-state

## Files of Interest
- `server.py` — event loop, cancellation, metrics
- `audio_buffer.py` — pre-buffered segment capture
- `index.html` — UI, interruption thresholds, playback queue

That’s the whole story: tight buffers, immediate cancellation, and MLX-only workloads keep everything under a second.


