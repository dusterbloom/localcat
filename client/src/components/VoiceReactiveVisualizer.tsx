"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { usePipecatClientMediaTrack } from '@pipecat-ai/client-react';

type PermissionState = 'idle' | 'pending' | 'granted' | 'denied';

interface Blob { id: number; size: number; color: string; }
interface BlobPhysics { id: number; x: number; y: number; vx: number; vy: number; }

interface VoiceReactiveVisualizerProps {
  isDarkMode: boolean;
  autoStart?: boolean;
  audioSource?: 'microphone' | 'output' | 'both';
  skipMicrophoneRequest?: boolean;
  externalShowControls?: boolean;
}

const NUM_BLOBS = 5;
const MIN_SPEED = 0.5;
const MAX_SPEED = 1.5;
const MIN_SIZE = 80;
const MAX_SIZE = 200;

const VoiceReactiveVisualizer: React.FC<VoiceReactiveVisualizerProps> = ({ isDarkMode, autoStart = false, audioSource = 'microphone', skipMicrophoneRequest = false, externalShowControls = false }) => {
  const [permissionState, setPermissionState] = useState<PermissionState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [blobs, setBlobs] = useState<Blob[]>([]);
  const [blobCount, setBlobCount] = useState(NUM_BLOBS);
  const [sensitivity, setSensitivity] = useState(1.0);

  const audioTrack = usePipecatClientMediaTrack("audio", "bot");
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const smoothedVolumeRef = useRef(0);
  const blobPhysicsRef = useRef<BlobPhysics[]>([]);
  const blobElementsRef = useRef<(HTMLDivElement | null)[]>([]);

  const createBlobs = useCallback(() => {
    const newBlobs: Blob[] = [];
    const newBlobPhysics: BlobPhysics[] = [];
    const colors = isDarkMode ? ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#F4A261'] : ['#FF006E', '#FB5607', '#FFBE0B', '#8338EC', '#3A86FF'];
    for (let i = 0; i < blobCount; i++) {
      const id = i;
      const size = Math.random() * (MAX_SIZE - MIN_SIZE) + MIN_SIZE;
      newBlobs.push({ id, size, color: colors[i % colors.length] });
      newBlobPhysics.push({ id, x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight, vx: (Math.random() - 0.5) * 2 * (MAX_SPEED - MIN_SPEED) + MIN_SPEED, vy: (Math.random() - 0.5) * 2 * (MAX_SPEED - MIN_SPEED) + MIN_SPEED });
    }
    setBlobs(newBlobs);
    blobPhysicsRef.current = newBlobPhysics;
    blobElementsRef.current = new Array(newBlobs.length).fill(null);
  }, [blobCount, isDarkMode]);

  const animationLoop = useCallback(() => {
    if (!analyserRef.current || !dataArrayRef.current) {
      animationFrameIdRef.current = requestAnimationFrame(animationLoop);
      return;
    }
    analyserRef.current.getByteFrequencyData(dataArrayRef.current);
    let sum = 0;
    for (const amplitude of dataArrayRef.current) sum += amplitude * amplitude;
    const rms = Math.sqrt(sum / dataArrayRef.current.length);
    const normalizedRms = Math.min(rms / 128, 1.0) * sensitivity;
    const SMOOTHING_FACTOR = 0.1;
    smoothedVolumeRef.current = SMOOTHING_FACTOR * normalizedRms + (1 - SMOOTHING_FACTOR) * smoothedVolumeRef.current;
    const volume = smoothedVolumeRef.current;
    const speedFactor = 1 + volume * 5;
    const scaleFactor = 1 + volume * 1.5;
    blobPhysicsRef.current = blobPhysicsRef.current.map(p => {
      let newX = p.x + p.vx * speedFactor;
      let newY = p.y + p.vy * speedFactor;
      const size = blobs.find(b => b.id === p.id)?.size || MAX_SIZE;
      if (newX > window.innerWidth + size / 2) newX = -size / 2;
      if (newX < -size / 2) newX = window.innerWidth + size / 2;
      if (newY > window.innerHeight + size / 2) newY = -size / 2;
      if (newY < -size / 2) newY = window.innerHeight + size / 2;
      return { ...p, x: newX, y: newY };
    });
    blobElementsRef.current.forEach((el, i) => {
      if (el) {
        const physics = blobPhysicsRef.current[i];
        if (physics) el.style.transform = `translate(${physics.x}px, ${physics.y}px) scale(${scaleFactor})`;
      }
    });
    animationFrameIdRef.current = requestAnimationFrame(animationLoop);
  }, [blobs, sensitivity]);

  useEffect(() => {
    if (!audioTrack) return;
    const setupAudio = async () => {
      try {
        const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioContextClass) throw new Error('AudioContext not supported');
        const context = new AudioContextClass();
        audioContextRef.current = context;
        const analyser = context.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>;
        const stream = new MediaStream([audioTrack]);
        const source = context.createMediaStreamSource(stream);
        source.connect(analyser);
        createBlobs();
        setPermissionState('granted');
      } catch (err) {
        console.error('🎵 Error setting up audio analysis:', err);
        setPermissionState('denied');
      }
    };
    setupAudio();
    return () => {
      audioContextRef.current?.close().catch(console.error);
      audioContextRef.current = null;
    };
  }, [audioTrack, createBlobs]);

  useEffect(() => {
    if (permissionState === 'granted') {
      animationFrameIdRef.current = requestAnimationFrame(animationLoop);
    }
    return () => {
      if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
    };
  }, [permissionState, animationLoop]);

  return (
    <div className="absolute inset-0 overflow-hidden">
      {blobs.map((blob, i) => (
        <div
          key={blob.id}
          ref={el => { blobElementsRef.current[i] = el; }}
          className="absolute rounded-full mix-blend-multiply filter blur-3xl opacity-40"
          style={{
            width: blob.size,
            height: blob.size,
            background: blob.color,
          }}
        />
      ))}
    </div>
  );
};

export default VoiceReactiveVisualizer;

