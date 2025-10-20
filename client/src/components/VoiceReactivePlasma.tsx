"use client";

import { useState, useEffect, useRef, useMemo } from 'react';
import FallbackPlasma from './FallbackPlasma';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let PlasmaVisualizer: any = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const webglModule = require("@pipecat-ai/voice-ui-kit/webgl");
  PlasmaVisualizer = webglModule.PlasmaVisualizer;
} catch {
  console.log("WebGL visualizer not available");
}

interface VoiceReactivePlasmaProps {
  isDarkMode: boolean;
  isLowPowerMode: boolean;
  showControls?: boolean;
}

export function VoiceReactivePlasma({ isDarkMode, isLowPowerMode, showControls = false }: VoiceReactivePlasmaProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [useReducedMotion, setUseReducedMotion] = useState(false);
  const [forceFallback, setForceFallback] = useState(false);
  const [cpuOverloaded, setCpuOverloaded] = useState(false);
  const [qualityLevel, setQualityLevel] = useState<'high' | 'medium' | 'low'>('medium');

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_DISABLE_WEBGL_PLASMA === 'true') {
      setForceFallback(true);
    }
  }, []);

  useEffect(() => {
    if (isLowPowerMode) return;
    
    let lastTime = performance.now();
    let frames = 0;
    let slowFrames = 0;
    let goodFrames = 0;
    let monitoring = true;

    function monitor(now: number) {
      frames++;
      const delta = now - lastTime;
      if (delta >= 1000) {
        const fps = (frames * 1000) / delta;
        frames = 0;
        lastTime = now;
        
        if (fps < 15) {
          slowFrames++;
          goodFrames = 0;
          if (slowFrames >= 2) {
            if (qualityLevel === 'high') {
              setQualityLevel('medium');
            } else if (qualityLevel === 'medium') {
              setQualityLevel('low');
            } else if (slowFrames >= 5) {
              setCpuOverloaded(true);
            }
            slowFrames = 0;
          }
        } else if (fps > 30) {
          goodFrames++;
          slowFrames = 0;
          if (goodFrames >= 5) {
            if (qualityLevel === 'low') setQualityLevel('medium');
            else if (qualityLevel === 'medium' && fps > 50) setQualityLevel('high');
            goodFrames = 0;
          }
        } else {
          slowFrames = 0;
          goodFrames = 0;
        }
      }
      if (monitoring) requestAnimationFrame(monitor);
    }
    requestAnimationFrame(monitor);
    return () => { monitoring = false; };
  }, [qualityLevel, isLowPowerMode]);

  const containerRef = useRef<HTMLDivElement>(null);
  const intersectionRef = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setUseReducedMotion(mediaQuery.matches);
    const handler = (e: MediaQueryListEvent) => setUseReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    intersectionRef.current = new IntersectionObserver(([entry]) => {
      setIsVisible(entry.isIntersecting);
    }, { threshold: 0.1 });
    intersectionRef.current.observe(containerRef.current);
    return () => intersectionRef.current?.disconnect();
  }, []);

  const fallbackGradient = useMemo(() => (
    <div className={`absolute inset-0 animate-pulse ${
      isDarkMode ? 'bg-gradient-to-br from-orange-900/20 to-purple-900/20' : 'bg-gradient-to-br from-orange-200/50 to-yellow-200/50'
    }`} />
  ), [isDarkMode]);

  const shouldShowFallback = !PlasmaVisualizer || useReducedMotion || forceFallback || cpuOverloaded || isLowPowerMode;
  const lastRenderTimeRef = useRef<number>(0);
  const TARGET_FPS: number = qualityLevel === 'high' ? 30 : qualityLevel === 'medium' ? 20 : 10;
  const FRAME_INTERVAL: number = 1000 / TARGET_FPS;
  useEffect(() => {
    if (!PlasmaVisualizer || shouldShowFallback) return;
    let rafId: number;
    const renderLoop = (time: number) => {
      if (typeof time === 'number' && time - lastRenderTimeRef.current >= FRAME_INTERVAL) {
        lastRenderTimeRef.current = time;
      }
      rafId = window.requestAnimationFrame(renderLoop);
    };
    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [PlasmaVisualizer, shouldShowFallback]);

  if (shouldShowFallback) {
    return (
      <div className="absolute inset-0 z-0 pointer-events-none">
        <FallbackPlasma isDarkMode={isDarkMode} showControls={showControls} />
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`absolute inset-0 z-0 pointer-events-none ${isDarkMode ? 'bg-black plasma-dark' : 'bg-white plasma-light'}`}>
      {isVisible ? (
        PlasmaVisualizer ? (
          <div style={{
            width: '100%',
            height: '100%',
            transform: qualityLevel === 'low' ? 'scale(1.5)' : qualityLevel === 'medium' ? 'scale(1.2)' : 'none',
            transformOrigin: 'center',
          }}>
            <PlasmaVisualizer
              state="connected"
              style={{
                backgroundColor: isDarkMode ? 'black' : 'white',
                width: qualityLevel === 'low' ? '66%' : qualityLevel === 'medium' ? '83%' : '100%',
                height: qualityLevel === 'low' ? '66%' : qualityLevel === 'medium' ? '83%' : '100%',
                imageRendering: qualityLevel === 'low' ? 'pixelated' : 'auto',
                filter: qualityLevel === 'low' ? 'blur(1px)' : qualityLevel === 'medium' ? 'blur(0.5px)' : 'none',
                willChange: 'transform',
                transform: 'translateZ(0)',
                backfaceVisibility: 'hidden',
              }}
            />
          </div>
        ) : (
          <FallbackPlasma isDarkMode={isDarkMode} />
        )
      ) : (
        fallbackGradient
      )}
    </div>
  );
}

