"use client";

import React, { useState } from "react";
import dynamic from 'next/dynamic';

const VoiceReactiveVisualizer = dynamic(
  () => import('./VoiceReactiveVisualizer'),
  { ssr: false }
);

interface FallbackPlasmaProps {
  isDarkMode: boolean;
  showControls?: boolean;
}

export function FallbackPlasma({ isDarkMode, showControls = false }: FallbackPlasmaProps) {
  const [useVoiceReactive, setUseVoiceReactive] = useState(true);

  if (useVoiceReactive) {
    return (
      <div className="absolute inset-0">
        <VoiceReactiveVisualizer 
          isDarkMode={isDarkMode} 
          autoStart={true} 
          audioSource="output"
          skipMicrophoneRequest={true}
          externalShowControls={showControls}
        />
        <button
          onClick={() => setUseVoiceReactive(false)}
          className={`
            absolute top-6 left-6 z-40
            px-4 py-2 rounded-full transition-all duration-300
            ${isDarkMode 
              ? 'bg-white/90 text-black hover:bg-white' 
              : 'bg-black/90 text-white hover:bg-black'}
            hover:scale-[1.02] active:scale-[0.98] transform
          `}
        >
          <span className="font-light text-xs tracking-[0.15em] uppercase">
            Simple Mode
          </span>
        </button>
      </div>
    );
  }

  return (
    <div
      className={`absolute inset-0 overflow-hidden ${
        isDarkMode ? "bg-black" : "bg-white"
      }`}
    >
      <div
        className={`w-full h-full animate-gradientMove ${
          isDarkMode
            ? "bg-gradient-to-br from-purple-900 via-orange-900 to-red-900"
            : "bg-gradient-to-br from-pink-200 via-orange-200 to-yellow-200"
        } bg-[length:400%_400%] opacity-70`}
        style={{
          animation: "gradientMove 12s ease infinite",
        }}
      />
      <button
        onClick={() => setUseVoiceReactive(true)}
        className={`
          absolute top-6 left-6 z-40
          px-4 py-2 rounded-full transition-all duration-300
          ${isDarkMode 
            ? 'bg-white/90 text-black hover:bg-white' 
            : 'bg-black/90 text-white hover:bg-black'}
          hover:scale-[1.02] active:scale-[0.98] transform
        `}
      >
        <span className="font-light text-xs tracking-[0.15em] uppercase">
          Voice Mode
        </span>
      </button>
      <style jsx>{`
        @keyframes gradientMove {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </div>
  );
}

export default FallbackPlasma;

