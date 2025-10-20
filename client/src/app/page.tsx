"use client";

import { useState, useEffect } from 'react';
import { ThemeProvider } from "@pipecat-ai/voice-ui-kit";
import { setupLinkConversion } from '../utils/linkFormatter.js';
import { LoadingScreen } from '../components/LoadingScreen';
import { VoiceApp } from '../components/VoiceApp';

export default function Home() {
  const [isLoading, setIsLoading] = useState(true);
  const [isTauri, setIsTauri] = useState(false);
  const enableClientTTS = process.env.NEXT_PUBLIC_ENABLE_CLIENT_TTS === 'true';
  const videoEnabled = process.env.NEXT_PUBLIC_ENABLE_VIDEO === "true";

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const isTauriApp = '__TAURI__' in window;
      setIsTauri(isTauriApp);
      console.log(`🚀 Running in ${isTauriApp ? 'Tauri bundle' : 'browser'}`);
    }
  }, []);

  useEffect(() => {
    console.log('🔍 Page mounted, setting up link conversion...');
    const timer = setTimeout(() => {
      const observer = setupLinkConversion();
      return () => observer?.disconnect();
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  if (isLoading && !isTauri) {
    return <LoadingScreen onComplete={() => setIsLoading(false)} />;
  }

  return (
    <ThemeProvider>
      <VoiceApp videoEnabled={videoEnabled} useClientTTS={enableClientTTS} />
    </ThemeProvider>
  );
}
