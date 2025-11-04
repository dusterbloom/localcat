"use client";

import { useState, useEffect, useRef } from 'react';
import {
  ConsoleTemplate,
  FullScreenContainer,
  TranscriptOverlay,
} from "@pipecat-ai/voice-ui-kit";
import {
  PipecatClientProvider,
  PipecatClientAudio
} from '@pipecat-ai/client-react';
import { PipecatClient } from '@pipecat-ai/client-js';
import { WebSocketTransport } from '@pipecat-ai/websocket-transport';
import { VoiceReactivePlasma } from './VoiceReactivePlasma';
import { StreamingText } from './StreamingText';

interface VoiceAppProps {
  videoEnabled: boolean;
  useClientTTS?: boolean;
}

type AppState = "idle" | "connecting" | "connected" | "disconnected";

// Type guard for checking if transcript has final property
const hasFinPropertyString = (obj: unknown): obj is { final: boolean } => {
  return typeof obj === 'object' && obj !== null && 'final' in obj;
};

// Text formatting functions
const formatUserText = (text: string): string => {
  // Fix capitalization and add basic punctuation
  if (!text) return text;

  // Convert to proper case (first letter of each sentence capitalized)
  let formatted = text.toLowerCase();

  // Capitalize first letter
  formatted = formatted.charAt(0).toUpperCase() + formatted.slice(1);

  // Capitalize after periods, exclamation marks, question marks
  formatted = formatted.replace(/[.!?]\s+\w/g, match => match.toUpperCase());

  // Add period at end if no punctuation exists
  if (!/[.!?]$/.test(formatted.trim())) {
    formatted += '.';
  }

  return formatted;
};

// Filter out tool call JSON and XML from bot text
const isToolCallText = (text: string): boolean => {
  if (!text) return false;

  const stripped = text.trim();

  // Check if text is JSON object (likely tool call argument)
  if (stripped.startsWith('{') && stripped.endsWith('}') && stripped.includes('"')) {
    console.log('[ToolCallFilter] Blocking JSON object from UI:', text.substring(0, 50));
    return true;
  }

  // Check for XML tool call syntax
  const toolCallPatterns = [
    /<function\s*=/i,
    /<\/function>/i,
    /<think>/i,
    /<\/think>/i,
    /\{\s*"information"\s*:/,
    /\{\s*"query"\s*:/,
    /\{\s*"new_information"\s*:/,
  ];

  for (const pattern of toolCallPatterns) {
    if (pattern.test(text)) {
      console.log('[ToolCallFilter] Blocking tool call syntax from UI:', text.substring(0, 50));
      return true;
    }
  }

  return false;
};

const formatBotText = (text: string): string => {
  // Remove markdown asterisks and other formatting
  if (!text) return text;

  // Remove **bold** formatting
  let formatted = text.replace(/\*\*(.*?)\*\*/g, '$1');

  // Remove *italic* formatting
  formatted = formatted.replace(/\*(.*?)\*/g, '$1');

  // Remove other common markdown
  formatted = formatted.replace(/`([^`]+)`/g, '$1'); // inline code
  formatted = formatted.replace(/^\s*[-*+]\s+/gm, '• '); // bullet points

  return formatted;
};

export function VoiceApp({ videoEnabled, useClientTTS = false }: VoiceAppProps) {
  const [showDebugUI, setShowDebugUI] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isLowPowerMode, setIsLowPowerMode] = useState(false);
  const [client, setClient] = useState<PipecatClient | null>(null);
  const [appState, setAppState] = useState<AppState>("idle");
  const [isMicEnabled, setIsMicEnabled] = useState(true);
  const [isCameraEnabled, setIsCameraEnabled] = useState(false); // Always start with camera OFF for privacy
  const [showTranscript, setShowTranscript] = useState(false);
  const [transcriptExpanded, setTranscriptExpanded] = useState(false);
  const [fontSize, setFontSize] = useState('normal'); // 'small', 'normal', 'large'
  const [conversationHistory, setConversationHistory] = useState<Array<{role: 'user' | 'assistant', text: string, isStreaming?: boolean, id: string}>>([]);
  const [currentUserTranscript, setCurrentUserTranscript] = useState('');
  const [currentAssistantTranscript, setCurrentAssistantTranscript] = useState('');
  const [wasConnected, setWasConnected] = useState(false); // Track if we were ever connected
  const [autoReconnectAttempts, setAutoReconnectAttempts] = useState(0);
  const [showPerformanceMonitor, setShowPerformanceMonitor] = useState(false);
  const [performanceStats, setPerformanceStats] = useState({
    fps: 0,
    memoryUsage: 0,
    cpuTemp: 0,
    isWebGLActive: false
  });
  const [showVisualizerControls, setShowVisualizerControls] = useState(false);
  const transcriptScrollRef = useRef<HTMLDivElement>(null);
  const isConnectingRef = useRef(false); // Guard against concurrent connection attempts

  useEffect(() => {
    let mounted = true;
    let currentClient: PipecatClient | null = null;

    // Initialize PipecatClient
    const initClient = async () => {
      const transport = new WebSocketTransport();
      const pcClient = new PipecatClient({
        enableCam: false, // Start with camera disabled
        enableMic: true,  // Mic enabled by default
        transport: transport,
        callbacks: {
          // Debug: Log all bot events to see what we're getting
          onBotStartedSpeaking: () => console.log('🎙️ Bot started speaking'),
          onBotStoppedSpeaking: () => console.log('🎙️ Bot stopped speaking'),
          onTransportStateChanged: (state) => {
            switch (state) {
              case "connecting":
              case "authenticating":
                setAppState("connecting");
                break;
              case "ready":
              case "connected":
                setAppState("connected");
                setWasConnected(true);
                setAutoReconnectAttempts(0); // Reset attempts on successful connection
                break;
              case "disconnected":
              case "disconnecting":
                // If we were connected before, show disconnected state, otherwise idle
                setAppState(wasConnected ? "disconnected" : "idle");
                break;
              default:
                setAppState("idle");
                break;
            }
          },
          onError: (error) => {
            console.error("Pipecat error:", error);
            setAppState("disconnected");
          },
          onUserTranscript: (transcript) => {
            if (transcript && transcript.text) {
              // Handle both streaming (partial) and final transcripts
              // Check if transcript has 'final' property, otherwise assume final
              const isFinal = !hasFinPropertyString(transcript) || transcript.final !== false;

              if (!isFinal) {
                // Streaming/partial transcript - show in streaming bubble
                setCurrentUserTranscript(formatUserText(transcript.text));
              } else {
                // Final transcript - add to conversation history and clear streaming
                const messageId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                setConversationHistory(prev => [
                  ...prev,
                  {
                    role: 'user',
                    text: formatUserText(transcript.text),
                    id: messageId
                  }
                ]);
                setCurrentUserTranscript('');
              }
            }
          },
          onBotTranscript: (transcript) => {
            // Fallback: some TTS services emit assistant text via transcript events
            try {
              const raw = transcript?.text || '';
              if (!raw.trim()) return;

              // Filter out tool call JSON/XML
              if (isToolCallText(raw)) {
                console.log('[onBotTranscript] Filtered out tool call text');
                return;
              }

              const newText = formatBotText(raw);
              // Mirror the onBotTtsText handling so we save only full, non-duplicate sentences
              setCurrentAssistantTranscript(prevText => {
                if (!prevText || newText.length > prevText.length) {
                  const endsWithPunctuation = /[.!?]$/.test(newText.trim());
                  if (endsWithPunctuation) {
                    setConversationHistory(prev => {
                      const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');

                      // Normalize whitespace to catch variations like "memory, able" vs "memory,able"
                      const normalizeWhitespace = (text: string) =>
                        text
                          .replace(/\s+/g, ' ')  // Collapse multiple spaces
                          .replace(/\s*([,;:.!?])\s*/g, '$1')  // Remove spaces around punctuation
                          .trim();
                      const normalizedNewText = normalizeWhitespace(newText);

                      for (const msg of allAssistantMessages) {
                        const normalizedMsgText = normalizeWhitespace(msg.text);

                        // Check exact match (after normalization)
                        if (normalizedMsgText === normalizedNewText) {
                          // Duplicate detected — skip
                          return prev;
                        }

                        // Check substring match (one contains the other)
                        if (normalizedMsgText.includes(normalizedNewText) ||
                            normalizedNewText.includes(normalizedMsgText)) {
                          // Duplicate/partial detected — skip
                          return prev;
                        }
                      }
                      const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                      return [...prev, { role: 'assistant', text: newText, id: messageId }];
                    });
                    return '';
                  }
                  return newText;
                }
                return prevText;
              });
            } catch (e) {
              console.warn('onBotTranscript fallback handling failed:', e);
            }
          },
          // Try different possible TTS text event names
          onBotTtsText: (ttsData) => {
            console.log('🎵 onBotTtsText:', ttsData);
            if (ttsData && ttsData.text && ttsData.text.trim().length > 0) {
              // Filter out tool call JSON/XML
              if (isToolCallText(ttsData.text)) {
                console.log('[onBotTtsText] Filtered out tool call text');
                return;
              }

              const newText = formatBotText(ttsData.text);
              // Only update if the new text is longer (accumulating)
              setCurrentAssistantTranscript(prevText => {
                // If new text is longer or we have no previous text, update
                if (!prevText || newText.length > prevText.length) {
                  console.log('📈 TTS text growing:', prevText?.length || 0, '→', newText.length);

                  // Check if this text ends with sentence-ending punctuation
                  const endsWithPunctuation = /[.!?]$/.test(newText.trim());
                  if (endsWithPunctuation) {
                    console.log('🎯 Complete sentence detected:', newText);
                    // Check if this sentence was already saved (check ALL assistant messages)
                    setConversationHistory(prev => {
                      // Check ALL previous assistant messages, not just last 3
                      const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');

                      // Check if this exact text or a variation already exists
                      // Normalize whitespace to catch variations like "memory, able" vs "memory,able"
                      const normalizeWhitespace = (text: string) =>
                        text
                          .replace(/\s+/g, ' ')  // Collapse multiple spaces
                          .replace(/\s*([,;:.!?])\s*/g, '$1')  // Remove spaces around punctuation
                          .trim();
                      const normalizedNewText = normalizeWhitespace(newText);

                      for (const msg of allAssistantMessages) {
                        const normalizedMsgText = normalizeWhitespace(msg.text);

                        // Check exact match (after normalization)
                        if (normalizedMsgText === normalizedNewText) {
                          console.log('🚫 Duplicate sentence found (exact match after normalization):', newText);
                          return prev;
                        }

                        // Check substring match (one contains the other)
                        if (normalizedMsgText.includes(normalizedNewText) ||
                            normalizedNewText.includes(normalizedMsgText)) {
                          console.log('🚫 Duplicate/partial sentence found (substring match):', newText);
                          return prev;
                        }
                      }

                      // It's a genuinely new sentence, save it
                      const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                      console.log('💾 Saving new sentence:', newText);
                      return [
                        ...prev,
                        { role: 'assistant', text: newText, id: messageId }
                      ];
                    });
                    // Return empty string to start fresh for next sentence
                    return '';
                  }

                  return newText;
                }
                // Otherwise keep the longer existing text
                console.log('🚫 TTS text shrinking, keeping existing:', prevText.length, 'vs', newText.length);
                return prevText;
              });
            }
          },
          onBotTtsStarted: () => {
            console.log('Bot TTS started - keep accumulating text');
            // Don't clear! Let text accumulate across TTS sessions
            setShowTranscript(true);  // ADD THIS - Auto-open transcript when bot speaks
          },
          onBotTtsStopped: () => {
            console.log('Bot TTS stopped - check for any remaining text');
            // Fallback: save any remaining text that didn't end with punctuation
            setCurrentAssistantTranscript(prevText => {
              if (prevText.trim()) {
                console.log('💾 Checking to save remaining text without punctuation:', prevText);
                setConversationHistory(prev => {
                  // Check ALL assistant messages for duplicates, not just last 3
                  const allAssistantMessages = prev.filter(msg => msg.role === 'assistant');

                  for (const msg of allAssistantMessages) {
                    if (msg.text === prevText ||
                        msg.text.includes(prevText) ||
                        prevText.includes(msg.text)) {
                      console.log('🚫 Duplicate remaining text found, skipping:', prevText);
                      return prev;
                    }
                  }

                  // Save the remaining text
                  const messageId = `assistant-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
                  console.log('✅ Saving remaining text:', prevText);
                  return [...prev, { role: 'assistant', text: prevText, id: messageId }];
                });
              }
              return ''; // Clear for next response
            });
          },
        },
      });

      await pcClient.initDevices();
      currentClient = pcClient;

      // Only set client if component is still mounted
      if (mounted) {
        setClient(pcClient);
      } else {
        // Component unmounted during initialization, clean up
        pcClient.disconnect();
      }
    };

    initClient();

    // Cleanup function to prevent memory leaks and duplicate connections
    return () => {
      mounted = false;
      if (currentClient) {
        console.log('🧹 Cleaning up PipecatClient on unmount');
        currentClient.disconnect();
      }
    };
  }, []);

  // Reset connection state on unmount to prevent stale state in dev mode
  useEffect(() => {
    return () => {
      console.log('🔄 Resetting connection state on unmount');
      setWasConnected(false);
      setAutoReconnectAttempts(0);
    };
  }, []);

  // Auto-scroll transcript to bottom when new messages arrive
  useEffect(() => {
    if (transcriptScrollRef.current && showTranscript && conversationHistory.length > 0) {
      const scrollElement = transcriptScrollRef.current;
      scrollElement.scrollTop = scrollElement.scrollHeight;
    }
  }, [conversationHistory, showTranscript]);

  // Auto-reconnection logic for kings 👑
  useEffect(() => {
    // Ensure client exists and we're in the right state for auto-reconnect
    if (appState === "disconnected" && wasConnected && autoReconnectAttempts < 5 && client) {
      const delay = Math.min(1000 * Math.pow(2, autoReconnectAttempts), 10000); // Exponential backoff, max 10s
      console.log(`🔄 Auto-reconnecting in ${delay}ms (attempt ${autoReconnectAttempts + 1}/5)...`);

      const reconnectTimeout = setTimeout(async () => {
        if (appState === "disconnected") { // Still disconnected
          setAutoReconnectAttempts(prev => prev + 1);
          try {
            // Ensure clean disconnect before reconnecting
            if (client) {
              client.disconnect();
              // Small delay to ensure cleanup
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            await handleConnect();
          } catch (error) {
            console.error("Auto-reconnect failed:", error);
          }
        }
      }, delay);

      return () => clearTimeout(reconnectTimeout);
    }
  }, [appState, wasConnected, autoReconnectAttempts, client]);

  // Performance monitoring
  useEffect(() => {
    if (!showPerformanceMonitor) return;

    let frameCount = 0;
    let lastTime = performance.now();
    let monitoring = true;

    const updatePerformance = () => {
      frameCount++;
      const now = performance.now();

      // Update FPS every second
      if (now - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastTime));

        // Get memory usage (if available)
        const memoryInfo = (performance as typeof performance & { memory?: { usedJSHeapSize: number } }).memory;
        const memoryUsage = memoryInfo ? Math.round(memoryInfo.usedJSHeapSize / 1048576) : 0; // MB

        // Simply check if we're NOT in low power mode to determine if WebGL should be active
        // Don't try to access canvas context as it causes conflicts with THREE.js
        const isWebGLActive = !isLowPowerMode && !!document.querySelector('canvas');

        setPerformanceStats({
          fps,
          memoryUsage,
          cpuTemp: 0, // Browser can't access CPU temp directly
          isWebGLActive
        });

        frameCount = 0;
        lastTime = now;
      }

      if (monitoring) {
        requestAnimationFrame(updatePerformance);
      }
    };

    requestAnimationFrame(updatePerformance);

    return () => {
      monitoring = false;
    };
  }, [showPerformanceMonitor]);

  const handleConnect = async () => {
    if (!client || (appState !== "idle" && appState !== "disconnected")) return;

    // CRITICAL: Prevent concurrent connection attempts
    if (isConnectingRef.current) {
      console.log("🚫 Connection already in progress, skipping duplicate attempt");
      return;
    }

    try {
      isConnectingRef.current = true;
      console.log("📞 Initiating connection to server...");

      const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || "http://127.0.0.1:7860";
      const wsUrl = serverUrl.replace(/^http/, 'ws'); // Convert http:// to ws://
      await client.connect({
        wsUrl: `${wsUrl}/ws`,
      });

      console.log("✅ Connection request completed");
    } catch (error) {
      console.error("❌ Connection error:", error);
      setAppState("disconnected");
    } finally {
      isConnectingRef.current = false;
    }
  };

  const handleDisconnect = () => {
    if (!client) return;
    client.disconnect();
    setConversationHistory([]);
  };

  const toggleMicrophone = async () => {
    if (!client) return;

    try {
      if (isMicEnabled) {
        client.enableMic(false);
      } else {
        client.enableMic(true);
      }
      setIsMicEnabled(!isMicEnabled);
    } catch (error) {
      console.error('Error toggling microphone:', error);
    }
  };

  const toggleCamera = async () => {
    if (!client) return;

    try {
      if (isCameraEnabled) {
        client.enableCam(false);
      } else {
        client.enableCam(true);
      }
      setIsCameraEnabled(!isCameraEnabled);
    } catch (error) {
      console.error('Error toggling camera:', error);
    }
  };

  const exportConversation = () => {
    const conversation = conversationHistory.map(msg =>
      `**${msg.role === 'user' ? 'User' : 'Assistant'}:** ${msg.text}`
    ).join('\n\n');
    const blob = new Blob([conversation], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `localcat-conversation-${new Date().toISOString().slice(0, 19)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (showDebugUI && client) {
    // Debug/dev mode - use shared client to preserve connection
    const serverUrl = process.env.NEXT_PUBLIC_SERVER_URL || "http://127.0.0.1:7860";
    const wsUrl = serverUrl.replace(/^http/, 'ws');
    return (
      <PipecatClientProvider client={client}>
        <FullScreenContainer>
          <div className="relative w-full h-full">
            {/* Debug mode toggle button */}
            <button
              onClick={() => setShowDebugUI(false)}
              className="absolute top-2 right-40 z-50 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm"
            >
              Enhanced UI
            </button>

            <ConsoleTemplate
              transportType="websocket"
              connectParams={{
                wsUrl: `${wsUrl}/ws`,
              }}
              noUserVideo={!videoEnabled}
            />
          </div>
        </FullScreenContainer>
        <PipecatClientAudio />
      </PipecatClientProvider>
    );
  }

  // Enhanced UX mode with PlasmaVisualizer
  return (
    <PipecatClientProvider client={client!}>
      <FullScreenContainer className={isDarkMode ? 'bg-black' : 'bg-white'}>
        <div
          className={`relative w-full h-full group ${isDarkMode ? 'bg-black' : 'bg-white'}`}
        >
          {/* Invisible interaction areas for controls */}
          <div className="absolute inset-0 z-50">

            {/* Connect/Disconnect in center when needed - ALWAYS VISIBLE IN IDLE */}
            {appState === "idle" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <button
                    onClick={handleConnect}
                    className={`
                      relative px-12 py-4 rounded-full transition-all duration-300
                      ${isDarkMode
                        ? 'bg-white text-black hover:bg-gray-100'
                        : 'bg-black text-white hover:bg-gray-900'}
                      hover:scale-[1.02] active:scale-[0.98] transform
                    `}
                  >
                    <span className="font-light text-base tracking-[0.2em] uppercase">
                      Connect
                    </span>
                  </button>
                </div>
              </div>
            )}

            {appState === "connecting" && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className={`
                    animate-spin w-10 h-10 rounded-full mx-auto mb-4
                    ${isDarkMode
                      ? 'border-2 border-white/20 border-t-white/60'
                      : 'border-2 border-black/20 border-t-black/60'}
                  `}></div>
                  <p className={`
                    font-light text-xs tracking-[0.15em] uppercase
                    ${isDarkMode ? 'text-white/60' : 'text-black/60'}
                  `}>
                    Connecting
                  </p>
                </div>
              </div>
            )}


            {/* Always visible controls - no hover required */}
            {/* Top-left: Connection Status
            <div className="absolute top-6 left-6">
              <div className={`
                px-4 py-2 rounded-full backdrop-blur-sm transition-all duration-500
                ${isDarkMode
                  ? 'bg-transparent border border-black/30'
                  : 'bg-transparent border border-white/30'}
              `}>
                <span className={`
                  font-light text-xs tracking-[0.15em] uppercase
                  ${isDarkMode ? 'text-black' : 'text-white'}
                `}>
                  {appState === 'connected' ? '● Connected' :
                   appState === 'connecting' ? '◐ Connecting' :
                   appState === 'disconnected' && autoReconnectAttempts > 0 ? `🔄 Reconnecting ${autoReconnectAttempts}/5` :
                   appState === 'disconnected' ? '⚡ Disconnected' :
                   '○ Offline'}
                </span>
              </div>
            </div> */}

            {/* Top-right: Control buttons - always visible */}
            <div className="absolute top-6 right-6 flex gap-3">
              <button
                onClick={() => setIsDarkMode(!isDarkMode)}
                className={`
                  w-10 h-10 rounded-full transition-all duration-500 flex items-center justify-center
                  ${isDarkMode
                    ? 'bg-white/90 hover:bg-white text-black'
                    : 'bg-black/90 hover:bg-black text-white'}
                  hover:scale-[1.05] active:scale-[0.95] transform
                `}
              >
                <span className="text-sm">{isDarkMode ? '☀' : '☽'}</span>
              </button>
              <button
                onClick={() => setShowPerformanceMonitor(!showPerformanceMonitor)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500 flex items-center gap-2
                  ${isDarkMode
                    ? (showPerformanceMonitor ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                    : (showPerformanceMonitor ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
                title="Toggle performance monitor"
              >
                <span className="text-sm">📊</span>
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  Perf
                </span>
              </button>
              {isLowPowerMode && (
                <button
                  onClick={() => setShowVisualizerControls(!showVisualizerControls)}
                  className={`
                    w-10 h-10 rounded-full transition-all duration-500 flex items-center justify-center
                    ${isDarkMode
                      ? (showVisualizerControls ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                      : (showVisualizerControls ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                    hover:scale-[1.05] active:scale-[0.95] transform
                  `}
                  title="Visualizer settings"
                >
                  <span className="text-sm">⚙️</span>
                </button>
              )}
              <button
                onClick={() => setIsLowPowerMode(!isLowPowerMode)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500 flex items-center gap-2
                  ${isDarkMode
                    ? (isLowPowerMode ? 'bg-white text-black' : 'bg-white/90 hover:bg-white text-black')
                    : (isLowPowerMode ? 'bg-black text-white' : 'bg-black/90 hover:bg-black text-white')}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
                title={isLowPowerMode ? 'Disable low power mode' : 'Enable low power mode'}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 24 24" className="w-3 h-3">
                  <path d="M16 4h-1V2h-6v2H8C6.9 4 6 4.9 6 6v14c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2z"/>
                </svg>
                <span className="font-light text-xs tracking-[0.15em] uppercase">
                  {isLowPowerMode ? 'Low Power' : 'Normal'}
                </span>
              </button>
              <button
                onClick={() => setShowDebugUI(true)}
                className={`
                  px-4 py-2 rounded-full transition-all duration-500
                  ${isDarkMode
                    ? 'bg-white/90 hover:bg-white text-black'
                    : 'bg-black/90 hover:bg-black text-white'}
                  hover:scale-[1.02] active:scale-[0.98] transform
                `}
              >
                <span className="font-light text-xs tracking-[0.15em] uppercase">Debug</span>
              </button>
            </div>

            {/* Bottom-left: Audio/Video controls - ALWAYS VISIBLE (even when disconnected) */}
            {(appState === "connected" || appState === "disconnected") && (
              <div className="absolute bottom-6 left-6 flex gap-3">
                <button
                  onClick={toggleMicrophone}
                  disabled={appState !== "connected"}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${appState !== "connected" ? 'opacity-50 cursor-not-allowed' : ''}
                    ${isDarkMode
                      ? (isMicEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50')
                      : (isMicEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    ${appState === "connected" ? 'hover:scale-[1.02] active:scale-[0.98] transform' : ''}
                  `}
                >
                  <span className="text-sm">
                    {appState !== "connected" ? '🔌' : (isMicEnabled ? '🎤' : '🔇')}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState !== "connected" ? 'Disconnected' : (isMicEnabled ? 'Mic On' : 'Mic Off')}
                  </span>
                </button>

                <button
                  onClick={toggleCamera}
                  disabled={appState !== "connected"}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300 flex items-center gap-2
                    ${appState !== "connected" ? 'opacity-50 cursor-not-allowed' : ''}
                    ${isDarkMode
                      ? (isCameraEnabled ? 'bg-white/90 text-black' : 'bg-white/50 text-black/50')
                      : (isCameraEnabled ? 'bg-black/90 text-white' : 'bg-black/50 text-white/50')}
                    ${appState === "connected" ? 'hover:scale-[1.02] active:scale-[0.98] transform' : ''}
                  `}
                >
                  <span className="text-sm">
                    {appState !== "connected" ? '🔌' : (isCameraEnabled ? '📹' : '📷')}
                  </span>
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState !== "connected" ? 'Disconnected' : (isCameraEnabled ? 'Camera On' : 'Camera Off')}
                  </span>
                </button>
              </div>
            )}

            {/* Bottom-center: Disconnect/Reconnect button */}
            {(appState === "connected" || appState === "disconnected") && (
              <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2">
                <button
                  onClick={appState === "connected" ? handleDisconnect : async () => {
                    try {
                      // Manual reconnect - ensure clean state first
                      if (client) {
                        client.disconnect();
                        await new Promise(resolve => setTimeout(resolve, 100));
                      }
                      setAutoReconnectAttempts(0); // Reset auto attempts when manually connecting
                      await handleConnect();
                    } catch (error) {
                      console.error("Manual reconnect failed:", error);
                    }
                  }}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300
                    ${isDarkMode
                      ? 'bg-white/90 text-black hover:bg-white'
                      : 'bg-black/90 text-white hover:bg-black'}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {appState === "connected" ? 'Disconnect' : 'Reconnect'}
                  </span>
                </button>
              </div>
            )}

            {/* Bottom-right: Transcript toggle - always visible when connected */}
            {appState === "connected" && (
              <div className="absolute bottom-6 right-6">
                <button
                  onClick={() => setShowTranscript(!showTranscript)}
                  className={`
                    px-4 py-2 rounded-full transition-all duration-300
                    ${isDarkMode
                      ? (showTranscript ? 'bg-white text-black' : 'bg-white/90 text-black')
                      : (showTranscript ? 'bg-black text-white' : 'bg-black/90 text-white')}
                    hover:scale-[1.02] active:scale-[0.98] transform
                  `}
                >
                  <span className="font-light text-xs tracking-[0.15em] uppercase">
                    {showTranscript ? 'Hide' : 'Show'} Transcript
                  </span>
                </button>
              </div>
            )}
          </div>

          {/* Voice-reactive PlasmaVisualizer background - PURE, NO OVERLAYS */}
          <VoiceReactivePlasma
            isDarkMode={isDarkMode}
            isLowPowerMode={isLowPowerMode}
            showControls={showVisualizerControls}
          />

          {/* Performance Monitor Overlay */}
          {showPerformanceMonitor && (
            <div className="absolute top-20 left-6 z-50">
              <div className={`
                px-4 py-3 rounded-lg backdrop-blur-sm transition-all duration-500 text-xs
                ${isDarkMode
                  ? 'bg-black/80 border border-white/20 text-white'
                  : 'bg-white/80 border border-black/20 text-black'}
              `}>
                <div className="font-bold tracking-wide uppercase mb-2">⚡ Performance Monitor</div>
                <div className="space-y-1 font-mono">
                  <div className="flex justify-between gap-4">
                    <span>FPS:</span>
                    <span className={`font-bold ${
                      performanceStats.fps >= 30 ? 'text-green-500' :
                      performanceStats.fps >= 15 ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {performanceStats.fps}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span>Memory:</span>
                    <span className={`font-bold ${
                      performanceStats.memoryUsage < 100 ? 'text-green-500' :
                      performanceStats.memoryUsage < 200 ? 'text-yellow-500' : 'text-red-500'
                    }`}>
                      {performanceStats.memoryUsage}MB
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span>WebGL:</span>
                    <span className={`font-bold ${performanceStats.isWebGLActive ? 'text-blue-500' : 'text-gray-500'}`}>
                      {performanceStats.isWebGLActive ? 'ACTIVE' : 'OFF'}
                    </span>
                  </div>
                  <div className="border-t border-opacity-20 pt-2 mt-2">
                    <div className="text-xs opacity-70">
                      {performanceStats.fps < 15 && performanceStats.isWebGLActive &&
                        '⚠️ Low FPS detected - consider Low Power mode'}
                      {performanceStats.memoryUsage > 200 &&
                        '⚠️ High memory usage detected'}
                      {!performanceStats.isWebGLActive &&
                        '✅ Using fallback renderer'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>


        {/* Transcript Panel - Full width with transparent background */}
        {showTranscript && (
          <div className={`
            absolute transition-all duration-500 z-50
            ${transcriptExpanded
              ? 'bottom-0 left-0 right-0 top-32'
              : 'bottom-20 left-0 right-0'}
             ${isDarkMode ? 'bg-white/70' : 'bg-black/70'}
               backdrop-blur-sm
          `}>
            {/* Header */}
            <div className="flex justify-between items-center px-6 py-3">
              <h3 className={`
                font-bold tracking-[0.2em] uppercase
                ${isDarkMode ?  'text-black' : 'text-white'}
                ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-sm' : 'text-xs'}
              `}>
                Conversation
              </h3>
              <div className="flex gap-3">
                <button
                  onClick={() => setTranscriptExpanded(!transcriptExpanded)}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  {transcriptExpanded ? '▼' : '▲'}
                </button>
                <button
                  onClick={() => {
                    const sizes = ['small', 'normal', 'large'];
                    const currentIndex = sizes.indexOf(fontSize);
                    setFontSize(sizes[(currentIndex + 1) % 3]);
                  }}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  A{fontSize === 'large' ? '+' : fontSize === 'small' ? '-' : ''}
                </button>
                <button
                  onClick={exportConversation}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  Export
                </button>
                <button
                  onClick={() => setShowTranscript(false)}
                  className={`
                    text-xs transition-all duration-300
                    ${isDarkMode
                      ? 'text-black/60 hover:text-black'
                      : 'text-white/60 hover:text-white' }
                  `}
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Transcript Content */}
            <div
              ref={transcriptScrollRef}
              className={`
                px-6 pb-3 overflow-y-auto
                ${transcriptExpanded ? 'max-h-[calc(100vh-12rem)]' : 'max-h-24'}
              `}
              style={{ scrollBehavior: 'smooth' }}
            >
              {/* Show conversation history if available */}
              {conversationHistory.length > 0 ? (
                <div className="space-y-3">
                  {conversationHistory.map((message, idx) => (
                    <div key={message.id || idx} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} mb-3`}>
                      <div className={`max-w-[80%] rounded-lg p-3 ${
                        message.role === 'user'
                          ? 'bg-black text-white'
                          : (isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300')
                      }`}>
                        {message.role === 'assistant' && message.isStreaming ? (
                          <div className={`leading-relaxed break-words ${
                            fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                          }`}>
                            <StreamingText
                              text={message.text}
                              speed={30}
                              className={isDarkMode ? 'text-gray-100' : 'text-gray-900'}
                            />
                          </div>
                        ) : (
                          <div
                            className={`leading-relaxed break-words ${
                              fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                            }`}
                            dangerouslySetInnerHTML={{
                              __html: message.text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="underline hover:no-underline">$1</a>')
                            }}
                          />
                        )}
                        <div className={`text-xs mt-1 opacity-70 ${
                          message.role === 'user' ? 'text-white' : (isDarkMode ? 'text-gray-400' : 'text-gray-500')
                        }`}>
                          {new Date().toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Streaming user transcript */}
                  {currentUserTranscript && (
                    <div className="flex justify-end mb-3">
                      <div className="max-w-[80%] rounded-lg p-3 bg-black text-white opacity-90">
                        <div className={`leading-relaxed break-words ${
                          fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                        }`}>
                          <StreamingText
                            text={currentUserTranscript}
                            speed={30} // Faster for real-time speech
                            className="text-white"
                          />
                        </div>
                        <div className="text-xs mt-1 opacity-70 text-white flex items-center gap-1">
                          <span>Speaking...</span>
                          <div className="flex gap-1">
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse"></div>
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.1s' }}></div>
                            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Streaming assistant transcript */}
                  {currentAssistantTranscript && (
                    <div className="flex justify-start mb-3">
                      <div className={`max-w-[80%] rounded-lg p-3 ${
                        isDarkMode ? 'bg-gray-700 text-gray-100 border border-gray-600' : 'bg-gray-200 text-gray-900 border border-gray-300'
                      } opacity-90`}>
                        <div className={`leading-relaxed break-words ${
                          fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'
                        }`}>
                          <StreamingText
                            text={currentAssistantTranscript}
                            speed={30} // Same speed as user side
                            className={isDarkMode ? 'text-gray-100' : 'text-gray-900'}
                          />
                        </div>
                        <div className={`text-xs mt-1 opacity-70 flex items-center gap-1 ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          <span>Assistant...</span>
                          <div className="flex gap-1">
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`}></div>
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`} style={{ animationDelay: '0.1s' }}></div>
                            <div className={`w-1 h-1 rounded-full animate-pulse ${
                              isDarkMode ? 'bg-gray-400' : 'bg-gray-500'
                            }`} style={{ animationDelay: '0.2s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                </div>
              ) : (
                <TranscriptOverlay
                  participant="remote"
                  className={`
                    ${isDarkMode ? 'text-white/80' : 'text-black/80'}
                    ${fontSize === 'small' ? 'text-xs' : fontSize === 'large' ? 'text-base' : 'text-sm'}
                  `}
                />
              )}
            </div>
          </div>
        )}
      </FullScreenContainer>

      {/* Audio component */}
      <PipecatClientAudio />
    </PipecatClientProvider>
  );
}
