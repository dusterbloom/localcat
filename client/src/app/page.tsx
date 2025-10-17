"use client";

import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";
import { useEffect, useState } from "react";

export default function Home() {
  const [online, setOnline] = useState(true);

  // Minimal online/offline indicator
  useEffect(() => {
    setOnline(typeof navigator !== "undefined" ? navigator.onLine : true);
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Register a tiny service worker for offline app shell
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    if (process.env.NODE_ENV !== "production") return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      /* no-op */
    });
  }, []);

  return (
    <ThemeProvider>
      <FullScreenContainer>
        <ConsoleTemplate
          transportType="smallwebrtc"
          connectParams={{
            connectionUrl: "/api/offer",
          }}
          noUserVideo={true}
          transportOptions={{
            // Offline/local: use host-only ICE candidates (no public STUN)
            waitForICEGathering: true,
            iceServers: [],
          }}
        />

        {/* Tiny status pill (non-blocking) */}
        <div
          style={{
            position: "fixed",
            right: 12,
            bottom: 12,
            padding: "4px 8px",
            borderRadius: 9999,
            fontSize: 12,
            color: "#fff",
            background: online ? "#16a34a" : "#ef4444",
            opacity: online ? 0.6 : 0.9,
            zIndex: 1000,
          }}
        >
          {online ? "Online" : "Offline"}
        </div>
      </FullScreenContainer>
    </ThemeProvider>
  );
}
