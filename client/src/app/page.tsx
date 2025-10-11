"use client";

import {
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from "@pipecat-ai/voice-ui-kit";

export default function Home() {
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
      </FullScreenContainer>
    </ThemeProvider>
  );
}
