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
          waitForICEGathering: true,
          iceServers: [
            {
              urls: "stun:stun.l.google.com:19302",
            },
          ],
        }}
        />
      </FullScreenContainer>
    </ThemeProvider>
  );
}