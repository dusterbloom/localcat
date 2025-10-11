import type { Metadata } from "next";
import "@pipecat-ai/voice-ui-kit/styles.css";

export const metadata: Metadata = {
  title: "Voice UI Kit - Console Template Example",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
