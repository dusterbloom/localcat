## Migration Plan: SmallWebRTC → LiveKit Transport

**Current Problem**: SmallWebRTC transport has connectivity issues when WiFi is off and requires complex WebSocket signaling setup that doesn't align with the voice UI kit expectations.

**Solution**: Migrate to LiveKit transport which provides robust WebRTC infrastructure with better offline support and professional-grade signaling.

### Phase 1: Server-Side Setup

**1.1 Install LiveKit Dependencies**
```bash
pip install "pipecat-ai[livekit]"
```

**1.2 Set Up LiveKit Server (Self-Hosted)**
- Option A: Docker setup (recommended for development)
```bash
docker run -d \
  -p 7880:7880 \
  -p 50000-50100:50000-50100/udp \
  -e LIVEKIT_KEYS="devkey:secret" \
  livekit/livekit-server:latest
```

- Option B: Native installation
```bash
brew install livekit  # macOS
livekit-server --dev --bind 0.0.0.0
```

**1.3 Update Server Code**
```python
# Replace SmallWebRTCConnection with LiveKitTransport
from pipecat.transports.services.livekit import LiveKitTransport, LiveKitParams

# Update bot.py to use LiveKit
transport = LiveKitTransport(
    url="ws://localhost:7880",  # LiveKit WebSocket URL
    token="devkey",  # API key for room access
    room_name="voice-room",  # Room name
    params=LiveKitParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_out_enabled=False,  # No video needed
    )
)
```

### Phase 2: Client-Side Setup

**2.1 Install LiveKit Transport Package**
```bash
cd client
npm install @pipecat-ai/livekit-transport
# Remove small-webrtc transport
npm uninstall @pipecat-ai/small-webrtc-transport
```

**2.2 Update Client Configuration**
```typescript
// page.tsx - Update transport configuration
<ConsoleTemplate
  transportType="livekit"
  connectParams={{
    url: "ws://localhost:7880",
    token: "devkey",
    roomName: "voice-room",
  }}
  noUserVideo={true}
  transportOptions={{
    // LiveKit-specific options
    adaptiveStream: true,
    disconnectOnPageLeave: true,
  }}
/>
```

**2.3 Update Next.js Config**
```typescript
// next.config.ts - Remove API proxying since LiveKit handles its own signaling
const nextConfig: NextConfig = {
  // Remove rewrites for /api/*
};
```

### Phase 3: Configuration & Environment

**3.1 Environment Variables**
```bash
# .env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_ROOM=voice-room
```

**3.2 Production Considerations**
- Use hosted LiveKit Cloud or self-hosted with proper SSL certificates
- Generate proper API keys using LiveKit CLI
- Configure TURN servers for NAT traversal
- Set up Redis for state management in production

### Phase 4: Benefits & Advantages

**4.1 Immediate Benefits**
- ✅ Professional WebRTC infrastructure
- ✅ Better NAT/firewall traversal
- ✅ Built-in TURN server support
- ✅ Reliable signaling (no more 403 errors)
- ✅ Better debugging and monitoring tools

**4.2 Long-term Benefits**
- ✅ Scalable to multiple participants
- ✅ Recording capabilities
- ✅ SIP/PSTN integration options
- ✅ Room management features
- ✅ Production-ready monitoring

### Phase 5: Migration Timeline

**Week 1**: Server setup and basic LiveKit integration
**Week 2**: Client migration and testing
**Week 3**: Production configuration and deployment
**Week 4**: Performance optimization and monitoring

### Risk Mitigation

- **Rollback plan**: Keep SmallWebRTC code as backup during migration
- **Testing**: Extensive testing with WiFi on/off scenarios
- **Documentation**: Update all setup guides and READMEs
- **Monitoring**: Add LiveKit-specific logging and metrics

This migration provides a robust, production-ready WebRTC solution that should resolve the current connectivity issues while offering a path for future scaling and features.