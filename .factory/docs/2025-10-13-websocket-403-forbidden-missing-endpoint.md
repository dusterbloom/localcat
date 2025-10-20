## 403 Forbidden WebSocket Connection Analysis

**The Issue**: The server is rejecting WebSocket connections with "403 Forbidden" because there's **no WebSocket endpoint defined** - only a regular HTTP POST endpoint at `/api/offer`.

**Root Cause**:
```
INFO: 127.0.0.1:57776 - "WebSocket /" 403
INFO: connection rejected (403 Forbidden)
```

The client is trying to connect to WebSocket endpoint `/` (root) but the server only has:
```python
@app.post("/api/offer")  # HTTP POST endpoint, not WebSocket
```

**The Problem Flow**:
1. Client tries WebSocket connection to `/`
2. FastAPI/Uvicorn sees no WebSocket handler for `/`
3. Returns 403 Forbidden (no endpoint, not auth issue)
4. Connection rejected

**Solutions**:

### 1. Add WebSocket Endpoint (Recommended)
```python
@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Handle SmallWebRTC signaling here
```

### 2. Fix Client Connection URL
Client is connecting to wrong URL. Should connect to:
```
connectionUrl: "/api/offer"  # HTTP POST, not WebSocket
```
But SmallWebRTC transport expects WebSocket, so this needs server-side WebSocket support.

### 3. Use Proper SmallWebRTC Setup
The current setup mixes HTTP POST `/api/offer` with WebSocket transport. SmallWebRTC needs:
- Server: WebSocket endpoint for signaling
- Client: WebSocket connection to that endpoint

**Immediate Fix**:
Add WebSocket endpoint to handle SmallWebRTC signaling properly, or change the transport to use HTTP POST instead of WebSocket for signaling.

**Key Insight**: This isn't about Daily.js, Chrome flags, or network issues - it's a missing WebSocket endpoint on the server side.