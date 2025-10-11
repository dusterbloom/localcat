# TTS Latency Optimization Spec

## Problem Analysis
From the logs, TTS Time-to-First-Byte (TTFB) varies significantly:
- **Good performance**: 473ms, 748ms 
- **Problematic performance**: 976ms, 1241ms, 1085ms, 1503ms, 1398ms
- **Target**: Consistent <600ms TTFB, p95 <800ms

The variance suggests inconsistent chunking and model loading patterns.

## Root Cause Analysis
1. **Model Cold Starts**: Large variance suggests model isn't fully warmed
2. **Chunk Size Inconsistency**: Audio chunks range from 106KB to 854KB bytes
3. **Buffer Management**: Current 50ms buffer may be too aggressive
4. **Memory Pressure**: Large chunks may cause Metal framework delays

## Optimization Strategy

### Phase 1: Immediate Fixes (ml-model-optimizer)
1. **Model Prewarming Enhancement**
   - Implement persistent model warm-up on startup
   - Add warm-up generation with target text patterns
   - Pre-cache common phoneme sequences

2. **Adaptive Chunking**
   - Reduce max chunk size from 8192KB to 4096KB
   - Implement text-complexity based chunk sizing
   - Add dynamic buffer adjustment (30-80ms range)

3. **Memory Optimization**
   - Implement audio buffer pooling
   - Add Metal memory pressure monitoring
   - Optimize numpy array operations

### Phase 2: Performance Monitoring (performance-analyzer)
1. **Real-time Metrics**
   - Add per-request latency breakdown
   - Implement TTFB alerting (>800ms)
   - Track Metal framework performance

2. **Adaptive Configuration**
   - Auto-tune buffer size based on performance
   - Implement fallback strategies for high load
   - Add circuit breaker for sustained high latency

## Implementation Plan

### Week 1: Critical Fixes
- Enhance model prewarming in `kokoro_worker_optimized.py`
- Implement adaptive chunking algorithm
- Add real-time latency monitoring
- Target: Reduce TTFB variance to ±100ms

### Week 2: Performance Tuning
- Optimize buffer management
- Implement memory pooling
- Add performance dashboard
- Target: Achieve p95 <600ms TTFB

### Week 3: Validation & Monitoring
- Comprehensive load testing
- Performance regression prevention
- Production deployment validation
- Target: Sustained <600ms TTFB under load

## Success Metrics
- **Primary**: TTFB p95 <600ms, max <800ms
- **Secondary**: Latency variance <±100ms
- **Quality**: No degradation in audio quality
- **Stability**: 99.9% request success rate

## Risk Mitigation
- Gradual rollout with A/B testing
- Fallback to current chunking strategy
- Monitor for audio quality degradation
- Memory usage monitoring for leaks

## Testing Strategy
- Load testing with concurrent requests
- Audio quality validation
- Metal framework stress testing
- End-to-end latency measurement

This plan addresses the critical TTS latency issue while maintaining audio quality and system stability.