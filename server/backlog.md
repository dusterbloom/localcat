# LocalCat Server Development Backlog

## ✅ Completed: mem0 Memory Service Integration (2025-09-04)

### Problem
- mem0 + LM Studio compatibility errors:
  - async_mode parameter error
  - response_format.type json_object vs json_schema
  - JSON parsing failures with infer=True

### Solution: Custom Mem0 Wrapper
- Created custom_mem0_service.py 
- Two-model architecture: gemma3:4b (conversation) + qwen2.5-7b-instruct (memory)
- Dynamic JSON schema detection
- Fallback to infer=False for local models (GitHub issue #3391)
- Graceful error handling

### Files Modified:
- server/custom_mem0_service.py (NEW)
- server/bot.py (import + system role fix)
- .env (MEM0_BASE_URL, MEM0_MODEL variables)

### Result: Working memory persistence with local models

---

## 🔧 Next: Technical Debt Cleanup (Planned)

### Target Issues for Next Commit
Priority: **High-impact, Low-effort** items for quick wins

#### Immediate Fixes (< 1 hour total)
1. **Pipecat Transport Import Deprecations** 
   - Update `bot.py` imports from deprecated modules
   - `pipecat.transports.network.small_webrtc` → `pipecat.transports.smallwebrtc.transport`
   - **Effort**: 30 minutes

2. **Dependency Version Fixes**
   - Pin scikit-learn to compatible version (≤1.5.1) 
   - Update PyTorch to tested version (2.5.0)
   - **Effort**: 15 minutes

3. **Requirements.txt Cleanup**
   - Pin all dependency versions to prevent breaking changes
   - Remove unused `vllm` dependency
   - **Effort**: 15 minutes

#### Medium Priority Fixes
4. **WebSockets Legacy API Update**
   - Follow websockets upgrade guide for deprecated legacy module
   - **Effort**: 1-2 hours

### Expected Outcomes
- ✅ Clean startup (no deprecation warnings)
- ✅ Stable dependency versions
- ✅ Future-proof imports
- ✅ Reduced maintenance burden

### Files to Modify
- `requirements.txt` (version pinning)
- `bot.py` (import updates)
- Test startup to verify no warnings

### Success Criteria
```bash
python bot.py  # Should start with no warnings except normal INFO logs
```
