# Fix Cache Pollution in Memory Retrieval

**Priority**: Critical (Security/Privacy)
**Effort**: 1 day
**Assigned To**: Memory Systems Specialist

## Problem Statement

The provenance cache in `retrieval.py:54-58` uses `lru_cache` without session scoping, causing a critical privacy violation where User A's provenance checks can return cached results from User B.

**Current Issue**:
```python
# server/core/memory/retrieval.py:54-58
# PROBLEM: Cache key is ONLY (edge_id, user_id, session_id)
# but the LRU cache itself is GLOBAL across all users/sessions
self._cached_check_edge_visibility = lru_cache(maxsize=CacheConfig.PROVENANCE_CACHE_SIZE)(
    self._check_edge_visibility_impl
)
```

## Impact

- **Security**: User data leakage across sessions
- **Privacy**: Incorrect access control decisions
- **Compliance**: Potential violation of user data isolation requirements

## Success Metrics

- ✓ Each session has isolated cache storage
- ✓ No cross-session cache hits occur
- ✓ Performance maintained (<50ms memory retrieval time)
- ✓ All existing memory tests pass
- ✓ New test validates session isolation

## Implementation Approach

### Option 1: Session-Scoped Cache (Recommended)
```python
# server/core/memory/retrieval.py
class SessionScopedCache:
    """Cache isolated per session for privacy"""

    def __init__(self):
        self._caches = {}  # session_id -> LRUCache
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> callable:
        """Get cache for specific session"""
        with self._lock:
            if session_id not in self._caches:
                self._caches[session_id] = lru_cache(
                    maxsize=CacheConfig.PROVENANCE_CACHE_SIZE
                )(self._check_edge_visibility_impl)
            return self._caches[session_id]

    def clear_session(self, session_id: str):
        """Clear cache when session ends"""
        with self._lock:
            if session_id in self._caches:
                del self._caches[session_id]

# Usage in MemRetrieval.__init__:
self._cache = SessionScopedCache()

# Usage in methods:
cache_fn = self._cache.get_or_create(self.session_id)
visible = cache_fn(edge_id, user_id, session_id)
```

### Option 2: Include Session in Cache Key
```python
# Alternative: Modify cache key to include session context
@lru_cache(maxsize=CacheConfig.PROVENANCE_CACHE_SIZE)
def _check_edge_visibility_cached(self, edge_id: int, user_id: str, session_id: str, session_hash: str):
    """session_hash ensures cache isolation per session"""
    return self._check_edge_visibility_impl(edge_id, user_id, session_id)
```

## Testing Requirements

### Unit Tests
```python
# server/tests/unit/memory_retrieval/test_cache_isolation.py
def test_cache_isolation_across_sessions():
    """Verify session A cache doesn't affect session B"""

    # Session A: Check edge visibility (should cache DENY)
    retrieval_a = MemRetrieval(session_id="session_a", user_id="user1")
    assert retrieval_a._check_visibility(edge_id=123) == False

    # Session B: Same edge, different session (should NOT use Session A cache)
    retrieval_b = MemRetrieval(session_id="session_b", user_id="user1")

    # Modify underlying data to return ALLOW
    # Session B should see new result, not cached DENY from Session A
    assert retrieval_b._check_visibility(edge_id=123) == True

def test_cache_cleared_on_session_end():
    """Verify cache cleanup when session ends"""

    retrieval = MemRetrieval(session_id="test_session")
    retrieval._check_visibility(edge_id=123)  # Prime cache

    # End session
    retrieval.cleanup()

    # Verify cache no longer exists for this session
    assert "test_session" not in retrieval._cache._caches
```

### Integration Tests
```python
# server/tests/integration/test_session_privacy.py
def test_no_cross_user_cache_leakage():
    """End-to-end test: User A data never visible to User B"""

    # User A creates private memory
    processor_a = HotPathMemoryProcessor(user_id="alice", session_id="sess_a")
    processor_a.process_turn("My secret is 12345", role="user")

    # User B queries (should NOT see User A's data via cache)
    processor_b = HotPathMemoryProcessor(user_id="bob", session_id="sess_b")
    bullets = processor_b.hot.retrieve_bullets("What is the secret?")

    assert not any("12345" in bullet for bullet in bullets)
```

## Performance Validation

- Benchmark memory retrieval time before/after changes
- Ensure <50ms retrieval target maintained
- Profile cache hit rates per session
- Memory usage for session-scoped caches

```bash
# Performance test
pytest server/tests/performance/test_memory_retrieval.py -v
# Expected: All tests pass with <50ms p95 latency
```

## Files to Modify

1. **server/core/memory/retrieval.py** (lines 54-58)
   - Replace global lru_cache with SessionScopedCache
   - Add cache cleanup on session end

2. **server/tests/unit/memory_retrieval/** (new)
   - test_cache_isolation.py (new file)
   - test_session_scoped_cache.py (new file)

3. **server/tests/integration/** (existing)
   - test_session_privacy.py (new file)

## Definition of Done

- [ ] SessionScopedCache class implemented
- [ ] MemRetrieval updated to use session-scoped cache
- [ ] Unit tests pass: cache isolation validated
- [ ] Integration tests pass: no cross-user leakage
- [ ] Performance tests pass: <50ms retrieval maintained
- [ ] Code reviewed for security implications
- [ ] Documentation updated in retrieval.py docstrings

## Delegation Command

```bash
# Manager delegates to Memory Systems Specialist
droid exec memory-systems-specialist --auto medium -f tasks/fix_cache_pollution.md
```

---

**Related Issues**: Part of technical debt cleanup (Phase 1, Critical Priority)
**Blocks**: None (standalone fix)
**References**: Tech debt guardian report - Critical Issue #1
