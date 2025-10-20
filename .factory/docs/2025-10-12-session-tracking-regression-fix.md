# Session Tracking Regression Fix Plan

## Issue Identified
The session header shows "Total sessions: 0" instead of the correct count despite the database containing many sessions for the user.

## Root Causes Found

1. **Duplicate start_session calls**: In `HotPathMemoryProcessor.__init__`, `start_session` is called twice (lines 217 and 231), causing potential overwrites due to `INSERT OR REPLACE` logic.

2. **Stats fallback issue**: In `_build_session_header`, `total_sessions` defaults to `1` if not found in stats, but the default fallback in line 822 uses `current_turn` instead of `1`.

## Fix Strategy

### 1. Remove Duplicate Session Initialization
- Remove the redundant `start_session` call in lines 228-233 of `hotpath_processor.py`
- Keep only the initial call that captures stats

### 2. Fix Session Header Building  
- Correct the default value for `total_sessions` from `current_turn` to `1` in `_build_session_header`

### 3. Add Debug Logging
- Add logging to verify session stats are correctly retrieved and used

### 4. Create Test Cases
- Add unit tests to verify session counting works correctly
- Test both database and JSON session trackers

## Files to Modify
1. `server/core/memory/hotpath_processor.py` - Remove duplicate call, fix default
2. `server/tests/unit/test_session_tracking.py` - Add comprehensive tests

## Expected Outcome
- Session headers will show correct "Total sessions: X" count
- Session numbering will be consistent and accurate
- No duplicate session records in database