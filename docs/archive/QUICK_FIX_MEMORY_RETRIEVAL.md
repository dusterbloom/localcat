# Quick Fix: Memory Retrieval Not Working

## Problem
User 'fantastic' cannot retrieve any historical memory data because all data belongs to user 'peppi'.

## Evidence
```bash
$ grep USER_ID server/.env
USER_ID=fantastic

$ sqlite3 data/memory.db "SELECT COUNT(*) FROM conversation_turn WHERE session_id LIKE 'peppi%'"
433

$ sqlite3 data/memory.db "SELECT COUNT(*) FROM conversation_turn WHERE session_id LIKE 'fantastic%'"
0
```

## Quick Fix (30 seconds)

```bash
cd /Users/peppi/Dev/localcat

# Option A: Edit .env manually
# Change USER_ID=fantastic to USER_ID=peppi

# Option B: Use sed to change it
sed -i '' 's/USER_ID=fantastic/USER_ID=peppi/' server/.env

# Restart server
# All 428 historical edges will now be accessible
```

## Verify Fix

After restarting the server, test retrieval:

```bash
# In a conversation, ask:
# "What did we talk about before?"
# "What is my favorite color?"
# "Tell me about my dog"

# Check logs:
tail -f ~/Library/Logs/LocalCat/server.log | grep "Retrieval"

# Should see:
# [Retrieval] Returning X memory bullets from sources: {...}
# Instead of:
# [Retrieval] No memory context found for query
```

## Why This Works

The memory system checks session ownership:
- All historical sessions start with `peppi_*`
- Ownership is tracked in the `mention` table with `eid='peppi'`
- When `USER_ID=fantastic`, the system looks for `eid='fantastic'`
- No matches found → 0 results
- When `USER_ID=peppi`, the system looks for `eid='peppi'`
- 50+ matches found → retrieval works!

## Alternative: Migrate Database

If you really want to keep `USER_ID=fantastic`:

```bash
# Backup first
sqlite3 data/memory.db ".backup data/memory_backup_$(date +%Y%m%d).db"

# Run migration
sqlite3 data/memory.db << 'EOF'
UPDATE conversation_turn
SET session_id = REPLACE(session_id, 'peppi_', 'fantastic_')
WHERE session_id LIKE 'peppi_%';

UPDATE mention
SET session_id = REPLACE(session_id, 'peppi_', 'fantastic_')
WHERE session_id LIKE 'peppi_%';

UPDATE mention
SET eid = 'fantastic'
WHERE eid = 'peppi';
EOF

# Verify
sqlite3 data/memory.db "SELECT COUNT(*) FROM conversation_turn WHERE session_id LIKE 'fantastic%'"
# Should show 433
```

## For Persistent Memory

Also update these settings in `server/.env`:

```bash
MEMORY_MODE=persistent
VOICE_AGENT_SESSION_PERSISTENCE=true
```

---

**See full report:** `MEMORY_RETRIEVAL_INVESTIGATION_REPORT.md`
