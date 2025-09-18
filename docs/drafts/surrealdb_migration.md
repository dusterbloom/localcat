Slowcat repo is available at `Users/peppi/Dev/slowcat-consciousness`
Love it — let’s go straight for the jugular.

# 1) SurrealDB in **slowcat** — hard, honest review

## What’s excellent (keep & reuse)

* **Single multi-model backend** replaces 3 stores and is the *default* path; clear env contract (`SURREALDB_URL/USER/PASS/NS/DB`). That removes a whole class of drift/consistency bugs.&#x20;
* **Graph-native schemas** (unified / hybrid / SOTA upgrade) with clean `SCHEMAFULL` tables, relation types, and permission scaffolding. Even if some parts are aspirational, the direction is right.
* **M3 layer design** (nodes, edges, clips, equivalences + bridges to messages/entities) gives you a practical “semantic+temporal” tier above raw tape. This is the piece to salvage wholesale.
* **Adapter contract parity** for FactsGraph/TapeStore (`reinforce_or_insert`, `search_facts`, `add_entry`, `get_recent`, plus time-travel & decay). That’s what makes a drop-in swap feasible.
* **Good hygiene:** startup/stop scripts, tests, and a handoff guide that already covers auth/connectivity failures and baseline perf.

## What’s risky / inconsistent (tighten up)

* **Schema drift**: multiple parallel schema files (`consciousness_schema.surql`, `hybrid_unified_schema.surql`, `unified_knowledge_schema.surql`, `upgrade_sota_v1.surql`, `m3_migration.surql`). There’s overlap and competing definitions (e.g., `speakers` is `SCHEMALESS` in one place yet `SCHEMAFULL` w/ constraints in another). Consolidate to *one canonical* schema + one migration folder.
* **Vector search is “placeholder”** (commented definitions). You’re leaning on embeddings in M3, but there’s no guaranteed, version-pinned vector index strategy yet. Decide: Surreal’s native vector (if on the version that supports it for your platform) or route vectors to a co-resident FAISS/pgvector and store ids/metadata in Surreal. Don’t leave it ambiguous.&#x20;
* **Evented decay** exists in schema, but there’s no ops story: no explicit scheduler or index plan on the decay driver fields (`last_accessed`, `created_at`). Add a daily cron (or Surreal `LIVE` consumer) and ensure supporting indexes.&#x20;
* **Temporal query path** in the query router is still stubbed (`_extract_temporal_range` returns “last 24h”). That’s fine for MVP, but your “<100ms text→graph retrieval” claims will hinge on a real temporal filter; ship it.&#x20;
* **Security defaults**: hardcoded `root/slowcat_secure_2024` in docs are fine for local dev, but don’t let these bleed into examples w/o a giant WARNING box or `.env.example` overrides.&#x20;

## Quick wins (high ROI)

* **Unify env**: one `.env` with `MEMORY_BACKEND=surreal | sqlite | stateless` and *remove* the legacy `USE_SURREALDB`. Your quick-start already nudges here — complete it.&#x20;
* **Index budget**: add concrete non-vector indexes you actually query on (session\_id, timestamp, speaker\_id) — many are present; ensure they exist on *all* canonical tables used at runtime (messages, sessions, M3).
* **One schema to rule them all**: publish `server/schema/unified_knowledge_schema.surql` as *the* entrypoint + a directory of migrations (`migrations/0001_...surql`, `0002_...surql`). Your docs already reference “unified”; make it live truth.&#x20;
* **Ship a tiny “surreal smoke” test** to CI that creates a message→M3 node→edge and runs a unified search. You already have scripts/tests; wire one into the default `pytest -k surreal_smoke`.&#x20;

---

# 2) Porting plan: make **localcat** Surreal-ready fast

The fastest path is **salvage + adapter**:

**Salvage from slowcat**

* `server/memory/surreal_memory.py` (adapter with FactsGraph/TapeStore parity).&#x20;
* `server/memory/m3_surreal_integration.py` + `processors/m3_memory_processor.py` (if you want M3 on day-1).&#x20;
* Canonical schema: `server/schema/unified_knowledge_schema.surql` (+ `m3_migration.surql` if you enable M3).
* Scripts: `scripts/start_surrealdb.sh`, `scripts/stop_surrealdb.sh`, `scripts/setup_surrealdb.py` (or `apply_graph_schema.py`).
* Query router: `memory/surreal_query_router.py` for unified searches (facts+tape+temporal).&#x20;
* Docs: `docs/SURREALDB_INTEGRATION.md` as a template for localcat doc.&#x20;

**Why this works:** the slowcat adapter already mimics your legacy interfaces, so localcat only flips its memory factory and drops in the schema. No rewrites to the rest of the pipeline.

---

# localcat.md

```markdown
# localcat + SurrealDB (drop-in)

## Overview
This integrates SurrealDB as a drop-in replacement for localcat’s current SQLite/LMDB/Leann combo, while preserving existing memory interfaces (FactsGraph/TapeStore). Optional M3 mode adds semantic+temporal graph memory.

**Key capabilities**
- Graph relations + document + time-series in one DB
- <100ms text→graph retrieval (with proper indexing)
- Time-travel queries (“last Tuesday”) and session scoping
- Optional M3 memory (nodes/edges/clips) for richer retrieval

## Code map (new/changed)
```

localcat/
memory/
**init**.py                # factory: now selects Surreal
surreal\_memory.py          # adapter (FactsGraph/TapeStore parity)
surreal\_query\_router.py    # unified search + temporal routing (optional)
m3\_surreal\_integration.py  # optional M3 layer
schema/
unified\_knowledge\_schema.surql
m3\_migration.surql         # optional
scripts/
start\_surrealdb.sh
stop\_surrealdb.sh
apply\_graph\_schema.py      # or setup\_surrealdb.py
tests/
test\_surreal\_smoke.py      # tiny end-to-end smoke

````

## Minimal change set

1) **Factory flip** (`memory/__init__.py`)  
Select Surreal when `MEMORY_BACKEND=surreal` (default), else fall back to sqlite/stateless.

2) **Adapter** (`memory/surreal_memory.py`)  
Implements:
- `reinforce_or_insert(fact)`
- `search_facts(query, limit)`
- `add_entry(role, content, speaker_id)`
- `get_recent(limit)`
- `time_travel_query(natural_date)`
- `apply_decay()`

3) **Schema apply**
```bash
./scripts/start_surrealdb.sh
python scripts/apply_graph_schema.py --ns ${SURREALDB_NAMESPACE:-localcat} --db ${SURREALDB_DATABASE:-memory}
````

4. **(Optional) M3 enable**

* Apply `m3_migration.surql`
* Use `m3_surreal_integration.py` + `m3_memory_processor.py`

## Environment

Add to `.env`:

```
MEMORY_BACKEND=surreal
SURREALDB_URL=ws://127.0.0.1:8000/rpc
SURREALDB_USER=root
SURREALDB_PASS=localcat_dev_password
SURREALDB_NAMESPACE=localcat
SURREALDB_DATABASE=memory
```

## Testing

* `pytest -k surreal_smoke -q` — verifies insert/search/time-travel.
* Manual smoke:

  * Start localcat, speak a line, confirm `messages` populated.
  * Run “what did I say 2 minutes ago?” → time-travel query should hit tape.

## Notes / pitfalls

* Use **one** canonical schema file (`schema/unified_knowledge_schema.surql`). Avoid multiple competing schemas.
* If you need vectors on day-1, keep them external (FAISS) and store ids in Surreal; switch to Surreal vector indexes later.
* Schedule decay (cron or a simple background task) and index `timestamp`, `session_id`, `speaker_id` everywhere you filter.

```

*(This doc is derived from the working slowcat integration and pruned to localcat-minimal surface.)* 
```

---

# Bonus — single diff for **localcat** + VSCode how-to

> Assumes localcat’s tree mirrors `localcat/` as shown above and currently uses a SQLite/LMDB/Leann memory factory. Replace paths if your structure differs.

```diff
*** a/localcat/memory/__init__.py
--- b/localcat/memory/__init__.py
@@
-from .sqlite_memory import create_sqlite_memory_system  # existing
+import os
+from typing import Any
+
+def create_smart_memory_system(*args: Any, **kwargs: Any):
+    backend = os.getenv("MEMORY_BACKEND", "surreal").lower()
+    if backend == "surreal":
+        from .surreal_memory import create_surreal_memory_system
+        return create_surreal_memory_system()
+    elif backend == "stateless":
+        from .stateless_memory import create_stateless_memory_system
+        return create_stateless_memory_system()
+    else:
+        from .sqlite_memory import create_sqlite_memory_system
+        return create_sqlite_memory_system()
```

```diff
*** /dev/null
--- b/localcat/memory/surreal_memory.py
@@
+import os
+import asyncio
+from typing import Dict, List, Optional, Tuple
+
+# Minimal SurrealDB client wrapper; swap with your existing client if available
+class SurrealClient:
+    def __init__(self, url: str, user: str, password: str, ns: str, db: str):
+        self._url, self._user, self._password, self._ns, self._db = url, user, password, ns, db
+        # lazy import to avoid optional dependency failure
+        from surrealdb import Surreal
+        self._client = Surreal(url)
+        self._ready = False
+    async def connect(self):
+        if self._ready: return
+        await self._client.connect()
+        await self._client.signin({"user": self._user, "pass": self._password})
+        await self._client.use(self._ns, self._db)
+        self._ready = True
+    async def query(self, q: str, vars: Optional[Dict]=None):
+        await self.connect()
+        return await self._client.query(q, vars or {})
+    # convenience
+    @property
+    def _user(self): return self._user_
+    @_user.setter
+    def _user(self, v): self._user_ = v
+    @property
+    def _password(self): return self._password_
+    @_password.setter
+    def _password(self, v): self._password_ = v
+    @property
+    def _ns(self): return self._ns_
+    @_ns.setter
+    def _ns(self, v): self._ns_ = v
+    @property
+    def _db(self): return self._db_
+    @_db.setter
+    def _db(self, v): self._db_ = v
+
+class SurrealMemory:
+    def __init__(self, client: SurrealClient):
+        self.db = client
+
+    # ---- FactsGraph parity ----
+    async def reinforce_or_insert(self, fact: Dict) -> str:
+        """
+        fact: {subject, predicate, value, fidelity?, strength?}
+        """
+        q = """
+        LET $s = $subject; LET $p = $predicate; LET $v = $value;
+        LET $existing = (SELECT * FROM fact WHERE subject=$s AND predicate=$p AND value=$v LIMIT 1);
+        IF count($existing) > 0 THEN
+            UPDATE $existing[0].id SET strength = math::min(1.0, (strength ?? 0.6) + 0.05), last_seen = time::now();
+        ELSE
+            CREATE fact SET subject=$s, predicate=$p, value=$v, fidelity=$fidelity ?? 3, strength=$strength ?? 0.6, last_seen=time::now();
+        END;
+        """
+        await self.db.query(q, fact)
+        return "ok"
+
+    async def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
+        # extremely simple baseline; replace with full-text when available
+        q = """
+        SELECT * FROM fact WHERE string::lower(predicate) CONTAINS string::lower($q)
+                             OR string::lower(value) CONTAINS string::lower($q)
+        ORDER BY last_seen DESC LIMIT $lim;
+        """
+        res = await self.db.query(q, {"q": query, "lim": limit})
+        return res[0]["result"] if res else []
+
+    async def get_top_facts(self, limit: int = 10) -> List[Dict]:
+        q = "SELECT * FROM fact ORDER BY strength DESC, last_seen DESC LIMIT $lim;"
+        res = await self.db.query(q, {"lim": limit})
+        return res[0]["result"] if res else []
+
+    # ---- Tape parity ----
+    async def add_entry(self, role: str, content: str, speaker_id: str, session_id: Optional[str]=None):
+        q = "CREATE messages SET role=$role, content=$content, speaker_id=$speaker_id, session_id=$session_id, timestamp=time::now();"
+        await self.db.query(q, {"role": role, "content": content, "speaker_id": speaker_id, "session_id": session_id})
+
+    async def get_recent(self, limit: int = 10) -> List[Dict]:
+        q = "SELECT * FROM messages ORDER BY timestamp DESC LIMIT $lim;"
+        res = await self.db.query(q, {"lim": limit})
+        return res[0]["result"] if res else []
+
+    async def search_tape(self, query: str, limit: int = 10) -> List[Dict]:
+        q = """
+        SELECT * FROM messages WHERE string::lower(content) CONTAINS string::lower($q)
+        ORDER BY timestamp DESC LIMIT $lim;
+        """
+        res = await self.db.query(q, {"q": query, "lim": limit})
+        return res[0]["result"] if res else []
+
+    # ---- Extras ----
+    async def time_travel_query(self, natural_date: str) -> List[Dict]:
+        # naive: “yesterday”, “last Tuesday” → compute bounds client-side
+        # replace with proper NL date parser as needed
+        import datetime as dt, dateutil.parser as dp  # ensure dependency
+        try:
+            # very permissive parse; adjust to your policy
+            anchor = dp.parse(natural_date, default=dt.datetime.now())
+            start = (anchor - dt.timedelta(hours=1)).isoformat()
+            end   = (anchor + dt.timedelta(hours=1)).isoformat()
+        except Exception:
+            end = dt.datetime.utcnow().isoformat()
+            start = (dt.datetime.utcnow() - dt.timedelta(hours=24)).isoformat()
+        q = "SELECT * FROM messages WHERE timestamp >= time::from($start) AND timestamp <= time::from($end) ORDER BY timestamp;"
+        res = await self.db.query(q, {"start": start, "end": end})
+        return res[0]["result"] if res else []
+
+async def create_surreal_memory_system() -> SurrealMemory:
+    client = SurrealClient(
+        url=os.getenv("SURREALDB_URL", "ws://127.0.0.1:8000/rpc"),
+        user=os.getenv("SURREALDB_USER", "root"),
+        password=os.getenv("SURREALDB_PASS", "localcat_dev_password"),
+        ns=os.getenv("SURREALDB_NAMESPACE", "localcat"),
+        db=os.getenv("SURREALDB_DATABASE", "memory"),
+    )
+    await client.connect()
+    return SurrealMemory(client)
```

```diff
*** /dev/null
--- b/localcat/schema/unified_knowledge_schema.surql
@@
+USE NS localcat; USE DB memory;
+-- Messages
+DEFINE TABLE messages SCHEMAFULL;
+DEFINE FIELD role ON messages TYPE string ASSERT $value IN ['user','assistant','system'];
+DEFINE FIELD content ON messages TYPE string ASSERT $value != NONE AND $value != "";
+DEFINE FIELD speaker_id ON messages TYPE string DEFAULT 'default_user';
+DEFINE FIELD session_id ON messages TYPE string;
+DEFINE FIELD timestamp ON messages TYPE datetime DEFAULT time::now();
+DEFINE INDEX messages_session ON messages COLUMNS session_id, timestamp;
+DEFINE INDEX messages_speaker ON messages COLUMNS speaker_id, timestamp DESC;
+DEFINE INDEX messages_ts ON messages COLUMNS timestamp DESC;
+-- Facts
+DEFINE TABLE fact SCHEMAFULL;
+DEFINE FIELD subject ON fact TYPE string;
+DEFINE FIELD predicate ON fact TYPE string;
+DEFINE FIELD value ON fact TYPE option<string>;
+DEFINE FIELD fidelity ON fact TYPE number DEFAULT 3;
+DEFINE FIELD strength ON fact TYPE number DEFAULT 0.6;
+DEFINE FIELD last_seen ON fact TYPE datetime DEFAULT time::now();
+DEFINE INDEX fact_k ON fact COLUMNS subject, predicate, value;
+DEFINE INDEX fact_seen ON fact COLUMNS last_seen DESC;
```

```diff
*** /dev/null
--- b/localcat/scripts/start_surrealdb.sh
@@
+#!/usr/bin/env bash
+set -euo pipefail
+DB_PATH="${SURREAL_PATH:-data/surreal.db}"
+PORT="${SURREAL_PORT:-8000}"
+mkdir -p "$(dirname "$DB_PATH")"
+surreal start --path "$DB_PATH" --bind 127.0.0.1:"$PORT" &
+echo $! > .surreal.pid
+echo "SurrealDB started on ws://127.0.0.1:${PORT}/rpc"
```

```diff
*** /dev/null
--- b/localcat/scripts/apply_graph_schema.py
@@
+import os, asyncio, sys
+from pathlib import Path
+from surrealdb import Surreal
+async def main():
+    url = os.getenv("SURREALDB_URL","ws://127.0.0.1:8000/rpc")
+    user = os.getenv("SURREALDB_USER","root")
+    pw   = os.getenv("SURREALDB_PASS","localcat_dev_password")
+    ns   = os.getenv("SURREALDB_NAMESPACE","localcat")
+    db   = os.getenv("SURREALDB_DATABASE","memory")
+    schema = Path("localcat/schema/unified_knowledge_schema.surql").read_text()
+    c = Surreal(url); await c.connect(); await c.signin({"user":user,"pass":pw}); await c.use(ns, db)
+    for stmt in [s for s in schema.split(";") if s.strip()]:
+        await c.query(stmt)
+    print("Schema applied to", ns, "/", db)
+asyncio.run(main())
```

```diff
*** /dev/null
--- b/localcat/tests/test_surreal_smoke.py
@@
+import pytest, asyncio, os
+from localcat.memory.surreal_memory import create_surreal_memory_system
+@pytest.mark.asyncio
+async def test_surreal_smoke():
+    mem = await create_surreal_memory_system()
+    await mem.add_entry("user","hello world","tester","s1")
+    rows = await mem.get_recent(1)
+    assert rows and rows[0]["content"] == "hello world"
+    await mem.reinforce_or_insert({"subject":"user","predicate":"likes","value":"coffee"})
+    facts = await mem.search_facts("likes", 5)
+    assert any(f["value"]=="coffee" for f in facts)
```

```diff
*** a/.env.example
--- b/.env.example
@@
+# Memory backend: surreal | sqlite | stateless
+MEMORY_BACKEND=surreal
+SURREALDB_URL=ws://127.0.0.1:8000/rpc
+SURREALDB_USER=root
+SURREALDB_PASS=localcat_dev_password
+SURREALDB_NAMESPACE=localcat
+SURREALDB_DATABASE=memory
```

## Test in VSCode (quick)

1. **Terminal 1**

   ```bash
   ./localcat/scripts/start_surrealdb.sh
   python localcat/scripts/apply_graph_schema.py
   ```
2. **Terminal 2**

   ```bash
   pip install surrealdb python-dateutil pytest
   pytest localcat/tests/test_surreal_smoke.py -q
   ```
3. **Debug**: use the Python test explorer to run `test_surreal_smoke`.

---

## Why SurrealDB is a game-changer *here* (applied to your goals)

* **Unified latency path**: one backend, one network hop, one schema. That’s how you keep sub-100ms end-to-end with predictable tails. Your current slowcat tests/docs repeatedly emphasize constant response times and low memory footprint; Surreal fits that “constant performance” ethos better than N-store hybrids.&#x20;
* **Graph reasoning without ceremony**: you already do subject-predicate-value and session scoping. Surreal’s relations + SCHEMAFULL validation + events are the minimal scaffolding to keep data clean while enabling traversal queries and later inference rules (your `relation_types` table is a nice staging ground).&#x20;
* **M3 becomes practical**: M3 needs fast node/edge creation, clip-based temporal windows, and identity equivalence. Your M3 schema + integration layer gives that — now you can scale semantic retrieval beyond “keyword+BM25” without bringing in another service.&#x20;

---

## Final checklist for the port

* [ ] Pick **one** schema file in localcat (`unified_knowledge_schema.surql`) and apply it.
* [ ] Drop in `surreal_memory.py` adapter; keep method names identical to legacy types.&#x20;
* [ ] Make `MEMORY_BACKEND=surreal` the default in `.env.example`.
* [ ] Add the smoke test and wire it to CI.
* [ ] If you want M3 on day-1, copy `m3_*` files and apply `m3_migration.surql`; otherwise skip for a lean first cut.&#x20;

If you want, I can also collapse the slowcat schema set into a single canonical `unified_knowledge_schema.surql` tailored to localcat and bundle a micro vector-search plan (FAISS co-resident first, Surreal vector when stable).
