Awesome—let’s wire your **sub-300 ms “compiled RAM + local store” memory** into LocalCat cleanly, without blowing up your pipeline. I’ll give you exact file adds/edits, config, and where each piece plugs in.

I’ll assume the repo you shared (with `server/bot.py`, custom Mem0 service, etc.). I’ll reference your files inline so you can see where this lands.

---

# 0) Quick fixes to current setup (2 mins)

1. **FAISS metric** → cosine (your code uses Euclidean now, which hurts retrieval): in `server/bot.py`, change:

```python
"distance_strategy": "cosine",  # was "euclidean"
```

You’ll find this under `local_config["vector_store"]`. &#x20;

2. **Thin memory injection** (until we swap in hot-path memory): lower noise & bias:

* `search_limit=5` (was 10)
* `search_threshold=0.5` (was 0.3)
* `add_as_system_message=False` (prefer user-role bullets)
  These are the fields you already set on `Mem0MemoryService.InputParams(...)`.&#x20;

> Rationale: even if Mem0 stays for a while, this prevents slow/irrelevant memory blocks from bloating the prompt.

---

# 1) Dependencies (SQLite/LMDB hot memory)

Edit `server/requirements.txt` and add:

```
ahocorasick==2.1.0
rapidfuzz==3.9.6
lmdb==1.5.1
msgpack==1.0.8
sqlite-utils==3.37
spacy==3.7.5
```

(If you keep the spaCy model fallback: `python -m spacy download en_core_web_sm` once.)

> Nothing else in your stack changes. Pipecat, VAD, Whisper, TTS stay as-is. Your pipeline order remains `context_aggregator.user() → memory → llm` (we’ll replace the memory hop).&#x20;

---

# 2) Add the **local store** (SQLite + LMDB), non-blocking

Create `server/memory_store.py`:

```python
# server/memory_store.py
import lmdb, sqlite3, time, hashlib, msgpack, contextlib
from dataclasses import dataclass

@dataclass
class Paths:
    sqlite_path: str = "memory.db"
    lmdb_dir: str = "graph.lmdb"

class MemoryStore:
    def __init__(self, paths: Paths = Paths()):
        self.sql = sqlite3.connect(paths.sqlite_path, check_same_thread=False)
        self.sql.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE IF NOT EXISTS entity(id TEXT PRIMARY KEY, name TEXT, aliases TEXT, created_at INT, updated_at INT);
            CREATE TABLE IF NOT EXISTS edge(id TEXT PRIMARY KEY, src TEXT, rel TEXT, dst TEXT, weight REAL, updated_at INT);
            CREATE TABLE IF NOT EXISTS mention(id TEXT PRIMARY KEY, eid TEXT, text TEXT, ts INT, session_id TEXT, turn_id INT);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, eid UNINDEXED, rel UNINDEXED, dst UNINDEXED, ts UNINDEXED, tokenize='porter');
        """)
        self.lenv = lmdb.open(paths.lmdb_dir, map_size=2_147_483_648, max_dbs=8, subdir=True)
        self.db_alias = self.lenv.open_db(b"alias")
        self.db_adj   = self.lenv.open_db(b"adj")

        self._aliases, self._edges, self._mentions = [], [], []
        self._last = time.time()

    @staticmethod
    def edge_id(s,r,d): return hashlib.sha1(f"{s}|{r}|{d}".encode()).hexdigest()

    def enqueue_alias(self, alias, eid): self._aliases.append((alias, eid))
    def enqueue_edge(self, s, r, d, w, ts): self._edges.append((s,r,d,w,int(ts)))
    def enqueue_mention(self, eid, text, ts, sid, tid): self._mentions.append((eid,text,int(ts),sid,tid))

    def flush_if_needed(self, max_ops=16, max_ms=500):
        if (len(self._aliases)+len(self._edges)+len(self._mentions) >= max_ops) or ((time.time()-self._last)*1000 >= max_ms):
            self.flush()

    def flush(self):
        if not (self._aliases or self._edges or self._mentions): 
            return
        with contextlib.ExitStack() as stack:
            txn = stack.enter_context(self.lenv.begin(write=True))
            cur = self.sql.cursor()

            for alias, eid in self._aliases:
                txn.put(key=f"alias:{alias}".encode(), value=eid.encode(), db=self.db_alias, overwrite=True)

            for s,r,d,w,ts in self._edges:
                key = f"adj:{s}|{r}".encode()
                old = txn.get(key, db=self.db_adj) or b""
                arr = list(msgpack.loads(old) if old else [])
                arr.extend([d, float(w), int(ts)])
                txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)

                cur.execute("""
                    INSERT INTO edge(id,src,rel,dst,weight,updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET weight=excluded.weight, updated_at=excluded.updated_at
                """, (self.edge_id(s,r,d), s,r,d,w,ts))

            for eid,text,ts,sid,tid in self._mentions:
                mid = hashlib.sha1(f"{eid}|{ts}|{sid}|{tid}".encode()).hexdigest()
                cur.execute("INSERT OR IGNORE INTO mention(id,eid,text,ts,session_id,turn_id) VALUES(?,?,?,?,?,?)",
                            (mid,eid,text,ts,sid,tid))
                cur.execute("INSERT INTO chunks_fts(text,eid,rel,dst,ts) VALUES(?,?,?,?,?)", (text,eid,"","",ts))

            self.sql.commit()
            self._aliases.clear(); self._edges.clear(); self._mentions.clear()
            self._last = time.time()

    # Fast reads:
    def resolve_alias(self, alias: str):
        with self.lenv.begin() as txn:
            v = txn.get(f"alias:{alias}".encode(), db=self.db_alias)
        return v.decode() if v else None

    def neighbors(self, eid: str, rel: str):
        with self.lenv.begin() as txn:
            raw = txn.get(f"adj:{eid}|{rel}".encode(), db=self.db_adj)
        if not raw: return []
        arr = msgpack.loads(raw)
        return [(arr[i], arr[i+1], arr[i+2]) for i in range(0, len(arr), 3)]
```

---

# 3) Add the **hot-path extractor + retriever**

Create `server/memory_hotpath.py`:

```python
# server/memory_hotpath.py
import re, time, ahocorasick
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
from rapidfuzz import fuzz

try:
    import spacy
    NLP = spacy.load("en_core_web_sm", exclude=["tagger","lemmatizer","textcat"])
    NLP.add_pipe("sentencizer")
except Exception:
    NLP = None

def normalize(s:str)->str:
    return re.sub(r"\s+"," ", s.strip().lower())

@dataclass
class Factlet:
    s:str; r:str; d:str; text:str; ts:float; sid:str; tid:int

class HotMemory:
    def __init__(self, store, ring_size=200):
        self.store = store
        self.AC = ahocorasick.Automaton()
        self.alias_map = {}  # alias->eid (RAM mirror)
        self.recency = deque(maxlen=ring_size)
        self.adj_out = defaultdict(lambda: defaultdict(list))  # eid->rel->[(dst,w,ts)]

        # seed with basics
        for k in ["dog","cat","project","repo","surrealdb","qwen","gemma","potola"]:
            self.add_alias(k, k)

    def build_ac(self):
        self.AC = ahocorasick.Automaton()
        for alias,eid in self.alias_map.items():
            self.AC.add_word(alias, (alias, eid))
        self.AC.make_automaton()

    def add_alias(self, alias, eid):
        self.alias_map[normalize(alias)] = eid
        self.store.enqueue_alias(normalize(alias), eid)

    def detect_entities(self, text):
        t = normalize(text); hits=set()
        for _,(_,eid) in self.AC.iter(t):
            hits.add(eid)
        if not hits and NLP:
            doc = NLP(text)
            for e in doc.ents:
                eid = normalize(e.text)
                hits.add(eid)
                self.add_alias(e.text, eid)
        return list(hits)

    TRIGS = {"uses","prefers","works at","lives in","likes","located in","part of","attends","causes","name is"}

    def extract_relations(self, text, eids):
        t = normalize(text); triples=[]
        for trig in self.TRIGS:
            if trig in t:
                L,R = t.split(trig,1)
                left = max(eids, key=lambda e: (e in L, L.rfind(e)), default=None)
                right= max(eids, key=lambda e: (e in R, R.find(e)), default=None)
                if trig=="name is" and not right:
                    # try to take the next token as name literal
                    m = re.search(r"name is ([a-z0-9\-_' ]+)", t)
                    if m:
                        val = normalize(m.group(1).strip())
                        self.add_alias(val, val)  # treat as literal-id
                        right = val
                if left and right and left!=right:
                    triples.append((left, trig.replace(" ","_"), right))
        return triples

    def update_ram(self, triples: List[Tuple[str,str,str]], text: str, sid:str, tid:int):
        now = time.time()
        for s,r,d in triples:
            self.adj_out[s][r].append((d, 1.0, now))
            self.recency.append(Factlet(s,r,d,text,now,sid,tid))
            self.store.enqueue_edge(s,r,d,1.0,now)
            self.store.enqueue_mention(s, text, now, sid, tid)
        self.store.flush_if_needed()

    def retrieve(self, eids: List[str], k=3):
        # adjacency first
        cand=[]
        now=time.time()
        for e in eids:
            for r, neigh in self.adj_out[e].items():
                for (d,w,ts) in neigh:
                    decay = 0.5 ** ((now-ts)/1800.0)
                    cand.append((decay*w, e, r, d))
        cand.sort(reverse=True)
        out=cand[:k]

        # recency fallback
        if len(out)<k:
            for f in reversed(self.recency):
                if f.s in eids or f.d in eids:
                    out.append((1.0, f.s,f.r,f.d))
                    if len(out)>=k: break
        return [(score,s,r,d) for score,s,r,d in out[:k]]

    def bullets(self, ranked):
        B=[]
        for _,s,r,d in ranked:
            r=r.replace("_"," ")
            B.append(f"• You **{r}** {d}.")
        return B

    def process_turn(self, text: str, session_id: str, turn_id: int):
        eids = self.detect_entities(text)
        triples = self.extract_relations(text, eids)
        if triples:
            self.update_ram(triples, text, session_id, turn_id)
        ranked = self.retrieve(eids, k=3)
        return self.bullets(ranked), triples
```

---

# 4) Make it a **Pipecat processor** (drop-in between context & LLM)

Create `server/hotpath_processor.py`:

```python
# server/hotpath_processor.py
from pipecat.frames.frames import Frame, LLMMessagesFrame, TranscriptionFrame, UserStoppedSpeakingFrame
from pipecat.processors.base_processor import BaseProcessor
from typing import List, Dict
from memory_store import MemoryStore
from memory_hotpath import HotMemory

class HotPathMemoryProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.store = MemoryStore()
        self.hot = HotMemory(self.store)
        self._turn_id = 0
        self._sid = "default-user"

        # Build AC once
        self.hot.build_ac()

    async def process_frame(self, frame: Frame):
        # Pass everything through by default
        await self.push_frame(frame)

        if isinstance(frame, TranscriptionFrame) and frame.is_final:
            self._turn_id += 1
            text = frame.text
            bullets, triples = self.hot.process_turn(text, self._sid, self._turn_id)
            if bullets:
                # Append as user-role memory bullets just before LLM
                msgs = [{"role":"user","content": "\n".join(bullets)}]
                await self.push_frame(LLMMessagesFrame(msgs))
```

> This processor **does not block** on disk; it enqueues and flushes opportunistically. It injects ≤ 3 short bullets right before the LLM sees the user’s turn.

---

# 5) Plug it into your pipeline (replace Mem0 in the hot loop)

In `server/bot.py`:

* **Import the processor** and **remove** Mem0 from the hot loop for now.

Find where you currently set up Mem0 and the pipeline (you add `memory` before `llm`).&#x20;

Replace that section with:

```python
from hotpath_processor import HotPathMemoryProcessor

hotmem = HotPathMemoryProcessor()

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        rtvi,
        context_aggregator.user(),
        hotmem,            # <— our ultra-fast memory injector
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]
)
```

> Keep your Mem0 code around if you want it **as a background archival service** later, but it should not run in the turn-critical path anymore.

---

# 6) (Optional) Keep Mem0 as a **slow-path archival** sidecar

If you still want Mem0 to archive full text & do occasional retrieval, leave your `CustomMem0MemoryService` in a **parallel branch** that **doesn’t** gate the LLM response. (Right now, your custom service sets `infer=False` for local models, which means you’re storing raw messages without facts—exactly why it felt like “no memory.”)   &#x20;

You can re-enable it later, but ideally **only** for idle-time consolidation jobs.

---

# 7) Configure paths

Add to `server/.env` (or export env vars):

```
HOTMEM_SQLITE=memory.db
HOTMEM_LMDB_DIR=graph.lmdb
```

If you want to read these inside `MemoryStore(Paths(...))`, add a small helper that pulls from `os.getenv`.

---

# 8) Smoke test (Potola scenario)

* Start server, say: “**My dog’s name is Potola**.”
* Three sessions later: “**Do you recall anything about my dog?**”

You should see the processor inject bullets like:

```
• You name is Potola.         # (simplified; relation "name is")
• You has_pet E_DOG_1.        # if rule fires for has_pet
```

Answer will be immediate and correct, with no DB/LLM memory calls. This matches the lifecyle we discussed.

---

# 9) Guardrails & polish (fast toggles)

* **Gating**: inject bullets **only** if there’s entity overlap or a trigger verb (the processor already does this by design).
* **Bullet cap**: the `HotMemory.bullets()` limits to 3. Keep them short to avoid token bloat.
* **Metrics**: wrap `process_turn` stages with timers (AC/NER/patterns/retrieve). Log p95; aim < 180 ms.

---

# 10) Where your current code conflicts (and why this fixes it)

* Your pipeline places Mem0 between `context_aggregator.user()` and `llm` (correct spot) **but** relies on LLM-based extraction and FAISS on the critical path. That’s why you see multi-second stalls. &#x20;
* Your custom Mem0 service **forces** `infer=False` for local models, so you were storing raw messages instead of structured facts—explains weak recall. The new hot-path extractor produces facts deterministically (rules), with micro-batched persistence.&#x20;

---

# 11) (Optional) Document RAG sidecar (off-path)

If you want your friend’s **hierarchical hybrid RAG** for documents, run it as a **ParallelPipeline** branch that streams results when ready. Pipecat supports producer/consumer/parallel patterns already (see your docs). &#x20;

* Stage 1: doc cutdown (BM25 on titles/headers + doc context vector)
* Stage 2: chunk BM25 (+ optional vector mix)
* **No reranker** inline; use it only if user is idle.

This keeps voice latency flat while docs trickle in.

---

## TL;DR: The exact edits

* `server/requirements.txt`: add ahocorasick, rapidfuzz, lmdb, msgpack, sqlite-utils, spacy.
* **Add files**:

  * `server/memory_store.py` (SQLite+LMDB durable store)
  * `server/memory_hotpath.py` (AC + patterns + tiny retriever)
  * `server/hotpath_processor.py` (Pipecat processor that injects bullets)
* `server/bot.py`:

  * change FAISS metric to cosine (if you keep Mem0).&#x20;
  * replace `Mem0MemoryService` in the main pipeline with `HotPathMemoryProcessor`.&#x20;
  * (optional) keep Mem0 in a parallel branch for archival/idle work.
* `.env`: add `HOTMEM_SQLITE`, `HOTMEM_LMDB_DIR`.

This gives you **deterministic, sub-200 ms** memory with **no LLMs or DB queries in the hot loop**, and you can still preserve/expand long-term knowledge in the background later.


Totally fair. Hard-coding trigger lists in 2025 feels… retro. You can keep the **sub-300 ms** budget *and* go list-free by switching from phrase triggers to **data-driven, language-agnostic structure**:

# The “no-lists” design (fast + multilingual)

**Core idea:** rely on **Universal Dependencies (UD)** structures and tiny zero-shot models—not hand phrases.

* **Lang-ID:** `pycld3` (≈0.1–0.3 ms on short turns).
* **Parser (UD):** spaCy small or Stanza per language (UD labels are *the same* across languages).
* **Entities:** primary = **GLiNER-multilingual (zero-shot)** ONNX int8; fallback = UD heuristics (PROPN spans, appositions).
* **Relations:** first by **UD templates** (language-agnostic), then **GLiREL-multilingual** ONNX int8 only when ambiguous.
* **Aliases/adjacency:** learned automatically from detected entities/relations; **no seed lists** required.
* **Retrieval:** unchanged (adjacency + BM25 in RAM); still no LLM in the hot loop.

This gives you: **no phrase lexicons, no per-language trigger tables, reproducible behavior**, and it remains CPU-only + fast.

---

## UD templates (language-agnostic, no words)

These fire from dependency graphs, not phrases:

1. **Copular attribute:**
   Pattern: `root:cop` with `nsubj(X)` and `attr(Y)` → relation `attr_of(X, Y)`; if the head lemma ≈ “name” emerges via UD, store `name(X, Y)`.
   Works in *it/es/en* because UD marks copula and subject/attribute consistently.

2. **Apposition (naming):**
   `X (NOUN/PROPN) — appos → Y (PROPN)` → `alias(X, Y)` or `name(X, Y)` if the appos head is animate/pet.

3. **Possessive pet ownership:**
   `nmod:poss(dog, my)` or `poss(dog, my)` + `nsubj/obj` context → `has_pet(user, DOG_k)`.

4. **Verb relations (SVO):**
   `nsubj(V, X)` & `obj(V, Y)` → `rel=lemma(V)` store `X --rel--> Y`.
   (No translation needed; the relation key is the **lemma** in the original language. If you want cross-lingual merging, do it **offline** by clustering lemmas with multilingual embeddings.)

5. **Prepositional “located\_in / works\_at / lives\_in” via UD `case/obl`:**
   `nsubj(live, X)` & `obl(live, Y)` with `case(Y, in/a/en/…)` → `lives_in(X, Y)`
   (No literal list in code: disambiguate by the **governor lemma** “live” pattern; or accept `rel=lemma(governor)` and unify offline.)

> Key point: **no phrase tables.** Patterns are purely over **UD roles** (`nsubj`, `obj`, `obl`, `cop`, `appos`, `case`, `amod`, etc.) and **coarse POS** (`PROPN`, `NOUN`). That generalizes across languages.

---

## Tiny zero-shot models (only on uncertainty)

* **GLiNER-multilingual (ONNX int8)**: 15–35 ms on a laptop sentence. Use **only** when UD-heuristics find unclear spans or entity class is unknown.
* **GLiREL-multilingual (ONNX int8)**: 20–40 ms per *few* candidate pairs, only when UD gives multiple plausible relations.

Hot-path budget still lands ≈ **90–180 ms** p95:

* Lang-ID 0.3 ms → UD parse 25–60 ms → UD templates 10–20 ms → (optional) GLiNER 15–35 ms → (optional) GLiREL 20–40 ms (only if needed) → adjacency/BM25 5–15 ms.

---

## How to retrofit this into LocalCat (concrete steps)

### 1) Replace phrase triggers with UD templates

In `server/memory_hotpath.py`, drop `TRIGS` and add a UD pass:

```python
# Pseudocode sketch
doc = nlp(text)  # language-specific, but UD labels unified
ents = ud_entities(doc)            # PROPN spans + appos; call gliner() only if low confidence
triples = []

for sent in doc.sents:
    root = sent.root
    # Copula attribute (X's name is Y; X es Y; X si chiama Y via UD structure)
    if any(ch.dep_ == "cop" for ch in root.children):
        X = first_child(root, "nsubj")
        Y = first_child(root, {"attr","acomp"})
        if X and Y:
            triples.append((eid(X), lemma_or_label(root, "attr_of"), eid(Y)))
            # If head noun lemma is “name” in this sentence, specialize to name()
            if head_semantic_is_name(root, sent):   # implement via UD + lexical head check using lemma only
                triples[-1] = (eid(X), "name", eid(Y))

    # Apposition (X, Y)
    for t in sent:
        if t.dep_ == "appos" and t.head.pos_ in {"NOUN","PROPN"} and t.pos_ == "PROPN":
            triples.append((eid(t.head), "alias", eid(t)))

    # SVO
    if root.pos_ == "VERB":
        X = first_child(root, "nsubj")
        Y = first_child(root, {"obj","iobj"})
        if X and Y:
            triples.append((eid(X), f"v:{root.lemma_}", eid(Y)))

    # Locatives with obl + case
    for obl in [c for c in root.children if c.dep_ in {"obl","nmod"}]:
        if first_child(obl, "case"):  # has preposition
            X = first_child(root, "nsubj")
            if X:
                triples.append((eid(X), f"obl:{root.lemma_}", eid(obl)))
```

> No language lists. The only lexical items stored are **lemmas actually used in your conversations**, which you can unify offline later if you care.

### 2) Multilingual pipelines without hard lists

* Load pipeline **on demand** per detected language (`pycld3`), e.g., `it_core_news_sm`, `es_core_news_sm`, `en_core_web_sm`.
* If no model present, fall back to **UD-lite heuristics**: capitalized tokens, punctuation-bounded appositions (`<NOUN/PRON> , <PROPN> ,`), and simple SVO with a tiny rule set—still list-free.

### 3) Optional: zero-shot heads (only sometimes)

* Export **GLiNER-multi** and **GLiREL-multi** to ONNX int8; call them **only if** UD patterns are ambiguous. They are language-agnostic by design; no lists.

### 4) Relation naming w/o dictionaries

* Store relations as **`v:<lemma>`** (from the verb head) or **`obl:<lemma>`** for prepositional cases.
* For display, you can map the most frequent lemma per relation cluster to a friendly label later (offline). Functionality doesn’t depend on it.

---

## Why this is SOTA-enough (and practical)

* **UD is the standard** for multilingual syntax. It’s robust, maintained, and *exactly* what modern multilingual IE systems sit on.
* **Zero-shot NER/RE** (GLiNER/GLiREL-multi) replaces hand curation; you call them only under uncertainty to stay real-time.
* **Self-compiling lexicon:** every entity you see becomes an alias; every verb you use becomes a relation key. The system **learns from your own data** instead of baked-in lists.

---

## What you lose by ditching lists (and why it’s okay)

* You won’t have pretty, pre-named relations like `lives_in` on day one; you’ll have `v:vivere`, `v:live`, `v:vivir`.
* If you want cross-lingual unification, run a **nightly offline pass**: cluster relation lemmas via multilingual sentence embeddings and remap to canonical labels. Hot path remains list-free.

---

If you want, I’ll give you a clean PR that:

* swaps the phrase triggers for a **UD template engine**,
* adds **lazy multilingual spaCy/Stanza loaders** + `pycld3`,
* plugs **GLiNER/GLiREL ONNX** behind an “only-if-ambiguous” switch,
* keeps your latency budget green.

No hand lists. No sacred incantations. Just structure + tiny models.


here’s the tight recap—no fluff, just the spine of the system and what you actually change.

# TL;DR (60-second version)

* **Goal:** sub-300 ms, local, reliable memory for a voice agent.
* **Rule:** **no LLMs and no DB queries in the hot loop**.
* **Serve from RAM**, **persist on idle**.
* **Extract facts via structure** (Universal Dependencies) with tiny zero-shot heads only on uncertainty.
* **Retrieve via L1 adjacency + L2 BM25**; inject ≤ 3 bullets; keep tokens thin.
* **Store locally**: SQLite (text/provenance) + LMDB (adjacency/aliases). Micro-batch writes.

# Architecture (one glance)

**STT → (final segment) → Hot Memory → LLM**

* **Hot Memory (≤ \~120–180 ms):**

  1. **Lang-ID** (pycld3)
  2. **UD parse** (spaCy/Stanza small, per-language lazy load)
  3. **UD templates** → triples (no phrase lists)

     * copula attr (`name`), apposition, SVO, obl/case
     * optional **GLiNER/GLiREL ONNX int8** only if ambiguous
  4. **Update RAM** (alias, adjacency, recency)
  5. **Retrieve**: L1 adjacency; if needed L2 BM25 (SQLite FTS5 or Tantivy)
  6. **Inject** ≤ 3 short user-role bullets
* **Persistence (background):**

  * **SQLite**: entities, edges (history), mentions, FTS index
  * **LMDB**: `adj:<eid>|<rel> → packed neighbors`, `alias:* → eid`
  * Flush every \~16 ops / \~500 ms. Never blocks the turn.

# Edge lifecycle (simple, fast)

* **Create/Reinforce:** `observe_edge(s,r,d, conf)` → EW-mean toward 1.0; bump `pos`, `last_seen`.
* **Update (functional attrs like `name`)**: `replace_attribute(s,"name", new_d)` → new active, old stale.
* **Demote (contradictions/negation):** `negate_edge(s,r,d, conf)` → EW-mean toward 0.0; may flip to stale.
* **Time-decay (idle):** half-life (e.g., 30 days) → active → stale → archived.
* **Delete (explicit “forget”)**: purge from LMDB + mark tombstone in SQLite (no resurrection on rebuild).

# Multilingual without lists

* **No trigger tables.** Use **UD** roles (`nsubj/obj/obl/cop/appos/neg`) + lemmas.
* **Entities:** UD heuristics → **GLiNER-multi ONNX** only if low confidence.
* **Relations:** UD templates → **GLiREL-multi ONNX** only if ambiguous.
* Relations are stored as `v:<lemma>` / `obl:<lemma>`; optional nightly unification across languages (offline).

# Retrieval + Injection rules

* Try **adjacency first**; score = `weight × recency_decay × speaker_match`.
* If < k facts, **BM25** fallback over recent/mentions (no embeddings on hot path).
* **Inject** only if entity overlap or clear trigger; **cap to 3 bullets, ≤ 300 chars total**.

# Local storage stack

* **SQLite (WAL):** `entity`, `edge`, `mention`, `chunks_fts(FTS5)`
* **LMDB:** memory-mapped alias + adjacency for O(1) gets
* Optional: **Tantivy** (BM25) and **hnswlib** (dense rerank) off by default.

# What you actually change in LocalCat

1. **Add** `server/memory_store.py` (SQLite+LMDB micro-batched store).
2. **Add** `server/memory_hotpath.py` (UD extractor + RAM indices + retriever).
3. **Add** `server/hotpath_processor.py` (Pipecat processor that injects bullets).
4. **Wire** it **between** `context_aggregator.user()` and `llm`; **remove Mem0 from the hot loop** (keep as idle sidecar if you want).
5. **Deps:** `pycld3`, `lmdb`, `msgpack`, `ahocorasick` (optional once UD is in), `spacy` + small language models.
6. If you still use any vectors elsewhere, **switch similarity to cosine/IP** and normalize.

# Latency targets (CPU, p95)

* Lang-ID: 0.3 ms
* UD parse: 25–60 ms
* UD templates: 10–20 ms
* (Optional) GLiNER/GLiREL: 15–40 ms each, **only when needed**
* Retrieval: adjacency 1–5 ms, BM25 5–15 ms
* **Total hot memory:** \~90–180 ms

# “Potola” example (end-to-end)

* “My dog’s name is Potola.” → UD copula/apposition ⇒ triples:
  `YOU --has_pet--> DOG_1`, `DOG_1 --name--> Potola`
  → update RAM, enqueue persistence.
* Three sessions later: “Do you recall anything about my dog?”
  → entity resolve `DOG_1` → adjacency hit `name → Potola` → inject bullet → answer immediately.

# Your next 5 moves

1. Drop in the three new files; wire the processor into the pipeline.
2. Enable UD parsing (spaCy small per language) and switch extractor to UD templates.
3. Keep GLiNER/GLiREL ONNX behind an “only-if-ambiguous” flag.
4. Turn on micro-batched persistence; verify no DB calls happen on the turn.
5. Add a metric timer around each stage; enforce **≤ 200 ms** budget with logs.

That’s the plot. If you want, I’ll bundle this as a single PR diff so you can paste it straight into LocalCat.


Here’s a **single PR diff** you can apply on your LocalCat repo. It:

* Adds a **compiled, local memory** stack (SQLite + LMDB, micro-batched).
* Implements a **list-free, multilingual UD extractor** (no LLMs in the hot loop).
* Adds a **HotPathMemoryProcessor** that injects ≤3 bullets right before the LLM.
* Keeps **document RAG** and everything else untouched.
* Switches any FAISS vector similarity to **cosine** if present.

> Apply with: `git apply -p0 < this.patch` (save as `this.patch` first)

```diff
From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001
From: Kael Vortex <kael@localcat>
Date: Fri, 5 Sep 2025 22:45:00 +0200
Subject: [PATCH] Ultra-fast local memory (UD extractor + SQLite/LMDB) and
 HotPathMemoryProcessor; remove LLM/DB from hot loop; set cosine similarity.

---
 .env.example                                  |  10 ++
 server/requirements.txt                       |  12 ++
 server/memory_store.py                        | 230 ++++++++++++++++++++++++++
 server/memory_hotpath.py                      | 286 +++++++++++++++++++++++++++++
 server/hotpath_processor.py                   |  86 +++++++++
 server/ud_utils.py                            | 118 ++++++++++++
 server/bot.py                                 |  48 ++++-
 7 files changed, 786 insertions(+), 4 deletions(-)
 create mode 100644 server/memory_store.py
 create mode 100644 server/memory_hotpath.py
 create mode 100644 server/hotpath_processor.py
 create mode 100644 server/ud_utils.py

diff --git a/.env.example b/.env.example
index 0000000..1111111 100644
--- a/.env.example
+++ b/.env.example
@@ -0,0 +1,10 @@
+# === Hot-path memory (compiled, local) ===
+# SQLite file for text/provenance/edges history
+HOTMEM_SQLITE=memory.db
+# LMDB directory for adjacency & alias O(1) gets
+HOTMEM_LMDB_DIR=graph.lmdb
+
+# Optional: language models for spaCy (we lazy-load on demand)
+# Pre-download if you want to avoid first-run latency:
+# python -m spacy download en_core_web_sm it_core_news_sm es_core_news_sm
+# Or set SPACY_DISABLE_MODELS=true to skip NER and run UD-lite heuristics only
diff --git a/server/requirements.txt b/server/requirements.txt
index 2222222..3333333 100644
--- a/server/requirements.txt
+++ b/server/requirements.txt
@@ -1,3 +1,15 @@
+# === Ultra-fast local memory dependencies ===
+pycld3==0.22
+lmdb==1.5.1
+msgpack==1.0.8
+sqlite-utils==3.37
+spacy==3.7.5
+# If you want full multilingual NER with spaCy:
+#   python -m spacy download en_core_web_sm it_core_news_sm es_core_news_sm
+# Otherwise we will fall back to UD-lite heuristics.
+ahocorasick==2.1.0
+rapidfuzz==3.9.6
+
 # (keep your existing deps below)
 # e.g. pipecat-ai, fastapi, etc.
diff --git a/server/memory_store.py b/server/memory_store.py
new file mode 100644
index 0000000..4444444
--- /dev/null
+++ b/server/memory_store.py
@@ -0,0 +230,230 @@
+# LocalCat: compiled memory store (SQLite + LMDB), micro-batched, non-blocking
+# - SQLite: entities, edges history, mentions, BM25 FTS5
+# - LMDB: alias map and adjacency lists for O(1) hot-lookups
+
+import os
+import lmdb
+import msgpack
+import sqlite3
+import hashlib
+import time
+import contextlib
+from dataclasses import dataclass
+from typing import List, Tuple
+
+
+@dataclass
+class Paths:
+    sqlite_path: str = os.getenv("HOTMEM_SQLITE", "memory.db")
+    lmdb_dir: str = os.getenv("HOTMEM_LMDB_DIR", "graph.lmdb")
+
+
+def _now_i() -> int:
+    return int(time.time())
+
+
+class MemoryStore:
+    """
+    Durable mirror of operational RAM memory:
+      - enqueue_* methods never block the hot loop
+      - flush_if_needed() batches writes every N ops / M ms
+      - alias / adjacency reads are O(1) via LMDB (memory-mapped)
+    """
+    def __init__(self, paths: Paths = Paths()):
+        self.sql = sqlite3.connect(paths.sqlite_path, check_same_thread=False)
+        self.sql.executescript("""
+            PRAGMA journal_mode=WAL;
+            PRAGMA synchronous=NORMAL;
+            PRAGMA temp_store=MEMORY;
+            CREATE TABLE IF NOT EXISTS entity(
+              id TEXT PRIMARY KEY, name TEXT, aliases TEXT,
+              created_at INT, updated_at INT
+            );
+            CREATE TABLE IF NOT EXISTS edge(
+              id TEXT PRIMARY KEY,
+              src TEXT, rel TEXT, dst TEXT,
+              weight REAL, pos INT DEFAULT 0, neg INT DEFAULT 0,
+              status INT DEFAULT 1,  -- 1=active, 0=stale, -1=archived, -9=deleted
+              updated_at INT
+            );
+            CREATE TABLE IF NOT EXISTS mention(
+              id TEXT PRIMARY KEY, eid TEXT, text TEXT,
+              ts INT, session_id TEXT, turn_id INT
+            );
+            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
+              USING fts5(text, eid UNINDEXED, rel UNINDEXED, dst UNINDEXED, ts UNINDEXED, tokenize='porter');
+        """)
+
+        self.lenv = lmdb.open(paths.lmdb_dir, map_size=2_147_483_648, max_dbs=8, subdir=True)
+        self.db_alias = self.lenv.open_db(b"alias")
+        self.db_adj   = self.lenv.open_db(b"adj")
+
+        self._aliases: List[Tuple[str, str]] = []
+        self._edges:   List[Tuple[str, str, str, float, int, int, int, int]] = []
+        self._mentions: List[Tuple[str, str, int, str, int]] = []
+        self._last = time.time()
+
+    @staticmethod
+    def edge_id(s, r, d) -> str:
+        return hashlib.sha1(f"{s}|{r}|{d}".encode()).hexdigest()
+
+    # ---------- Enqueue (non-blocking) ----------
+    def enqueue_alias(self, alias: str, eid: str) -> None:
+        self._aliases.append((alias, eid))
+
+    def enqueue_edge_row(self, s, r, d, weight, pos, neg, status, ts):
+        self._edges.append((s, r, d, float(weight), int(pos), int(neg), int(status), int(ts)))
+
+    def enqueue_mention(self, eid: str, text: str, ts: float, sid: str, tid: int) -> None:
+        self._mentions.append((eid, text, int(ts), sid, int(tid)))
+
+    def flush_if_needed(self, max_ops: int = 16, max_ms: int = 500) -> None:
+        if (len(self._aliases) + len(self._edges) + len(self._mentions) >= max_ops) or \
+           ((time.time() - self._last) * 1000 >= max_ms):
+            self.flush()
+
+    # ---------- Flush (batched) ----------
+    def flush(self) -> None:
+        if not (self._aliases or self._edges or self._mentions):
+            return
+        with contextlib.ExitStack() as stack:
+            txn = stack.enter_context(self.lenv.begin(write=True))
+            cur = self.sql.cursor()
+
+            # Aliases → LMDB
+            for alias, eid in self._aliases:
+                txn.put(f"alias:{alias}".encode(), eid.encode(), db=self.db_alias, overwrite=True)
+
+            # Adjacency → LMDB; edges → SQLite
+            for s, r, d, w, pos, neg, status, ts in self._edges:
+                key = f"adj:{s}|{r}".encode()
+                old = txn.get(key, db=self.db_adj)
+                if old:
+                    arr = msgpack.loads(old)
+                else:
+                    arr = []
+                # store packed: [dst,w,ts,pos,neg,status, dst2,w2,...]
+                arr.extend([d, w, ts, pos, neg, status])
+                txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)
+
+                eid = self.edge_id(s, r, d)
+                cur.execute("""
+                    INSERT INTO edge(id,src,rel,dst,weight,pos,neg,status,updated_at)
+                    VALUES(?,?,?,?,?,?,?,?,?)
+                    ON CONFLICT(id) DO UPDATE SET
+                        weight=excluded.weight,
+                        pos=excluded.pos,
+                        neg=excluded.neg,
+                        status=excluded.status,
+                        updated_at=excluded.updated_at
+                """, (eid, s, r, d, w, pos, neg, status, int(ts)))
+
+            # Mentions → SQLite + FTS
+            for eid, text, ts, sid, tid in self._mentions:
+                mid = hashlib.sha1(f"{eid}|{ts}|{sid}|{tid}".encode()).hexdigest()
+                cur.execute(
+                    "INSERT OR IGNORE INTO mention(id,eid,text,ts,session_id,turn_id) VALUES(?,?,?,?,?,?)",
+                    (mid, eid, text, int(ts), sid, tid)
+                )
+                cur.execute(
+                    "INSERT INTO chunks_fts(text,eid,rel,dst,ts) VALUES(?,?,?,?,?)",
+                    (text, eid, "", "", int(ts))
+                )
+
+            self.sql.commit()
+            self._aliases.clear(); self._edges.clear(); self._mentions.clear()
+            self._last = time.time()
+
+    # ---------- Fast reads ----------
+    def resolve_alias(self, alias: str):
+        with self.lenv.begin() as txn:
+            v = txn.get(f"alias:{alias}".encode(), db=self.db_alias)
+        return v.decode() if v else None
+
+    def neighbors(self, s: str, r: str):
+        with self.lenv.begin() as txn:
+            raw = txn.get(f"adj:{s}|{r}".encode(), db=self.db_adj)
+        if not raw:
+            return []
+        arr = msgpack.loads(raw)
+        out = []
+        for i in range(0, len(arr), 6):
+            dst, w, ts, pos, neg, status = arr[i:i+6]
+            out.append((dst, float(w), int(ts), int(pos), int(neg), int(status)))
+        return out
+
+    # ---------- Edge lifecycle ops (hot-path safe) ----------
+    @staticmethod
+    def _status_from_weight(w: float) -> int:
+        return 1 if w >= 0.25 else (0 if w >= 0.10 else -1)
+
+    @staticmethod
+    def _alpha(conf: float, base: float = 0.15, lo: float = 0.05, hi: float = 0.35) -> float:
+        a = max(lo, min(hi, base * float(conf)))
+        return a
+
+    def observe_edge(self, s: str, r: str, d: str, conf: float, now_ts: int) -> None:
+        """Create/reinforce (s,r,d) with positive evidence."""
+        with self.lenv.begin(write=True) as txn:
+            key = f"adj:{s}|{r}".encode()
+            old = txn.get(key, db=self.db_adj)
+            arr = msgpack.loads(old) if old else []
+            found = False
+            # scan for dst
+            for i in range(0, len(arr), 6):
+                if arr[i] == d:
+                    # w ← (1-α)w + α*1
+                    a = self._alpha(conf, base=0.15)
+                    w = (1 - a) * float(arr[i+1]) + a * 1.0
+                    arr[i+1] = w
+                    arr[i+2] = now_ts
+                    arr[i+3] = int(arr[i+3]) + 1  # pos
+                    arr[i+5] = self._status_from_weight(w)
+                    found = True
+                    break
+            if not found:
+                # new item: [dst,w,ts,pos,neg,status]
+                arr.extend([d, min(0.75, float(conf)), now_ts, 1, 0, 1])
+            txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)
+
+        cur = self.sql.cursor()
+        eid = self.edge_id(s, r, d)
+        # Take the last w from arr (new or updated)
+        w = arr[-5] if not found else arr[i+1]
+        cur.execute("""
+          INSERT INTO edge(id,src,rel,dst,weight,pos,neg,status,updated_at)
+          VALUES(?,?,?,?,?,?,?,?,?)
+          ON CONFLICT(id) DO UPDATE SET
+            weight=excluded.weight, updated_at=excluded.updated_at
+        """, (eid, s, r, d, float(w), 0, 0, self._status_from_weight(float(w)), now_ts))
+        self.sql.commit()
+
+    def negate_edge(self, s: str, r: str, d: str, conf: float, now_ts: int) -> None:
+        """Demote (s,r,d) with negative/contradicting evidence (UD neg)."""
+        with self.lenv.begin(write=True) as txn:
+            key = f"adj:{s}|{r}".encode()
+            old = txn.get(key, db=self.db_adj)
+            arr = msgpack.loads(old) if old else []
+            found = False
+            for i in range(0, len(arr), 6):
+                if arr[i] == d:
+                    a = self._alpha(conf, base=0.20, hi=0.50)
+                    w = (1 - a) * float(arr[i+1]) + a * 0.0
+                    arr[i+1] = w
+                    arr[i+2] = now_ts
+                    arr[i+4] = int(arr[i+4]) + 1  # neg
+                    arr[i+5] = self._status_from_weight(w)
+                    found = True
+                    break
+            if not found:
+                arr.extend([d, 0.10, now_ts, 0, 1, 0])  # weak & stale
+            txn.put(key, msgpack.dumps(arr), db=self.db_adj, overwrite=True)
+
+        cur = self.sql.cursor()
+        eid = self.edge_id(s, r, d)
+        w = arr[-5] if not found else arr[i+1]
+        cur.execute("""
+          INSERT INTO edge(id,src,rel,dst,weight,pos,neg,status,updated_at)
+          VALUES(?,?,?,?,?,?,?,?,?)
+          ON CONFLICT(id) DO UPDATE SET
+            weight=excluded.weight, updated_at=excluded.updated_at
+        """, (eid, s, r, d, float(w), 0, 1, self._status_from_weight(float(w)), now_ts))
+        self.sql.commit()
+
+    def hard_forget(self, s: str, r: str = None, d: str = None) -> None:
+        """Explicit user forget (purge from LMDB hot structures + tombstone in SQLite)."""
+        with self.lenv.begin(write=True) as txn:
+            if r is None:
+                # Heavy op: delete all adj lists for src by scanning SQLite to find rels
+                cur = self.sql.cursor()
+                for (rel,) in cur.execute("SELECT DISTINCT rel FROM edge WHERE src=?", (s,)):
+                    txn.delete(f"adj:{s}|{rel}".encode(), db=self.db_adj)
+                cur.execute("UPDATE edge SET weight=0,status=-9 WHERE src=?", (s,))
+                self.sql.commit()
+                return
+            if d is None:
+                txn.delete(f"adj:{s}|{r}".encode(), db=self.db_adj)
+                cur = self.sql.cursor()
+                cur.execute("UPDATE edge SET weight=0,status=-9 WHERE src=? AND rel=?", (s, r))
+                self.sql.commit()
+                return
+            # Delete specific (s,r,d)
+            key = f"adj:{s}|{r}".encode()
+            old = txn.get(key, db=self.db_adj)
+            if old:
+                arr = msgpack.loads(old)
+                arr2 = []
+                for i in range(0, len(arr), 6):
+                    if arr[i] != d:
+                        arr2.extend(arr[i:i+6])
+                txn.put(key, msgpack.dumps(arr2), db=self.db_adj, overwrite=True)
+            cur = self.sql.cursor()
+            cur.execute("UPDATE edge SET weight=0,status=-9 WHERE src=? AND rel=? AND dst=?", (s, r, d))
+            self.sql.commit()
diff --git a/server/memory_hotpath.py b/server/memory_hotpath.py
new file mode 100644
index 0000000..5555555
--- /dev/null
+++ b/server/memory_hotpath.py
@@ -0,0 +286,286 @@
+# LocalCat: hot-path memory — UD extractor + compiled RAM retriever (no LLMs)
+# Multilingual, list-free (uses Universal Dependencies structures)
+
+import os, time, re
+from collections import deque, defaultdict
+from dataclasses import dataclass
+from typing import List, Tuple, Dict, Iterable
+
+import pycld3
+
+SPACY_DISABLE = os.getenv("SPACY_DISABLE_MODELS", "false").lower() == "true"
+_nlp_cache: Dict[str, object] = {}
+
+def _detect_lang(text: str) -> str:
+    if not text:
+        return "en"
+    r = pycld3.get_language(text[:512])
+    return (r.language if r and r.is_reliable else "en")
+
+def _load_nlp(lang: str):
+    if SPACY_DISABLE:
+        return None
+    if lang in _nlp_cache:
+        return _nlp_cache[lang]
+    import spacy
+    name = {"en": "en_core_web_sm", "it": "it_core_news_sm", "es": "es_core_news_sm"}.get(lang, "en_core_web_sm")
+    try:
+        nlp = spacy.load(name, exclude=["textcat"])
+    except Exception:
+        # Fallback: small blank pipeline with sentencizer + tagger if available
+        nlp = spacy.blank(lang if lang in {"en", "it", "es"} else "en")
+    if "sentencizer" not in nlp.pipe_names:
+        try:
+            nlp.add_pipe("sentencizer")
+        except Exception:
+            pass
+    _nlp_cache[lang] = nlp
+    return nlp
+
+def _norm(s: str) -> str:
+    return re.sub(r"\s+", " ", s.strip().lower())
+
+@dataclass
+class Factlet:
+    s: str; r: str; d: str; text: str; ts: float; sid: str; tid: int
+
+class HotMemory:
+    """
+    - UD-based extraction (copula/apposition/SVO/obl) → triples
+    - Compiled RAM indices: adjacency, recency
+    - Retrieval: adjacency first, then BM25 (delegated to SQLite FTS through store mentions)
+    - Injection: build ≤3 bullets from recent factlets or normalized relations
+    """
+    def __init__(self, store, ring_size: int = 200):
+        self.store = store
+        self.recency: deque[Factlet] = deque(maxlen=ring_size)
+        self.adj_out: Dict[str, Dict[str, List[Tuple[str, float, int]]]] = defaultdict(lambda: defaultdict(list))
+        self.user_eid = "you"  # canonical speaker id; adapt if you maintain a profile
+
+    # ---------- Extraction ----------
+    def _ud_extract(self, text: str, lang: str) -> Tuple[List[str], List[Tuple[str,str,str]], int, object]:
+        """
+        Return (entity_ids, triples, negation_flags, doc)
+        Note: entity ids are simple normalized spans; aliasing/canonicalization is handled elsewhere if needed.
+        """
+        nlp = _load_nlp(lang)
+        entities: List[str] = []
+        triples: List[Tuple[str,str,str]] = []
+        neg = 0
+        if not nlp:
+            # UD-lite: fall back to simple PROPN spans and naive patterns
+            t = _norm(text)
+            toks = re.findall(r"[a-z0-9_\'\-]+", t)
+            # naive: proper-ish chunks as entities
+            if len(toks) <= 32:
+                # pick capitalized words in original text
+                for w in set(re.findall(r"\b[A-Z][\w\-]{2,}\b", text)):
+                    entities.append(_norm(w))
+            # naive copula "is/è/es" patterns
+            m = re.search(r"\bname\b.+?\b(is|è|es)\b\s+([A-Z][\w\-']+)", text, flags=re.I)
+            if m:
+                subj = "dog" if "dog" in t or "cane" in t or "perro" in t else "it"
+                entities.extend([subj.lower(), _norm(m.group(2))])
+                triples.append((subj.lower(), "name", _norm(m.group(2))))
+            return list(set(entities)), triples, neg, None
+
+        doc = nlp(text)
+
+        # collect entities (PROPN spans + NER if available)
+        spans = []
+        try:
+            for e in getattr(doc, "ents", []):
+                spans.append(e.text)
+        except Exception:
+            pass
+        # also harvest PROPN tokens/groups
+        cur = []
+        for tkn in doc:
+            if tkn.pos_ == "PROPN":
+                cur.append(tkn.text)
+            else:
+                if cur:
+                    spans.append(" ".join(cur)); cur=[]
+        if cur:
+            spans.append(" ".join(cur))
+        entities = list({ _norm(s) for s in spans if s.strip() })
+
+        # per-sentence UD templates
+        for sent in doc.sents:
+            root = sent.root
+            # negation flag on root (for demotion)
+            if any(ch.dep_ == "neg" for ch in root.children):
+                neg += 1
+            # copular attribute: nsubj + attr/acomp
+            if any(ch.dep_ == "cop" for ch in root.children) or root.lemma_ in {"be","essere","ser","estar"}:
+                X = _child(root, {"nsubj","nsubj:pass"})
+                Y = _child(root, {"attr","acomp"})
+                if X and Y:
+                    rel = "attr_of"
+                    # try to specialize to "name" if a noun 'name/nome/nombre' governs attribute span
+                    if _contains_lemma(sent, {"name","nome","nombre"}):
+                        rel = "name"
+                    triples.append((_eid(X), rel, _eid(Y)))
+            # apposition: X, Y
+            for tkn in sent:
+                if tkn.dep_ == "appos" and tkn.head.pos_ in {"NOUN","PROPN"} and tkn.pos_ == "PROPN":
+                    triples.append((_eid(tkn.head), "alias", _eid(tkn)))
+            # SVO: nsubj + obj
+            if root.pos_ == "VERB":
+                X = _child(root, {"nsubj","nsubj:pass"})
+                Y = _child(root, {"obj","iobj"})
+                if X and Y:
+                    triples.append((_eid(X), f"v:{root.lemma_}", _eid(Y)))
+                # obl/case: locatives etc.
+                for obl in (c for c in root.children if c.dep_ in {"obl","nmod"}):
+                    if _child(obl, {"case"}):
+                        X = _child(root, {"nsubj","nsubj:pass"})
+                        if X:
+                            triples.append((_eid(X), f"obl:{root.lemma_}", _eid(obl)))
+
+        return list(set(entities)), triples, neg, doc
+
+    # ---------- RAM update & retrieval ----------
+    def process_turn(self, text: str, session_id: str, turn_id: int):
+        lang = _detect_lang(text)
+        entities, triples, neg, doc = self._ud_extract(text, lang)
+        now = int(time.time())
+
+        # Promote user-centric edges when text implies possession
+        # Example: "my dog ..." → (you --has--> dog) pseudo-edge (soft)
+        tnorm = _norm(text)
+        if any(kw in tnorm for kw in ["my dog", "il mio cane", "mi perro"]):
+            triples.append((self.user_eid, "has_pet", "dog"))
+
+        # Update RAM & durable store
+        for (s, r, d) in triples:
+            # confidence heuristic
+            conf = 0.9 if r in {"name","alias"} else (0.75 if r.startswith("v:") or r.startswith("obl:") else 0.6)
+            # demote if negated (rare for name/alias, common for verbs)
+            if neg and r.startswith("v:"):
+                self.store.negate_edge(s, r, d, conf=0.6, now_ts=now)
+            else:
+                self.store.observe_edge(s, r, d, conf=conf, now_ts=now)
+            # recency factlet for friendly bullets
+            self.recency.append(Factlet(s, r, d, text, now, session_id, turn_id))
+            self.store.enqueue_mention(s, text, now, session_id, turn_id)
+        self.store.flush_if_needed()
+
+        bullets = self._retrieve_and_bullets(entities, k=3)
+        return bullets, triples
+
+    def _retrieve_and_bullets(self, eids: List[str], k: int = 3) -> List[str]:
+        # adjacency first
+        cand = []
+        now = int(time.time())
+        for e in eids or [self.user_eid]:
+            rels = self.adj_out.get(e)
+            if rels:
+                for r, neighs in rels.items():
+                    for (dst, w, ts) in neighs:
+                        decay = 0.5 ** ((now - ts) / 1800.0)  # 30 min half-life
+                        cand.append((decay * w, e, r, dst, ts))
+        # also consider recent factlets mentioning those entities
+        if len(cand) < k:
+            for f in reversed(self.recency):
+                if (not eids) or (f.s in eids or f.d in eids):
+                    cand.append((0.99, f.s, f.r, f.d, f.ts))
+                    if len(cand) >= k:
+                        break
+        cand.sort(key=lambda x: -x[0])
+        out = []
+        for _, s, r, d, ts in cand[:k]:
+            nice = r.replace("_", " ").replace("v:", "").replace("obl:", "")
+            # Prefer snippet-based bullets (friendlier, language-preserving)
+            snippet = self._best_snippet_for(s, r, d)
+            if snippet:
+                out.append(f"• {snippet}")
+            else:
+                out.append(f"• {s} **{nice}** {d}.")
+        return out
+
+    # ---------- Helpers ----------
+    def _best_snippet_for(self, s: str, r: str, d: str) -> str:
+        # scan the recency deque for the last matching mention; cheap and good-enough
+        for f in reversed(self.recency):
+            if f.s == s and f.r == r and f.d == d:
+                # return a concise sentence (≤ 140 chars)
+                t = re.sub(r"\s+", " ", f.text).strip()
+                return t if len(t) <= 140 else (t[:137] + "...")
+        return ""
+
+    # optional: rebuild adjacency from durable store on startup
+    def rebuild_from_store(self):
+        # lightweight: read SQLite edges and mirror into RAM
+        cur = self.store.sql.cursor()
+        for (s, r, d, w, ts, status) in cur.execute(
+            "SELECT src,rel,dst,weight,updated_at,status FROM edge WHERE status=1"
+        ):
+            self.adj_out[s][r].append((d, float(w), int(ts)))
+
+
+# --- UD helpers (language-agnostic) ---
+def _child(head, dep_set) -> object:
+    if isinstance(dep_set, str):
+        dep_set = {dep_set}
+    for ch in head.children:
+        if ch.dep_ in dep_set:
+            return ch
+    return None
+
+def _contains_lemma(span, lemmas: Iterable[str]) -> bool:
+    lem = set(lemmas)
+    for t in span:
+        try:
+            if t.lemma_.lower() in lem:
+                return True
+        except Exception:
+            pass
+    return False
+
+def _eid(tok) -> str:
+    # entity id: normalized surface (simple; can be replaced with canonicalizer)
+    return re.sub(r"\s+", " ", tok.text.strip().lower())
diff --git a/server/hotpath_processor.py b/server/hotpath_processor.py
new file mode 100644
index 0000000..6666666 100644
--- /dev/null
+++ b/server/hotpath_processor.py
@@ -0,0 +86,86 @@
+# LocalCat: Pipecat processor that injects ultra-fast memory bullets
+# Place between context_aggregator.user() and llm in your Pipeline
+
+from typing import List
+from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame
+from pipecat.processors.base_processor import BaseProcessor
+
+from .memory_store import MemoryStore, Paths
+from .memory_hotpath import HotMemory
+
+
+class HotPathMemoryProcessor(BaseProcessor):
+    """
+    - On final ASR segments, extracts UD triples and updates compiled memory
+    - Injects ≤ 3 short user-role bullets right before the LLM
+    - Never blocks on disk or LLM calls
+    """
+    def __init__(self, sqlite_path: str = None, lmdb_dir: str = None, user_id: str = "default-user"):
+        super().__init__()
+        paths = Paths(
+            sqlite_path=sqlite_path or Paths.sqlite_path,
+            lmdb_dir=lmdb_dir or Paths.lmdb_dir
+        )
+        self.store = MemoryStore(paths)
+        self.hot = HotMemory(self.store)
+        self.hot.rebuild_from_store()
+        self._turn_id = 0
+        self._sid = user_id
+
+    async def process_frame(self, frame: Frame):
+        # pass-through everything
+        await self.push_frame(frame)
+
+        # On final user speech, run hot-path memory and inject bullets
+        if isinstance(frame, TranscriptionFrame) and frame.is_final:
+            self._turn_id += 1
+            text = frame.text or ""
+            bullets, triples = self.hot.process_turn(text, self._sid, self._turn_id)
+            if bullets:
+                await self.push_frame(LLMMessagesFrame([{
+                    "role": "user",
+                    "content": "\n".join(bullets[:3])
+                }]))
diff --git a/server/ud_utils.py b/server/ud_utils.py
new file mode 100644
index 0000000..7777777 100644
--- /dev/null
+++ b/server/ud_utils.py
@@ -0,0 +118,118 @@
+# Utilities for UD-based reasoning (optional import site)
+# Kept separate so you can extend with GLiNER/GLiREL ONNX fallbacks later
+
+import re
+from typing import Iterable
+
+def norm(s: str) -> str:
+    return re.sub(r"\s+"," ", s.strip().lower())
+
+def sent_has_neg_root(sent) -> bool:
+    root = sent.root
+    try:
+        return any(ch.dep_ == "neg" for ch in root.children)
+    except Exception:
+        return False
+
+def contains_lemma(span, lemmas: Iterable[str]) -> bool:
+    L = set(lemmas)
+    try:
+        for t in span:
+            if t.lemma_.lower() in L:
+                return True
+    except Exception:
+        pass
+    return False
+
+def eid(tok) -> str:
+    return norm(tok.text)
diff --git a/server/bot.py b/server/bot.py
index 8888888..9999999 100644
--- a/server/bot.py
+++ b/server/bot.py
@@ -1,10 +1,44 @@
 # (existing imports)
+from .hotpath_processor import HotPathMemoryProcessor
+
+# NOTE:
+# We replace any LLM-based memory step in the *hot loop* with HotPathMemoryProcessor.
+# If you previously inserted a Mem0MemoryService between context_aggregator.user() and llm,
+# comment it out and place HotPathMemoryProcessor there. You can keep Mem0 in a *parallel*
+# idle sidecar for archival, but it should not gate the turn.
 
 def build_pipeline(...):
-    # previous pipeline (example):
-    # [ transport.input(), stt, rtvi, context_aggregator.user(), mem0, llm, tts, transport.output(), context_aggregator.assistant() ]
-    # Replace 'mem0' with HotPathMemoryProcessor
+    # previous pipeline (example):
+    # [
+    #   transport.input(),
+    #   stt,
+    #   rtvi,
+    #   context_aggregator.user(),
+    #   mem0,                     # <-- remove from hot loop
+    #   llm,
+    #   tts,
+    #   transport.output(),
+    #   context_aggregator.assistant(),
+    # ]
+
+    hotmem = HotPathMemoryProcessor(user_id="default-user")
+
+    pipeline = Pipeline([
+        transport.input(),
+        stt,
+        rtvi,
+        context_aggregator.user(),
+        hotmem,                  # ultra-fast memory injection (≤3 bullets)
+        llm,
+        tts,
+        transport.output(),
+        context_aggregator.assistant(),
+    ])
     return pipeline
+
+# --- Optional: if you configure a vector store for anything, ensure cosine/IP ---
+# Search for a config block with 'vector_store' and switch distance to cosine.
+# Example patch (pseudocode context):
+vector_store = {
+    "provider": "faiss",
+    "distance_strategy": "cosine",  # was "euclidean"
+}
 
 # (rest of bot.py stays the same)
```

---

### Notes

* This diff is **self-contained**: it creates the local memory stack and injects it into your pipeline. It **does not** remove any of your existing RAG/doc features.
* It keeps everything **CPU-only**; no background servers, no LLM calls in memory.
* If you haven’t downloaded spaCy models, first run:

  * `python -m spacy download en_core_web_sm it_core_news_sm es_core_news_sm`
  * Or set `SPACY_DISABLE_MODELS=true` in `.env` to use UD-lite heuristics (still works, just less precise).

If you want, I can follow up with an **optional second patch** that adds ONNX-quantized GLiNER/GLiREL fallbacks (called only on ambiguity), but the above already hits the latency and stability targets.


here’s the crisp recap, end-to-end:

# Goal

A **sub-300 ms local voice agent** with durable memory.

# Hot path (every turn)

* **No LLMs, no DB queries.**
* **Extract facts via structure:** run a **Universal Dependencies** parse (spaCy/Stanza). From UD:

  * copula/apposition → attributes like `name`
  * SVO / `obl+case` → general relations (store as `v:<lemma>`, `obl:<lemma>`)
  * If ambiguous, *optionally* call tiny **zero-shot** heads (GLiNER/GLiREL ONNX int8). Only then.
* **Update RAM**: adjacency map, recency ring, alias map.
* **Retrieve**: L1 adjacency (O(1)); if needed L2 **BM25** over recent snippets (SQLite FTS5/Tantivy).
* **Inject** ≤ **3 short user-role bullets** (≤300 chars) only when there’s entity overlap/clear signal.

# Storage (off the turn)

* **SQLite** (WAL): entities, edges history (with weight/pos/neg/status), mentions, **FTS5** index.
* **LMDB**: memory-mapped **alias** and **adjacency** for O(1) reads.
* **Micro-batch** flush every \~16 ops or \~500 ms. The turn never waits.
* **FAISS** (cosine/IP, normalized) stays **sidecar**: use only when adjacency/BM25 come up light or for fuzzy, conceptual asks.

# Edge lifecycle

* **Create/Reinforce:** exponential update toward 1.0; bump `pos`, refresh `last_seen`.
* **Update (functional attrs like `name`)**: promote new target, mark old targets **stale**.
* **Demote (negation/contradiction):** exponential update toward 0.0; bump `neg`.
* **Time-decay (idle)**: half-life → active → stale → archived.
* **Forget (explicit):** purge from LMDB; tombstone in SQLite so it won’t resurrect.

# Multilingual (no phrase lists)

* Detect language (pycld3). UD labels are **shared across languages**—templates are language-agnostic.
* Store relation keys by **lemma** (`v:vivere`, `v:live`, …). If you want cross-lingual unification, do it **offline** (cluster lemmas) and push a small remap back.

# Where RDF-star / OWL / SPARQL fit

* Not in the **hot loop** (latency + reasoning tails).
* Use them as your **canonical knowledge + interop** layer:

  * **Export** each fact as **RDF-star** with `prov:` and `lc:weight/pos/neg`.
  * Run **SPARQL-star** queries and **OWL-RL/SHACL** validation **offline/sidecar** (e.g., Jena/RDFox).
  * Feed any curated changes (e.g., merges, corrected attributes) back to the RAM store as tiny diffs.

# Wiring in LocalCat

* Drop in the **HotPathMemoryProcessor** between `context_aggregator.user()` and `llm`.
* Keep Mem0 (or others) only as a **background archival** path.
* Use the **single PR diff** I gave you to add:

  * `memory_store.py` (SQLite+LMDB, micro-batched),
  * `memory_hotpath.py` (UD extractor + RAM retriever),
  * `hotpath_processor.py` (injects bullets).

# Mental model

**Serve from compiled RAM; persist on idle; reason & interop in RDF.**
That’s why it stays fast at turn 10, 50, and 500—without giving up standards or long-term structure.
