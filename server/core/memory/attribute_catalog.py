"""
Attribute Catalog for data-driven slot detection (low maintenance).

Provides:
- SQLite schema for attribute_catalog and FTS mirror (attribute_fts)
- Builder that derives attributes from per-user graph edges (src=user)
- FTS-based slot detection for queries

Kept optional and behind environment flags to avoid impacting hot path by default.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple, Optional
from loguru import logger


class AttributeCatalog:
    """Attribute catalog backed by SQLite with FTS mirror."""

    def __init__(self, store):
        self.store = store
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.store.sql.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attribute_catalog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    rel TEXT NOT NULL,
                    synonyms TEXT,
                    examples TEXT,
                    popularity INTEGER DEFAULT 0,
                    last_seen INTEGER DEFAULT 0,
                    UNIQUE(user_id, rel)
                )
                """
            )

            # FTS mirror of the catalog (contentless for simplicity)
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS attribute_fts USING fts5(
                    user_id UNINDEXED,
                    rel,
                    name,
                    synonyms,
                    examples
                )
                """
            )

            # Triggers to keep FTS in sync
            cur.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS attribute_ai AFTER INSERT ON attribute_catalog BEGIN
                    INSERT INTO attribute_fts(rowid, user_id, rel, name, synonyms, examples)
                    VALUES (new.id, new.user_id, new.rel, new.name, COALESCE(new.synonyms,''), COALESCE(new.examples,''));
                END;
                CREATE TRIGGER IF NOT EXISTS attribute_ad AFTER DELETE ON attribute_catalog BEGIN
                    INSERT INTO attribute_fts(attribute_fts, rowid, user_id, rel, name, synonyms, examples)
                    VALUES ('delete', old.id, old.user_id, old.rel, old.name, COALESCE(old.synonyms,''), COALESCE(old.examples,''));
                END;
                CREATE TRIGGER IF NOT EXISTS attribute_au AFTER UPDATE ON attribute_catalog BEGIN
                    INSERT INTO attribute_fts(attribute_fts, rowid, user_id, rel, name, synonyms, examples)
                    VALUES ('delete', old.id, old.user_id, old.rel, old.name, COALESCE(old.synonyms,''), COALESCE(old.examples,''));
                    INSERT INTO attribute_fts(rowid, user_id, rel, name, synonyms, examples)
                    VALUES (new.id, new.user_id, new.rel, new.name, COALESCE(new.synonyms,''), COALESCE(new.examples,''));
                END;
                """
            )

            self.store.sql.commit()
        except Exception as e:
            logger.debug(f"[AttributeCatalog] Schema init failed (non-fatal): {e}")

    def build_for_user(self, user_id: str, limit: int = 100, min_support: int = 1) -> int:
        """Populate/refresh catalog rows for a user based on edge statistics.

        Returns number of upserts performed.
        """
        if not user_id:
            return 0

        upserts = 0
        try:
            cur = self.store.sql.cursor()
            rows = cur.execute(
                """
                SELECT rel, COUNT(*) AS c, MAX(updated_at) AS last_seen
                FROM edge
                WHERE src = ? AND status >= 0
                GROUP BY rel
                HAVING c >= ?
                ORDER BY c DESC
                LIMIT ?
                """,
                (user_id, int(min_support), int(limit)),
            ).fetchall()

            for rel, pop, last_seen in rows:
                name, synonyms, examples = self._derive_from_rel(rel, user_id)
                cur.execute(
                    """
                    INSERT INTO attribute_catalog(user_id, name, rel, synonyms, examples, popularity, last_seen)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(user_id, rel) DO UPDATE SET
                        name=excluded.name,
                        synonyms=excluded.synonyms,
                        examples=excluded.examples,
                        popularity=excluded.popularity,
                        last_seen=excluded.last_seen
                    """,
                    (
                        user_id,
                        name,
                        rel,
                        json.dumps(synonyms, ensure_ascii=False),
                        examples,
                        int(pop),
                        int(last_seen or 0),
                    ),
                )
                upserts += 1

            self.store.sql.commit()
        except Exception as e:
            logger.debug(f"[AttributeCatalog] build_for_user failed: {e}")

        return upserts

    def _derive_from_rel(self, rel: str, user_id: str) -> Tuple[str, List[str], str]:
        """Derive a display name, synonyms, and examples given a relation id.

        This is intentionally simple; it can be extended based on observed edges.
        """
        # Name heuristics
        if rel.startswith('pref:'):
            key = rel.split(':', 1)[1]
            name = f"favorite {key}"
        elif rel == 'name':
            name = 'name'
        elif rel.startswith('v:'):
            name = rel.split(':', 1)[1].replace('_', ' ')
        else:
            name = rel.replace('_', ' ')

        # Synonyms from Enhanced FTS expansions when available
        synonyms: List[str] = []
        try:
            from .enhanced_fts import EnhancedFTS
            fts = EnhancedFTS(self.store)
            # pick synonyms for tokens in name we know expansions for
            for token in name.split():
                token = token.lower()
                if token in getattr(fts, 'expansions', {}):
                    synonyms.extend(fts.expansions[token][:2])
        except Exception:
            pass

        # One example from provenance if present
        examples = ''
        try:
            cur = self.store.sql.cursor()
            row = cur.execute(
                """
                SELECT t.text
                FROM edge e
                JOIN edge_source es ON es.edge_id = e.id
                JOIN conversation_turn t ON t.id = es.turn_id
                WHERE e.src = ? AND e.rel = ? AND e.status >= 0
                ORDER BY es.extracted_at DESC
                LIMIT 1
                """,
                (user_id, rel),
            ).fetchone()
            if row and row[0]:
                examples = row[0]
        except Exception:
            pass

        return name, list(dict.fromkeys(synonyms)), examples

    def detect_slot(self, text: str, user_id: str, min_score: float = 0.8) -> Tuple[Optional[str], float]:
        """Detect slot (relation id) by querying attribute_fts.

        Returns (rel, score) if above threshold, else (None, 0.0).
        """
        if not text or not text.strip():
            return None, 0.0

        try:
            cur = self.store.sql.cursor()
            # Build simple match query; restrict to user_id to avoid cross-user leaks
            # Use FTS match on name/synonyms/examples
            safe_query = ' '.join([tok for tok in text.lower().split() if tok.isalnum()])
            rows = cur.execute(
                """
                SELECT rel, bm25(attribute_fts) AS score
                FROM attribute_fts
                WHERE user_id = ? AND attribute_fts MATCH ?
                ORDER BY score LIMIT 3
                """,
                (user_id, safe_query),
            ).fetchall()
            if not rows:
                return None, 0.0
            # Convert bm25 (lower is better in SQLite bm25) to [0,1]
            rel, bm = rows[0]
            score = 1.0 / (1.0 + float(bm)) if bm is not None else 0.0
            if score >= float(os.getenv("MEMORY_SLOT_MIN_SCORE", str(min_score))):
                return str(rel), score
        except Exception as e:
            logger.debug(f"[AttributeCatalog] detect_slot failed: {e}")

        return None, 0.0

