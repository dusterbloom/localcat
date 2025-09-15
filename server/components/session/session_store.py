"""
Session Storage System
======================

Comprehensive session management that stores:
1. Verbatim conversation data (user/assistant messages)
2. Session metadata (id, user, timestamps, summary)
3. Proper linkage between sessions, summaries, and extracted knowledge

This replaces the fragmented session handling in the original MemoryStore.
"""

import sqlite3
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

@dataclass
class SessionMessage:
    """Single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: int
    turn_id: int
    
@dataclass
class SessionMetadata:
    """Session metadata"""
    session_id: str
    user_id: str
    start_time: int
    end_time: Optional[int] = None
    summary: Optional[str] = None
    message_count: int = 0
    extraction_count: int = 0
    
class SessionStore:
    """Comprehensive session storage with verbatim data and metadata"""
    
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize session database with proper schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Sessions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time INTEGER NOT NULL,
                end_time INTEGER,
                summary TEXT,
                message_count INTEGER DEFAULT 0,
                extraction_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        
        # Messages table for verbatim storage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                turn_id INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # Session summaries table for LEANN integration
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                summary_type TEXT DEFAULT 'auto',
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # Session-to-knowledge linkage table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_knowledge_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                edge_id TEXT NOT NULL,
                link_type TEXT DEFAULT 'extracted',
                confidence REAL DEFAULT 1.0,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # Indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON sessions(user_id, start_time)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_time ON session_messages(session_id, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_session ON session_knowledge_links(session_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_edge ON session_knowledge_links(edge_id)")
        
        self.conn.commit()
        logger.info(f"🗃️ SessionStore initialized at {self.db_path}")
    
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Create a new session"""
        if session_id is None:
            # Create hash-based session ID: timestamp + user_id hash
            import hashlib
            timestamp = int(time.time())
            user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]  # First 8 chars of MD5 hash
            session_id = f"sess_{timestamp}_{user_hash}"
        
        now = int(time.time())
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions (session_id, user_id, start_time, message_count, extraction_count)
            VALUES (?, ?, ?, 0, 0)
        """, (session_id, user_id, now))
        self.conn.commit()
        
        logger.info(f"📝 Created session: {session_id} for user: {user_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, turn_id: int):
        """Add a message to a session"""
        timestamp = int(time.time())
        
        # Store the message
        self.conn.execute("""
            INSERT INTO session_messages (session_id, role, content, timestamp, turn_id)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, timestamp, turn_id))
        
        # Update session message count
        self.conn.execute("""
            UPDATE sessions 
            SET message_count = message_count + 1, 
                end_time = ?,
                updated_at = ?
            WHERE session_id = ?
        """, (timestamp, timestamp, session_id))
        
        self.conn.commit()
        logger.debug(f"💬 Added {role} message to session {session_id}")
    
    def add_session_summary(self, session_id: str, summary: str, summary_type: str = "auto"):
        """Add a summary for a session"""
        self.conn.execute("""
            INSERT INTO session_summaries (session_id, summary_text, summary_type)
            VALUES (?, ?, ?)
        """, (session_id, summary, summary_type))
        
        # Update session summary
        self.conn.execute("""
            UPDATE sessions 
            SET summary = ?, updated_at = ?
            WHERE session_id = ?
        """, (summary, int(time.time()), session_id))
        
        self.conn.commit()
        logger.info(f"📄 Added summary for session {session_id}")
    
    def link_knowledge_to_session(self, session_id: str, edge_id: str, link_type: str = "extracted", confidence: float = 1.0):
        """Link extracted knowledge to a session"""
        self.conn.execute("""
            INSERT INTO session_knowledge_links (session_id, edge_id, link_type, confidence)
            VALUES (?, ?, ?, ?)
        """, (session_id, edge_id, link_type, confidence))
        
        # Update extraction count
        self.conn.execute("""
            UPDATE sessions 
            SET extraction_count = extraction_count + 1,
                updated_at = ?
            WHERE session_id = ?
        """, (int(time.time()), session_id))
        
        self.conn.commit()
        logger.debug(f"🔗 Linked knowledge {edge_id} to session {session_id}")
    
    def get_session_conversation(self, session_id: str) -> List[SessionMessage]:
        """Get full conversation for a session"""
        cursor = self.conn.execute("""
            SELECT role, content, timestamp, turn_id
            FROM session_messages 
            WHERE session_id = ?
            ORDER BY timestamp, turn_id
        """, (session_id,))
        
        return [SessionMessage(role=row[0], content=row[1], timestamp=row[2], turn_id=row[3]) 
                for row in cursor.fetchall()]
    
    def get_session_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata"""
        cursor = self.conn.execute("""
            SELECT session_id, user_id, start_time, end_time, summary, message_count, extraction_count
            FROM sessions 
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if row:
            return SessionMetadata(
                session_id=row[0], user_id=row[1], start_time=row[2], 
                end_time=row[3], summary=row[4], message_count=row[5], extraction_count=row[6]
            )
        return None
    
    def get_user_sessions(self, user_id: str, limit: int = 10) -> List[SessionMetadata]:
        """Get recent sessions for a user"""
        cursor = self.conn.execute("""
            SELECT session_id, user_id, start_time, end_time, summary, message_count, extraction_count
            FROM sessions 
            WHERE user_id = ?
            ORDER BY start_time DESC
            LIMIT ?
        """, (user_id, limit))
        
        return [SessionMetadata(
            session_id=row[0], user_id=row[1], start_time=row[2], 
            end_time=row[3], summary=row[4], message_count=row[5], extraction_count=row[6]
        ) for row in cursor.fetchall()]
    
    def get_session_knowledge(self, session_id: str) -> List[Tuple[str, str, float]]:
        """Get knowledge linked to a session"""
        cursor = self.conn.execute("""
            SELECT edge_id, link_type, confidence
            FROM session_knowledge_links 
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,))
        
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    
    def get_conversation_context(self, session_id: str, max_messages: int = 10) -> str:
        """Get formatted conversation context for retrieval"""
        messages = self.get_session_conversation(session_id)
        if not messages:
            return ""
        
        # Take last max_messages messages
        recent_messages = messages[-max_messages:]
        
        context = f"Conversation from session {session_id}:\n"
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context += f"{role}: {msg.content}\n"
        
        return context.strip()
    
    def get_all_sessions_for_leann(self) -> List[Dict[str, Any]]:
        """Get all session data for LEANN indexing"""
        cursor = self.conn.execute("""
            SELECT s.session_id, s.user_id, s.start_time, s.end_time, s.summary,
                   sm.content as session_text
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, GROUP_CONCAT(content, ' ') as content
                FROM session_messages
                GROUP BY session_id
            ) sm ON s.session_id = sm.session_id
            WHERE s.summary IS NOT NULL OR sm.content IS NOT NULL
            ORDER BY s.start_time DESC
        """)
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'user_id': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'summary': row[4],
                'content': row[5] or ''
            })
        
        return sessions
    
    def cleanup_old_sessions(self, days: int = 30):
        """Clean up sessions older than specified days"""
        cutoff_time = int(time.time()) - (days * 24 * 60 * 60)
        
        # Get sessions to be deleted
        cursor = self.conn.execute("""
            SELECT session_id FROM sessions 
            WHERE start_time < ?
        """, (cutoff_time,))
        
        old_sessions = [row[0] for row in cursor.fetchall()]
        
        if old_sessions:
            logger.info(f"🧹 Cleaning up {len(old_sessions)} old sessions")
            self.conn.execute("""
                DELETE FROM sessions WHERE start_time < ?
            """, (cutoff_time,))
            self.conn.commit()
        
        return len(old_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session storage statistics"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM session_messages")
        message_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM session_summaries")
        summary_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM session_knowledge_links")
        link_count = cursor.fetchone()[0]

        return {
            'sessions': session_count,
            'messages': message_count,
            'summaries': summary_count,
            'knowledge_links': link_count,
            'avg_messages_per_session': message_count / max(session_count, 1),
            'sessions_with_summaries': summary_count
        }

    def get_user_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive session statistics for a specific user"""
        # Get user's sessions
        cursor = self.conn.execute("""
            SELECT session_id, start_time, end_time, message_count, summary
            FROM sessions
            WHERE user_id = ?
            ORDER BY start_time DESC
        """, (user_id,))

        sessions = cursor.fetchall()
        total_sessions = len(sessions)

        if total_sessions == 0:
            return {
                'user_id': user_id,
                'total_sessions': 0,
                'total_time_spent_seconds': 0,
                'total_messages': 0,
                'avg_session_duration': 0,
                'avg_messages_per_session': 0,
                'sessions_with_summaries': 0,
                'recent_sessions': []
            }

        # Calculate statistics
        total_time_spent = 0
        total_messages = 0
        sessions_with_summaries = 0
        recent_sessions = []

        for session in sessions:
            session_id, start_time, end_time, message_count, summary = session
            duration = (end_time or int(time.time())) - start_time
            total_time_spent += duration
            total_messages += message_count or 0
            if summary:
                sessions_with_summaries += 1

            # Include recent session details
            if len(recent_sessions) < 5:
                recent_sessions.append({
                    'session_id': session_id,
                    'start_time': start_time,
                    'duration_seconds': duration,
                    'message_count': message_count or 0,
                    'has_summary': bool(summary)
                })

        return {
            'user_id': user_id,
            'total_sessions': total_sessions,
            'total_time_spent_seconds': total_time_spent,
            'total_messages': total_messages,
            'avg_session_duration': total_time_spent / total_sessions,
            'avg_messages_per_session': total_messages / total_sessions,
            'sessions_with_summaries': sessions_with_summaries,
            'recent_sessions': recent_sessions,
            'first_session_time': sessions[-1][1] if sessions else None,
            'last_session_time': sessions[0][2] or sessions[0][1] if sessions else None
        }

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific session"""
        # Get session metadata
        metadata = self.get_session_metadata(session_id)
        if not metadata:
            return {}

        # Get conversation timeline
        cursor = self.conn.execute("""
            SELECT role, timestamp, turn_id
            FROM session_messages
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))

        messages = cursor.fetchall()

        if not messages:
            return {
                'session_id': session_id,
                'duration_seconds': metadata.end_time - metadata.start_time if metadata.end_time else None,
                'message_count': 0,
                'turn_count': 0,
                'avg_response_time': None,
                'timeline': []
            }

        # Calculate analytics
        turn_count = max(msg[2] for msg in messages) if messages else 0
        user_messages = [msg for msg in messages if msg[0] == 'user']
        assistant_messages = [msg for msg in messages if msg[0] == 'assistant']

        # Calculate response times (time between user message and next assistant message)
        response_times = []
        for i in range(len(messages) - 1):
            if messages[i][0] == 'user' and messages[i + 1][0] == 'assistant':
                response_time = messages[i + 1][1] - messages[i][1]
                response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else None

        # Create timeline
        timeline = []
        for i, (role, timestamp, turn_id) in enumerate(messages):
            timeline.append({
                'sequence': i + 1,
                'role': role,
                'timestamp': timestamp,
                'turn_id': turn_id,
                'time_from_start': timestamp - messages[0][1]
            })

        return {
            'session_id': session_id,
            'duration_seconds': metadata.end_time - metadata.start_time if metadata.end_time else None,
            'message_count': len(messages),
            'turn_count': turn_count,
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'avg_response_time_seconds': avg_response_time,
            'extraction_count': metadata.extraction_count,
            'has_summary': bool(metadata.summary),
            'timeline': timeline,
            'start_time': metadata.start_time,
            'end_time': metadata.end_time
        }

    def get_user_session_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's session history with summaries for navigation"""
        cursor = self.conn.execute("""
            SELECT s.session_id, s.start_time, s.end_time, s.message_count,
                   s.extraction_count, s.summary, ss.summary_text
            FROM sessions s
            LEFT JOIN session_summaries ss ON s.session_id = ss.session_id
            WHERE s.user_id = ?
            ORDER BY s.start_time DESC
            LIMIT ?
        """, (user_id, limit))

        sessions = []
        for row in cursor.fetchall():
            session_id, start_time, end_time, message_count, extraction_count, summary, summary_text = row
            duration = (end_time or int(time.time())) - start_time

            sessions.append({
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'message_count': message_count or 0,
                'extraction_count': extraction_count or 0,
                'summary': summary or summary_text,  # Use either field summary
                'has_detailed_summary': bool(summary_text)
            })

        return sessions


# Global session store instance
_session_store = None

def get_session_store(db_path: str = "sessions.db") -> SessionStore:
    """Get or create the global session store instance"""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(db_path)
    return _session_store

def reset_session_store():
    """Reset the global session store instance (for testing)"""
    global _session_store
    _session_store = None