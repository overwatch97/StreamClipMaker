"""
review_store.py
Dedicated SQLite interface for the StreamClipMaker Review Session Engine and ML dataset infrastructure.
"""
import sqlite3
import logging
import uuid
import datetime
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = "performance.db"

def _init_review_db():
    with sqlite3.connect(DB_PATH) as conn:
        # clips table: core asset tracking and ML metrics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT UNIQUE NOT NULL,
                event_id TEXT NOT NULL,
                variant_group_id TEXT NOT NULL,
                variant_type TEXT NOT NULL,
                category TEXT NOT NULL,
                review_status TEXT DEFAULT 'pending',
                source_video TEXT,
                clip_path TEXT,
                start_time REAL DEFAULT 0.0,
                end_time REAL DEFAULT 0.0,
                duration REAL DEFAULT 0.0,
                view_count INTEGER DEFAULT 0,
                replay_count INTEGER DEFAULT 0,
                skip_count INTEGER DEFAULT 0,
                avg_review_time_ms REAL DEFAULT 0.0,
                model_generated_score REAL DEFAULT 0.0,
                model_version TEXT DEFAULT 'v1',
                generation_reason TEXT,
                generation_signals TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # reviews table: editorial decisions (soft-deletable via is_active)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT NOT NULL,
                score TEXT NOT NULL,
                decision_version TEXT NOT NULL,
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                reviewer_id TEXT DEFAULT 'system',
                review_time_ms REAL DEFAULT 0.0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # review_tags table: granular analytical tags
        conn.execute('''
            CREATE TABLE IF NOT EXISTS review_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # review_sessions table: lifecycle and analytics
        conn.execute('''
            CREATE TABLE IF NOT EXISTS review_sessions (
                id TEXT PRIMARY KEY,
                reviewer_id TEXT DEFAULT 'system',
                session_status TEXT DEFAULT 'active',
                queue_mode TEXT DEFAULT 'chronological',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                total_reviewed INTEGER DEFAULT 0
            )
        ''')
        
        # review_session_items table: persistent queue ordering
        conn.execute('''
            CREATE TABLE IF NOT EXISTS review_session_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                clip_id TEXT NOT NULL,
                queue_position INTEGER NOT NULL,
                reviewed BOOLEAN DEFAULT 0,
                reviewed_at TIMESTAMP
            )
        ''')
        
        # review_session_metrics table: reviewer telemetry
        conn.execute('''
            CREATE TABLE IF NOT EXISTS review_session_metrics (
                session_id TEXT PRIMARY KEY,
                avg_decision_time REAL DEFAULT 0.0,
                best_rate REAL DEFAULT 0.0,
                undo_rate REAL DEFAULT 0.0,
                replay_rate REAL DEFAULT 0.0,
                completion_rate REAL DEFAULT 0.0,
                fatigue_score REAL DEFAULT 0.0
            )
        ''')

def ingest_clip(clip_data: Dict[str, Any]):
    """Registers a newly generated clip into the clips table."""
    _init_review_db()
    
    # Extract stringified JSON for generation_signals if it's a dict/list
    signals = clip_data.get('generation_signals', '{}')
    if isinstance(signals, (dict, list)):
        signals = json.dumps(signals)
        
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT OR IGNORE INTO clips (
                clip_id, event_id, variant_group_id, variant_type, category,
                source_video, clip_path, start_time, end_time, duration,
                model_generated_score, model_version, generation_reason, generation_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            clip_data.get('clip_id'),
            clip_data.get('event_id', 'unknown_event'),
            clip_data.get('variant_group_id', 'unknown_group'),
            clip_data.get('variant_type', 'default'),
            clip_data.get('category', 'neutral'),
            clip_data.get('source_video', ''),
            clip_data.get('clip_path', ''),
            clip_data.get('start_time', 0.0),
            clip_data.get('end_time', 0.0),
            clip_data.get('duration', 0.0),
            clip_data.get('model_generated_score', 0.0),
            clip_data.get('model_version', 'v1'),
            clip_data.get('generation_reason', 'auto_generated'),
            signals
        ))
        logger.info(f"Ingested clip into review_store: {clip_data.get('clip_id')}")

def start_review_session(reviewer_id: str = "system", queue_mode: str = "chronological", variant_filter: str = "All", category_filter: str = "All") -> Dict[str, Any]:
    """Creates a deterministic session snapshot and returns session state."""
    _init_review_db()
    session_id = str(uuid.uuid4())
    
    with sqlite3.connect(DB_PATH) as conn:
        # Create session record
        conn.execute('''
            INSERT INTO review_sessions (id, reviewer_id, session_status, queue_mode)
            VALUES (?, ?, 'active', ?)
        ''', (session_id, reviewer_id, queue_mode))
        
        # Build query for pending clips
        query = "SELECT clip_id FROM clips WHERE review_status = 'pending'"
        params = []
        
        if variant_filter != "All":
            query += " AND variant_type = ?"
            params.append(variant_filter.lower())
            
        if category_filter != "All":
            query += " AND category = ?"
            params.append(category_filter.lower())
            
        if queue_mode == "chronological":
            query += " ORDER BY created_at ASC"
        elif queue_mode == "randomized":
            query += " ORDER BY RANDOM()"
        elif queue_mode == "high_score_candidates":
            query += " ORDER BY model_generated_score DESC"
        else:
            query += " ORDER BY created_at ASC"
            
        cursor = conn.execute(query, params)
        pending_clips = cursor.fetchall()
        
        # Freeze queue into review_session_items
        for idx, row in enumerate(pending_clips):
            clip_id = row[0]
            conn.execute('''
                INSERT INTO review_session_items (session_id, clip_id, queue_position)
                VALUES (?, ?, ?)
            ''', (session_id, clip_id, idx))
            
    logger.info(f"Started session {session_id} with {len(pending_clips)} clips queued.")
    return {"session_id": session_id, "queued_count": len(pending_clips)}

def get_next_session_item(session_id: str) -> Optional[Dict[str, Any]]:
    """Returns the next unreviewed clip payload for a given session."""
    _init_review_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Get next item
        cursor = conn.execute('''
            SELECT c.*, rsi.queue_position 
            FROM review_session_items rsi
            JOIN clips c ON rsi.clip_id = c.clip_id
            WHERE rsi.session_id = ? AND rsi.reviewed = 0
            ORDER BY rsi.queue_position ASC
            LIMIT 1
        ''', (session_id,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def submit_clip_review(session_id: str, clip_id: str, score: str, tags: List[str], review_time_ms: float, replay_count: int, decision_version: str = "v2_editorial_quality"):
    """Records a review, updates clip metrics, and marks the session item as reviewed."""
    _init_review_db()
    reviewer_id = "system" # Can be fetched from session later
    
    with sqlite3.connect(DB_PATH) as conn:
        # Mark session item reviewed
        conn.execute('''
            UPDATE review_session_items 
            SET reviewed = 1, reviewed_at = CURRENT_TIMESTAMP
            WHERE session_id = ? AND clip_id = ?
        ''', (session_id, clip_id))
        
        # Insert active review
        conn.execute('''
            INSERT INTO reviews (clip_id, score, decision_version, session_id, reviewer_id, review_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (clip_id, score, decision_version, session_id, reviewer_id, review_time_ms))
        
        # Insert tags
        for tag in tags:
            conn.execute('INSERT INTO review_tags (clip_id, tag) VALUES (?, ?)', (clip_id, tag))
            
        # Update clip state and rolling metrics
        # Simple rolling average logic for avg_review_time_ms (if needed) - here we just update
        conn.execute('''
            UPDATE clips 
            SET review_status = 'reviewed',
                replay_count = replay_count + ?,
                avg_review_time_ms = CASE 
                    WHEN avg_review_time_ms = 0 THEN ?
                    ELSE (avg_review_time_ms + ?) / 2.0
                END
            WHERE clip_id = ?
        ''', (replay_count, review_time_ms, review_time_ms, clip_id))
        
        # Update session metrics (simplified)
        conn.execute('UPDATE review_sessions SET total_reviewed = total_reviewed + 1 WHERE id = ?', (session_id,))
        
    logger.info(f"Submitted review for {clip_id}: {score}")

def undo_last_review(session_id: str) -> Optional[Dict[str, Any]]:
    """Soft-deletes the last review and reverts the queue state."""
    _init_review_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        
        # Find the last reviewed item in the session
        cursor = conn.execute('''
            SELECT clip_id FROM review_session_items
            WHERE session_id = ? AND reviewed = 1
            ORDER BY reviewed_at DESC
            LIMIT 1
        ''', (session_id,))
        last_item = cursor.fetchone()
        
        if not last_item:
            return {"error": "No actions to undo in this session."}
            
        clip_id = last_item["clip_id"]
        
        # Un-review session item
        conn.execute('''
            UPDATE review_session_items 
            SET reviewed = 0, reviewed_at = NULL 
            WHERE session_id = ? AND clip_id = ?
        ''', (session_id, clip_id))
        
        # Soft-delete the review
        conn.execute('''
            UPDATE reviews SET is_active = 0 
            WHERE session_id = ? AND clip_id = ? AND is_active = 1
        ''', (session_id, clip_id))
        
        # Restore clip status
        conn.execute("UPDATE clips SET review_status = 'pending' WHERE clip_id = ?", (clip_id,))
        
        # Update session totals
        conn.execute("UPDATE review_sessions SET total_reviewed = MAX(0, total_reviewed - 1) WHERE id = ?", (session_id,))
        
    logger.info(f"Undid last review for clip {clip_id} in session {session_id}")
    return {"status": "success", "clip_id": clip_id}

def get_queue_stats(session_id: str) -> Dict[str, Any]:
    _init_review_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute('''
            SELECT 
                COUNT(*) as total_in_session,
                SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) as reviewed_today
            FROM review_session_items
            WHERE session_id = ?
        ''', (session_id,))
        session_stats = cursor.fetchone()
        
        # Best rated (active)
        cursor = conn.execute("SELECT COUNT(*) FROM reviews WHERE session_id = ? AND score = 'BEST' AND is_active = 1", (session_id,))
        best_rated = cursor.fetchone()[0]
        
        total = session_stats[0] if session_stats and session_stats[0] else 0
        reviewed = session_stats[1] if session_stats and session_stats[1] else 0
        
        return {
            "remaining": total - reviewed,
            "reviewed_today": reviewed,
            "best_rated": best_rated,
            "total_in_session": total
        }
