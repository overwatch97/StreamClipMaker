"""
performance_store.py
Stores performance data for each variant using SQLite.
"""
import sqlite3
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DB_PATH = "performance.db"

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS variants (
                variant_id TEXT PRIMARY KEY,
                event_type TEXT,
                hook TEXT,
                caption_style TEXT,
                views INTEGER DEFAULT 0,
                watch_time REAL DEFAULT 0.0,
                retention REAL DEFAULT 0.0,
                likes INTEGER DEFAULT 0
            )
        ''')
        # Ensure new schema columns exist without wiping the DB
        try:
            conn.execute('ALTER TABLE variants ADD COLUMN parent_event_id TEXT DEFAULT "unknown"')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE variants ADD COLUMN output_path TEXT DEFAULT ""')
        except sqlite3.OperationalError:
            pass

def store_initial_data(outputs: List[Dict[str, Any]]):
    _init_db()
    
    # Import locally to avoid circular dependencies if any
    import review_store
    
    with sqlite3.connect(DB_PATH) as conn:
        for out in outputs:
            # 1. Old variants table for backward compatibility
            conn.execute('''
                INSERT OR IGNORE INTO variants 
                (variant_id, event_type, hook, caption_style, parent_event_id, output_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (out['variant_id'], out.get('event_type'), out.get('hook'), out.get('caption_style'), out.get('parent_event_id', 'unknown'), out.get('output_path', '')))
            
    for out in outputs:
        # 2. New review architecture
        review_store.ingest_clip({
            "clip_id": out['variant_id'],
            "event_id": out.get('parent_event_id', 'unknown_event'),
            "variant_group_id": out.get('parent_event_id', 'unknown_event'), # Can separate later if needed
            "variant_type": "facecam" if "facecam" in out.get('output_path', '').lower() else "clean",
            "category": out.get('event_type', 'neutral'),
            "source_video": "",
            "clip_path": out.get('output_path', ''),
            "start_time": out.get('start_time', 0.0),
            "end_time": out.get('end_time', 0.0),
            "duration": out.get('duration', 0.0),
            "model_generated_score": out.get('score', 0.0),
            "model_version": "v1",
            "generation_reason": out.get('hook', 'auto_generated'),
            "generation_signals": "{}"
        })
            
    logger.info(f"Stored {len(outputs)} variants in performance store and review queue.")

def update_metrics(variant_id: str, views: int, watch_time: float, retention: float, likes: int):
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            UPDATE variants 
            SET views = ?, watch_time = ?, retention = ?, likes = ?
            WHERE variant_id = ?
        ''', (views, watch_time, retention, likes, variant_id))

        
def get_all_variants() -> List[Dict[str, Any]]:
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM variants')
        return [dict(row) for row in cursor.fetchall()]
