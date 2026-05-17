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
            conn.execute('ALTER TABLE variants ADD COLUMN human_rating TEXT DEFAULT "PENDING"')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE variants ADD COLUMN feedback_tags TEXT DEFAULT ""')
        except sqlite3.OperationalError:
            pass
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
    with sqlite3.connect(DB_PATH) as conn:
        for out in outputs:
            conn.execute('''
                INSERT OR IGNORE INTO variants 
                (variant_id, event_type, hook, caption_style, parent_event_id, output_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (out['variant_id'], out.get('event_type'), out.get('hook'), out.get('caption_style'), out.get('parent_event_id', 'unknown'), out.get('output_path', '')))
    logger.info(f"Stored {len(outputs)} variants in performance store.")

def update_metrics(variant_id: str, views: int, watch_time: float, retention: float, likes: int):
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            UPDATE variants 
            SET views = ?, watch_time = ?, retention = ?, likes = ?
            WHERE variant_id = ?
        ''', (views, watch_time, retention, likes, variant_id))

def store_human_feedback(variant_id: str, rating: str, tags: str = ""):
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            UPDATE variants 
            SET human_rating = ?, feedback_tags = ?
            WHERE variant_id = ?
        ''', (rating, tags, variant_id))
    logger.info(f"Stored human feedback for {variant_id}: {rating} ({tags})")
        
def get_all_variants() -> List[Dict[str, Any]]:
    _init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute('SELECT * FROM variants')
        return [dict(row) for row in cursor.fetchall()]
