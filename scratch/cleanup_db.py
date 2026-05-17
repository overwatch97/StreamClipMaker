import sqlite3
import os

DB_PATH = 'performance.db'

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    # Delete test clips
    conn.execute("DELETE FROM variants WHERE variant_id LIKE 'test_%'")
    conn.commit()
    print("Test clips removed successfully.")
    
    # Check remaining
    cursor = conn.execute("SELECT count(*) FROM variants")
    print(f"Remaining clips: {cursor.fetchone()[0]}")
    conn.close()
else:
    print("Database not found.")
