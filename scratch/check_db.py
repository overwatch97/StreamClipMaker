import sqlite3
import os

DB_PATH = 'performance.db'

if not os.path.exists(DB_PATH):
    print(f"Database {DB_PATH} does not exist.")
else:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute("SELECT count(*) FROM variants")
        count = cursor.fetchone()[0]
        print(f"Total variants in database: {count}")
        
        if count > 0:
            print("\nRecent Variants:")
            cursor = conn.execute("SELECT variant_id, event_type, output_path FROM variants LIMIT 5")
            for row in cursor.fetchall():
                print(f" - {row[0]} ({row[1]}) -> {row[2]}")
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}")
