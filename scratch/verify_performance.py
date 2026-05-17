
import os
import sys
sys.path.append(os.getcwd())
from performance_store import store_initial_data, update_metrics, get_all_variants

def verify_performance_layer():
    print("--- Verifying Performance Layer ---")
    test_variants = [{"variant_id": "test_clip_001", "event_type": "combat", "hook": "Mayhem", "caption_style": "dynamic"}]
    store_initial_data(test_variants)
    update_metrics("test_clip_001", views=1200, watch_time=45.5, retention=0.85, likes=150)
    results = get_all_variants()
    
    if any(r['variant_id'] == "test_clip_001" for r in results):
        match = [r for r in results if r['variant_id'] == "test_clip_001"][0]
        print("OK: Database initialized.")
        print(f"OK: Verified Data -> Views: {match['views']}, Likes: {match['likes']}, Retention: {match['retention']}")
    else:
        print("ERROR: Data not found.")

if __name__ == "__main__":
    verify_performance_layer()
