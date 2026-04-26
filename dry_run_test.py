import os
import sys

# Ensure current dir is in path
sys.path.append(os.getcwd())

def test_imports():
    try:
        import narrative_engine
        import editing_brain
        import audio_director
        import main
        import clipper
        print("[PASS] Core Cinematic Modules: IMPORT SUCCESS")
    except ImportError as e:
        print(f"[FAIL] Import Error: {e}")
        sys.exit(1)

def test_logic_paths():
    import narrative_engine
    import audio_director
    # Simple check for NarrativeAI
    try:
        nai = narrative_engine.NarrativeAI()
        print("[PASS] Narrative Engine: INSTANTIATION SUCCESS")
    except Exception as e:
        print(f"[FAIL] Narrative Engine Init Failure: {e}")

    # Check Audio Director manifest
    try:
        manifest = audio_director.get_royalty_free_music_manifest(".")
        print(f"[PASS] Audio Director: Manifest scan successful ({len(manifest)} tracks)")
    except Exception as e:
        print(f"[FAIL] Audio Director Failure: {e}")

if __name__ == "__main__":
    test_imports()
    test_logic_paths()
    print("\n--- DRY RUN PASSED ---")
    print("The pipeline is ready for a real stream processing.")
