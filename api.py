from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import sqlite3
import os
import json
import re
import subprocess
from typing import List, Optional

# Import the existing Python pipeline modules
from performance_store import get_all_variants, store_human_feedback, DB_PATH
from learning_store import update_learning_store, _load_store, STORE_PATH

app = FastAPI(title="StreamClipMaker Training Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output_clips" # Default output directory

class FeedbackRequest(BaseModel):
    rating: str
    tags: str = ""

@app.get("/api/clips")
def get_clips():
    """Returns all variants from the performance DB along with their video URLs."""
    if not os.path.exists(DB_PATH):
        return []
    
    return get_all_variants()

@app.get("/api/variant-groups")
def get_variant_groups():
    """Returns variants grouped by parent_event_id for A/B comparison."""
    if not os.path.exists(DB_PATH):
        return {}
    
    variants = get_all_variants()
    groups = {}
    for v in variants:
        pid = v.get("parent_event_id", "unknown")
        # For testing, we might want to group even 'unknown' if we need to, but let's separate them by variant ID prefix if unknown.
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(v)
        
    # Only return groups with multiple variants for true A/B testing
    return {pid: group for pid, group in groups.items() if len(group) > 1}

@app.get("/api/video/{variant_id}")
def get_video(variant_id: str, request: Request):
    """Serve the video file for a specific variant with proper headers."""
    variants = get_all_variants()
    variant = next((v for v in variants if v["variant_id"] == variant_id), None)
    
    video_path = None
    if variant and variant.get("output_path") and os.path.exists(variant["output_path"]):
        video_path = variant["output_path"]
    elif os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4") and variant_id in f:
                video_path = os.path.join(OUTPUT_DIR, f)
                break
                
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
    # Check pixel format to see if browser supports it (browsers need yuv420p)
    needs_transcode = False
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "stream=pix_fmt", 
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        pix_fmt = subprocess.check_output(probe_cmd, text=True, stderr=subprocess.DEVNULL).strip().split('\n')[0]
        if pix_fmt and "yuv420p" not in pix_fmt:
            needs_transcode = True
    except Exception:
        pass

    if needs_transcode:
        # Transcode on the fly to yuv420p for web compatibility without modifying NAS files
        def ffmpeg_stream():
            cmd = [
                "ffmpeg", "-i", video_path,
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-b:a", "128k",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov",
                "pipe:1"
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                process.kill()
        
        return StreamingResponse(
            ffmpeg_stream(),
            media_type="video/mp4",
            headers={"Cache-Control": "no-cache"}
        )

        
    file_size = os.stat(video_path).st_size
    range_header = request.headers.get("range")
    
    if not range_header:
        headers = {"Accept-Ranges": "bytes"}
        return FileResponse(video_path, headers=headers, media_type="video/mp4", filename=os.path.basename(video_path))
        
    # Parse the range header
    byte1, byte2 = 0, None
    match = re.search(r"bytes=(\d+)-(.*)", range_header)
    if match:
        byte1 = int(match.group(1))
        if match.group(2):
            byte2 = int(match.group(2))
            
    start = byte1
    end = file_size - 1 if byte2 is None else min(byte2, file_size - 1)
    length = end - start + 1
    
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Type": "video/mp4",
    }
    
    def file_iterator(file_path, offset, bytes_to_read):
        with open(file_path, "rb") as f:
            f.seek(offset, os.SEEK_SET)
            remaining = bytes_to_read
            while remaining > 0:
                chunk_size = 1024 * 1024  # 1MB chunks
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        file_iterator(video_path, start, length),
        status_code=206,
        headers=headers,
        media_type="video/mp4"
    )

@app.post("/api/feedback/{variant_id}")
def submit_feedback(variant_id: str, feedback: FeedbackRequest):
    """Stores human feedback in performance.db and updates learning if BEST."""
    try:
        # Update performance.db
        store_human_feedback(variant_id, feedback.rating, feedback.tags)
        
        # If rating is BEST, promote it to the learning store
        if feedback.rating == "BEST":
            # We need to construct a 'winner' object as expected by learning_store.py
            # winner needs: event_type, hook_type, caption_style, final_score
            variants = get_all_variants()
            winner_row = next((v for v in variants if v["variant_id"] == variant_id), None)
            
            if winner_row:
                winner_obj = {
                    "event_type": winner_row.get("event_type", "neutral"),
                    "hook_type": winner_row.get("hook", "neutral"),
                    "caption_style": winner_row.get("caption_style", "dynamic"),
                    "final_score": 1.0 # BEST means it scored perfectly in human review
                }
                update_learning_store(winner_obj)
                
        return {"status": "success", "variant_id": variant_id, "rating": feedback.rating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/learning")
def get_learning():
    """Returns the current state of the learning store."""
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
