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
from performance_store import get_all_variants, DB_PATH
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

import review_store

class ReviewSessionRequest(BaseModel):
    queue_mode: str = "chronological"
    variant_filter: str = "All"
    category_filter: str = "All"

class ClipReviewRequest(BaseModel):
    session_id: str
    score: str
    tags: List[str] = []
    review_time_ms: float = 0.0
    replay_count: int = 0
    decision_version: str = "v2_editorial_quality"

class UndoRequest(BaseModel):
    session_id: str

@app.post("/api/review/session/start")
def start_review_session(req: ReviewSessionRequest):
    """Creates a deterministic queue snapshot and returns a session_id."""
    return review_store.start_review_session(
        reviewer_id="system", 
        queue_mode=req.queue_mode, 
        variant_filter=req.variant_filter, 
        category_filter=req.category_filter
    )

@app.get("/api/variant-groups")
def get_variant_groups():
    """Returns variants grouped by variant_group_id for A/B comparison."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM clips")
        rows = cursor.fetchall()
        
    groups = {}
    for r in rows:
        vid = r["variant_group_id"]
        if vid not in groups:
            groups[vid] = []
        groups[vid].append(dict(r))
        
    # Only return groups with multiple variants for true A/B testing
    return {vid: group for vid, group in groups.items() if len(group) > 1}

@app.get("/api/review/next")
def get_next_review_item(session_id: str):
    """Returns the next unreviewed clip payload for the active session."""
    item = review_store.get_next_session_item(session_id)
    if not item:
        return {"status": "complete", "message": "No more clips in this session."}
    return {"status": "ok", "clip": item}

@app.get("/api/video/{clip_id}")
def get_video(clip_id: str, request: Request):
    """Serve the video file for a specific clip."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT clip_path FROM clips WHERE clip_id = ?", (clip_id,))
        row = cursor.fetchone()
        
    video_path = None
    if row and row["clip_path"]:
        db_path = row["clip_path"]
        if os.path.exists(db_path):
            video_path = db_path
        else:
            # Fallback 1: Check if file exists in parent/grandparent directory of db_path
            filename = os.path.basename(db_path)
            parent_dir = os.path.dirname(db_path)
            grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
            
            candidates = []
            if grandparent_dir:
                candidates.append(os.path.join(grandparent_dir, filename))
            if parent_dir:
                candidates.append(os.path.join(parent_dir, filename))
                
            for c in candidates:
                if c and os.path.exists(c):
                    video_path = c
                    break
                    
            # Fallback 2: Check if filename exists anywhere under parent directories up to Streamclipperoutput
            if not video_path:
                curr = parent_dir
                for _ in range(3):
                    if not curr or curr == os.path.dirname(curr):
                        break
                    if os.path.basename(curr).lower() in ["streamclipperoutput", "videos"]:
                        found = False
                        for root, _, files in os.walk(curr):
                            if filename in files:
                                video_path = os.path.join(root, filename)
                                found = True
                                break
                        if found:
                            break
                    curr = os.path.dirname(curr)

    # Fallback 3: Check local output directory
    if not video_path and os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4") and clip_id in f:
                video_path = os.path.join(OUTPUT_DIR, f)
                break

    # Fallback 4: Dynamic Placeholder generation
    if not video_path or not os.path.exists(video_path):
        placeholder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        os.makedirs(placeholder_dir, exist_ok=True)
        placeholder_path = os.path.join(placeholder_dir, f"placeholder_{clip_id}.mp4")
        if not os.path.exists(placeholder_path):
            try:
                # Generate a 5-second vertical video (720x1280) with a test pattern and timer
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=5:size=720x1280:rate=30",
                    "-pix_fmt", "yuv420p",
                    placeholder_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception:
                pass
        if os.path.exists(placeholder_path):
            video_path = placeholder_path

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
        
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
                chunk_size = 1024 * 1024
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

@app.post("/api/review/{clip_id}")
def submit_review(clip_id: str, req: ClipReviewRequest):
    """Records a review, tracks metrics, and marks the item resolved."""
    try:
        review_store.submit_clip_review(
            session_id=req.session_id,
            clip_id=clip_id,
            score=req.score,
            tags=req.tags,
            review_time_ms=req.review_time_ms,
            replay_count=req.replay_count,
            decision_version=req.decision_version
        )
        
        # If BEST, promote to learning store
        if req.score == "BEST":
            event_type = "neutral"
            hook_type = "neutral"
            caption_style = "dynamic"
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT category, generation_reason FROM clips WHERE clip_id = ?",
                    (clip_id,)
                )
                clip_row = cursor.fetchone()
                if clip_row:
                    event_type = clip_row["category"] or "neutral"
                    hook_type = clip_row["generation_reason"] or "neutral"
                    
                cursor = conn.execute(
                    "SELECT caption_style FROM variants WHERE variant_id = ?",
                    (clip_id,)
                )
                variant_row = cursor.fetchone()
                if variant_row:
                    caption_style = variant_row["caption_style"] or "dynamic"
            
            winner_obj = {
                "event_type": event_type,
                "hook_type": hook_type,
                "caption_style": caption_style,
                "final_score": 1.0
            }
            update_learning_store(winner_obj)
            
        return {"status": "success", "clip_id": clip_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/review/undo")
def undo_review(req: UndoRequest):
    """Soft-deletes the active review and rolls back the session queue."""
    result = review_store.undo_last_review(req.session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/api/review/stats")
def get_queue_stats(session_id: str):
    """Retrieves session metrics for the UI."""
    return review_store.get_queue_stats(session_id)

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
