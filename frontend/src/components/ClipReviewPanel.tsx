import { useState, useEffect, useRef } from 'react';
import { ThumbsUp, ThumbsDown, Star, RotateCcw, AlertTriangle, Zap, UserCheck } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Types
interface ClipPayload {
  clip_id: string;
  variant_type: string;
  category: string;
  generation_reason: string;
  generation_signals: string;
  model_generated_score: number;
}

interface QueueStats {
  remaining: number;
  reviewed_today: number;
  best_rated: number;
  total_in_session: number;
}

export default function ClipReviewPanel() {
  // Session State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  
  // Filters
  const [variantFilter, setVariantFilter] = useState('All');
  const [categoryFilter, setCategoryFilter] = useState('All');
  
  // High-Speed Preload State
  const [currentClip, setCurrentClip] = useState<ClipPayload | null>(null);
  const [stats, setStats] = useState<QueueStats>({ remaining: 0, reviewed_today: 0, best_rated: 0, total_in_session: 0 });
  
  // Playback Refs
  const videoRefActive = useRef<HTMLVideoElement>(null);
  
  // Telemetry
  const clipLoadTime = useRef<number>(Date.now());
  const replayCount = useRef<number>(0);
  
  // UI State
  const [feedbackFlash, setFeedbackFlash] = useState<'GOOD' | 'BAD' | 'BEST' | 'UNDO' | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [sessionCompleted, setSessionCompleted] = useState(false);

  // Initialize Session
  const startSession = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/review/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          queue_mode: 'chronological',
          variant_filter: variantFilter,
          category_filter: categoryFilter
        })
      });
      const data = await res.json();
      setSessionId(data.session_id);
      setSessionActive(true);
      setSessionCompleted(false);
      
      // Load first clip immediately
      await fetchNextClip(data.session_id, true);
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  };

  // Fetch Next Clip
  const fetchNextClip = async (sid: string, isInitial: boolean = false) => {
    try {
      const res = await fetch(`http://localhost:8000/api/review/next?session_id=${sid}`);
      const data = await res.json();
      
      if (data.status === 'complete' || !data.clip) {
        if (isInitial) setSessionCompleted(true);
        return;
      }
      
      if (isInitial) {
        setCurrentClip(data.clip);
        clipLoadTime.current = Date.now();

      }
      
      fetchStats(sid);
    } catch (err) {
      console.error("Failed to fetch next clip:", err);
    }
  };

  const fetchStats = async (sid: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/review/stats?session_id=${sid}`);
      setStats(await res.json());
    } catch (err) {}
  };

  const handleFeedback = async (rating: 'GOOD' | 'BAD' | 'BEST', tag: string = '') => {
    if (!currentClip || !sessionId || isTransitioning) return;
    
    setIsTransitioning(true);
    setFeedbackFlash(rating);
    setTimeout(() => setFeedbackFlash(null), 300);

    const reviewTimeMs = Date.now() - clipLoadTime.current;

    try {
      await fetch(`http://localhost:8000/api/review/${currentClip.clip_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          session_id: sessionId,
          score: rating, 
          tags: tag ? [tag] : [],
          review_time_ms: reviewTimeMs,
          replay_count: replayCount.current
        })
      });
      
      // Fast swap
      setCurrentClip(null); // Triggers exit animation
      setTimeout(() => {
        fetchNextClip(sessionId, true);
        setIsTransitioning(false);
      }, 300);
      
    } catch (err) {
      console.error("Failed to submit feedback", err);
      setIsTransitioning(false);
    }
  };

  const handleUndo = async () => {
    if (!sessionId || isTransitioning) return;
    setIsTransitioning(true);
    setFeedbackFlash('UNDO');
    setTimeout(() => setFeedbackFlash(null), 300);

    try {
      await fetch('http://localhost:8000/api/review/undo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      
      // Reload current queue
      fetchNextClip(sessionId, true);
      setIsTransitioning(false);
    } catch (err) {
      console.error("Failed to undo", err);
      setIsTransitioning(false);
    }
  };

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch(e.key.toLowerCase()) {
        case 'w':
          handleFeedback('BEST');
          break;
        case 'a':
          handleFeedback('GOOD');
          break;
        case 'd':
          handleFeedback('BAD');
          break;
        case 'z':
          handleUndo();
          break;
        case ' ':
          e.preventDefault();
          if (videoRefActive.current) {
            if (videoRefActive.current.paused) videoRefActive.current.play();
            else videoRefActive.current.pause();
          }
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentClip, sessionId, isTransitioning]);

  if (!sessionActive) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2">Editorial Review Session</h1>
          <p className="text-gray-400">Initialize a deterministic review queue snapshot.</p>
        </div>
        
        <div className="flex gap-8 bg-surface p-8 rounded-xl border border-gray-800 shadow-2xl">
          <div className="flex flex-col gap-2">
            <span className="text-xs uppercase text-gray-500 font-bold">Variant Filter</span>
            <select className="bg-gray-900 border border-gray-700 rounded p-2 text-white" value={variantFilter} onChange={e => setVariantFilter(e.target.value)}>
              <option value="All">All Variants</option>
              <option value="facecam">Facecam Only</option>
              <option value="clean">Clean Only</option>
            </select>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-xs uppercase text-gray-500 font-bold">Category Filter</span>
            <select className="bg-gray-900 border border-gray-700 rounded p-2 text-white" value={categoryFilter} onChange={e => setCategoryFilter(e.target.value)}>
              <option value="All">All Categories</option>
              <option value="combat">Combat</option>
              <option value="funny">Funny</option>
              <option value="dialogue">Dialogue</option>
            </select>
          </div>
        </div>

        <button 
          onClick={startSession}
          className="px-8 py-4 bg-primary text-white font-bold rounded-lg hover:bg-primary/80 transition shadow-[0_0_20px_rgba(var(--color-primary),0.4)] flex items-center gap-3"
        >
          <Zap size={24} />
          Start Review Session
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full p-4 gap-4 bg-background">
      {/* Top Filter & Stats Bar */}
      <div className="flex justify-between items-center bg-surface p-4 rounded-xl border border-gray-800 shadow-lg shrink-0">
        <div className="flex gap-4 items-center">
          <span className="px-3 py-1 bg-gray-900 border border-gray-700 rounded-full text-xs font-mono uppercase text-primary">
            Session: {sessionId?.substring(0, 8)}
          </span>
          <span className="px-3 py-1 bg-gray-900 border border-gray-700 rounded-full text-xs font-mono uppercase">
            {variantFilter} / {categoryFilter}
          </span>
        </div>
        
        <div className="flex gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-200">{stats.remaining}</div>
            <div className="text-[10px] uppercase text-gray-500 font-bold tracking-wider">Remaining</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-success">{stats.reviewed_today}</div>
            <div className="text-[10px] uppercase text-gray-500 font-bold tracking-wider">Reviewed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-warning">{stats.best_rated}</div>
            <div className="text-[10px] uppercase text-gray-500 font-bold tracking-wider">Best Rated</div>
          </div>
        </div>
      </div>

      {sessionCompleted ? (
        <div className="flex-1 flex flex-col items-center justify-center bg-surface rounded-xl border border-gray-800">
          <UserCheck size={64} className="text-success mb-4" />
          <h2 className="text-3xl font-bold text-white mb-2">Queue Empty</h2>
          <p className="text-gray-400">All clips in this session have been reviewed.</p>
          <button onClick={() => setSessionActive(false)} className="mt-8 px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded text-sm uppercase font-bold text-gray-300">
            Start New Session
          </button>
        </div>
      ) : (
        <div className="flex flex-1 gap-6 min-h-0">
          {/* Active Video Player */}
          <div className="flex-1 relative bg-black rounded-xl border border-gray-800 overflow-hidden shadow-2xl flex items-center justify-center">
            <AnimatePresence mode="wait">
              {currentClip && (
                <motion.video 
                  key={currentClip.clip_id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, x: -50 }}
                  transition={{ duration: 0.2 }}
                  ref={videoRefActive}
                  src={`http://localhost:8000/api/video/${currentClip.clip_id}`}
                  className="h-full object-contain"
                  autoPlay
                  loop
                  controls
                  onEnded={() => replayCount.current++}
                />
              )}
            </AnimatePresence>
            
            {/* Feedback Animation Overlay */}
            <AnimatePresence>
              {feedbackFlash && (
                <motion.div 
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1.2 }}
                  exit={{ opacity: 0, scale: 1.5 }}
                  transition={{ duration: 0.3 }}
                  className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-10 pointer-events-none"
                >
                  <div className="flex flex-col items-center">
                    {feedbackFlash === 'GOOD' && <ThumbsUp size={120} className="text-success drop-shadow-lg" />}
                    {feedbackFlash === 'BAD' && <ThumbsDown size={120} className="text-danger drop-shadow-lg" />}
                    {feedbackFlash === 'BEST' && <Star size={120} className="text-warning drop-shadow-lg" />}
                    {feedbackFlash === 'UNDO' && <RotateCcw size={120} className="text-primary drop-shadow-lg" />}
                    <h2 className="text-4xl font-bold mt-4 tracking-widest">{feedbackFlash}</h2>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right Panel: Explainability & Controls */}
          <div className="w-[420px] flex flex-col gap-4 overflow-y-auto pr-2">
            
            {/* Explainability Panel */}
            <div className="bg-surface p-5 rounded-xl border border-gray-800 flex flex-col shadow-lg shrink-0">
              <div className="flex items-center gap-2 mb-4 text-gray-400 border-b border-gray-800 pb-2">
                <AlertTriangle size={18} className="text-primary" />
                <h3 className="font-semibold uppercase tracking-wider text-xs">Explainability Signals</h3>
              </div>
              
              {currentClip ? (
                <div className="space-y-4">
                  <div>
                    <span className="text-[10px] text-gray-500 uppercase tracking-widest block mb-1">Why was this generated?</span>
                    <div className="bg-gray-900 p-3 rounded border border-gray-800 text-sm font-medium text-gray-200">
                      {currentClip.generation_reason}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-900 p-3 rounded border border-gray-800 text-center">
                      <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Model Score</div>
                      <div className="text-lg font-mono text-success">
                        {(currentClip.model_generated_score * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-gray-900 p-3 rounded border border-gray-800 text-center">
                      <div className="text-[10px] text-gray-500 uppercase font-bold mb-1">Category</div>
                      <div className="text-sm font-mono text-primary capitalize">
                        {currentClip.category}
                      </div>
                    </div>
                  </div>
                  
                  <div>
                     <span className="text-[10px] text-gray-500 uppercase tracking-widest block mb-1">Fused Signals JSON</span>
                     <pre className="text-[10px] font-mono text-gray-400 bg-black p-2 rounded border border-gray-800 overflow-x-auto">
                        {currentClip.generation_signals}
                     </pre>
                  </div>
                </div>
              ) : (
                <div className="text-center text-sm text-gray-500 py-4">Loading metadata...</div>
              )}
            </div>

            {/* Voting Panel */}
            <div className="bg-surface p-5 rounded-xl border border-gray-800 flex flex-col gap-3 shadow-lg shrink-0">
              <h3 className="font-semibold text-gray-400 uppercase tracking-wider text-xs mb-2">Editorial Vote</h3>
              
              <button 
                onClick={() => handleFeedback('BEST')}
                className="w-full py-4 bg-warning/10 hover:bg-warning/20 border border-warning/50 text-warning rounded-lg flex items-center justify-center gap-3 transition-colors group"
              >
                <Star size={24} className="group-hover:scale-110 transition-transform" />
                <div className="text-left">
                  <span className="block font-bold text-lg">BEST</span>
                  <span className="text-[10px] uppercase font-mono opacity-70">Hotkey: W</span>
                </div>
              </button>
              
              <div className="flex gap-3">
                <button 
                  onClick={() => handleFeedback('GOOD')}
                  className="flex-1 py-4 bg-success/10 hover:bg-success/20 border border-success/50 text-success rounded-lg flex flex-col items-center justify-center gap-1 transition-colors group"
                >
                  <ThumbsUp size={20} className="group-hover:-translate-y-1 transition-transform" />
                  <span className="font-bold text-sm">GOOD</span>
                  <span className="text-[10px] uppercase font-mono opacity-70">[ A ]</span>
                </button>
                <button 
                  onClick={() => handleFeedback('BAD')}
                  className="flex-1 py-4 bg-danger/10 hover:bg-danger/20 border border-danger/50 text-danger rounded-lg flex flex-col items-center justify-center gap-1 transition-colors group"
                >
                  <ThumbsDown size={20} className="group-hover:translate-y-1 transition-transform" />
                  <span className="font-bold text-sm">BAD</span>
                  <span className="text-[10px] uppercase font-mono opacity-70">[ D ]</span>
                </button>
              </div>

              <div className="mt-4 pt-4 border-t border-gray-800 flex justify-center">
                <button 
                  onClick={handleUndo}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-xs uppercase font-bold transition-colors"
                >
                  <RotateCcw size={14} />
                  Undo Last Vote [Z]
                </button>
              </div>
            </div>
            
          </div>
        </div>
      )}
    </div>
  );
}
