import React, { useState, useEffect, useRef } from 'react';
import { ThumbsUp, ThumbsDown, Star, SkipForward, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ClipReviewPanel() {
  const [clips, setClips] = useState<any[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Feedback overlay animation state
  const [feedbackFlash, setFeedbackFlash] = useState<'GOOD' | 'BAD' | 'BEST' | null>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/clips')
      .then(res => res.json())
      .then(data => {
        setClips(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch clips", err);
        setLoading(false);
      });
  }, []);

  const currentClip = clips[currentIndex];

  const handleFeedback = async (rating: 'GOOD' | 'BAD' | 'BEST', tag: string = '') => {
    if (!currentClip) return;
    
    setFeedbackFlash(rating);
    setTimeout(() => setFeedbackFlash(null), 500);

    try {
      await fetch(`http://localhost:8000/api/feedback/${currentClip.variant_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, tags: tag })
      });
      // Move to next clip automatically
      nextClip();
    } catch (err) {
      console.error("Failed to submit feedback", err);
    }
  };

  const nextClip = () => {
    if (currentIndex < clips.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  };

  const prevClip = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch(e.key.toLowerCase()) {
        case 'a':
          handleFeedback('BAD');
          break;
        case 'd':
          handleFeedback('GOOD');
          break;
        case 'f':
          handleFeedback('BEST');
          break;
        case ' ':
          e.preventDefault();
          if (videoRef.current) {
            if (videoRef.current.paused) videoRef.current.play();
            else videoRef.current.pause();
          }
          break;
        case 'arrowleft':
          if (videoRef.current) videoRef.current.currentTime -= 2;
          break;
        case 'arrowright':
          if (videoRef.current) videoRef.current.currentTime += 2;
          break;
        case 'q':
          prevClip();
          break;
        case 'e':
          nextClip();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIndex, currentClip]);

  if (loading) return <div className="flex h-full items-center justify-center">Loading clips...</div>;
  if (!clips.length) return <div className="flex h-full items-center justify-center text-gray-500">No clips available in performance.db</div>;

  return (
    <div className="flex h-full p-6 gap-6 bg-background">
      {/* Video Player Column */}
      <div className="flex-1 flex flex-col items-center justify-center relative bg-black rounded-lg border border-gray-800 overflow-hidden shadow-2xl">
        <video 
          ref={videoRef}
          src={`http://localhost:8000/api/video/${currentClip.variant_id}`}
          className="h-full object-contain"
          autoPlay
          loop
          controls
        />
        
        <AnimatePresence>
          {feedbackFlash && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1.2 }}
              exit={{ opacity: 0, scale: 1.5 }}
              transition={{ duration: 0.3 }}
              className={`absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-10`}
            >
              <div className="flex flex-col items-center">
                {feedbackFlash === 'GOOD' && <ThumbsUp size={120} className="text-success drop-shadow-lg" />}
                {feedbackFlash === 'BAD' && <ThumbsDown size={120} className="text-danger drop-shadow-lg" />}
                {feedbackFlash === 'BEST' && <Star size={120} className="text-warning drop-shadow-lg" />}
                <h2 className="text-4xl font-bold mt-4 tracking-widest">{feedbackFlash}</h2>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Progress bar info overlay */}
        <div className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded text-sm text-gray-300 backdrop-blur-md">
          {currentIndex + 1} / {clips.length}
        </div>
      </div>

      {/* Side Panel: Metadata & Controls */}
      <div className="w-[400px] flex flex-col gap-4">
        <div className="bg-surface p-5 rounded-xl border border-gray-800 flex flex-col shadow-lg">
          <div className="flex items-center gap-2 mb-4 text-gray-400">
            <Info size={18} />
            <h3 className="font-semibold uppercase tracking-wider text-xs">Clip Metadata</h3>
          </div>
          
          <div className="space-y-4">
            <div>
              <span className="text-xs text-gray-500 uppercase">Event Type</span>
              <p className="text-lg font-medium text-gray-200 capitalize">{currentClip.event_type}</p>
            </div>
            
            <div>
              <span className="text-xs text-gray-500 uppercase">Hook Variant</span>
              <div className="bg-gray-800/50 p-3 rounded-lg border border-gray-700 mt-1">
                <p className="text-sm text-gray-300 italic">"{currentClip.hook}"</p>
              </div>
            </div>
            
            <div>
              <span className="text-xs text-gray-500 uppercase">Caption Style</span>
              <p className="text-sm font-medium text-gray-400">{currentClip.caption_style}</p>
            </div>
          </div>
        </div>

        {/* Actions Panel */}
        <div className="bg-surface p-5 rounded-xl border border-gray-800 flex flex-col gap-3 flex-1 shadow-lg">
          <h3 className="font-semibold text-gray-400 uppercase tracking-wider text-xs mb-2">Training Actions</h3>
          
          <button 
            onClick={() => handleFeedback('BEST')}
            className="flex-1 bg-warning/10 hover:bg-warning/20 border border-warning/50 text-warning rounded-lg flex items-center justify-center gap-3 transition-colors group"
          >
            <Star size={24} className="group-hover:scale-110 transition-transform" />
            <div className="text-left">
              <span className="block font-bold text-lg">BEST (Winner)</span>
              <span className="text-xs opacity-70">Promotes pattern [F]</span>
            </div>
          </button>
          
          <div className="flex gap-3 flex-1">
            <button 
              onClick={() => handleFeedback('GOOD')}
              className="flex-1 bg-success/10 hover:bg-success/20 border border-success/50 text-success rounded-lg flex flex-col items-center justify-center gap-2 transition-colors group"
            >
              <ThumbsUp size={24} className="group-hover:-translate-y-1 transition-transform" />
              <span className="font-bold">GOOD [D]</span>
            </button>
            <button 
              onClick={() => handleFeedback('BAD')}
              className="flex-1 bg-danger/10 hover:bg-danger/20 border border-danger/50 text-danger rounded-lg flex flex-col items-center justify-center gap-2 transition-colors group"
            >
              <ThumbsDown size={24} className="group-hover:translate-y-1 transition-transform" />
              <span className="font-bold">BAD [A]</span>
            </button>
          </div>

          <div className="mt-2 flex flex-wrap gap-2">
            {['NEEDS BETTER ENDING', 'GREAT HOOK / BAD PAYOFF', 'BORING START', 'TOO LONG'].map(tag => (
              <button 
                key={tag}
                onClick={() => handleFeedback('BAD', tag)}
                className="px-2 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-[10px] uppercase font-bold text-gray-400 transition-colors"
              >
                {tag}
              </button>
            ))}
          </div>

          <div className="mt-auto pt-4 border-t border-gray-800 flex justify-between">
            <button onClick={prevClip} className="p-2 text-gray-500 hover:text-white transition-colors">
              <span className="text-xs font-mono bg-gray-800 px-2 py-1 rounded">Q</span> Prev
            </button>
            <button onClick={nextClip} className="p-2 text-gray-500 hover:text-white transition-colors">
              Next <span className="text-xs font-mono bg-gray-800 px-2 py-1 rounded">E</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
