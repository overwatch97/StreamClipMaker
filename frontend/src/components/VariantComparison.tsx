import { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipBack, SplitSquareHorizontal, Trophy } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function VariantComparison() {
  const [groups, setGroups] = useState<Record<string, any[]>>({});
  const [groupIds, setGroupIds] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRefA = useRef<HTMLVideoElement>(null);
  const videoRefB = useRef<HTMLVideoElement>(null);

  const [winnerFlash, setWinnerFlash] = useState<'A' | 'B' | 'TIE' | null>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/variant-groups')
      .then(res => res.json())
      .then(data => {
        setGroups(data);
        setGroupIds(Object.keys(data));
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch variant groups", err);
        setLoading(false);
      });
  }, []);

  const currentGroupId = groupIds[currentIndex];
  const currentGroup = groups[currentGroupId] || [];
  const variantA = currentGroup[0];
  const variantB = currentGroup[1];

  const handlePlayPause = () => {
    if (isPlaying) {
      videoRefA.current?.pause();
      videoRefB.current?.pause();
    } else {
      videoRefA.current?.play();
      videoRefB.current?.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRefA.current) videoRefA.current.currentTime = time;
    if (videoRefB.current) videoRefB.current.currentTime = time;
  };

  const resetVideos = () => {
    if (videoRefA.current) videoRefA.current.currentTime = 0;
    if (videoRefB.current) videoRefB.current.currentTime = 0;
  };

  const handleVote = async (winner: 'A' | 'B' | 'TIE') => {
    setWinnerFlash(winner);
    setTimeout(() => setWinnerFlash(null), 800);

    let winnerId = null;
    if (winner === 'A') winnerId = variantA.clip_id;
    if (winner === 'B') winnerId = variantB.clip_id;

    if (winnerId) {
      try {
        await fetch(`http://localhost:8000/api/review/${winnerId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            session_id: 'ab_test_session',
            score: 'BEST', 
            tags: ['A/B Winner'],
            review_time_ms: 0,
            replay_count: 0
          })
        });
      } catch (err) {
        console.error("Failed to submit A/B feedback", err);
      }
    }

    // Move to next group
    setTimeout(() => {
      if (currentIndex < groupIds.length - 1) {
        setCurrentIndex(prev => prev + 1);
        setIsPlaying(true); // Auto-play next
      }
    }, 500);
  };

  if (loading) return <div className="flex h-full items-center justify-center">Loading comparisons...</div>;
  if (groupIds.length === 0) return (
    <div className="flex h-full flex-col items-center justify-center text-gray-500 gap-4">
      <SplitSquareHorizontal size={48} className="opacity-20" />
      <p>No variant groups found.</p>
      <p className="text-sm">The pipeline needs to generate multiple variants for the same event.</p>
    </div>
  );

  return (
    <div className="flex flex-col h-full p-6 gap-6 bg-background">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
            <SplitSquareHorizontal className="text-primary" /> Variant Comparison
          </h2>
          <p className="text-gray-400 mt-1">Synchronized A/B testing to isolate hook performance.</p>
        </div>
        <div className="text-sm bg-gray-800 px-3 py-1 rounded text-gray-300">
          Group {currentIndex + 1} / {groupIds.length}
        </div>
      </div>

      <div className="flex-1 flex gap-6 min-h-0 relative">
        <AnimatePresence>
          {winnerFlash && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.8, y: -50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 1.2 }}
              className="absolute inset-0 z-50 flex items-center justify-center pointer-events-none"
            >
              <div className="bg-warning text-black px-12 py-6 rounded-2xl shadow-[0_0_100px_rgba(234,179,8,0.5)] flex flex-col items-center">
                <Trophy size={64} className="mb-2" />
                <h1 className="text-5xl font-bold font-black uppercase tracking-widest">
                  {winnerFlash === 'TIE' ? 'TIE' : `VARIANT ${winnerFlash} WINS`}
                </h1>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Variant A */}
        <div className="flex-1 flex flex-col bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-lg relative">
          <div className="absolute top-4 left-4 z-10 bg-black/60 px-4 py-1 rounded-full font-bold text-xl border border-gray-700 backdrop-blur-md">A</div>
          <div className="flex-1 bg-black relative">
            <video 
              ref={videoRefA}
              src={`http://localhost:8000/api/video/${variantA.clip_id}`}
              className="w-full h-full object-contain"
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onTimeUpdate={() => setCurrentTime(videoRefA.current?.currentTime ?? 0)}
              onLoadedMetadata={() => setDuration(videoRefA.current?.duration ?? 0)}
            />
          </div>
          <div className="p-4 border-t border-gray-800 bg-surface">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Variant Type</span>
              <span className="text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-300">{variantA.variant_type}</span>
            </div>
            <p className="text-lg font-medium text-white italic">"{variantA.generation_reason}"</p>
          </div>
        </div>

        {/* Variant B */}
        <div className="flex-1 flex flex-col bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-lg relative">
          <div className="absolute top-4 left-4 z-10 bg-black/60 px-4 py-1 rounded-full font-bold text-xl border border-gray-700 backdrop-blur-md">B</div>
          <div className="flex-1 bg-black relative">
            <video 
              ref={videoRefB}
              src={`http://localhost:8000/api/video/${variantB.clip_id}`}
              className="w-full h-full object-contain"
            />
          </div>
          <div className="p-4 border-t border-gray-800 bg-surface">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Variant Type</span>
              <span className="text-xs bg-gray-800 px-2 py-0.5 rounded text-gray-300">{variantB.variant_type}</span>
            </div>
            <p className="text-lg font-medium text-white italic">"{variantB.generation_reason}"</p>
          </div>
        </div>
      </div>

      {/* Synchronized Controls */}
      <div className="bg-surface p-4 border border-gray-800 rounded-xl flex items-center gap-6 shadow-lg">
        <div className="flex items-center gap-2">
          <button onClick={resetVideos} className="p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors">
            <SkipBack size={20} />
          </button>
          <button onClick={handlePlayPause} className="p-3 bg-primary hover:bg-primary/80 text-white rounded-lg transition-colors w-12 flex justify-center">
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
          </button>
        </div>

        <input 
          type="range" 
          min="0" 
          max={duration || 100} 
          step="0.01"
          className="flex-1 h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer"
          onChange={handleSeek}
          value={currentTime}
        />

        <div className="flex items-center gap-3 ml-auto border-l border-gray-800 pl-6">
          <span className="text-sm text-gray-400 font-medium mr-2">VOTE:</span>
          <button onClick={() => handleVote('A')} className="px-6 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg font-bold transition-transform hover:scale-105">
            A WINS
          </button>
          <button onClick={() => handleVote('TIE')} className="px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-gray-400 transition-colors">
            TIE
          </button>
          <button onClick={() => handleVote('B')} className="px-6 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-lg font-bold transition-transform hover:scale-105">
            B WINS
          </button>
        </div>
      </div>
    </div>
  );
}
