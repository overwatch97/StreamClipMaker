import React from 'react';
import { Activity, PlaySquare, Eye, ListVideo } from 'lucide-react';

export default function TimelineInspector() {
  return (
    <div className="flex flex-col h-full bg-background p-6 overflow-hidden">
      <div className="mb-6">
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
          <Activity className="text-accent" /> Timeline Inspector
        </h2>
        <p className="text-gray-400 mt-1">Deep dive into the signals (Motion, Audio, Emotion) that formed this clip.</p>
      </div>

      <div className="flex-1 flex flex-col lg:flex-row gap-6 min-h-0">
        
        {/* Left Col: Video Player Placeholder */}
        <div className="flex-1 bg-surface border border-gray-800 rounded-xl overflow-hidden shadow-lg flex flex-col relative">
           <div className="absolute top-4 left-4 z-10 bg-black/60 px-3 py-1 rounded text-xs border border-gray-700 backdrop-blur-md">Clip View</div>
           <div className="flex-1 flex items-center justify-center bg-black">
              <PlaySquare size={64} className="text-gray-700 opacity-50" />
           </div>
        </div>

        {/* Right Col: Timeline Graphs */}
        <div className="flex-[2] flex flex-col gap-4 min-h-0">
          
          <div className="bg-surface border border-gray-800 rounded-xl p-4 shadow-lg flex flex-col justify-center items-center relative h-1/3">
             <span className="absolute top-3 left-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Audio Intensity</span>
             {/* Mock Graph using pure CSS to simulate a waveform */}
             <div className="w-full h-12 flex items-end gap-1 px-4 mt-6">
                {Array.from({ length: 40 }).map((_, i) => {
                  const h = Math.random() * 100;
                  const isPeak = i === 25;
                  return (
                    <div 
                      key={i} 
                      className={`flex-1 rounded-t-sm transition-all ${isPeak ? 'bg-primary shadow-[0_0_10px_#3b82f6]' : 'bg-primary/20'}`}
                      style={{ height: `${h}%` }}
                    />
                  );
                })}
             </div>
             {/* Markers */}
             <div className="absolute bottom-2 w-full flex justify-between px-8 text-[10px] text-gray-600">
               <span>Clip Start</span>
               <span className="text-primary font-bold">Peak</span>
               <span>Payoff End</span>
             </div>
          </div>

          <div className="bg-surface border border-gray-800 rounded-xl p-4 shadow-lg flex flex-col justify-center items-center relative h-1/3">
             <span className="absolute top-3 left-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Motion / Combat Signals</span>
             <div className="w-full h-12 flex items-end gap-1 px-4 mt-6">
                {Array.from({ length: 40 }).map((_, i) => {
                  const h = Math.max(10, Math.sin(i / 5) * 50 + 50);
                  const isPayoff = i > 30;
                  return (
                    <div 
                      key={i} 
                      className={`flex-1 rounded-t-sm transition-all ${isPayoff ? 'bg-success/50' : 'bg-warning/30'}`}
                      style={{ height: `${h}%` }}
                    />
                  );
                })}
             </div>
          </div>
          
          <div className="bg-surface border border-gray-800 rounded-xl p-4 shadow-lg flex-1 relative overflow-y-auto">
             <span className="absolute top-3 left-4 text-xs font-bold text-gray-500 uppercase tracking-wider">Detection Logs</span>
             <div className="mt-8 space-y-2 text-xs font-mono text-gray-400">
               <p>[00:00:02.1] Motion threshold exceeded (0.42 &gt; 0.25)</p>
               <p>[00:00:04.5] Audio peak detected (0.89)</p>
               <p className="text-warning">[00:00:06.0] Peak Prominence validated</p>
               <p className="text-success">[00:00:08.5] Payoff Detected: Silence + Sentence Complete</p>
               <p className="text-primary">[00:00:08.5] Ending clip extension.</p>
             </div>
          </div>

        </div>
      </div>
    </div>
  );
}
