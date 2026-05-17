import React, { useState, useEffect } from 'react';
import { BrainCircuit, Activity, Zap, ShieldAlert } from 'lucide-react';

export default function LearningDashboard() {
  const [learningData, setLearningData] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/api/learning')
      .then(res => res.json())
      .then(data => {
        setLearningData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch learning data", err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="flex h-full items-center justify-center">Loading insights...</div>;

  const eventTypes = Object.keys(learningData);

  if (eventTypes.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center text-gray-500 gap-4">
        <BrainCircuit size={48} className="opacity-20" />
        <p>No learning patterns established yet.</p>
        <p className="text-sm">Review clips and select "BEST" to train the system.</p>
      </div>
    );
  }

  return (
    <div className="p-8 h-full overflow-y-auto bg-background">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <BrainCircuit className="text-accent" /> Learning Insights
          </h2>
          <p className="text-gray-400 mt-2">Observe how the AI editorial layer is evolving its taste.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {eventTypes.map(type => {
            const pattern = learningData[type];
            const confidencePct = Math.round((pattern.confidence || 0) * 100);
            
            return (
              <div key={type} className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg flex flex-col">
                <div className="flex justify-between items-start mb-6">
                  <h3 className="text-xl font-bold capitalize text-gray-200">{type}</h3>
                  {pattern.recovery_mode && (
                    <span className="bg-danger/20 text-danger text-xs px-2 py-1 rounded border border-danger/30 flex items-center gap-1">
                      <ShieldAlert size={12} /> Recovery
                    </span>
                  )}
                </div>

                <div className="space-y-4 flex-1">
                  <div>
                    <span className="text-xs text-gray-500 uppercase">Winning Hook Pattern</span>
                    <p className="text-lg font-medium text-primary mt-1">{pattern.best_hook_type}</p>
                  </div>
                  
                  <div>
                    <span className="text-xs text-gray-500 uppercase">Winning Caption Style</span>
                    <p className="text-md font-medium text-gray-300 mt-1">{pattern.best_caption_style}</p>
                  </div>

                  <div className="pt-4 mt-auto">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">Confidence Level</span>
                      <span className={confidencePct > 70 ? 'text-success' : 'text-warning'}>{confidencePct}%</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${confidencePct > 70 ? 'bg-success' : 'bg-warning'}`} 
                        style={{ width: `${confidencePct}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-gray-800 flex justify-between text-xs text-gray-500">
                  <span className="flex items-center gap-1"><Activity size={14} /> Samples: {pattern.sample_count}</span>
                  <span className="flex items-center gap-1"><Zap size={14} /> Score: {pattern.historical_score?.toFixed(2) || '0.00'}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
