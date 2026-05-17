import React, { useState } from 'react';
import { Play, TrendingUp, ListVideo, SplitSquareHorizontal } from 'lucide-react';
import ClipReviewPanel from './components/ClipReviewPanel';
import VariantComparison from './components/VariantComparison';
import LearningDashboard from './components/LearningDashboard';
import QueueDashboard from './components/QueueDashboard';
import TimelineInspector from './components/TimelineInspector';

function App() {
  const [activeTab, setActiveTab] = useState('review');

  return (
    <div className="h-screen bg-background text-gray-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-surface border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="bg-primary text-white p-1 rounded-md">
            <Play size={20} />
          </div>
          <h1 className="text-xl font-bold tracking-tight">StreamClipMaker Cockpit</h1>
          <span className="ml-2 text-xs bg-gray-800 px-2 py-1 rounded-full text-gray-400 border border-gray-700">AI Feedback Layer</span>
        </div>
        
        <nav className="flex space-x-1">
          <button 
            onClick={() => setActiveTab('review')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition-colors ${activeTab === 'review' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}`}
          >
            <Play size={16} /> Fast Review
          </button>
          <button 
            onClick={() => setActiveTab('compare')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition-colors ${activeTab === 'compare' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}`}
          >
            <SplitSquareHorizontal size={16} /> A/B Compare
          </button>
          <button 
            onClick={() => setActiveTab('timeline')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition-colors ${activeTab === 'timeline' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}`}
          >
            <TrendingUp size={16} /> Timeline
          </button>
          <button 
            onClick={() => setActiveTab('learning')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition-colors ${activeTab === 'learning' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}`}
          >
            <TrendingUp size={16} /> Learning
          </button>
          <button 
            onClick={() => setActiveTab('queue')}
            className={`px-4 py-2 rounded-md flex items-center gap-2 text-sm font-medium transition-colors ${activeTab === 'queue' ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'}`}
          >
            <ListVideo size={16} /> Queue
          </button>
        </nav>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-hidden relative">
        {activeTab === 'review' && <ClipReviewPanel />}
        {activeTab === 'compare' && <VariantComparison />}
        {activeTab === 'timeline' && <TimelineInspector />}
        {activeTab === 'learning' && <LearningDashboard />}
        {activeTab === 'queue' && <QueueDashboard />}
      </main>
    </div>
  );
}

export default App;
