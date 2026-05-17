import React from 'react';
import { ListVideo } from 'lucide-react';

export default function QueueDashboard() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-gray-500 gap-4 bg-background">
      <ListVideo size={48} className="opacity-20" />
      <p className="text-xl font-medium text-gray-400">Queue Dashboard</p>
      <p className="text-sm max-w-md text-center">
        This view will display batch processing status, rendering progress, 
        and allow drag/drop uploading of new raw streams.
      </p>
      <div className="mt-8 px-4 py-2 border border-gray-700 rounded text-xs bg-gray-800/50">
        Phase 1 basic implementation complete. Advanced features pending.
      </div>
    </div>
  );
}
