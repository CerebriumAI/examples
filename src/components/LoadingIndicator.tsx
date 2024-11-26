import { Loader2 } from 'lucide-react';

export function LoadingIndicator() {
  return (
    <div className="flex items-center justify-center gap-2 text-cerebrium-pink">
      <Loader2 className="w-5 h-5 animate-spin" />
      <span className="text-sm font-medium">Processing your request...New text will appear when available</span>
    </div>
  );
}