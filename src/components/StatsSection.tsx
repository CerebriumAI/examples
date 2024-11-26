import { Zap } from 'lucide-react';

interface StatProps {
  title: string;
  description: string;
  average: string;
}

function Stat({ title, description, average }: StatProps) {
    return (
      <div className="flex flex-col items-center">
        <div className="flex items-center gap-2 text-cerebrium-pink mb-2">
          <Zap className="w-5 h-5" />
          <span className="text-2xl font-bold">{title}</span>
        </div>
        <p className="text-gray-600">{description}</p>
        <span className="text-sm text-gray-500">{average}</span> {/* Display average stats */}
      </div>
    );
  }

function Separator() {
  return <div className="h-12 w-px bg-gray-200" />;
}

export function StatsSection() {
    return (
      <div className="flex items-center justify-center gap-8">
        <Stat title="20x" average="2-4s cold starts" description="Faster Cold starts" />
        <Separator />
        <Stat title="24x" average="8-14s builds" description="Faster Builds" />
        <Separator />
        <Stat title="5x" average="35ms added latency" description="Better Performance" />
      </div>
    );
  }