import { StatsPanel } from '../stats/StatsPanel';
import { NodeSelector } from '../graph/NodeSelector';
import { GraphViewer } from '../graph/GraphViewer';
import { NodeDetailPanel } from '../graph/NodeDetailPanel';

export function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 px-6 py-3">
        <h1 className="text-lg font-bold text-gray-800">BitoGuard 詐騙偵測儀表板</h1>
      </header>
      <div className="flex flex-col lg:flex-row h-[calc(100vh-52px)]">
        {/* Left: Stats Panel (25%) */}
        <aside className="w-full lg:w-1/4 bg-white border-r border-gray-200 overflow-y-auto">
          <StatsPanel />
        </aside>
        {/* Right: Main area (75%) */}
        <main className="w-full lg:w-3/4 overflow-y-auto">
          <NodeSelector />
          <div className="px-4">
            <GraphViewer />
          </div>
          <NodeDetailPanel />
        </main>
      </div>
    </div>
  );
}
