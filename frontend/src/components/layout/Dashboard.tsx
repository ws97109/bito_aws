import { useDashboard } from '../../context/DashboardContext';
import { StatsPanel } from '../stats/StatsPanel';
import { NodeSelector } from '../graph/NodeSelector';
import { GraphViewer } from '../graph/GraphViewer';
import { NodeDetailPanel } from '../graph/NodeDetailPanel';
import { DashboardSwitcher } from '../fpfn/DashboardSwitcher';
import { FpFnStatsPanel } from '../fpfn/FpFnStatsPanel';
import { FpFnNodeSelector } from '../fpfn/FpFnNodeSelector';
import { ShapPanel } from '../fpfn/ShapPanel';
import { PredictStatsPanel } from '../predict/PredictStatsPanel';
import { PredictNodeSelector } from '../predict/PredictNodeSelector';
import { PredictDetailPanel } from '../predict/PredictDetailPanel';
import { FeatureInfoPanel } from '../features/FeatureInfoPanel';
import { FeaturesStatsPanel } from '../features/FeaturesStatsPanel';

export function Dashboard() {
  const { state } = useDashboard();
  const isFpFnMode = state.dashboardMode === 'fp-fn';
  const isPredictMode = state.dashboardMode === 'predict';
  const isFeaturesMode = state.dashboardMode === 'features';

  const renderLeftPanel = () => {
    if (isFeaturesMode) return <FeaturesStatsPanel />;
    if (isPredictMode) return <PredictStatsPanel />;
    if (isFpFnMode) return <FpFnStatsPanel />;
    return <StatsPanel />;
  };

  const renderRightContent = () => {
    if (isFeaturesMode) return <FeatureInfoPanel />;

    if (isPredictMode) {
      return (
        <main className="flex-1 overflow-y-auto min-w-0 min-h-0">
          <div className="flex flex-col gap-4">
            <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl p-4 flex flex-col">
              <PredictNodeSelector />
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl px-4 py-3">
              <PredictDetailPanel />
            </div>
          </div>
        </main>
      );
    }

    if (isFpFnMode) {
      return (
        <main className="flex-1 overflow-y-auto min-w-0 min-h-0">
          <div className="flex flex-col gap-4">
            <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl p-4 flex flex-col">
              <FpFnNodeSelector />
              <div className="mt-3" style={{ height: '500px' }} onWheel={e => e.stopPropagation()}>
                <GraphViewer />
              </div>
            </div>
            <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl px-4 py-3">
              <ShapPanel />
            </div>
          </div>
        </main>
      );
    }

    return (
      <main className="flex-1 overflow-y-auto min-w-0 min-h-0">
        <div className="flex flex-col gap-4">
          <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl p-4 flex flex-col">
            <NodeSelector />
            <div className="mt-3" style={{ height: '500px' }} onWheel={e => e.stopPropagation()}>
              <GraphViewer />
            </div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl px-4 py-3">
            <NodeDetailPanel />
          </div>
        </div>
      </main>
    );
  };

  return (
    <div className="flex flex-col h-screen text-slate-200 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <header className="bg-slate-900/80 backdrop-blur-md shadow-xl px-6 py-4 z-20 border-b border-slate-700/60">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-sky-500/20 border border-sky-500/40 text-sky-400 text-lg">
              &#128737;
            </div>
            <div>
              <h1 className="text-lg font-bold text-sky-400 leading-tight">BitoGuard 詐騙偵測儀表板</h1>
              <p className="text-xs text-slate-500 leading-tight">即時交易圖分析平台</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block"></span>
            系統運行中
          </div>
        </div>
      </header>

      <div className="flex-1 flex flex-col lg:flex-row gap-4 p-4 overflow-hidden min-h-0">
        {/* Left Panel */}
        <aside className="w-full lg:w-96 bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl overflow-y-auto flex-shrink-0">
          <div className="p-4 space-y-4">
            <DashboardSwitcher />
            {renderLeftPanel()}
          </div>
        </aside>

        {/* Right Content */}
        {renderRightContent()}
      </div>
    </div>
  );
}
