import { useDashboard } from '../../context/DashboardContext';
import type { FpFnMode } from '../../types/index';

export function FpFnStatsPanel() {
  const { state, dispatch } = useDashboard();
  const { fpNodes, fnNodes } = state;

  const fpCount = fpNodes.length;
  const fnCount = fnNodes.length;

  const TABS: { mode: FpFnMode; label: string; count: number; desc: string }[] = [
    { mode: 'fp', label: 'FP', count: fpCount, desc: '白→黑 誤判' },
    { mode: 'fn', label: 'FN', count: fnCount, desc: '黑→白 漏判' },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 pb-3 border-b border-slate-700">
        <span className="text-slate-400">&#9878;</span>
        <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">誤判分析</h2>
      </div>

      <div className="flex gap-2">
        {TABS.map(tab => (
          <button
            key={tab.mode}
            onClick={() => dispatch({ type: 'SET_FPFN_MODE', mode: tab.mode })}
            className={`flex-1 py-3 rounded-lg text-center transition-colors focus:outline-none ring-1
              ${state.fpFnMode === tab.mode
                ? tab.mode === 'fp'
                  ? 'bg-orange-500/20 text-orange-300 ring-orange-500/50'
                  : 'bg-red-500/20 text-red-300 ring-red-500/50'
                : 'bg-slate-700/40 text-slate-400 ring-slate-600/40 hover:bg-slate-700/70'
              }`}
          >
            <p className="text-base font-bold">{tab.count.toLocaleString()}</p>
            <p className="text-[10px] font-semibold uppercase tracking-wide mt-0.5">{tab.label}</p>
            <p className="text-[10px] opacity-70 mt-0.5">{tab.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
