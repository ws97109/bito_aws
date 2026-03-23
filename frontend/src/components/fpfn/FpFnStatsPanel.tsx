import { useDashboard } from '../../context/DashboardContext';
import type { FpFnMode } from '../../types/index';

const MODEL_METRICS = [
  { label: 'AUC-ROC', value: '0.859', color: 'sky' },
  { label: 'F1 Score', value: '0.362', color: 'amber' },
  { label: 'Precision', value: '0.481', color: 'emerald' },
  { label: 'Recall',    value: '0.290', color: 'violet' },
];

const colorMap: Record<string, string> = {
  sky:     'bg-sky-900/40 text-sky-300 ring-sky-500/30',
  amber:   'bg-amber-900/40 text-amber-300 ring-amber-500/30',
  emerald: 'bg-emerald-900/40 text-emerald-300 ring-emerald-500/30',
  violet:  'bg-violet-900/40 text-violet-300 ring-violet-500/30',
};

const TABS: { mode: FpFnMode; label: string; count: number; desc: string }[] = [
  { mode: 'fp', label: 'FP', count: 237, desc: '白→黑 誤判' },
  { mode: 'fn', label: 'FN', count: 203, desc: '黑→白 漏判' },
];

export function FpFnStatsPanel() {
  const { state, dispatch } = useDashboard();

  return (
    <div className="space-y-4">
      {/* Model performance */}
      <div>
        <div className="flex items-center gap-2 pb-3 border-b border-slate-700">
          <span className="text-slate-400">&#128202;</span>
          <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">模型效能</h2>
        </div>
        <div className="grid grid-cols-2 gap-2 mt-3">
          {MODEL_METRICS.map(m => (
            <div key={m.label} className={`rounded-lg p-2.5 ring-1 ${colorMap[m.color]}`}>
              <p className="text-[10px] uppercase tracking-wider opacity-70">{m.label}</p>
              <p className="text-lg font-bold mt-0.5">{m.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* FP / FN toggle */}
      <div>
        <div className="flex items-center gap-2 pb-3 border-b border-slate-700">
          <span className="text-slate-400">&#9878;</span>
          <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">誤判分析</h2>
        </div>
        <div className="flex gap-2 mt-3">
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
              <p className="text-base font-bold">{tab.count}</p>
              <p className="text-[10px] font-semibold uppercase tracking-wide mt-0.5">{tab.label}</p>
              <p className="text-[10px] opacity-70 mt-0.5">{tab.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Confusion matrix hint */}
      <div className="bg-slate-800/40 rounded-lg p-3 ring-1 ring-slate-700/50 space-y-1.5">
        <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-2">混淆矩陣摘要</p>
        <div className="grid grid-cols-2 gap-1.5 text-xs">
          <div className="bg-emerald-900/30 ring-1 ring-emerald-700/40 rounded p-2 text-center">
            <p className="text-emerald-300 font-bold text-sm">448</p>
            <p className="text-emerald-400/70 text-[10px]">TN 正確正常</p>
          </div>
          <div className="bg-orange-900/30 ring-1 ring-orange-700/40 rounded p-2 text-center">
            <p className="text-orange-300 font-bold text-sm">237</p>
            <p className="text-orange-400/70 text-[10px]">FP 誤判詐騙</p>
          </div>
          <div className="bg-red-900/30 ring-1 ring-red-700/40 rounded p-2 text-center">
            <p className="text-red-300 font-bold text-sm">203</p>
            <p className="text-red-400/70 text-[10px]">FN 漏網之魚</p>
          </div>
          <div className="bg-sky-900/30 ring-1 ring-sky-700/40 rounded p-2 text-center">
            <p className="text-sky-300 font-bold text-sm">52</p>
            <p className="text-sky-400/70 text-[10px]">TP 正確詐騙</p>
          </div>
        </div>
      </div>
    </div>
  );
}
