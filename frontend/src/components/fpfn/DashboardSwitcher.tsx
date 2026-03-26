import { useDashboard } from '../../context/DashboardContext';
import type { DashboardMode } from '../../types/index';

const MODES: { mode: DashboardMode; label: string; icon: string }[] = [
  { mode: 'fraud',   label: '詐騙偵測',   icon: '&#128737;' },
  { mode: 'fp-fn',   label: 'FP/FN 分析', icon: '&#9878;' },
  { mode: 'predict', label: 'Predict',    icon: '&#128269;' },
];

export function DashboardSwitcher() {
  const { state, dispatch } = useDashboard();

  return (
    <div className="flex gap-1.5 pb-3 border-b border-slate-700">
      {MODES.map(({ mode, label, icon }) => (
        <button
          key={mode}
          onClick={() => dispatch({ type: 'SET_DASHBOARD_MODE', mode })}
          className={`flex-1 flex items-center justify-center gap-1.5 px-2 py-2 text-xs font-semibold rounded-lg transition-colors focus:outline-none focus:ring-1 focus:ring-sky-500/50
            ${state.dashboardMode === mode
              ? 'bg-sky-500/20 text-sky-400 ring-1 ring-sky-500/50'
              : 'bg-slate-700/40 text-slate-400 hover:bg-slate-700/70 hover:text-slate-300'
            }`}
          dangerouslySetInnerHTML={{ __html: `${icon} ${label}` }}
        />
      ))}
    </div>
  );
}
