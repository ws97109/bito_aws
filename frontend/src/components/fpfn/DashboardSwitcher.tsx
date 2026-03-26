import { useDashboard } from '../../context/DashboardContext';
import type { DashboardMode } from '../../types/index';

const MODES: { mode: DashboardMode; label: string; icon: string; accent: string }[] = [
  { mode: 'features', label: '特徵說明', icon: '📋', accent: 'purple' },
  { mode: 'fraud',    label: '詐騙偵測', icon: '🛡️', accent: 'sky'    },
  { mode: 'fp-fn',    label: 'FP/FN 分析', icon: '⚖️', accent: 'sky'  },
  { mode: 'predict',  label: 'Predict',  icon: '🔍', accent: 'sky'    },
];

const ACTIVE_CLASS: Record<string, string> = {
  purple: 'bg-purple-500/20 text-purple-400 ring-1 ring-purple-500/50',
  sky:    'bg-sky-500/20 text-sky-400 ring-1 ring-sky-500/50',
};

const FOCUS_CLASS: Record<string, string> = {
  purple: 'focus:ring-purple-500/50',
  sky:    'focus:ring-sky-500/50',
};

export function DashboardSwitcher() {
  const { state, dispatch } = useDashboard();

  return (
    <div className="flex gap-1.5 pb-3 border-b border-slate-700">
      {MODES.map(({ mode, label, icon, accent }) => {
        const isActive = state.dashboardMode === mode;
        return (
          <button
            key={mode}
            onClick={() => dispatch({ type: 'SET_DASHBOARD_MODE', mode })}
            className={`flex-1 flex items-center justify-center gap-1 px-2 py-2 text-xs font-semibold rounded-lg transition-colors focus:outline-none focus:ring-1 ${FOCUS_CLASS[accent]}
              ${isActive
                ? ACTIVE_CLASS[accent]
                : 'bg-slate-700/40 text-slate-400 hover:bg-slate-700/70 hover:text-slate-300'
              }`}
          >
            <span>{icon}</span>
            <span>{label}</span>
          </button>
        );
      })}
    </div>
  );
}
