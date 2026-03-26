import { useStats } from '../../hooks/useStats';
import { useDashboard } from '../../context/DashboardContext';
import { useFraudNodes } from '../../hooks/useFraudNodes';
import { Spinner } from '../common/Spinner';
import { ErrorMessage } from '../common/ErrorMessage';
import { RelationStats } from './RelationStats';
import { RiskNodeList } from './RiskNodeList';

const MODEL_METRICS = [
  { label: 'AUC-ROC', value: '0.8612', color: 'sky' },
  { label: 'AUC-PR',  value: '0.3071', color: 'violet' },
  { label: 'Recall',  value: '0.4634', color: 'emerald' },
  { label: 'F1 Score',value: '0.3572', color: 'amber' },
];

const colorMap: Record<string, string> = {
  sky:     'bg-sky-900/40 text-sky-300 ring-sky-500/30',
  amber:   'bg-amber-900/40 text-amber-300 ring-amber-500/30',
  emerald: 'bg-emerald-900/40 text-emerald-300 ring-emerald-500/30',
  violet:  'bg-violet-900/40 text-violet-300 ring-violet-500/30',
};

export function StatsPanel() {
  const { stats, loading, error } = useStats();
  const { loadStats } = useDashboard();
  const { fraudNodes } = useFraudNodes();

  if (loading && !stats) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={loadStats} />;
  if (!stats) return null;

  return (
    <div className="space-y-4">
      {/* 模型效能 */}
      <div className="flex items-center gap-2 pb-3 border-b border-slate-700">
        <span className="text-slate-400">&#128202;</span>
        <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">模型效能</h2>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {MODEL_METRICS.map(m => (
          <div key={m.label} className={`rounded-lg p-2.5 ring-1 ${colorMap[m.color]}`}>
            <p className="text-[10px] uppercase tracking-wider opacity-70">{m.label}</p>
            <p className="text-lg font-bold mt-0.5">{m.value}</p>
          </div>
        ))}
      </div>

      {/* 關係統計 */}
      <div className="bg-slate-800/40 rounded-lg p-3 ring-1 ring-slate-700/50">
        <RelationStats counts={stats.relation_counts} />
      </div>

      {/* 高風險節點列表 */}
      <div className="bg-slate-800/40 rounded-lg p-3 ring-1 ring-slate-700/50">
        <RiskNodeList nodes={fraudNodes} />
      </div>
    </div>
  );
}
