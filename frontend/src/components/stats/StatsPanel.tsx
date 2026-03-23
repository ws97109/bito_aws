import { useStats } from '../../hooks/useStats';
import { useDashboard } from '../../context/DashboardContext';
import { Spinner } from '../common/Spinner';
import { ErrorMessage } from '../common/ErrorMessage';
import { StatCard } from './StatCard';
import { RiskBarChart } from './RiskBarChart';
import { RelationStats } from './RelationStats';

export function StatsPanel() {
  const { stats, loading, error } = useStats();
  const { loadStats } = useDashboard();

  if (loading && !stats) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={loadStats} />;
  if (!stats) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 pb-3 border-b border-slate-700">
        <span className="text-slate-400">&#128202;</span>
        <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">統計摘要</h2>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <StatCard title="總節點數"  value={stats.total_nodes}                            accentColor="sky"     icon="&#9679;" />
        <StatCard title="詐騙節點"  value={stats.fraud_nodes}                            accentColor="red"     icon="&#9888;" />
        <StatCard title="正常節點"  value={stats.normal_nodes}                           accentColor="emerald" icon="&#10003;" />
        <StatCard title="詐騙比例"  value={`${(stats.fraud_ratio * 100).toFixed(1)}%`}   accentColor="orange"  icon="&#128308;" />
      </div>

      <div className="bg-slate-800/40 rounded-lg p-3 ring-1 ring-slate-700/50">
        <RiskBarChart data={stats.risk_distribution} />
      </div>

      <div className="bg-slate-800/40 rounded-lg p-3 ring-1 ring-slate-700/50">
        <RelationStats counts={stats.relation_counts} />
      </div>
    </div>
  );
}
