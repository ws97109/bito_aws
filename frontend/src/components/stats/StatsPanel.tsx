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
    <div className="p-4 space-y-3">
      <h2 className="text-base font-semibold text-gray-700">統計摘要</h2>
      <div className="grid grid-cols-2 gap-2">
        <StatCard title="總節點數" value={stats.total_nodes} />
        <StatCard title="詐騙節點數" value={stats.fraud_nodes} />
        <StatCard title="正常節點數" value={stats.normal_nodes} />
        <StatCard title="詐騙比例" value={`${(stats.fraud_ratio * 100).toFixed(1)}%`} />
      </div>
      <RiskBarChart data={stats.risk_distribution} />
      <RelationStats counts={stats.relation_counts} />
    </div>
  );
}
