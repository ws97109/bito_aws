import { useEffect } from 'react';
import { useDashboard } from '../context/DashboardContext';

export function useStats() {
  const { state, loadStats } = useDashboard();

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 60_000);
    return () => clearInterval(interval);
  }, [loadStats]);

  return { stats: state.stats, loading: state.loading.stats, error: state.error.stats };
}
