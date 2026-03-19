import { useEffect } from 'react';
import { useDashboard } from '../context/DashboardContext';

export function useFraudNodes() {
  const { state, loadFraudNodes } = useDashboard();

  useEffect(() => {
    loadFraudNodes();
  }, [loadFraudNodes]);

  return { fraudNodes: state.fraudNodes, loading: state.loading.fraudNodes, error: state.error.fraudNodes };
}
