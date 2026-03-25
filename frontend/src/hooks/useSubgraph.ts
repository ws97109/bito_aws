import { useEffect } from 'react';
import { useDashboard } from '../context/DashboardContext';

export function useSubgraph() {
  const { state, loadSubgraph } = useDashboard();
  const { selectedUserId, subgraphCache } = state;

  useEffect(() => {
    if (selectedUserId === null) return;
    // Skip if already cached (cache check is in reducer via SELECT_USER)
    if (subgraphCache.has(selectedUserId)) return;
    loadSubgraph(selectedUserId, 2);
  }, [selectedUserId, subgraphCache, loadSubgraph]);

  return {
    subgraph: state.subgraph,
    loading: state.loading.subgraph,
    error: state.error.subgraph,
    isLargeGraph: false,
  };
}
