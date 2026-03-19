import { useEffect } from 'react';
import { useDashboard } from '../context/DashboardContext';

export function useSubgraph() {
  const { state, loadSubgraph } = useDashboard();
  const { selectedUserId, subgraph, subgraphCache } = state;

  useEffect(() => {
    if (selectedUserId === null) return;
    // Skip if already cached (cache check is in reducer via SELECT_USER)
    if (subgraphCache.has(selectedUserId)) return;
    loadSubgraph(selectedUserId, 2);
  }, [selectedUserId, subgraphCache, loadSubgraph]);

  // Auto-downgrade: if subgraph has > 200 nodes, reload with 1-hop
  useEffect(() => {
    if (selectedUserId === null || !subgraph) return;
    if (subgraph.nodes.length > 200) {
      loadSubgraph(selectedUserId, 1);
    }
  }, [subgraph, selectedUserId, loadSubgraph]);

  return {
    subgraph: state.subgraph,
    loading: state.loading.subgraph,
    error: state.error.subgraph,
    isLargeGraph: (state.subgraph?.nodes.length ?? 0) > 200,
  };
}
