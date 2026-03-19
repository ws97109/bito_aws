import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { useFraudNodes } from '../../hooks/useFraudNodes';
import type { FraudNode } from '../../types/index';

export function getFilteredNodes(nodes: FraudNode[], keyword: string): FraudNode[] {
  return nodes.filter(n => n.risk_score >= 0.5 && String(n.user_id).includes(keyword.trim()));
}

export function NodeSelector() {
  const { fraudNodes, loading } = useFraudNodes();
  const { state, dispatch, loadSubgraph } = useDashboard();
  const [keyword, setKeyword] = useState('');
  const [debouncedKeyword, setDebouncedKeyword] = useState('');

  // 300ms debounce
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedKeyword(keyword), 300);
    return () => clearTimeout(timer);
  }, [keyword]);

  const filtered = getFilteredNodes(fraudNodes, debouncedKeyword);

  const handleSelect = useCallback((userId: number) => {
    dispatch({ type: 'SELECT_USER', userId });
    // Only load subgraph if not cached
    if (!state.subgraphCache.has(userId)) {
      loadSubgraph(userId, 2);
    }
  }, [dispatch, loadSubgraph, state.subgraphCache]);

  return (
    <div className="p-4">
      <label className="block text-sm font-medium text-gray-700 mb-1">選擇詐騙節點</label>
      <input
        type="text"
        placeholder="搜尋 user_id..."
        value={keyword}
        onChange={e => setKeyword(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm mb-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      {loading ? (
        <p className="text-sm text-gray-400">載入中...</p>
      ) : filtered.length === 0 ? (
        <p className="text-sm text-gray-400">目前無確定詐騙節點</p>
      ) : (
        <select
          size={Math.min(filtered.length, 8)}
          className="w-full border border-gray-300 rounded-md text-sm"
          value={state.selectedUserId ?? ''}
          onChange={e => handleSelect(Number(e.target.value))}
        >
          <option value="" disabled>請選擇詐騙節點</option>
          {filtered.map(n => (
            <option key={n.user_id} value={n.user_id}>
              {n.user_id} | 風險分數: {n.risk_score.toFixed(2)}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
