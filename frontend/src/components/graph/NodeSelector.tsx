import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { useFraudNodes } from '../../hooks/useFraudNodes';
import type { FraudNode } from '../../types/index';
import { Spinner } from '../common/Spinner';
import { hasGraphData } from '../../utils/graphDataStore';

export function getFilteredNodes(nodes: FraudNode[], keyword: string): FraudNode[] {
  return nodes.filter(n => n.risk_score >= 0.4 && String(n.user_id).includes(keyword.trim()));
}

export function NodeSelector() {
  const { fraudNodes, loading } = useFraudNodes();
  const { state, dispatch, loadSubgraph, loadNodeDetail } = useDashboard();
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
    loadNodeDetail(userId);
    if (!state.subgraphCache.has(userId)) {
      loadSubgraph(userId, 2);
    }
  }, [dispatch, loadSubgraph, loadNodeDetail, state.subgraphCache]);

  const getRiskBadge = (score: number) => {
    if (score > 0.9) return { cls: 'bg-red-900/60 text-red-300 ring-1 ring-red-500/50',    label: '極高' };
    if (score > 0.7) return { cls: 'bg-orange-900/60 text-orange-300 ring-1 ring-orange-500/50', label: '高' };
    return              { cls: 'bg-yellow-900/60 text-yellow-300 ring-1 ring-yellow-500/50', label: '中' };
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-slate-400 text-xs">&#9741;</span>
        <label htmlFor="node-search" className="text-xs font-bold uppercase tracking-wider text-slate-300">
          選擇詐騙節點
        </label>
        <span className="ml-auto text-xs text-slate-500">{filtered.length} 筆</span>
      </div>

      {/* Search input with icon */}
      <div className="relative mb-2">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-xs pointer-events-none">&#128269;</span>
        <input
          id="node-search"
          type="text"
          placeholder="搜尋 user_id..."
          value={keyword}
          onChange={e => setKeyword(e.target.value)}
          className="w-full pl-8 pr-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm shadow-sm placeholder-slate-500 text-slate-200
                     focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
        />
      </div>

      <div className="max-h-44 overflow-y-auto ring-1 ring-slate-700 rounded-lg">
        {loading ? (
          <div className="flex items-center justify-center h-16">
            <Spinner />
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center gap-2 h-10 text-center">
            <span className="text-slate-600">&#128269;</span>
            <p className="text-xs text-slate-500">無符合條件的詐騙節點</p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-700/40">
            {filtered.map(n => {
              const badge = getRiskBadge(n.risk_score);
              const isSelected = state.selectedUserId === n.user_id;
              return (
                <li key={n.user_id}>
                  <button
                    onClick={() => handleSelect(n.user_id)}
                    className={`w-full text-left px-3 py-1.5 focus:outline-none transition-colors
                                ${isSelected
                                  ? 'bg-indigo-500/25 border-l-2 border-indigo-400'
                                  : 'border-l-2 border-transparent hover:bg-slate-700/40'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-sky-400 text-xs flex items-center gap-1.5">
                        ID: {n.user_id}
                        <span className="text-slate-500 font-normal">{n.risk_score.toFixed(3)}</span>
                        {hasGraphData(n.user_id)
                          ? <span title="有圖形資料" className="text-emerald-400 text-[10px]">&#9679;</span>
                          : <span title="無圖形資料" className="text-slate-600 text-[10px]">&#9675;</span>}
                      </span>
                      <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${badge.cls}`}>
                        {badge.label} {(n.risk_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
