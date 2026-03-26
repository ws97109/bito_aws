import { useState, useEffect, useCallback, useMemo } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { Spinner } from '../common/Spinner';
import { hasGraphData } from '../../utils/graphDataStore';

export function PredictNodeSelector() {
  const { state, dispatch, loadSubgraph } = useDashboard();
  const { predictNodes, selectedUserId, loading } = state;
  const [keyword, setKeyword] = useState('');
  const [debouncedKeyword, setDebouncedKeyword] = useState('');
  const [filterMode, setFilterMode] = useState<'all' | 'blacklist' | 'normal'>('all');

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedKeyword(keyword), 300);
    return () => clearTimeout(timer);
  }, [keyword]);

  const filtered = useMemo(() => {
    let nodes = predictNodes;
    if (filterMode === 'blacklist') nodes = nodes.filter(n => n.is_blacklist === 1);
    else if (filterMode === 'normal') nodes = nodes.filter(n => n.is_blacklist === 0);
    if (debouncedKeyword.trim()) {
      nodes = nodes.filter(n => String(n.user_id).includes(debouncedKeyword.trim()));
    }
    return nodes;
  }, [predictNodes, debouncedKeyword, filterMode]);

  const handleSelect = useCallback((userId: number) => {
    dispatch({ type: 'SELECT_USER', userId });
    if (!state.subgraphCache.has(userId)) {
      loadSubgraph(userId, 2);
    }
  }, [dispatch, loadSubgraph, state.subgraphCache]);

  const getScoreBadge = (riskScore: number) => {
    if      (riskScore >= 0.8415) return { cls: 'bg-red-900/60 text-red-300 ring-1 ring-red-500/50',         label: '極高' };
    else if (riskScore >= 0.6)    return { cls: 'bg-orange-900/60 text-orange-300 ring-1 ring-orange-500/50', label: '高' };
    else if (riskScore >= 0.4)    return { cls: 'bg-yellow-900/60 text-yellow-300 ring-1 ring-yellow-500/50', label: '中' };
    else if (riskScore >= 0.2)    return { cls: 'bg-sky-900/60 text-sky-300 ring-1 ring-sky-500/50',          label: '中低' };
    else                          return { cls: 'bg-slate-700/60 text-slate-300 ring-1 ring-slate-500/50',    label: '低' };
  };

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs font-bold uppercase tracking-wider text-violet-400">Predict 預測節點</span>
        <span className="ml-auto text-xs text-slate-500">{filtered.length} 筆</span>
      </div>
      <p className="text-[10px] text-slate-500 mb-2">模型預測結果：黑名單 / 正常用戶</p>

      {/* Filter tabs */}
      <div className="flex gap-1.5 mb-2">
        {([
          { key: 'all', label: `全部 (${predictNodes.length})` },
          { key: 'blacklist', label: `黑名單 (${predictNodes.filter(n => n.is_blacklist === 1).length})` },
          { key: 'normal', label: `正常 (${predictNodes.filter(n => n.is_blacklist === 0).length})` },
        ] as const).map(tab => (
          <button
            key={tab.key}
            onClick={() => setFilterMode(tab.key)}
            className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-colors ${
              filterMode === tab.key
                ? 'bg-violet-500/20 text-violet-400 ring-1 ring-violet-500/50'
                : 'bg-slate-700/40 text-slate-400 hover:bg-slate-700/70'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="relative mb-2">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-xs pointer-events-none">&#128269;</span>
        <input
          type="text"
          placeholder="搜尋 user_id..."
          value={keyword}
          onChange={e => setKeyword(e.target.value)}
          className="w-full pl-8 pr-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm shadow-sm placeholder-slate-500 text-slate-200
                     focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
        />
      </div>

      {/* Node list */}
      <div className="max-h-44 overflow-y-auto ring-1 ring-slate-700 rounded-lg">
        {loading.predictNodes ? (
          <div className="flex items-center justify-center h-16"><Spinner /></div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center gap-2 h-10">
            <p className="text-xs text-slate-500">無符合條件的節點</p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-700/40">
            {filtered.slice(0, 200).map(node => {
              const isSelected = selectedUserId === node.user_id;
              const isBlack = node.is_blacklist === 1;
              const badge = getScoreBadge(node.risk_score);
              return (
                <li key={node.user_id}>
                  <button
                    onClick={() => handleSelect(node.user_id)}
                    className={`w-full text-left px-3 py-1.5 focus:outline-none transition-colors
                      ${isSelected
                        ? 'bg-indigo-500/25 border-l-2 border-indigo-400'
                        : 'border-l-2 border-transparent hover:bg-slate-700/40'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className={`font-semibold text-xs ${isBlack ? 'text-red-400' : 'text-emerald-400'} flex items-center gap-1.5`}>
                        ID: {node.user_id}
                        <span className="text-slate-500 font-normal">{node.risk_score.toFixed(3)}</span>
                        {hasGraphData(node.user_id)
                          ? <span title="有圖形資料" className="text-emerald-400 text-[10px]">&#9679;</span>
                          : <span title="無圖形資料" className="text-slate-600 text-[10px]">&#9675;</span>}
                      </span>
                      <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${badge.cls}`}>
                        {badge.label}
                      </span>
                    </div>
                  </button>
                </li>
              );
            })}
            {filtered.length > 200 && (
              <li>
                <p className="text-xs text-slate-500 text-center py-1.5">顯示前 200 筆，請搜尋縮小範圍</p>
              </li>
            )}
          </ul>
        )}
      </div>
    </div>
  );
}
