import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import type { FpFnNode } from '../../types/index';
import { Spinner } from '../common/Spinner';
import { hasGraphData } from '../../utils/graphDataStore';

export function getFilteredFpFnNodes(nodes: FpFnNode[], keyword: string): FpFnNode[] {
  return nodes.filter(n => String(n.user_id).includes(keyword.trim()));
}

export function FpFnNodeSelector() {
  const { state, dispatch, loadSubgraph } = useDashboard();
  const [keyword, setKeyword] = useState('');
  const [debouncedKeyword, setDebouncedKeyword] = useState('');

  const nodes = state.fpFnMode === 'fp' ? state.fpNodes : state.fnNodes;
  const isFp = state.fpFnMode === 'fp';
  const loading = state.loading.fpFnNodes;

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedKeyword(keyword), 300);
    return () => clearTimeout(timer);
  }, [keyword]);

  // Reset keyword when mode switches
  useEffect(() => { setKeyword(''); }, [state.fpFnMode]);

  const filtered = getFilteredFpFnNodes(nodes, debouncedKeyword);

  const handleSelect = useCallback((userId: number) => {
    dispatch({ type: 'SELECT_USER', userId });
    if (!state.subgraphCache.has(userId)) {
      loadSubgraph(userId, 2);
    }
  }, [dispatch, loadSubgraph, state.subgraphCache]);

  const getScoreBadge = (node: FpFnNode) => {
    if (isFp) {
      // FP: normal predicted as fraud — orange tones
      return node.risk_score > 0.7
        ? { cls: 'bg-orange-900/60 text-orange-300 ring-1 ring-orange-500/50', label: '中高' }
        : { cls: 'bg-yellow-900/60 text-yellow-300 ring-1 ring-yellow-500/50', label: '中' };
    } else {
      // FN: fraud predicted as normal — show how well hidden
      return node.risk_score < 0.2
        ? { cls: 'bg-slate-700/60 text-slate-300 ring-1 ring-slate-500/50', label: '極低' }
        : { cls: 'bg-blue-900/60 text-blue-300 ring-1 ring-blue-500/50', label: '低' };
    }
  };

  const accentClass = isFp ? 'text-orange-400' : 'text-red-400';
  const title = isFp ? 'FP 誤判節點（白→黑）' : 'FN 漏判節點（黑→白）';
  const desc = isFp
    ? '正常用戶被預測為詐騙'
    : '詐騙用戶未被模型發現';

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-xs font-bold uppercase tracking-wider ${accentClass}`}>{title}</span>
        <span className="ml-auto text-xs text-slate-500">{filtered.length} 筆</span>
      </div>
      <p className="text-[10px] text-slate-500 mb-2">{desc}</p>

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

      <div className="max-h-44 overflow-y-auto ring-1 ring-slate-700 rounded-lg">
        {loading ? (
          <div className="flex items-center justify-center h-16"><Spinner /></div>
        ) : filtered.length === 0 ? (
          <div className="flex items-center justify-center gap-2 h-10">
            <p className="text-xs text-slate-500">無符合條件的節點</p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-700/40">
            {filtered.map(n => {
              const badge = getScoreBadge(n);
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
                      <span className={`font-semibold text-xs ${accentClass} flex items-center gap-1.5`}>
                        ID: {n.user_id}
                        <span className="text-slate-500 font-normal">{n.risk_score.toFixed(3)}</span>
                        {hasGraphData(n.user_id)
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
          </ul>
        )}
      </div>
    </div>
  );
}
