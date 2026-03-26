import { useState, useEffect, useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { useFraudNodes } from '../../hooks/useFraudNodes';
import type { FraudNode } from '../../types/index';
import { Spinner } from '../common/Spinner';
import { hasGraphData, findUsersByWalletId, type WalletSearchResult } from '../../utils/graphDataStore';

type SearchMode = 'node' | 'wallet';

export function getFilteredNodes(nodes: FraudNode[], keyword: string): FraudNode[] {
  return nodes.filter(n => n.risk_score >= 0.4 && String(n.user_id).includes(keyword.trim()));
}

export function NodeSelector() {
  const { fraudNodes, loading } = useFraudNodes();
  const { state, dispatch, loadSubgraph, loadNodeDetail } = useDashboard();

  const [mode, setMode] = useState<SearchMode>('node');
  const [keyword, setKeyword] = useState('');
  const [debouncedKeyword, setDebouncedKeyword] = useState('');
  const [walletResults, setWalletResults] = useState<WalletSearchResult[]>([]);
  const [walletLoading, setWalletLoading] = useState(false);

  // 300ms debounce
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedKeyword(keyword), 300);
    return () => clearTimeout(timer);
  }, [keyword]);

  // Wallet search when debounced keyword changes and mode is wallet
  useEffect(() => {
    if (mode !== 'wallet') return;
    if (!debouncedKeyword.trim()) {
      setWalletResults([]);
      return;
    }
    setWalletLoading(true);
    findUsersByWalletId(debouncedKeyword)
      .then(results => setWalletResults(results))
      .catch(() => setWalletResults([]))
      .finally(() => setWalletLoading(false));
  }, [debouncedKeyword, mode]);

  const handleModeChange = useCallback((next: SearchMode) => {
    setMode(next);
    setKeyword('');
    setDebouncedKeyword('');
    setWalletResults([]);
  }, []);

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

  const filteredNodes = getFilteredNodes(fraudNodes, debouncedKeyword);
  const resultCount = mode === 'node' ? filteredNodes.length : walletResults.length;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-slate-400 text-xs">&#9741;</span>
        <label className="text-xs font-bold uppercase tracking-wider text-slate-300">
          選擇詐騙節點
        </label>
        <span className="ml-auto text-xs text-slate-500">{resultCount} 筆</span>
      </div>

      {/* Mode toggle */}
      <div className="flex mb-2 rounded-lg overflow-hidden ring-1 ring-slate-600/60 bg-slate-800/60">
        <button
          onClick={() => handleModeChange('node')}
          className={`flex-1 py-1.5 text-xs font-medium transition-colors focus:outline-none
            ${mode === 'node'
              ? 'bg-indigo-600/70 text-indigo-100'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40'}`}
        >
          節點 user_id
        </button>
        <button
          onClick={() => handleModeChange('wallet')}
          className={`flex-1 py-1.5 text-xs font-medium transition-colors focus:outline-none
            ${mode === 'wallet'
              ? 'bg-indigo-600/70 text-indigo-100'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40'}`}
        >
          錢包 ID
        </button>
      </div>

      {/* Search input */}
      <div className="relative mb-2">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-xs pointer-events-none">&#128269;</span>
        <input
          id="node-search"
          type="text"
          placeholder={mode === 'node' ? '搜尋 user_id...' : '搜尋錢包 ID（如 abc123）...'}
          value={keyword}
          onChange={e => setKeyword(e.target.value)}
          className="w-full pl-8 pr-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-sm shadow-sm placeholder-slate-500 text-slate-200
                     focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
        />
      </div>

      {/* Results list */}
      <div className="max-h-44 overflow-y-auto ring-1 ring-slate-700 rounded-lg">
        {mode === 'node' ? (
          /* ── Node search results ── */
          loading ? (
            <div className="flex items-center justify-center h-16">
              <Spinner />
            </div>
          ) : filteredNodes.length === 0 ? (
            <div className="flex items-center justify-center gap-2 h-10 text-center">
              <span className="text-slate-600">&#128269;</span>
              <p className="text-xs text-slate-500">無符合條件的詐騙節點</p>
            </div>
          ) : (
            <ul className="divide-y divide-slate-700/40">
              {filteredNodes.map(n => {
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
          )
        ) : (
          /* ── Wallet search results ── */
          walletLoading ? (
            <div className="flex items-center justify-center h-16">
              <Spinner />
            </div>
          ) : !debouncedKeyword.trim() ? (
            <div className="flex items-center justify-center gap-2 h-10 text-center">
              <span className="text-slate-600">&#128176;</span>
              <p className="text-xs text-slate-500">輸入錢包 ID 以搜尋關聯節點</p>
            </div>
          ) : walletResults.length === 0 ? (
            <div className="flex items-center justify-center gap-2 h-10 text-center">
              <span className="text-slate-600">&#128269;</span>
              <p className="text-xs text-slate-500">找不到符合的錢包 ID</p>
            </div>
          ) : (
            <ul className="divide-y divide-slate-700/40">
              {walletResults.map((r, i) => {
                const badge = getRiskBadge(r.riskScore);
                const isSelected = state.selectedUserId === r.userId;
                const relLabel = r.relationType === 'R1' ? '資金流入' : '資金流出';
                const relCls   = r.relationType === 'R1' ? 'text-emerald-400' : 'text-orange-400';
                return (
                  <li key={`${r.walletId}-${r.userId}-${i}`}>
                    <button
                      onClick={() => handleSelect(r.userId)}
                      className={`w-full text-left px-3 py-2 focus:outline-none transition-colors
                                  ${isSelected
                                    ? 'bg-indigo-500/25 border-l-2 border-indigo-400'
                                    : 'border-l-2 border-transparent hover:bg-slate-700/40'}`}
                    >
                      {/* Wallet ID row */}
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="text-slate-500 text-[10px]">&#128176;</span>
                        <span className="text-[10px] text-slate-400 font-mono truncate max-w-[160px]">
                          {r.walletId}
                        </span>
                        <span className={`text-[10px] font-medium ${relCls}`}>{relLabel}</span>
                      </div>
                      {/* User node row */}
                      <div className="flex justify-between items-center">
                        <span className="font-semibold text-sky-400 text-xs flex items-center gap-1.5">
                          ID: {r.userId}
                          <span className="text-slate-500 font-normal">{r.riskScore.toFixed(3)}</span>
                          {hasGraphData(r.userId)
                            ? <span title="有圖形資料" className="text-emerald-400 text-[10px]">&#9679;</span>
                            : <span title="無圖形資料" className="text-slate-600 text-[10px]">&#9675;</span>}
                        </span>
                        <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${badge.cls}`}>
                          {badge.label} {(r.riskScore * 100).toFixed(0)}%
                        </span>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )
        )}
      </div>
    </div>
  );
}
