import { useState } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { hasGraphData } from '../../utils/graphDataStore';
import type { FraudNode } from '../../types/index';

type RiskTab = 'high' | 'mid' | 'low';

interface Props {
  nodes: FraudNode[];
}

export function RiskNodeList({ nodes }: Props) {
  const { state, dispatch, loadSubgraph, loadNodeDetail } = useDashboard();
  const [tab, setTab] = useState<RiskTab>('high');

  const filtered = nodes.filter(n =>
    tab === 'high' ? n.risk_score >= 0.8743 :
    tab === 'mid'  ? n.risk_score >= 0.65 && n.risk_score < 0.8743 :
                     n.risk_score >= 0.45 && n.risk_score < 0.65
  );

  const counts = {
    high: nodes.filter(n => n.risk_score >= 0.8743).length,
    mid:  nodes.filter(n => n.risk_score >= 0.65 && n.risk_score < 0.8743).length,
    low:  nodes.filter(n => n.risk_score >= 0.45 && n.risk_score < 0.65).length,
  };

  const handleSelect = (userId: number) => {
    dispatch({ type: 'SELECT_USER', userId });
    loadNodeDetail(userId);
    if (!state.subgraphCache.has(userId)) loadSubgraph(userId, 2);
  };

  const tabs: { key: RiskTab; label: string; color: string; active: string }[] = [
    { key: 'high', label: '高風險', color: 'text-red-400',    active: 'bg-red-500/20 border-red-500/60' },
    { key: 'mid',  label: '中風險', color: 'text-orange-400', active: 'bg-orange-500/20 border-orange-500/60' },
    { key: 'low',  label: '低風險', color: 'text-yellow-400', active: 'bg-yellow-500/20 border-yellow-500/60' },
  ];

  return (
    <div>
      <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
        <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
        風險節點列表
      </h3>

      {/* Tabs */}
      <div className="flex gap-1 mb-2">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 py-1 text-xs font-semibold rounded border transition-colors
              ${tab === t.key ? `${t.active} ${t.color}` : 'border-slate-700 text-slate-500 hover:text-slate-300'}`}
          >
            {t.label}<br/>
            <span className="text-[10px] font-normal">({counts[t.key]})</span>
          </button>
        ))}
      </div>

      {/* List */}
      <div className="max-h-44 overflow-y-auto ring-1 ring-slate-700 rounded-lg">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-10">
            <p className="text-xs text-slate-500">無符合節點</p>
          </div>
        ) : (
          <ul className="divide-y divide-slate-700/40">
            {filtered.map(n => {
              const isSelected = state.selectedUserId === n.user_id;
              const riskColor = n.risk_score >= 0.8743 ? 'text-red-400' : n.risk_score >= 0.65 ? 'text-orange-400' : 'text-yellow-400';
              return (
                <li key={n.user_id}>
                  <button
                    onClick={() => handleSelect(n.user_id)}
                    className={`w-full text-left px-3 py-1.5 focus:outline-none transition-colors
                      ${isSelected ? 'bg-indigo-500/25 border-l-2 border-indigo-400' : 'border-l-2 border-transparent hover:bg-slate-700/40'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sky-400 text-xs font-semibold flex items-center gap-1">
                        {n.user_id}
                        {hasGraphData(n.user_id)
                          ? <span className="text-emerald-400 text-[10px]">&#9679;</span>
                          : <span className="text-slate-600 text-[10px]">&#9675;</span>}
                      </span>
                      <span className={`font-mono text-xs font-semibold ${riskColor}`}>
                        {(n.risk_score * 100).toFixed(0)}%
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
