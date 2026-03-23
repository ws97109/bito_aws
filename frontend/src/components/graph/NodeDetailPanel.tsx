import { useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';

export function getShapColor(contribution: number): string {
  if (contribution > 0) return 'text-red-400';
  if (contribution < 0) return 'text-green-400';
  return 'text-slate-500';
}

export function NodeDetailPanel() {
  const { state, dispatch, loadSubgraph } = useDashboard();
  const { selectedNode, loading, error } = state;

  const handleSetAsCenter = useCallback(() => {
    if (!selectedNode) return;
    dispatch({ type: 'SELECT_USER', userId: selectedNode.user_id });
    if (!state.subgraphCache.has(selectedNode.user_id)) {
      loadSubgraph(selectedNode.user_id, 2);
    }
  }, [selectedNode, dispatch, loadSubgraph, state.subgraphCache]);

  if (loading.nodeDetail) {
    return <div className="text-sm text-slate-400">載入節點資訊...</div>;
  }

  if (error.nodeDetail) {
    return <div className="text-sm text-red-500">{error.nodeDetail}</div>;
  }

  if (!selectedNode) {
    return (
      <p className="text-xs text-slate-500 py-1 text-center">&#9675; 點擊圖中節點以查看詳細資訊</p>
    );
  }

  const { user_id, risk_score, status, account_age_days, shap_features, neighbor_counts } = selectedNode;
  const isFraud = status === 1;
  const maxShap = Math.max(...shap_features.slice(0, 3).map(f => Math.abs(f.contribution)), 0.001);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">&#9654;</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">節點詳細資訊</h3>
        </div>
        <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full ${isFraud ? 'bg-red-900/60 text-red-300 ring-1 ring-red-500/50' : 'bg-emerald-900/60 text-emerald-300 ring-1 ring-emerald-500/50'}`}>
          {isFraud ? '&#9888; 詐騙' : '&#10003; 正常'}
        </span>
      </div>

      {/* Metric cards */}
      <div className={`grid grid-cols-2 gap-2 p-3 rounded-lg ring-1 ${isFraud ? 'bg-red-900/10 ring-red-700/40' : 'bg-slate-800/40 ring-slate-700/50'}`}>
        <div className="rounded-md bg-slate-700/40 p-2.5">
          <p className="text-xs text-slate-400 uppercase tracking-wider">User ID</p>
          <p className="text-base font-bold text-sky-400 mt-0.5">{user_id}</p>
        </div>
        <div className="rounded-md bg-slate-700/40 p-2.5">
          <p className="text-xs text-slate-400 uppercase tracking-wider">風險分數</p>
          <p className={`text-base font-bold mt-0.5 ${risk_score > 0.9 ? 'text-red-400' : risk_score > 0.7 ? 'text-orange-400' : 'text-yellow-400'}`}>
            {risk_score.toFixed(3)}
          </p>
        </div>
        <div className="rounded-md bg-slate-700/40 p-2.5">
          <p className="text-xs text-slate-400 uppercase tracking-wider">狀態</p>
          <p className={`text-base font-bold mt-0.5 ${isFraud ? 'text-red-400' : 'text-emerald-400'}`}>
            {isFraud ? '詐騙' : '正常'}
          </p>
        </div>
        <div className="rounded-md bg-slate-700/40 p-2.5">
          <p className="text-xs text-slate-400 uppercase tracking-wider">帳戶年齡</p>
          <p className="text-base font-bold text-slate-100 mt-0.5">{account_age_days} 天</p>
        </div>
      </div>

      {/* SHAP features */}
      <div>
        <h4 className="text-xs uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
          SHAP Top-3 特徵貢獻
        </h4>
        {shap_features.length === 0 ? (
          <p className="text-xs text-slate-400">SHAP 資料不可用</p>
        ) : (
          <div className="space-y-1.5">
            {shap_features.slice(0, 3).map((f, i) => (
              <div key={i} className="relative overflow-hidden bg-slate-700/30 rounded-md p-2.5">
                {/* Magnitude bar */}
                <div
                  className={`absolute inset-y-0 left-0 rounded-md ${f.contribution > 0 ? 'bg-red-500/15' : 'bg-emerald-500/15'}`}
                  style={{ width: `${(Math.abs(f.contribution) / maxShap) * 100}%` }}
                />
                <div className="relative flex justify-between items-center">
                  <span className="text-slate-300 text-xs">{f.feature_name}</span>
                  <span className={`font-mono font-semibold text-xs ${getShapColor(f.contribution)}`}>
                    {f.contribution > 0 ? '+' : ''}{f.contribution.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Neighbor counts */}
      <div>
        <h4 className="text-xs uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
          鄰居數量
        </h4>
        <div className="flex gap-2">
          <span className="bg-sky-900/40 text-sky-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-sky-500/30">R1 · {neighbor_counts.r1}</span>
          <span className="bg-amber-900/40 text-amber-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-amber-500/30">R2 · {neighbor_counts.r2}</span>
          <span className="bg-emerald-900/40 text-emerald-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-emerald-500/30">R3 · {neighbor_counts.r3}</span>
        </div>
      </div>

      <div className="border-t border-slate-700 pt-3">
        <button
          onClick={handleSetAsCenter}
          className="w-full py-2 text-sm border border-indigo-500/70 text-indigo-400 hover:bg-indigo-500/20 font-semibold rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-colors"
        >
          &#9654; 設為中心節點
        </button>
      </div>
    </div>
  );
}
