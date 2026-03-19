import { useCallback } from 'react';
import { useDashboard } from '../../context/DashboardContext';

// Task 10.1: Pure function for SHAP contribution color
export function getShapColor(contribution: number): string {
  if (contribution > 0) return 'text-red-600';
  if (contribution < 0) return 'text-green-600';
  return 'text-gray-500';
}

// Task 10.3: NodeDetailPanel component
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
    return <div className="p-4 text-sm text-gray-400">載入節點資訊...</div>;
  }

  if (error.nodeDetail) {
    return <div className="p-4 text-sm text-red-500">{error.nodeDetail}</div>;
  }

  if (!selectedNode) {
    return (
      <div className="p-4 text-sm text-gray-400">
        點擊圖中節點以查看詳細資訊
      </div>
    );
  }

  const { user_id, risk_score, status, account_age_days, shap_features, neighbor_counts } = selectedNode;

  return (
    <div className="p-4 border-t border-gray-200 space-y-3">
      <h3 className="text-sm font-semibold text-gray-700">節點詳細資訊</h3>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div><span className="text-gray-500">User ID:</span> <span className="font-medium">{user_id}</span></div>
        <div><span className="text-gray-500">風險分數:</span> <span className="font-medium">{risk_score.toFixed(3)}</span></div>
        <div><span className="text-gray-500">狀態:</span> <span className={status === 1 ? 'text-red-600 font-medium' : 'text-green-600 font-medium'}>{status === 1 ? '詐騙' : '正常'}</span></div>
        <div><span className="text-gray-500">帳戶年齡:</span> <span className="font-medium">{account_age_days} 天</span></div>
      </div>

      <div>
        <p className="text-xs font-medium text-gray-600 mb-1">SHAP Top-3 特徵</p>
        {shap_features.length === 0 ? (
          <p className="text-xs text-gray-400">SHAP 資料不可用</p>
        ) : (
          <div className="space-y-1">
            {shap_features.slice(0, 3).map((f, i) => (
              <div key={i} className="flex justify-between text-xs">
                <span className="text-gray-600">{f.feature_name}</span>
                <span className={getShapColor(f.contribution)}>
                  {f.contribution > 0 ? '+' : ''}{f.contribution.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div>
        <p className="text-xs font-medium text-gray-600 mb-1">鄰居數量</p>
        <div className="flex gap-3 text-xs text-gray-600">
          <span>R1: {neighbor_counts.r1}</span>
          <span>R2: {neighbor_counts.r2}</span>
          <span>R3: {neighbor_counts.r3}</span>
        </div>
      </div>

      <button
        onClick={handleSetAsCenter}
        className="w-full py-1.5 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
      >
        設為中心節點
      </button>
    </div>
  );
}
