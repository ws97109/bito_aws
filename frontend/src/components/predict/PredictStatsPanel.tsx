import { useDashboard } from '../../context/DashboardContext';

export function PredictStatsPanel() {
  const { state } = useDashboard();
  const { predictNodes, loading } = state;

  if (loading.predictNodes) {
    return <div className="text-sm text-slate-400">載入 Predict 資料...</div>;
  }

  const total = predictNodes.length;
  const blacklist = predictNodes.filter(n => n.is_blacklist === 1).length;
  const normal = total - blacklist;
  const blacklistRatio = total > 0 ? (blacklist / total * 100).toFixed(2) : '0';

  // Risk distribution
  const buckets = [0, 0, 0, 0, 0];
  for (const n of predictNodes) {
    const idx = Math.min(Math.floor(n.risk_score / 0.2), 4);
    buckets[idx]++;
  }
  const labels = ['[0, 0.2)', '[0.2, 0.4)', '[0.4, 0.6)', '[0.6, 0.8)', '[0.8, 1.0]'];
  const colors = ['bg-emerald-500', 'bg-sky-500', 'bg-yellow-500', 'bg-orange-500', 'bg-red-500'];

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <span className="w-0.5 h-3.5 bg-violet-500 rounded-full inline-block"></span>
        Predict 用戶概覽
      </h3>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-slate-700/40 rounded-lg p-2.5 ring-1 ring-slate-600/50">
          <p className="text-xs text-slate-400">總用戶</p>
          <p className="text-lg font-bold text-slate-100">{total.toLocaleString()}</p>
        </div>
        <div className="bg-slate-700/40 rounded-lg p-2.5 ring-1 ring-slate-600/50">
          <p className="text-xs text-slate-400">預測黑名單</p>
          <p className="text-lg font-bold text-red-400">{blacklist.toLocaleString()}</p>
        </div>
        <div className="bg-slate-700/40 rounded-lg p-2.5 ring-1 ring-slate-600/50">
          <p className="text-xs text-slate-400">預測正常</p>
          <p className="text-lg font-bold text-emerald-400">{normal.toLocaleString()}</p>
        </div>
        <div className="bg-slate-700/40 rounded-lg p-2.5 ring-1 ring-slate-600/50">
          <p className="text-xs text-slate-400">黑名單比例</p>
          <p className="text-lg font-bold text-amber-400">{blacklistRatio}%</p>
        </div>
      </div>

      {/* Risk distribution */}
      <div>
        <h4 className="text-xs uppercase tracking-wider text-slate-400 mb-2">風險分布</h4>
        <div className="space-y-1.5">
          {labels.map((label, i) => {
            const pct = total > 0 ? (buckets[i] / total * 100) : 0;
            return (
              <div key={label} className="flex items-center gap-2">
                <span className="text-xs text-slate-400 w-20 shrink-0">{label}</span>
                <div className="flex-1 bg-slate-700/50 rounded-full h-3 overflow-hidden">
                  <div className={`h-full rounded-full ${colors[i]}`} style={{ width: `${pct}%` }} />
                </div>
                <span className="text-xs text-slate-300 w-10 text-right">{buckets[i]}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
