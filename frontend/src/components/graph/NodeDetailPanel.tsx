import { useDashboard } from '../../context/DashboardContext';
import type { ShapFeature } from '../../types/index';

// ── Waterfall chart (same style as FP/FN ShapPanel) ──────────────────────────

const BASE_VALUE = -2.885;

interface NodeWaterfallProps {
  features: ShapFeature[];
  baseValue: number;
}

function NodeWaterfall({ features, baseValue }: NodeWaterfallProps) {
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);

  const finalValue = baseValue + sorted.reduce((s, f) => s + f.contribution, 0);

  let running = finalValue;
  const rows = sorted.map(f => {
    const end = running;
    const start = running - f.contribution;
    running = start;
    return { feature: f, start, end };
  });

  const allX = [baseValue, finalValue, ...rows.flatMap(r => [r.start, r.end])];
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const xPad = (xMax - xMin) * 0.15;
  const xLo = xMin - xPad;
  const xHi = xMax + xPad;

  const ROW_H   = 26;
  const BAR_H   = 14;
  const LABEL_W = 200;
  const TOTAL_W = 600;
  const BAR_AREA = TOTAL_W - LABEL_W - 6;
  const PAD_T   = 28;
  const PAD_B   = 28;
  const TOTAL_H = rows.length * ROW_H + PAD_T + PAD_B;

  const toX = (v: number) => LABEL_W + ((v - xLo) / (xHi - xLo)) * BAR_AREA;

  return (
    <svg
      viewBox={`0 0 ${TOTAL_W} ${TOTAL_H}`}
      width="100%"
      style={{ display: 'block' }}
    >
      {/* Final prediction label at top */}
      <text
        x={toX(finalValue)} y={PAD_T - 14}
        textAnchor="middle" fontSize="9"
        fill="#64748b"
      >
        預測值
      </text>
      <text
        x={toX(finalValue)} y={PAD_T - 4}
        textAnchor="middle" fontSize="11" fontFamily="monospace"
        fill="#94a3b8" fontWeight="bold"
      >
        f(x) = {finalValue.toFixed(3)}
      </text>
      <line
        x1={toX(finalValue)} y1={PAD_T}
        x2={toX(finalValue)} y2={PAD_T + 4}
        stroke="#64748b" strokeWidth="1"
      />

      {/* Feature rows */}
      {rows.map((row, i) => {
        const y      = PAD_T + i * ROW_H + ROW_H / 2;
        const isPos  = row.feature.contribution >= 0;
        const x1     = toX(Math.min(row.start, row.end));
        const x2     = toX(Math.max(row.start, row.end));
        const barW   = Math.max(x2 - x1, 2);
        const fill   = isPos ? '#ef4444' : '#3b82f6';
        const fillBg = isPos ? 'rgba(239,68,68,0.12)' : 'rgba(59,130,246,0.12)';
        const txtClr = isPos ? '#fca5a5' : '#93c5fd';
        const valTxt = (isPos ? '+' : '') + row.feature.contribution.toFixed(2);
        const connX  = toX(row.start);

        return (
          <g key={i}>
            {i < rows.length - 1 && (
              <line
                x1={connX} y1={y + BAR_H / 2}
                x2={connX} y2={y + ROW_H - BAR_H / 2}
                stroke="#475569" strokeWidth="1" strokeDasharray="3,2"
              />
            )}
            {i === rows.length - 1 && (
              <line
                x1={connX} y1={y + BAR_H / 2}
                x2={connX} y2={TOTAL_H - PAD_B + 3}
                stroke="#475569" strokeWidth="1" strokeDasharray="3,2"
              />
            )}

            <text
              x={4} y={y + 4}
              textAnchor="start" fontSize="10"
              fill="#e2e8f0"
            >
              {row.feature.feature_name}
            </text>

            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fillBg} />
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fill} opacity={0.82} />

            <text
              x={isPos ? x2 + 4 : x1 - 4}
              y={y + 4}
              textAnchor={isPos ? 'start' : 'end'}
              fontSize="10" fontFamily="monospace" fontWeight="bold"
              fill={txtClr}
            >
              {valTxt}
            </text>
          </g>
        );
      })}

      {/* Base value label at bottom */}
      <text
        x={toX(baseValue)} y={TOTAL_H - PAD_B + 12}
        textAnchor="middle" fontSize="9"
        fill="#64748b"
      >
        基準值
      </text>
      <text
        x={toX(baseValue)} y={TOTAL_H - PAD_B + 23}
        textAnchor="middle" fontSize="11" fontFamily="monospace"
        fill="#94a3b8" fontWeight="bold"
      >
        E[f(x)] = {baseValue.toFixed(3)}
      </text>
      <line
        x1={toX(baseValue)} y1={TOTAL_H - PAD_B - 3}
        x2={toX(baseValue)} y2={TOTAL_H - PAD_B + 3}
        stroke="#64748b" strokeWidth="1"
      />
    </svg>
  );
}

// ── NodeDetailPanel ──────────────────────────────────────────────────────────

export function NodeDetailPanel() {
  const { state } = useDashboard();
  const { selectedNode, loading, error } = state;

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

  const { user_id, risk_score, status, shap_features, neighbor_counts } = selectedNode;
  const isFraud = status === 1;

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
          <p className={`text-base font-bold mt-0.5 ${risk_score > 0.9 ? 'text-red-400' : risk_score > 0.7 ? 'text-orange-400' : risk_score > 0.4 ? 'text-yellow-400' : 'text-emerald-400'}`}>
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
          <p className="text-xs text-slate-400 uppercase tracking-wider">鄰居數量</p>
          <p className="text-base font-bold text-slate-100 mt-0.5">{neighbor_counts.r1 + neighbor_counts.r2 + neighbor_counts.r3}</p>
        </div>
      </div>

      {/* SHAP waterfall */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-700">
          <span className="w-0.5 h-4 bg-sky-500 rounded-full inline-block flex-shrink-0"></span>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">SHAP 瀑布圖 — 特徵貢獻分析</h4>
          <span className="ml-auto text-[10px] text-slate-500 font-mono">User {user_id}</span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-[10px] text-slate-400">
          <div className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-500 opacity-80"></span>
            正向貢獻（增加詐騙機率）
          </div>
          <div className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-blue-500 opacity-80"></span>
            負向貢獻（降低詐騙機率）
          </div>
        </div>

        {/* Chart area */}
        <div className="bg-slate-900/40 rounded-lg px-2 py-1.5 ring-1 ring-slate-700/40">
          {shap_features.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-4">SHAP 資料不可用</p>
          ) : (
            <NodeWaterfall features={shap_features} baseValue={BASE_VALUE} />
          )}
        </div>
      </div>

      {/* Neighbor counts */}
      <div>
        <h4 className="text-xs uppercase tracking-wider text-slate-400 mb-2 flex items-center gap-1.5">
          <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
          鄰居關係
        </h4>
        <div className="flex gap-2">
          <span className="bg-sky-900/40 text-sky-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-sky-500/30">錢包→帳戶 · {neighbor_counts.r1}</span>
          <span className="bg-amber-900/40 text-amber-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-amber-500/30">帳戶→帳戶 · {neighbor_counts.r2}</span>
          <span className="bg-emerald-900/40 text-emerald-300 rounded-full px-3 py-1 text-xs font-semibold ring-1 ring-emerald-500/30">帳戶→錢包 · {neighbor_counts.r3}</span>
        </div>
      </div>

    </div>
  );
}
