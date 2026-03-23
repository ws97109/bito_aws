import { useEffect } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import type { ShapWaterfallFeature } from '../../types/index';
import { Spinner } from '../common/Spinner';

// ── Waterfall chart ───────────────────────────────────────────────────────────

interface WaterfallChartProps {
  features: ShapWaterfallFeature[];
  baseValue: number;
}

function WaterfallChart({ features, baseValue }: WaterfallChartProps) {
  // Sort by |contribution| desc, cap at 8 rows for readability
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 8);

  const finalValue = baseValue + sorted.reduce((s, f) => s + f.contribution, 0);

  // Build bar positions top→bottom (largest contribution row = top)
  let running = finalValue;
  const rows = sorted.map(f => {
    const end = running;
    const start = running - f.contribution;
    running = start;
    return { feature: f, start, end };
  });

  // x-axis range
  const allX = [baseValue, finalValue, ...rows.flatMap(r => [r.start, r.end])];
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const xPad = (xMax - xMin) * 0.12;
  const xLo = xMin - xPad;
  const xHi = xMax + xPad;

  // Layout constants
  const ROW_H   = 22;
  const BAR_H   = 12;
  const LABEL_W = 160;
  const TOTAL_W = 560;
  const BAR_AREA = TOTAL_W - LABEL_W - 6;
  const PAD_T   = 20;
  const PAD_B   = 20;
  const TOTAL_H = rows.length * ROW_H + PAD_T + PAD_B;

  const toX = (v: number) => LABEL_W + ((v - xLo) / (xHi - xLo)) * BAR_AREA;

  return (
    <svg
      viewBox={`0 0 ${TOTAL_W} ${TOTAL_H}`}
      width="100%"
      style={{ display: 'block' }}
    >
      {/* Final prediction value at top */}
      <text
        x={toX(finalValue)} y={PAD_T - 7}
        textAnchor="middle" fontSize="10" fontFamily="monospace"
        fill="#94a3b8" fontWeight="bold"
      >
        {finalValue.toFixed(3)}
      </text>
      <line
        x1={toX(finalValue)} y1={PAD_T - 3}
        x2={toX(finalValue)} y2={PAD_T + 3}
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
        const fillBg = isPos ? 'rgba(239,68,68,0.18)' : 'rgba(59,130,246,0.18)';
        const txtClr = isPos ? '#fca5a5' : '#93c5fd';
        const valTxt = (isPos ? '+' : '') + row.feature.contribution.toFixed(2);
        const connX  = toX(row.start); // connector anchor = start of THIS bar = end of NEXT

        return (
          <g key={i}>
            {/* Dashed connector to next bar */}
            {i < rows.length - 1 && (
              <line
                x1={connX} y1={y + BAR_H / 2}
                x2={connX} y2={y + ROW_H - BAR_H / 2}
                stroke="#475569" strokeWidth="1" strokeDasharray="3,2"
              />
            )}
            {/* Connector from last bar down to base value tick */}
            {i === rows.length - 1 && (
              <line
                x1={connX} y1={y + BAR_H / 2}
                x2={connX} y2={TOTAL_H - PAD_B + 3}
                stroke="#475569" strokeWidth="1" strokeDasharray="3,2"
              />
            )}

            {/* Feature value label — right-aligned in label area */}
            <text
              x={LABEL_W - 68} y={y + 4}
              textAnchor="end" fontSize="9" fontFamily="monospace"
              fill="#64748b"
            >
              {row.feature.feature_value}
            </text>

            {/* "= feature_name" */}
            <text
              x={LABEL_W - 64} y={y + 4}
              textAnchor="start" fontSize="9.5"
              fill="#94a3b8"
            >
              {`= ${row.feature.feature_name}`}
            </text>

            {/* Background wash */}
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={2} fill={fillBg} />
            {/* Bar */}
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={2} fill={fill} opacity={0.82} />

            {/* Contribution label outside bar */}
            <text
              x={isPos ? x2 + 3 : x1 - 3}
              y={y + 4}
              textAnchor={isPos ? 'start' : 'end'}
              fontSize="9.5" fontFamily="monospace" fontWeight="bold"
              fill={txtClr}
            >
              {valTxt}
            </text>
          </g>
        );
      })}

      {/* Base value at bottom */}
      <text
        x={toX(baseValue)} y={TOTAL_H - PAD_B + 16}
        textAnchor="middle" fontSize="10" fontFamily="monospace"
        fill="#94a3b8" fontWeight="bold"
      >
        {baseValue.toFixed(3)}
      </text>
      <line
        x1={toX(baseValue)} y1={TOTAL_H - PAD_B - 3}
        x2={toX(baseValue)} y2={TOTAL_H - PAD_B + 3}
        stroke="#64748b" strokeWidth="1"
      />
    </svg>
  );
}

// ── ShapPanel ─────────────────────────────────────────────────────────────────

export function ShapPanel() {
  const { state, loadShapWaterfall } = useDashboard();
  const { shapWaterfall, fpFnMode, selectedUserId, loading, error } = state;

  // Load (or reload) whenever mode or selected user changes
  useEffect(() => {
    loadShapWaterfall(fpFnMode, selectedUserId ?? undefined);
  }, [fpFnMode, selectedUserId, loadShapWaterfall]);

  const isFp    = fpFnMode === 'fp';
  const title   = isFp ? 'FP SHAP 瀑布圖 — 為何誤判為詐騙?' : 'FN SHAP 瀑布圖 — 為何漏判詐騙?';
  const insight = isFp
    ? '共用 IP 與高風險鄰居是主要誤判原因。建議將圖結構距離納入後處理規則以降低 FP。'
    : '帳戶年齡與低交易金額掩蓋了詐騙特徵，導致模型低估風險。孤立節點結構是漏判盲點。';

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center gap-2 pb-2 border-b border-slate-700">
        <span className="w-0.5 h-4 bg-sky-500 rounded-full inline-block flex-shrink-0"></span>
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">{title}</h3>
        {selectedUserId != null ? (
          <span className="ml-auto text-[10px] text-slate-500 font-mono">User {selectedUserId}</span>
        ) : (
          <span className="ml-auto text-[10px] text-slate-600 italic">群體平均</span>
        )}
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
        {loading.shapWaterfall ? (
          <div className="flex items-center justify-center h-20"><Spinner /></div>
        ) : error.shapWaterfall ? (
          <p className="text-xs text-red-400 text-center py-4">{error.shapWaterfall}</p>
        ) : shapWaterfall ? (
          <WaterfallChart features={shapWaterfall.features} baseValue={shapWaterfall.base_value} />
        ) : null}
      </div>

      {/* Insight */}
      <div className="bg-slate-800/40 ring-1 ring-slate-700/50 rounded-lg p-2.5">
        <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">&#128161; 模型分析</p>
        <p className="text-xs text-slate-300 leading-relaxed">{insight}</p>
      </div>
    </div>
  );
}
