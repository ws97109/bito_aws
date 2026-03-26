import { useEffect } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import type { ShapWaterfallFeature } from '../../types/index';
import { Spinner } from '../common/Spinner';
import { LLM_Model } from './LLM_Model';

// ── Waterfall chart ───────────────────────────────────────────────────────────

interface WaterfallChartProps {
  features: ShapWaterfallFeature[];
  baseValue: number;
}

function WaterfallChart({ features, baseValue }: WaterfallChartProps) {
  // Sort by |contribution| desc, cap at 10 rows
  const sorted = [...features]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 10);

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
  const xPad = (xMax - xMin) * 0.15;
  const xLo = xMin - xPad;
  const xHi = xMax + xPad;

  // Layout constants — wider label area for Chinese text
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

            {/* Feature name — left-aligned */}
            <text
              x={4} y={y + 4}
              textAnchor="start" fontSize="10"
              fill="#e2e8f0"
            >
              {row.feature.feature_name}
            </text>

            {/* Feature value — right-aligned before bar area */}
            <text
              x={LABEL_W - 6} y={y + 4}
              textAnchor="end" fontSize="9" fontFamily="monospace"
              fill="#64748b"
            >
              {row.feature.feature_value}
            </text>

            {/* Background wash */}
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fillBg} />
            {/* Bar */}
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fill} opacity={0.82} />

            {/* Contribution label outside bar */}
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

  // Dynamic insight from actual SHAP data
  const insight = (() => {
    if (!shapWaterfall?.features?.length) return '';
    const sorted = [...shapWaterfall.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    const top3 = sorted.slice(0, 3).map(f => f.feature_name);
    const posFeats = sorted.filter(f => f.contribution > 0).slice(0, 2).map(f => f.feature_name);
    const negFeats = sorted.filter(f => f.contribution < 0).slice(0, 2).map(f => f.feature_name);
    if (isFp) {
      return `主要誤判因素：${posFeats.join('、') || top3.join('、')}。${negFeats.length ? `降低風險的特徵：${negFeats.join('、')}。` : ''}建議針對高 SHAP 貢獻特徵設計後處理規則以降低 FP。`;
    }
    return `主要漏判因素：${negFeats.join('、') || top3.join('、')} 掩蓋了詐騙特徵。${posFeats.length ? `偵測到的風險信號：${posFeats.join('、')}。` : ''}建議加強對這些特徵組合的監控。`;
  })();

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

      {/* LLM misclassification analysis */}
      <LLM_Model />
    </div>
  );
}
