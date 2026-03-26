import { useState } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import type { ShapFeature, NeighborPeer } from '../../types/index';

// ── Waterfall chart ───────────────────────────────────────────────────────────

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

  const ROW_H    = 26;
  const BAR_H    = 14;
  const LABEL_W  = 200;
  const TOTAL_W  = 600;
  const BAR_AREA = TOTAL_W - LABEL_W - 6;
  const PAD_T    = 8;
  const PAD_B    = 8;
  const TOTAL_H  = rows.length * ROW_H + PAD_T + PAD_B;

  const toX = (v: number) => LABEL_W + ((v - xLo) / (xHi - xLo)) * BAR_AREA;

  return (
    <svg viewBox={`0 0 ${TOTAL_W} ${TOTAL_H}`} width="100%" style={{ display: 'block' }}>
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
              <line x1={connX} y1={y + BAR_H / 2} x2={connX} y2={y + ROW_H - BAR_H / 2}
                stroke="#475569" strokeWidth="1" strokeDasharray="3,2" />
            )}
            <text x={4} y={y + 4} textAnchor="start" fontSize="10" fill="#e2e8f0">
              {row.feature.feature_name}
            </text>
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fillBg} />
            <rect x={x1} y={y - BAR_H / 2} width={barW} height={BAR_H} rx={3} fill={fill} opacity={0.82} />
            <text x={isPos ? x2 + 4 : x1 - 4} y={y + 4}
              textAnchor={isPos ? 'start' : 'end'}
              fontSize="10" fontFamily="monospace" fontWeight="bold" fill={txtClr}>
              {valTxt}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── NeighborList ──────────────────────────────────────────────────────────────

interface NeighborListProps {
  peers: NeighborPeer[];
  label: string;
  color: string; // tailwind text color
}

function NeighborList({ peers, label, color }: NeighborListProps) {
  return (
    <div className="mt-1.5">
      <p className={`text-[10px] uppercase tracking-wider font-semibold mb-1 ${color}`}>{label}</p>
      <div className="space-y-1">
        {peers.map(p => {
          const riskClr = p.risk_score >= 0.8743 ? 'text-red-400'
            : p.risk_score >= 0.65 ? 'text-orange-400'
            : p.risk_score >= 0.45 ? 'text-yellow-400'
            : 'text-emerald-400';
          const label = p.node_type === 'wallet'
            ? (p.node_label ?? `wallet_${p.peer_id}`)
            : `User ${p.peer_id}`;
          return (
            <div key={p.peer_id}
              className="flex items-center justify-between px-2.5 py-1.5 rounded-md bg-slate-800/60 ring-1 ring-slate-600/40">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className={`text-[10px] flex-shrink-0 ${p.node_type === 'wallet' ? 'text-violet-400' : 'text-sky-400'}`}>
                  {p.node_type === 'wallet' ? '◆' : '●'}
                </span>
                <span className="text-xs text-slate-300 truncate" title={label}>{label}</span>
                {p.status === 1 && (
                  <span className="text-[9px] bg-red-900/60 text-red-300 px-1 rounded flex-shrink-0">詐騙</span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                <span className={`font-mono text-xs font-semibold ${riskClr}`}>
                  {(p.risk_score * 100).toFixed(0)}%
                </span>
                <span className="text-[10px] text-slate-500">×{p.tx_count}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── NeighborSection (expandable button) ───────────────────────────────────────

interface NeighborSectionProps {
  label: string;
  count: number;
  activeColor: string;
  textColor: string;
  children: React.ReactNode;
}

function NeighborSection({ label, count, activeColor, textColor, children }: NeighborSectionProps) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen(o => !o)}
        className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg text-xs font-semibold ring-1 transition-colors
          ${open ? `${activeColor} ${textColor}` : `bg-slate-700/30 ring-slate-600/40 text-slate-400 hover:text-slate-200`}`}
      >
        <span>{label} · {count}</span>
        <span className="text-[10px]">{open ? '▲' : '▼'}</span>
      </button>
      {open && count > 0 && (
        <div className="mt-1 px-1">
          {children}
        </div>
      )}
      {open && count === 0 && (
        <p className="text-[10px] text-slate-500 text-center py-2">無連接節點</p>
      )}
    </div>
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

  const { user_id, risk_score, status, shap_features, neighbor_counts, neighbor_details } = selectedNode;
  const isFraud = status === 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className="text-slate-400">&#9654;</span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">節點詳細資訊</h3>
        </div>
        <span className={`px-2.5 py-0.5 text-xs font-semibold rounded-full ${isFraud ? 'bg-red-900/60 text-red-300 ring-1 ring-red-500/50' : 'bg-emerald-900/60 text-emerald-300 ring-1 ring-emerald-500/50'}`}>
          {isFraud ? '⚠ 詐騙' : '✓ 正常'}
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
        <div className="bg-slate-900/40 rounded-lg px-2 py-1.5 ring-1 ring-slate-700/40">
          {shap_features.length === 0 ? (
            <p className="text-xs text-slate-400 text-center py-4">SHAP 資料不可用</p>
          ) : (
            <NodeWaterfall features={shap_features} baseValue={BASE_VALUE} />
          )}
        </div>
      </div>

      {/* Neighbor relation buttons */}
      <div className="space-y-2">
        <h4 className="text-xs uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
          鄰居關係
        </h4>

        {/* R1: 錢包→帳戶 */}
        <NeighborSection
          label="錢包→帳戶" count={neighbor_counts.r1}
          activeColor="bg-sky-500/20" textColor="text-sky-300"
        >
          <NeighborList peers={neighbor_details.r1} label="來源錢包" color="text-sky-400" />
        </NeighborSection>

        {/* R2: 帳戶→帳戶 */}
        <NeighborSection
          label="帳戶→帳戶" count={neighbor_counts.r2}
          activeColor="bg-amber-500/20" textColor="text-amber-300"
        >
          {neighbor_details.r2_out.length > 0 && (
            <NeighborList peers={neighbor_details.r2_out} label="轉出對象"
              color="text-amber-400" />
          )}
          {neighbor_details.r2_in.length > 0 && (
            <NeighborList peers={neighbor_details.r2_in} label="轉入來源"
              color="text-amber-300" />
          )}
        </NeighborSection>

        {/* R3: 帳戶→錢包 */}
        <NeighborSection
          label="帳戶→錢包" count={neighbor_counts.r3}
          activeColor="bg-emerald-500/20" textColor="text-emerald-300"
        >
          <NeighborList peers={neighbor_details.r3} label="目標錢包"
            color="text-emerald-400" />
        </NeighborSection>
      </div>

    </div>
  );
}
