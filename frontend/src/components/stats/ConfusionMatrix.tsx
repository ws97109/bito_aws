import { useEffect, useState } from 'react';
import { getConfusionMatrix } from '../../utils/graphDataStore';
import type { ConfusionMatrixData } from '../../utils/graphDataStore';

function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }
function fmt(v: number) { return v.toLocaleString(); }

export function ConfusionMatrix() {
  const [data, setData] = useState<ConfusionMatrixData | null>(null);

  useEffect(() => { getConfusionMatrix().then(setData); }, []);

  if (!data) {
    return (
      <div className="flex items-center gap-2 py-4 justify-center text-xs text-slate-500">
        <span className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse" />
        計算混淆矩陣中...
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
        <span className="w-0.5 h-3.5 bg-purple-500 rounded-full inline-block" />
        混淆矩陣摘要
        <span className="ml-auto text-[10px] text-slate-500 normal-case tracking-normal">
          閾值 {data.threshold}
        </span>
      </h3>

      {/* 2×2 矩陣 */}
      <div className="mb-3">
        {/* 表頭：預測 */}
        <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-[10px] text-center mb-1">
          <div />
          <div className="text-red-400 font-semibold py-0.5">預測：詐騙</div>
          <div className="text-emerald-400 font-semibold py-0.5">預測：正常</div>
        </div>

        {/* TP / FN row */}
        <div className="grid grid-cols-[auto_1fr_1fr] gap-1 mb-1">
          <div className="flex items-center justify-end pr-1.5 text-[10px] text-red-400 font-semibold whitespace-nowrap">
            實際：詐騙
          </div>
          {/* TP */}
          <div className="bg-emerald-500/15 ring-1 ring-emerald-500/40 rounded-lg p-2 text-center">
            <div className="text-xs text-emerald-400 font-bold mb-0.5">TP</div>
            <div className="text-base font-bold text-white">{fmt(data.tp)}</div>
            <div className="text-[10px] text-emerald-400/70">正確偵測</div>
          </div>
          {/* FN */}
          <div className="bg-orange-500/10 ring-1 ring-orange-500/30 rounded-lg p-2 text-center">
            <div className="text-xs text-orange-400 font-bold mb-0.5">FN</div>
            <div className="text-base font-bold text-white">{fmt(data.fn)}</div>
            <div className="text-[10px] text-orange-400/70">漏報詐騙</div>
          </div>
        </div>

        {/* FP / TN row */}
        <div className="grid grid-cols-[auto_1fr_1fr] gap-1">
          <div className="flex items-center justify-end pr-1.5 text-[10px] text-emerald-400 font-semibold whitespace-nowrap">
            實際：正常
          </div>
          {/* FP */}
          <div className="bg-red-500/10 ring-1 ring-red-500/30 rounded-lg p-2 text-center">
            <div className="text-xs text-red-400 font-bold mb-0.5">FP</div>
            <div className="text-base font-bold text-white">{fmt(data.fp)}</div>
            <div className="text-[10px] text-red-400/70">誤報為詐騙</div>
          </div>
          {/* TN */}
          <div className="bg-slate-700/30 ring-1 ring-slate-600/40 rounded-lg p-2 text-center">
            <div className="text-xs text-slate-400 font-bold mb-0.5">TN</div>
            <div className="text-base font-bold text-white">{fmt(data.tn)}</div>
            <div className="text-[10px] text-slate-400/70">正確放行</div>
          </div>
        </div>
      </div>

      {/* 指標列 */}
      <div className="grid grid-cols-2 gap-1.5">
        {[
          { label: '準確率 Accuracy',    value: pct(data.accuracy),    color: 'text-sky-400' },
          { label: '精確率 Precision',   value: pct(data.precision),   color: 'text-violet-400' },
          { label: '召回率 Recall',      value: pct(data.recall),      color: 'text-emerald-400' },
          { label: 'F1 Score',           value: pct(data.f1),          color: 'text-yellow-400' },
          { label: '特異度 Specificity', value: pct(data.specificity), color: 'text-cyan-400' },
          { label: '總樣本數',           value: fmt(data.total),       color: 'text-slate-300' },
        ].map(m => (
          <div key={m.label} className="bg-slate-900/50 rounded-md px-2 py-1.5 flex justify-between items-center">
            <span className="text-[10px] text-slate-500">{m.label}</span>
            <span className={`text-xs font-bold ${m.color}`}>{m.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
