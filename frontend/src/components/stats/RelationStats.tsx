import type { StatsResponse } from '../../types/index';

interface RelationStatsProps {
  counts: StatsResponse['relation_counts'];
}

export function RelationStats({ counts }: RelationStatsProps) {
  const relations = [
    { key: 'R1', label: '錢包→帳戶', value: counts.r1, dotColor: 'bg-sky-500',     barColor: 'bg-sky-500/30',     textColor: 'text-sky-300' },
    { key: 'R2', label: '帳戶→帳戶', value: counts.r2, dotColor: 'bg-amber-500',   barColor: 'bg-amber-500/30',   textColor: 'text-amber-300' },
    { key: 'R3', label: '帳戶→錢包', value: counts.r3, dotColor: 'bg-emerald-500', barColor: 'bg-emerald-500/30', textColor: 'text-emerald-300' },
  ];

  const maxVal = Math.max(...relations.map(r => r.value));

  return (
    <div>
      <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
        <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
        關係類型統計
      </h3>
      <div className="space-y-2 text-sm">
        {relations.map((rel) => (
          <div key={rel.key} className="relative overflow-hidden bg-slate-700/30 rounded-md p-2.5">
            {/* Progress bar background */}
            <div
              className={`absolute inset-y-0 left-0 rounded-md transition-all ${rel.barColor}`}
              style={{ width: `${(rel.value / maxVal) * 100}%` }}
            />
            <div className="relative flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ring-2 ring-offset-1 ring-offset-slate-800 ${rel.dotColor}`}></span>
                <div>
                  <span className={`font-semibold text-xs ${rel.textColor}`}>{rel.key}</span>
                  <span className="text-slate-400 text-xs ml-1">{rel.label}</span>
                </div>
              </div>
              <span className="font-semibold text-slate-100 text-sm">{rel.value.toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
