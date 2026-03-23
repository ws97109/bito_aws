interface StatCardProps {
  title: string;
  value: string | number;
  accentColor?: 'sky' | 'red' | 'emerald' | 'orange';
  icon?: string;
}

const accentMap = {
  sky:     { border: 'border-t-sky-500',     text: 'text-sky-400',     bg: 'bg-sky-500/10' },
  red:     { border: 'border-t-red-500',     text: 'text-red-400',     bg: 'bg-red-500/10' },
  emerald: { border: 'border-t-emerald-500', text: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  orange:  { border: 'border-t-orange-500',  text: 'text-orange-400',  bg: 'bg-orange-500/10' },
};

export function StatCard({ title, value, accentColor = 'sky', icon }: StatCardProps) {
  const accent = accentMap[accentColor];
  return (
    <div className={`rounded-lg p-3 border-t-2 ${accent.border} bg-slate-700/40 ring-1 ring-slate-600/50`}>
      <div className="flex items-center gap-1.5 mb-2">
        {icon && <span className={`text-sm ${accent.text}`}>{icon}</span>}
        <p className={`text-xs uppercase tracking-wider font-medium ${accent.text}`}>{title}</p>
      </div>
      <p className="text-2xl font-bold text-slate-100">{value}</p>
    </div>
  );
}
