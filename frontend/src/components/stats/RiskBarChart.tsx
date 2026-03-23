import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { StatsResponse } from '../../types/index';

interface RiskBarChartProps {
  data: StatsResponse['risk_distribution'];
}

const RISK_COLORS = ['#10b981', '#84cc16', '#f59e0b', '#f97316', '#ef4444'];

export function RiskBarChart({ data }: RiskBarChartProps) {
  const chartData = data.map(item => ({ ...item, range: item.range }));
  return (
    <div>
      <h3 className="text-xs uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
        <span className="w-0.5 h-3.5 bg-sky-500 rounded-full inline-block"></span>
        風險分布
      </h3>
      <ResponsiveContainer width="100%" height={175}>
        <BarChart data={chartData} margin={{ top: 16, right: 4, bottom: 4, left: -24 }}>
          <XAxis dataKey="range" tick={{ fontSize: 10, fill: '#64748b' }} interval={0} />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} allowDecimals={false} />
          <Tooltip
            cursor={{ fill: 'rgba(100, 116, 139, 0.2)' }}
            contentStyle={{
              background: 'rgba(15, 23, 42, 0.9)',
              backdropFilter: 'blur(8px)',
              border: '1px solid #334155',
              color: '#cbd5e1',
              borderRadius: '0.5rem',
              fontSize: '12px',
              padding: '8px 12px',
            }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} label={{ position: 'top', fontSize: 9, fill: '#64748b' }}>
            {chartData.map((_, index) => (
              <Cell key={index} fill={RISK_COLORS[index % RISK_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
