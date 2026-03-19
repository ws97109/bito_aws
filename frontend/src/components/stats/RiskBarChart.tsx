import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import type { StatsResponse } from '../../types/index';

interface RiskBarChartProps {
  data: StatsResponse['risk_distribution'];
}

export function RiskBarChart({ data }: RiskBarChartProps) {
  return (
    <div className="mt-4">
      <p className="text-sm font-medium text-gray-600 mb-2">風險分布</p>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <XAxis dataKey="range" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip />
          <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
