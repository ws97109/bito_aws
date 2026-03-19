import type { StatsResponse } from '../../types/index';

interface RelationStatsProps {
  counts: StatsResponse['relation_counts'];
}

export function RelationStats({ counts }: RelationStatsProps) {
  return (
    <div className="mt-4">
      <p className="text-sm font-medium text-gray-600 mb-2">關係類型統計</p>
      <div className="space-y-1 text-sm">
        <div className="flex justify-between"><span>R1（共用 IP）</span><span className="font-medium">{counts.r1}</span></div>
        <div className="flex justify-between"><span>R2（加密貨幣內轉）</span><span className="font-medium">{counts.r2}</span></div>
        <div className="flex justify-between"><span>R3（共用錢包）</span><span className="font-medium">{counts.r3}</span></div>
      </div>
    </div>
  );
}
