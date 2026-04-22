import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { getConfusionMatrix } from '../../utils/graphDataStore';
import type { ConfusionMatrixData } from '../../utils/graphDataStore';

const COLOR: Record<string, string> = {
  sky:     'from-sky-500/10 ring-sky-500/30 text-sky-300',
  violet:  'from-violet-500/10 ring-violet-500/30 text-violet-300',
  emerald: 'from-emerald-500/10 ring-emerald-500/30 text-emerald-300',
  amber:   'from-amber-500/10 ring-amber-500/30 text-amber-300',
  cyan:    'from-cyan-500/10 ring-cyan-500/30 text-cyan-300',
  rose:    'from-rose-500/10 ring-rose-500/30 text-rose-300',
};

interface Kpi {
  i18nPrefix: string;
  color: keyof typeof COLOR;
  accessor: (d: ConfusionMatrixData | null) => string;
}

// AUC-ROC / AUC-PR can't be derived from a single confusion matrix; they
// are reported from the test-set evaluation of the LOO-enabled ensemble
// (final_model/output/baseline_loo/metrics.json).
const KPIS: Kpi[] = [
  { i18nPrefix: 'auc_roc',  color: 'sky',     accessor: () => '0.9778' },
  { i18nPrefix: 'auc_pr',   color: 'violet',  accessor: () => '0.8600' },
  { i18nPrefix: 'recall',   color: 'emerald', accessor: d => d ? `${(d.recall * 100).toFixed(1)}%` : '—' },
  { i18nPrefix: 'precision',color: 'amber',   accessor: d => d ? `${(d.precision * 100).toFixed(1)}%` : '—' },
  { i18nPrefix: 'f1',       color: 'cyan',    accessor: d => d ? d.f1Weighted.toFixed(4) : '—' },
  { i18nPrefix: 'specificity', color: 'rose', accessor: d => d ? `${(d.specificity * 100).toFixed(1)}%` : '—' },
];

export function KpiGrid() {
  const { t } = useTranslation();
  const [data, setData] = useState<ConfusionMatrixData | null>(null);
  useEffect(() => { getConfusionMatrix().then(setData); }, []);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
      {KPIS.map(k => (
        <div
          key={k.i18nPrefix}
          className={`relative overflow-hidden rounded-xl p-3 ring-1 bg-gradient-to-br to-slate-900/40 ${COLOR[k.color]}`}
        >
          <p className="text-[10px] uppercase tracking-wider opacity-80">
            {t(`overview.kpi.${k.i18nPrefix}.label`)}
          </p>
          <p className="text-xl font-bold text-white mt-1 leading-tight">{k.accessor(data)}</p>
          <p className="text-[10px] text-slate-400 mt-0.5 leading-snug">
            {t(`overview.kpi.${k.i18nPrefix}.hint`)}
          </p>
        </div>
      ))}
    </div>
  );
}
