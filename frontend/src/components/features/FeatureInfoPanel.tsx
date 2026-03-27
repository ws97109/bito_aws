import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { getShapTop20AllUsers, getShapTop20Blacklist, getShapTop10FP, getShapTop10FN } from '../../utils/graphDataStore';
import type { ShapTop20Entry } from '../../utils/graphDataStore';

// ── 特徵類別色彩對應 ──────────────────────────────────────────────────────────

const FEATURE_COLOR: Record<string, string> = {
  swap_sum: '#6366f1', swap_count: '#6366f1',
  tx_interval_median: '#06b6d4', tx_interval_mean: '#06b6d4', tx_interval_min: '#06b6d4', tx_interval_std: '#06b6d4',
  account_age_days: '#3b82f6',
  crypto_wit_sum: '#eab308', crypto_wit_max: '#eab308', crypto_wit_mean: '#eab308', crypto_wit_count: '#eab308',
  crypto_dep_sum: '#f59e0b', crypto_dep_count: '#f59e0b', crypto_dep_mean: '#f59e0b', crypto_dep_max: '#f59e0b',
  twd_dep_sum: '#10b981', twd_dep_count: '#10b981', twd_net_flow: '#10b981',
  ip_night_ratio: '#f97316', ip_unique_count: '#f97316',
  career_freq: '#a855f7',
  weekend_tx_ratio: '#14b8a6',
  reg_hour: '#3b82f6',
  kyc_speed_sec: '#3b82f6',
  is_app_user: '#3b82f6',
  trading_market_order_ratio: '#8b5cf6',
  total_tx_count: '#14b8a6', first_to_last_tx_days: '#14b8a6', velocity_ratio_7d_vs_30d: '#14b8a6', composite_risk_score: '#14b8a6',
  fund_stay_sec: '#06b6d4',
  dep_to_first_wit_hours: '#ef4444', twd_to_crypto_out_ratio: '#ef4444', tx_amount_cv: '#ef4444',
  rapid_kyc_then_trade: '#ef4444', crypto_out_in_ratio: '#ef4444', same_day_in_out_count: '#ef4444',
  if_score: '#f59e0b', hbos_score: '#f59e0b', lof_score: '#f59e0b',
};

const DEFAULT_COLOR = '#64748b';

function barColor(feature: string): string {
  return FEATURE_COLOR[feature] ?? DEFAULT_COLOR;
}

// ── 靜態特徵類別說明 ──────────────────────────────────────────────────────────

const FEATURE_CATEGORIES = [
  { icon:'👤', name:'用戶人口特徵',  accent:'sky',     count:15, description:'KYC 異常（秒級完成暗示自動化）、深夜註冊、性別/職業風險。代表特徵：kyc_speed_sec, account_age_days, is_female, reg_hour, reg_is_night' },
  { icon:'💵', name:'法幣交易行為',  accent:'emerald', count:14, description:'淨流出、入金/提領比異常、Smurfing 偵測。代表特徵：twd_dep_sum, twd_net_flow, twd_withdraw_ratio, twd_smurf_flag' },
  { icon:'🪙', name:'虛幣交易行為', accent:'yellow',  count:15, description:'多錢包分散提領、跨鏈移轉。代表特徵：crypto_wit_sum, crypto_wallet_hash_nunique, crypto_protocol_diversity' },
  { icon:'📊', name:'掛單/一鍵買賣', accent:'violet',  count:9,  description:'單向購買、市價單洗量。代表特徵：trading_buy_ratio, swap_sum, trading_market_order_ratio, total_trading_volume' },
  { icon:'🌐', name:'IP & 資金流速', accent:'orange',  count:5,  description:'多 IP 切換、深夜操作、資金快進快出。代表特徵：ip_unique_count, ip_night_ratio, ip_max_shared, fund_stay_sec' },
  { icon:'🕸️', name:'交易圖拓撲',   accent:'pink',    count:5,  description:'資金樞紐、詐騙集團聚集。代表特徵：pagerank_score, connected_component_size, betweenness_centrality' },
  { icon:'🔗', name:'跨表衍生',      accent:'teal',    count:4,  description:'近期活動加速。代表特徵：total_tx_count, weekend_tx_ratio, velocity_ratio_7d_vs_30d' },
  { icon:'🚩', name:'AML 紅旗指標',  accent:'red',     count:6,  description:'法幣入→幣出漏斗、Smurfing 拆單。代表特徵：twd_to_crypto_out_ratio, same_day_in_out_count, tx_amount_cv' },
  { icon:'⏱️', name:'時序模式',      accent:'cyan',    count:7,  description:'規律性操作偵測、爆發交易。代表特徵：tx_interval_mean, tx_interval_min, amount_p90_p10_ratio, active_days' },
  { icon:'🎯', name:'複合風險分數',  accent:'rose',    count:1,  description:'多維度風險加權綜合。代表特徵：composite_risk_score' },
  { icon:'🔬', name:'異常偵測分數',  accent:'amber',   count:3,  description:'Isolation Forest、HBOS、LOF 三種無監督方法產生的異常評分（後續步驟加入）' },
  { icon:'🧠', name:'GNN 嵌入',      accent:'indigo',  count:16, description:'HeteroSAGE + GAT 在用戶–錢包異構圖上學習的節點表示，經 PCA 降維至 16 維（後續步驟加入）' },
];

const ACCENT: Record<string, { border:string; bg:string; badge:string; title:string }> = {
  sky:     { border:'border-sky-500/30',     bg:'bg-sky-500/5',     badge:'bg-sky-500/15 text-sky-300',     title:'text-sky-300' },
  emerald: { border:'border-emerald-500/30', bg:'bg-emerald-500/5', badge:'bg-emerald-500/15 text-emerald-300', title:'text-emerald-300' },
  yellow:  { border:'border-yellow-500/30',  bg:'bg-yellow-500/5',  badge:'bg-yellow-500/15 text-yellow-300',  title:'text-yellow-300' },
  violet:  { border:'border-violet-500/30',  bg:'bg-violet-500/5',  badge:'bg-violet-500/15 text-violet-300',  title:'text-violet-300' },
  orange:  { border:'border-orange-500/30',  bg:'bg-orange-500/5',  badge:'bg-orange-500/15 text-orange-300',  title:'text-orange-300' },
  pink:    { border:'border-pink-500/30',    bg:'bg-pink-500/5',    badge:'bg-pink-500/15 text-pink-300',    title:'text-pink-300' },
  cyan:    { border:'border-cyan-500/30',    bg:'bg-cyan-500/5',    badge:'bg-cyan-500/15 text-cyan-300',    title:'text-cyan-300' },
  red:     { border:'border-red-500/30',     bg:'bg-red-500/5',     badge:'bg-red-500/15 text-red-300',     title:'text-red-300' },
  teal:    { border:'border-teal-500/30',    bg:'bg-teal-500/5',    badge:'bg-teal-500/15 text-teal-300',    title:'text-teal-300' },
  amber:   { border:'border-amber-500/30',   bg:'bg-amber-500/5',   badge:'bg-amber-500/15 text-amber-300',   title:'text-amber-300' },
  indigo:  { border:'border-indigo-500/30',  bg:'bg-indigo-500/5',  badge:'bg-indigo-500/15 text-indigo-300',  title:'text-indigo-300' },
  slate:   { border:'border-slate-500/30',   bg:'bg-slate-500/5',   badge:'bg-slate-500/15 text-slate-300',   title:'text-slate-300' },
  rose:    { border:'border-rose-500/30',    bg:'bg-rose-500/5',    badge:'bg-rose-500/15 text-rose-300',    title:'text-rose-300' },
};

// ── 自定義 Tooltip ────────────────────────────────────────────────────────────

function ShapTooltip({ active, payload }: { active?: boolean; payload?: { payload: ShapTop20Entry }[] }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
      <div className="font-semibold text-white mb-1">#{d.rank} {d.label}</div>
      <div className="text-slate-400 font-mono mb-2">{d.feature}</div>
      <div className="space-y-1">
        <div className="flex justify-between gap-4">
          <span className="text-slate-400">平均 |SHAP|</span>
          <span className="text-sky-300 font-semibold">{d.shap.toFixed(4)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-slate-400">佔總重要性</span>
          <span className="text-purple-300">{d.pct}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-slate-400">出現頻率</span>
          <span className="text-emerald-300">{d.freq.toLocaleString()} 人</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-slate-400">累積佔比</span>
          <span className="text-amber-300">{d.cumPct}</span>
        </div>
      </div>
    </div>
  );
}

// ── 單一 SHAP 長條圖 ──────────────────────────────────────────────────────────

function ShapBarChart({ data, title, subtitle, color }: {
  data: ShapTop20Entry[];
  title: string;
  subtitle: string;
  color: string;
}) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-32 text-xs text-slate-500">
        <span className="w-3 h-3 rounded-full animate-pulse mr-2" style={{ background: color }} />
        載入中...
      </div>
    );
  }

  // 直接使用原始順序：rank1(最重要)在陣列最前 → Recharts 由上往下渲染 → 多到少
  const chartData = [...data];

  return (
    <div>
      <div className="mb-3">
        <div className="text-sm font-semibold text-slate-200">{title}</div>
        <div className="text-xs text-slate-500 mt-0.5">{subtitle}</div>
      </div>
      <ResponsiveContainer width="100%" height={520}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 60, bottom: 4, left: 130 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickFormatter={v => v.toFixed(3)}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: '#cbd5e1', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={125}
          />
          <Tooltip content={<ShapTooltip />} cursor={{ fill: 'rgba(148,163,184,0.06)' }} />
          <ReferenceLine x={0} stroke="#475569" />
          <Bar dataKey="shap" radius={[0, 3, 3, 0]} maxBarSize={18}>
            {chartData.map((entry) => (
              <Cell key={entry.feature} fill={barColor(entry.feature)} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* 圖例：累積佔比 */}
      <div className="mt-3 flex items-center gap-1.5 text-xs text-slate-500">
        <span>Top 20 累積佔比：</span>
        <span className="text-slate-300 font-semibold">{data[data.length - 1]?.cumPct ?? '—'}</span>
      </div>
    </div>
  );
}

// ── 對比視圖 ─────────────────────────────────────────────────────────────────

function CompareChart({ allData, blacklistData }: { allData: ShapTop20Entry[]; blacklistData: ShapTop20Entry[] }) {
  if (!allData.length || !blacklistData.length) {
    return <div className="text-xs text-slate-500 text-center py-8">載入中...</div>;
  }

  // 合併兩組的 feature 集合（union of top 20）
  const featureSet = new Set([...allData.map(d => d.feature), ...blacklistData.map(d => d.feature)]);
  const allMap = new Map(allData.map(d => [d.feature, d]));
  const blMap  = new Map(blacklistData.map(d => [d.feature, d]));

  const merged = Array.from(featureSet).map(f => {
    const a = allMap.get(f);
    const b = blMap.get(f);
    return {
      feature: f,
      label: a?.label ?? b?.label ?? f,
      all: a?.shap ?? 0,
      blacklist: b?.shap ?? 0,
      diff: (b?.shap ?? 0) - (a?.shap ?? 0),
    };
  }).sort((x, y) => Math.max(y.all, y.blacklist) - Math.max(x.all, x.blacklist));

  const chartData = [...merged]; // 已降冪排列，最重要在頂端

  return (
    <div>
      <div className="mb-3">
        <div className="text-sm font-semibold text-slate-200">整體 vs 黑名單 SHAP 重要性對比</div>
        <div className="text-xs text-slate-500 mt-0.5">同一特徵在全體用戶（藍）與黑名單用戶（紅）的平均 |SHAP| 值差異</div>
      </div>

      {/* 圖例 */}
      <div className="flex gap-4 mb-3 text-xs">
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-sky-500 inline-block" />整體用戶</div>
        <div className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-red-500 inline-block" />黑名單用戶</div>
      </div>

      <ResponsiveContainer width="100%" height={620}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 60, bottom: 4, left: 130 }}
          barGap={2}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: '#94a3b8', fontSize: 10 }}
            tickFormatter={v => v.toFixed(3)}
            axisLine={{ stroke: '#475569' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: '#cbd5e1', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={125}
          />
          <Tooltip
            cursor={{ fill: 'rgba(148,163,184,0.06)' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as typeof merged[0];
              return (
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs shadow-xl">
                  <div className="font-semibold text-white mb-2">{d.label}</div>
                  <div className="space-y-1">
                    <div className="flex justify-between gap-4"><span className="text-sky-400">整體用戶</span><span className="font-mono text-white">{d.all.toFixed(4)}</span></div>
                    <div className="flex justify-between gap-4"><span className="text-red-400">黑名單</span><span className="font-mono text-white">{d.blacklist.toFixed(4)}</span></div>
                    <div className={`flex justify-between gap-4 border-t border-slate-700 pt-1 ${d.diff > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                      <span>差異</span><span className="font-mono">{d.diff > 0 ? '+' : ''}{d.diff.toFixed(4)}</span>
                    </div>
                  </div>
                </div>
              );
            }}
          />
          <Bar dataKey="all"       name="整體用戶" fill="#38bdf8" fillOpacity={0.75} radius={[0,2,2,0]} maxBarSize={10} />
          <Bar dataKey="blacklist" name="黑名單"   fill="#f87171" fillOpacity={0.85} radius={[0,2,2,0]} maxBarSize={10} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── 主元件 ────────────────────────────────────────────────────────────────────

type Tab = 'all' | 'blacklist' | 'fp' | 'fn' | 'compare' | 'categories';

export function FeatureInfoPanel() {
  const [allData,       setAllData]       = useState<ShapTop20Entry[]>([]);
  const [blacklistData, setBlacklistData] = useState<ShapTop20Entry[]>([]);
  const [fpData,        setFpData]        = useState<ShapTop20Entry[]>([]);
  const [fnData,        setFnData]        = useState<ShapTop20Entry[]>([]);
  const [loading,       setLoading]       = useState(true);
  const [activeTab,     setActiveTab]     = useState<Tab>('all');

  useEffect(() => {
    Promise.all([
      getShapTop20AllUsers(),
      getShapTop20Blacklist(),
      getShapTop10FP(),
      getShapTop10FN(),
    ]).then(([a, b, fp, fn]) => {
      setAllData(a);
      setBlacklistData(b);
      setFpData(fp);
      setFnData(fn);
      setLoading(false);
    });
  }, []);

  const TABS: { id: Tab; label: string }[] = [
    { id: 'all',        label: '📈 整體重要性' },
    { id: 'blacklist',  label: '🚫 黑名單重要性' },
    { id: 'fp',         label: '⚠️ FP 誤判特徵' },
    { id: 'fn',         label: '🔍 FN 漏判特徵' },
    { id: 'compare',    label: '🔄 對比分析' },
    { id: 'categories', label: '🗂️ 特徵類別' },
  ];

  return (
    <main className="flex-1 overflow-y-auto min-w-0 min-h-0">
      <div className="flex flex-col gap-4">

        {/* 標頭 */}
        <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-purple-500/30 rounded-xl shadow-2xl p-5">
          <div className="flex items-start gap-4">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-purple-500/20 border border-purple-500/40 text-2xl flex-shrink-0">📋</div>
            <div className="flex-1 min-w-0">
              <h2 className="text-base font-bold text-purple-300 mb-1">模型特徵說明</h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                特徵工程產出 <span className="text-white font-semibold">81 個基礎特徵</span>，
                經零方差移除（−1）、高相關去重 &gt;0.95（−13）、LightGBM 零重要性篩選（−2）後剩 <span className="text-white font-semibold">65 個</span>，
                再加入異常偵測分數（+3）與 GNN 嵌入（+16），移除受保護屬性 is_female、age（−2），
                最終以 <span className="text-purple-300 font-semibold">82 維特徵</span> 輸入 XGBoost / LightGBM / CatBoost Stacking 集成。
              </p>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-4 gap-2">
            {[
              { label:'特徵類別',   value:10,  unit:'大類' },
              { label:'基礎特徵數', value:81,  unit:'維' },
              { label:'篩選後',     value:65,  unit:'維' },
              { label:'最終輸入',   value:82,  unit:'維' },
            ].map(({ label, value, unit }) => (
              <div key={label} className="bg-slate-900/60 rounded-lg p-3 text-center">
                <div className="text-lg font-bold text-white">{value}<span className="text-xs text-slate-400 ml-0.5">{unit}</span></div>
                <div className="text-xs text-slate-500 mt-0.5">{label}</div>
              </div>
            ))}
          </div>

          {/* Tab 切換 */}
          <div className="mt-4 flex gap-2 flex-wrap">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors
                  ${activeTab === tab.id
                    ? 'bg-purple-500/20 text-purple-300 ring-1 ring-purple-500/40'
                    : 'bg-slate-700/40 text-slate-400 hover:text-slate-300'}`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab 內容 */}
        <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl p-5">
          {loading ? (
            <div className="flex items-center justify-center gap-2 text-xs text-slate-400 py-16">
              <span className="w-3 h-3 rounded-full bg-purple-400 animate-pulse" />
              載入 SHAP 資料中...
            </div>
          ) : (
            <>
              {activeTab === 'all' && (
                <ShapBarChart
                  data={allData}
                  title="Top 20 特徵重要性（全體用戶）"
                  subtitle="來源：shap_top20_all_users.csv — 所有用戶的平均 |SHAP| 值排名"
                  color="#38bdf8"
                />
              )}
              {activeTab === 'blacklist' && (
                <ShapBarChart
                  data={blacklistData}
                  title="Top 20 特徵重要性（黑名單用戶）"
                  subtitle="來源：shap_top20_blacklist.csv — 僅黑名單用戶的平均 |SHAP| 值排名"
                  color="#f87171"
                />
              )}
              {activeTab === 'fp' && (
                <ShapBarChart
                  data={fpData.slice(0, 3)}
                  title="Top 3 特徵重要性（FP：白預測成黑）"
                  subtitle="來源：shap_values_fp.csv — 被誤判為黑名單的白名單用戶，平均 |SHAP| 值前三名"
                  color="#fb923c"
                />
              )}
              {activeTab === 'fn' && (
                <ShapBarChart
                  data={fnData.slice(0, 3)}
                  title="Top 3 特徵重要性（FN：黑預測成白）"
                  subtitle="來源：shap_values_fn.csv — 被漏判為白名單的黑名單用戶，平均 |SHAP| 值前三名"
                  color="#a78bfa"
                />
              )}
              {activeTab === 'compare' && (
                <CompareChart allData={allData} blacklistData={blacklistData} />
              )}
              {activeTab === 'categories' && (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  {FEATURE_CATEGORIES.map(cat => {
                    const c = ACCENT[cat.accent];
                    return (
                      <div key={cat.name} className={`ring-1 ${c.border} rounded-xl p-4 flex items-start gap-3 ${c.bg}`}>
                        <div className={`flex items-center justify-center w-9 h-9 rounded-lg border ${c.border} text-lg flex-shrink-0`}>{cat.icon}</div>
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className={`text-sm font-semibold ${c.title}`}>{cat.name}</span>
                            <span className={`text-xs px-1.5 py-0.5 rounded-full ${c.badge}`}>{cat.count} 個特徵</span>
                          </div>
                          <p className="text-xs text-slate-500 leading-snug">{cat.description}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>

        {/* 模型訓練流程 */}
        <div className="bg-slate-800/50 backdrop-blur-sm ring-1 ring-slate-700/60 rounded-xl shadow-2xl p-5 mb-2">
          <h3 className="text-sm font-semibold text-slate-300 mb-3">模型訓練流程</h3>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {[
              { label:'特徵工程 (81)',    cls:'bg-sky-500/20 text-sky-300 border-sky-500/30' },
              { label:'篩選 81→65',      cls:'bg-purple-500/20 text-purple-300 border-purple-500/30' },
              { label:'異常偵測 (+3)',    cls:'bg-amber-500/20 text-amber-300 border-amber-500/30' },
              { label:'GNN 嵌入 (+16)',   cls:'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' },
              { label:'移除受保護屬性 (−2)', cls:'bg-red-500/20 text-red-300 border-red-500/30' },
              { label:'最終 82 維',       cls:'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' },
              { label:'Stacking 集成',   cls:'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' },
              { label:'SHAP 解釋',       cls:'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' },
            ].reduce<React.ReactNode[]>((acc, step, i) => {
              if (i > 0) acc.push(<span key={`arr-${i}`} className="text-slate-500">→</span>);
              acc.push(<span key={step.label} className={`px-2 py-1 rounded-md border font-medium ${step.cls}`}>{step.label}</span>);
              return acc;
            }, [])}
          </div>
        </div>

      </div>
    </main>
  );
}
