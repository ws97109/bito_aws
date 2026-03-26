/**
 * LLM_Model — Groq LLM-powered misclassification analysis panel.
 *
 * Reads the currently loaded SHAP waterfall data from DashboardContext and
 * sends it to the Groq API (llama-3.3-70b-versatile) with a structured
 * prompt asking an XAI expert to explain why this user was misclassified.
 *
 * Displayed at the bottom of the FP/FN SHAP panel when a specific user is
 * selected. Hidden when viewing the group average.
 */

import { useState, useRef } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import { analyzeMisclassification } from '../../api/llmApi';

// ── Simple markdown-like renderer (bold, bullet lines) ────────────────────────

function renderAnalysis(text: string) {
  const lines = text.split('\n');
  return lines.map((line, idx) => {
    // Section headings: lines starting with ## or **n.**
    if (/^#{1,3}\s/.test(line)) {
      return (
        <p key={idx} className="text-[11px] font-bold text-sky-300 mt-3 mb-0.5">
          {line.replace(/^#{1,3}\s/, '')}
        </p>
      );
    }
    // Numbered list items or bullet items
    if (/^\d+\.\s\*\*/.test(line) || /^[-•]\s/.test(line)) {
      // Extract bold label if present
      const cleaned = line.replace(/^[-•\d.]+\s/, '');
      const parts = cleaned.split(/\*\*(.+?)\*\*/g);
      return (
        <p key={idx} className="text-[11px] text-slate-300 leading-relaxed ml-2 mt-1">
          {'• '}
          {parts.map((part, pi) =>
            pi % 2 === 1 ? (
              <span key={pi} className="font-semibold text-slate-100">{part}</span>
            ) : (
              part
            ),
          )}
        </p>
      );
    }
    // Bold inline: **text**
    if (line.includes('**')) {
      const parts = line.split(/\*\*(.+?)\*\*/g);
      return (
        <p key={idx} className="text-[11px] text-slate-300 leading-relaxed mt-0.5">
          {parts.map((part, pi) =>
            pi % 2 === 1 ? (
              <span key={pi} className="font-semibold text-slate-100">{part}</span>
            ) : (
              part
            ),
          )}
        </p>
      );
    }
    // Separator
    if (/^---+$/.test(line.trim())) {
      return <hr key={idx} className="border-slate-700 my-2" />;
    }
    // Empty line → small gap
    if (!line.trim()) return <div key={idx} className="h-1" />;
    // Default paragraph
    return (
      <p key={idx} className="text-[11px] text-slate-300 leading-relaxed">
        {line}
      </p>
    );
  });
}

// ── LLM_Model component ───────────────────────────────────────────────────────

export function LLM_Model() {
  const { state } = useDashboard();
  const { shapWaterfall, fpFnMode, selectedUserId } = state;

  const [analysis, setAnalysis] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Only render when a specific user is selected and SHAP data is available
  if (selectedUserId == null || !shapWaterfall?.features?.length) return null;

  // Find the risk score for the selected user
  const nodes = fpFnMode === 'fp' ? state.fpNodes : state.fnNodes;
  const node = nodes.find(n => n.user_id === selectedUserId);
  const riskScore = node?.risk_score ?? shapWaterfall.base_value;

  const handleAnalyze = async () => {
    // Cancel any ongoing request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setAnalysis(null);

    try {
      const result = await analyzeMisclassification(
        {
          mode: fpFnMode,
          userId: selectedUserId,
          riskScore,
          features: shapWaterfall.features,
          baseValue: shapWaterfall.base_value,
        },
        abortRef.current.signal,
      );
      setAnalysis(result);
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      setError(err instanceof Error ? err.message : '分析失敗，請稍後再試');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    abortRef.current?.abort();
    setAnalysis(null);
    setError(null);
    setLoading(false);
  };

  const isFp = fpFnMode === 'fp';
  const modeTag = isFp ? 'FP 誤判' : 'FN 漏判';
  const accentColor = isFp ? 'text-orange-400' : 'text-red-400';
  const btnColor = isFp
    ? 'bg-orange-600 hover:bg-orange-500 active:bg-orange-700'
    : 'bg-red-700 hover:bg-red-600 active:bg-red-800';

  return (
    <div className="mt-3 border-t border-slate-700/60 pt-3 space-y-2">
      {/* Section header */}
      <div className="flex items-center gap-2">
        <span className="w-0.5 h-4 bg-violet-500 rounded-full inline-block flex-shrink-0" />
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
          🤖 LLM 誤判原因分析
        </h3>
        <span className={`ml-auto text-[10px] font-mono font-semibold ${accentColor}`}>
          {modeTag} · User {selectedUserId}
        </span>
      </div>

      {/* Description */}
      <p className="text-[10px] text-slate-500 leading-relaxed">
        由 Groq LLaMA-3.3-70B 結合上方 SHAP 特徵貢獻，深度解析此用戶被{isFp ? '誤判為詐騙' : '漏判詐騙'}的原因。
      </p>

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className={`px-3 py-1 rounded text-[11px] font-semibold text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${btnColor}`}
        >
          {loading ? (
            <span className="flex items-center gap-1.5">
              <svg className="animate-spin h-3 w-3 text-white" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              分析中…
            </span>
          ) : analysis ? '重新分析' : '開始 LLM 分析'}
        </button>
        {(analysis || error) && !loading && (
          <button
            onClick={handleClear}
            className="px-2 py-1 rounded text-[11px] text-slate-400 hover:text-slate-200 transition-colors"
          >
            清除
          </button>
        )}
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-950/40 ring-1 ring-red-800/50 rounded-lg p-2.5">
          <p className="text-[11px] text-red-400">{error}</p>
        </div>
      )}

      {/* Analysis result */}
      {analysis && (
        <div className="bg-slate-900/60 ring-1 ring-violet-700/30 rounded-lg p-3 space-y-0.5 max-h-96 overflow-y-auto">
          <div className="flex items-center gap-1.5 mb-2 pb-1.5 border-b border-slate-700/50">
            <span className="text-[10px] uppercase tracking-wider text-violet-400 font-semibold">
              LLaMA-3.3-70B 分析結果
            </span>
            <span className="ml-auto text-[10px] text-slate-600 font-mono">
              Groq
            </span>
          </div>
          {renderAnalysis(analysis)}
        </div>
      )}
    </div>
  );
}
