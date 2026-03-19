import { describe, it, expect } from 'vitest';
import { getNodeColor, getLinkDash } from '../components/graph/GraphViewer';
import { getShapColor } from '../components/graph/NodeDetailPanel';
import { getFilteredNodes } from '../components/graph/NodeSelector';
import type { SubgraphNode, SubgraphEdge, FraudNode } from '../types/index';

// ── getNodeColor ──────────────────────────────────────────────────────────────

describe('getNodeColor', () => {
  it('returns red for status=1 nodes', () => {
    const node: SubgraphNode = { user_id: 1, risk_score: 0.9, status: 1 };
    expect(getNodeColor(node)).toBe('#ef4444');
  });

  it('returns red for status=1 even when risk_score < 0.5', () => {
    const node: SubgraphNode = { user_id: 2, risk_score: 0.1, status: 1 };
    expect(getNodeColor(node)).toBe('#ef4444');
  });

  it('returns orange for status=0 with risk_score >= 0.5', () => {
    const node: SubgraphNode = { user_id: 3, risk_score: 0.5, status: 0 };
    expect(getNodeColor(node)).toBe('#f97316');
  });

  it('returns orange for status=0 with risk_score = 0.99', () => {
    const node: SubgraphNode = { user_id: 4, risk_score: 0.99, status: 0 };
    expect(getNodeColor(node)).toBe('#f97316');
  });

  it('returns blue for status=0 with risk_score < 0.5', () => {
    const node: SubgraphNode = { user_id: 5, risk_score: 0.49, status: 0 };
    expect(getNodeColor(node)).toBe('#3b82f6');
  });

  it('returns blue for status=0 with risk_score = 0', () => {
    const node: SubgraphNode = { user_id: 6, risk_score: 0, status: 0 };
    expect(getNodeColor(node)).toBe('#3b82f6');
  });
});

// ── getLinkDash ───────────────────────────────────────────────────────────────

describe('getLinkDash', () => {
  it('returns [] (solid) for R2', () => {
    const edge: SubgraphEdge = { source: 1, target: 2, relation_type: 'R2' };
    expect(getLinkDash(edge)).toEqual([]);
  });

  it('returns [4, 2] (dashed) for R1', () => {
    const edge: SubgraphEdge = { source: 1, target: 2, relation_type: 'R1' };
    expect(getLinkDash(edge)).toEqual([4, 2]);
  });

  it('returns [1, 2] (dotted) for R3', () => {
    const edge: SubgraphEdge = { source: 1, target: 2, relation_type: 'R3' };
    expect(getLinkDash(edge)).toEqual([1, 2]);
  });
});

// ── getShapColor ──────────────────────────────────────────────────────────────

describe('getShapColor', () => {
  it('returns red class for positive contribution', () => {
    expect(getShapColor(0.5)).toBe('text-red-600');
  });

  it('returns green class for negative contribution', () => {
    expect(getShapColor(-0.3)).toBe('text-green-600');
  });

  it('returns gray class for zero contribution', () => {
    expect(getShapColor(0)).toBe('text-gray-500');
  });
});

// ── getFilteredNodes ──────────────────────────────────────────────────────────

describe('getFilteredNodes', () => {
  const nodes: FraudNode[] = [
    { user_id: 1042, risk_score: 0.987 },
    { user_id: 1187, risk_score: 0.5 },
    { user_id: 2001, risk_score: 0.49 },  // below threshold
    { user_id: 3000, risk_score: 0.8 },
  ];

  it('filters out nodes with risk_score < 0.5', () => {
    const result = getFilteredNodes(nodes, '');
    expect(result.find(n => n.user_id === 2001)).toBeUndefined();
  });

  it('includes nodes with risk_score >= 0.5', () => {
    const result = getFilteredNodes(nodes, '');
    expect(result.map(n => n.user_id)).toContain(1042);
    expect(result.map(n => n.user_id)).toContain(1187);
    expect(result.map(n => n.user_id)).toContain(3000);
  });

  it('filters by keyword matching user_id', () => {
    const result = getFilteredNodes(nodes, '104');
    expect(result).toHaveLength(1);
    expect(result[0].user_id).toBe(1042);
  });

  it('returns empty array when no nodes match keyword', () => {
    const result = getFilteredNodes(nodes, '9999');
    expect(result).toHaveLength(0);
  });

  it('trims whitespace from keyword', () => {
    const result = getFilteredNodes(nodes, '  1042  ');
    expect(result).toHaveLength(1);
    expect(result[0].user_id).toBe(1042);
  });

  it('returns all qualifying nodes when keyword is empty', () => {
    const result = getFilteredNodes(nodes, '');
    expect(result).toHaveLength(3); // 2001 excluded (risk_score < 0.5)
  });
});
