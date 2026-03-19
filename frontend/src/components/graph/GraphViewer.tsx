import { useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useDashboard } from '../../context/DashboardContext';
import { useSubgraph } from '../../hooks/useSubgraph';
import { Spinner } from '../common/Spinner';
import { ErrorMessage } from '../common/ErrorMessage';
import type { SubgraphNode, SubgraphEdge } from '../../types/index';

export function getNodeColor(node: SubgraphNode): string {
  if (node.status === 1) return '#ef4444';
  if (node.risk_score >= 0.5) return '#f97316';
  return '#3b82f6';
}

export function getLinkDash(edge: SubgraphEdge): number[] {
  if (edge.relation_type === 'R2') return [];
  if (edge.relation_type === 'R1') return [4, 2];
  return [1, 2];
}

export function GraphViewer() {
  const { state, loadNodeDetail, loadSubgraph } = useDashboard();
  const { subgraph, loading, error, isLargeGraph } = useSubgraph();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  const handleNodeClick = useCallback((node: any) => {
    loadNodeDetail(node.user_id);
  }, [loadNodeDetail]);

  const handleResetView = useCallback(() => {
    graphRef.current?.zoomToFit(400);
  }, []);

  if (!state.selectedUserId) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
        請從上方選擇一個詐騙節點以查看交易網路圖
      </div>
    );
  }

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={() => loadSubgraph(state.selectedUserId!, 2)} />;
  if (!subgraph) return null;

  const graphData = {
    nodes: subgraph.nodes.map(n => ({ ...n, id: n.user_id })),
    links: subgraph.edges.map(e => ({ ...e, source: e.source, target: e.target })),
  };

  return (
    <div className="relative border border-gray-200 rounded-lg overflow-hidden" style={{ height: 480 }}>
      {isLargeGraph && (
        <div className="absolute top-2 left-2 z-10 bg-yellow-100 text-yellow-800 text-xs px-2 py-1 rounded">
          節點數超過 200，已自動降級為 1-hop 顯示
        </div>
      )}
      <button
        onClick={handleResetView}
        className="absolute top-2 right-2 z-10 px-2 py-1 text-xs bg-white border border-gray-300 rounded shadow-sm hover:bg-gray-50"
      >
        重置視角
      </button>
      {/* @ts-ignore */}
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeColor={(node: any) => getNodeColor(node as SubgraphNode)}
        linkLineDash={(link: any) => getLinkDash(link as SubgraphEdge)}
        minZoom={0.1}
        maxZoom={5}
        onNodeClick={handleNodeClick}
        nodeLabel={(node: any) => `User ${node.user_id} | Risk: ${(node as SubgraphNode).risk_score.toFixed(2)}`}
        nodeRelSize={6}
      />
    </div>
  );
}
