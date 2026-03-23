import { useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { useDashboard } from '../../context/DashboardContext';
import { useSubgraph } from '../../hooks/useSubgraph';
import { Spinner } from '../common/Spinner';
import { ErrorMessage } from '../common/ErrorMessage';
import type { SubgraphNode, SubgraphEdge } from '../../types/index';

export function getNodeColor(node: SubgraphNode): string {
  if (node.status === 1) return '#ef4444'; // red-500
  if (node.risk_score >= 0.5) return '#f97316'; // orange-500
  return '#4f46e5'; // indigo-600
}

export function getLinkColor(edge: SubgraphEdge): string {
  if (edge.relation_type === 'R1') return '#0ea5e9'; // sky-500
  if (edge.relation_type === 'R2') return '#f59e0b'; // amber-500
  if (edge.relation_type === 'R3') return '#10b981'; // emerald-500
  return '#94a3b8'; // slate-400
}

export function getLinkDash(edge: SubgraphEdge): number[] | null {
  if (edge.relation_type === 'R2') return null; // Solid line
  if (edge.relation_type === 'R1') return [4, 2];
  return [2, 3];
}

export function GraphViewer() {
  const { state, loadNodeDetail, loadSubgraph } = useDashboard();
  const { subgraph, loading, error, isLargeGraph } = useSubgraph();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);

  const handleNodeClick = useCallback((node: any) => {
    loadNodeDetail(node.user_id);
    if (node.x && node.y) {
      graphRef.current?.centerAt(node.x, node.y, 1000);
      graphRef.current?.zoom(2.5, 500);
    }
  }, [loadNodeDetail]);

  const handleZoomIn = useCallback(() => graphRef.current?.zoom(graphRef.current.zoom() * 1.5, 200), []);
  const handleZoomOut = useCallback(() => graphRef.current?.zoom(graphRef.current.zoom() / 1.5, 200), []);
  const handleResetView = useCallback(() => graphRef.current?.zoomToFit(400, 100), []);

  if (!state.selectedUserId) {
    return (
      <div className="flex flex-col items-center justify-center h-full border border-dashed border-slate-700 rounded-lg bg-slate-800/20">
        <span className="text-5xl text-slate-600 mb-3">&#128202;</span>
        <p className="text-slate-300 font-medium text-sm">選擇節點以載入關係圖</p>
        <p className="text-slate-500 text-xs mt-1">支援 1-hop 和 2-hop 鄰居展開</p>
      </div>
    );
  }

  if (loading) return (
    <div className="flex items-center justify-center h-full"><Spinner /></div>
  );
  if (error) return (
    <div className="flex items-center justify-center h-full">
      <ErrorMessage message={error} onRetry={() => loadSubgraph(state.selectedUserId!, 2)} />
    </div>
  );
  if (!subgraph) return null;

  const graphData = {
    nodes: subgraph.nodes.map(n => ({ ...n, id: n.user_id })),
    links: subgraph.edges.map(e => ({ ...e, source: e.source, target: e.target })),
  };

  return (
    <div className="relative border border-slate-700 rounded-lg overflow-hidden h-full bg-slate-900/30">
      {isLargeGraph && (
        <div className="absolute top-3 left-3 z-10 bg-amber-900/70 text-amber-300 border border-amber-700/50 text-xs px-2.5 py-1 rounded-md shadow-sm backdrop-blur-sm">
          &#9888; 節點數超過 200，已自動降級為 1-hop 顯示
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute top-3 right-3 z-10 flex flex-col gap-1.5">
        {[
          { label: '&#43;',    title: 'Zoom In',    handler: handleZoomIn },
          { label: '&#8722;', title: 'Zoom Out',   handler: handleZoomOut },
          { label: '&#8635;', title: 'Reset View', handler: handleResetView },
        ].map(btn => (
          <button
            key={btn.title}
            onClick={btn.handler}
            title={btn.title}
            className="w-9 h-9 font-bold flex items-center justify-center bg-slate-800/90 border border-slate-600/70 rounded-lg shadow hover:bg-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500 text-slate-300 backdrop-blur-sm transition-colors"
            dangerouslySetInnerHTML={{ __html: btn.label }}
          />
        ))}
      </div>

      {/* Graph legend */}
      <div className="absolute bottom-3 left-3 z-10 bg-slate-900/85 backdrop-blur-sm border border-slate-700/60 rounded-lg p-2.5 text-xs space-y-1.5">
        <p className="text-slate-400 font-semibold uppercase tracking-wider text-[10px] mb-1.5">圖例</p>
        <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block flex-shrink-0"></span><span className="text-slate-300">詐騙節點</span></div>
        <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-orange-500 inline-block flex-shrink-0"></span><span className="text-slate-300">高風險</span></div>
        <div className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block flex-shrink-0"></span><span className="text-slate-300">正常</span></div>
        <div className="border-t border-slate-700/60 pt-1.5 mt-1 space-y-1.5">
          <div className="flex items-center gap-1.5"><span className="inline-block w-5 h-0.5 bg-sky-500"></span><span className="text-slate-300">R1 共用 IP</span></div>
          <div className="flex items-center gap-1.5"><span className="inline-block w-5 h-0.5 bg-amber-500"></span><span className="text-slate-300">R2 加密內轉</span></div>
          <div className="flex items-center gap-1.5"><span className="inline-block w-5 h-0.5 bg-emerald-500"></span><span className="text-slate-300">R3 共用錢包</span></div>
        </div>
      </div>

      {/* @ts-ignore */}
      <ForceGraph2D
        ref={graphRef}
        graphData={graphData}
        nodeColor={(node: any) => getNodeColor(node as SubgraphNode)}
        linkColor={(link: any) => getLinkColor(link as SubgraphEdge)}
        linkLineDash={(link: any) => getLinkDash(link as SubgraphEdge)}
        linkWidth={1.5}
        minZoom={0.1}
        maxZoom={8}
        onNodeClick={handleNodeClick}
        nodeLabel={(node: any) => `User ${node.user_id} | Risk: ${(node as SubgraphNode).risk_score.toFixed(2)}`}
        nodeRelSize={6}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObjectMode={() => 'after'}
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D) => {
          if (node.user_id === state.selectedUserId) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 8, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          }
        }}
      />
    </div>
  );
}
