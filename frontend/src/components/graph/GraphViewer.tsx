import { useRef, useCallback, useEffect, useState, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import { useDashboard } from '../../context/DashboardContext';
import { useSubgraph } from '../../hooks/useSubgraph';
import { Spinner } from '../common/Spinner';
import { ErrorMessage } from '../common/ErrorMessage';
import type { SubgraphNode, SubgraphEdge } from '../../types/index';

// ── Colour helpers ─────────────────────────────────────────────────────────────

export function getNodeColor(node: SubgraphNode): string {
  if (node.node_type === 'wallet') return '#8b5cf6';
  if (node.status === 1) return '#ef4444';
  if (node.risk_score >= 0.5) return '#f97316';
  return '#4f46e5';
}

export function getLinkColor(edge: SubgraphEdge): string {
  if (edge.relation_type === 'R1') return '#0ea5e9';  // sky     — 錢包→帳戶
  if (edge.relation_type === 'R2') return '#f59e0b';  // amber   — 帳戶→帳戶
  if (edge.relation_type === 'R3') return '#10b981';  // emerald — 帳戶→錢包
  return '#94a3b8';
}

// getLinkDash kept for tests that import it; 3-D graph doesn't use dashes
export function getLinkDash(_edge: SubgraphEdge): number[] | null { return null; }

// ── Legend row (outside canvas, always visible) ────────────────────────────────

function GraphLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-3 py-2 bg-slate-800/70 rounded-lg border border-slate-700/50 text-xs flex-shrink-0">
      <span className="text-slate-500 text-[10px] uppercase tracking-wider font-semibold">圖例</span>

      <div className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-red-500 flex-shrink-0"></span>
        <span className="text-slate-300">詐騙節點</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-orange-500 flex-shrink-0"></span>
        <span className="text-slate-300">高風險</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 flex-shrink-0"></span>
        <span className="text-slate-300">正常</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span
          className="w-3 h-3 flex-shrink-0 inline-block bg-violet-500"
          style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
        ></span>
        <span className="text-slate-300">錢包節點</span>
      </div>

      <div className="border-l border-slate-700/60 pl-3 flex items-center gap-1.5">
        <span className="inline-block w-5 h-0.5 bg-sky-500"></span>
        <span className="text-slate-300">錢包→帳戶</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="inline-block w-5 h-0.5 bg-amber-500"></span>
        <span className="text-slate-300">帳戶→帳戶</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="inline-block w-5 h-0.5 bg-emerald-500"></span>
        <span className="text-slate-300">帳戶→錢包</span>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export function GraphViewer() {
  const { state, dispatch, loadNodeDetail, loadSubgraph } = useDashboard();
  const { subgraph, loading, error, isLargeGraph } = useSubgraph();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });

  // Refs to break dependency cycles in stable callbacks
  const selectedWalletIdRef = useRef(state.selectedWalletId);
  const selectedUserIdRef = useRef(state.selectedUserId);
  const subgraphRef = useRef(subgraph);
  useEffect(() => { selectedWalletIdRef.current = state.selectedWalletId; }, [state.selectedWalletId]);
  useEffect(() => { selectedUserIdRef.current = state.selectedUserId; }, [state.selectedUserId]);
  useEffect(() => { subgraphRef.current = subgraph; }, [subgraph]);

  // Track container size so the 3-D canvas fills exactly
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setDimensions({ width, height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Auto-fit camera once force simulation settles; fly to selected wallet if present.
  // Uses refs so this callback never changes reference — prevents ForceGraph3D from
  // restarting the simulation every time subgraph or selectedWalletId updates.
  const handleEngineStop = useCallback(() => {
    const walletId = selectedWalletIdRef.current;
    const sg = subgraphRef.current;
    if (walletId && sg) {
      const walletNode = sg.nodes.find(n =>
        n.node_type === 'wallet' &&
        (n.node_label === walletId || String(n.user_id) === walletId)
      );
      if (walletNode) {
        const graphNode = (graphRef.current?.graphData()?.nodes ?? [])
          .find((n: any) => n.user_id === walletNode.user_id);
        if (graphNode) {
          const { x = 0, y = 0, z = 0 } = graphNode;
          const distance = 120;
          graphRef.current?.cameraPosition(
            { x: x + distance, y: y + distance / 2, z: z + distance },
            { x, y, z },
            800,
          );
          return;
        }
      }
    }
    graphRef.current?.zoomToFit(600, 80);
  }, []); // intentionally empty — reads live values via refs

  const handleNodeClick = useCallback((node: any) => {
    const n = node as SubgraphNode;
    if (n.node_type === 'wallet') {
      // Find connected users via current subgraph edges
      const edges = subgraph?.edges ?? [];
      const connectedUserIds = edges
        .filter(e => e.source === n.user_id || e.target === n.user_id)
        .map(e => e.source === n.user_id ? e.target : e.source);
      const connectedUsers = (subgraph?.nodes ?? [])
        .filter(nd => connectedUserIds.includes(nd.user_id) && nd.node_type === 'user')
        .sort((a, b) => b.risk_score - a.risk_score);
      if (connectedUsers.length > 0) loadNodeDetail(connectedUsers[0].user_id);
    } else {
      loadNodeDetail(n.user_id);
    }
    // Camera flyto (keep existing)
    const distance = 120;
    const { x = 0, y = 0, z = 0 } = node;
    graphRef.current?.cameraPosition(
      { x: x + distance, y: y + distance / 2, z: z + distance },
      { x, y, z },
      800,
    );
  }, [loadNodeDetail, subgraph]);

  const handleZoomIn  = useCallback(() => {
    const cam = graphRef.current?.camera();
    if (!cam) return;
    const f = 0.65;
    graphRef.current.cameraPosition({ x: cam.position.x * f, y: cam.position.y * f, z: cam.position.z * f }, undefined, 200);
  }, []);

  const handleZoomOut = useCallback(() => {
    const cam = graphRef.current?.camera();
    if (!cam) return;
    const f = 1.5;
    graphRef.current.cameraPosition({ x: cam.position.x * f, y: cam.position.y * f, z: cam.position.z * f }, undefined, 200);
  }, []);

  const handleResetView = useCallback(() => graphRef.current?.zoomToFit(500, 80), []);

  // Memoize graphData so ForceGraph3D doesn't see new arrays on every render
  const graphData = useMemo(() => ({
    nodes: subgraph?.nodes.map(n => ({ ...n, id: n.user_id })) ?? [],
    links: subgraph?.edges.map(e => ({ ...e })) ?? [],
  }), [subgraph]);

  const nodeColor = useCallback((node: any) => getNodeColor(node as SubgraphNode), []);

  const nodeVal = useCallback((node: any) => {
    const n = node as SubgraphNode;
    const walletId = selectedWalletIdRef.current;
    const isSelectedWallet = n.node_type === 'wallet' && walletId !== null &&
      (n.node_label === walletId || String(n.user_id) === walletId);
    if (isSelectedWallet) return 4;
    if (n.user_id === selectedUserIdRef.current && !walletId) return 4;
    return n.node_type === 'wallet' ? 1.2 : 1;
  }, []);

  const linkColor = useCallback((link: any) => getLinkColor(link as SubgraphEdge), []);
  const linkArrowColor = useCallback((link: any) => getLinkColor(link as SubgraphEdge), []);

  const nodeLabel = useCallback((node: any) => {
    const n = node as SubgraphNode;
    if (n.node_type === 'wallet') return `Wallet: ${n.node_label ?? n.user_id}`;
    return `User ${n.node_label?.replace('user_', '') ?? n.user_id} | Risk: ${n.risk_score.toFixed(2)}`;
  }, []);

  // Custom 3-D object: gold ring+sphere for selected wallet or center user, plain octahedron for other wallets.
  // Uses refs so deps stay empty — prevents ForceGraph3D from rebuilding all node objects on every state change.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodeThreeObject = useCallback((node: any): any => {
    const n = node as SubgraphNode;
    const walletId = selectedWalletIdRef.current;
    const userId = selectedUserIdRef.current;
    const isSelectedWallet = n.node_type === 'wallet' && walletId !== null &&
      (n.node_label === walletId || String(n.user_id) === walletId);
    if (isSelectedWallet || (n.user_id === userId && !walletId && n.node_type !== 'wallet')) {
      const group = new THREE.Group();
      group.add(new THREE.Mesh(
        new THREE.SphereGeometry(6, 16, 16),
        new THREE.MeshLambertMaterial({ color: 0xfbbf24 }),
      ));
      group.add(new THREE.Mesh(
        new THREE.TorusGeometry(9, 1.2, 8, 32),
        new THREE.MeshLambertMaterial({ color: 0xfde68a }),
      ));
      return group;
    }
    if (n.node_type === 'wallet') {
      return new THREE.Mesh(
        new THREE.OctahedronGeometry(4),
        new THREE.MeshLambertMaterial({ color: 0x8b5cf6 }),
      );
    }
    return undefined;
  }, []); // intentionally empty — reads live values via refs

  return (
    <div className="flex flex-col h-full gap-2">
      {/* Legend — always rendered above the canvas, never obscured by the 3D scene */}
      <GraphLegend />

      {/* 3-D canvas area */}
      <div
        ref={containerRef}
        className="flex-1 relative rounded-lg overflow-hidden min-h-0 border border-slate-700 bg-slate-900/30"
      >
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
          {state.selectedNode && (
            <button
              title="設為中心節點"
              onClick={() => {
                const uid = state.selectedNode!.user_id;
                dispatch({ type: 'SELECT_USER', userId: uid });
                if (!state.subgraphCache.has(uid)) loadSubgraph(uid, 2);
              }}
              className="w-9 h-9 text-xs font-bold flex items-center justify-center bg-indigo-700/80 border border-indigo-500/70 rounded-lg shadow hover:bg-indigo-600 focus:outline-none focus:ring-1 focus:ring-indigo-400 text-indigo-200 backdrop-blur-sm transition-colors"
            >
              ⊙
            </button>
          )}
        </div>

        {!state.selectedUserId ? (
          <div className="flex flex-col items-center justify-center h-full">
            <span className="text-5xl text-slate-600 mb-3">&#128202;</span>
            <p className="text-slate-300 font-medium text-sm">選擇節點以載入關係圖</p>
            <p className="text-slate-500 text-xs mt-1">支援 1-hop 和 2-hop 鄰居展開</p>
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center h-full"><Spinner /></div>
        ) : error ? (
          <div className="flex items-center justify-center h-full">
            <ErrorMessage message={error} onRetry={() => loadSubgraph(state.selectedUserId!, 2)} />
          </div>
        ) : subgraph && subgraph.nodes.length > 0 ? (
          /* @ts-ignore */
          <ForceGraph3D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            nodeColor={nodeColor}
            nodeVal={nodeVal}
            nodeThreeObject={nodeThreeObject}
            linkColor={linkColor}
            linkWidth={1.2}
            linkOpacity={0.7}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkDirectionalArrowColor={linkArrowColor}
            onNodeClick={handleNodeClick}
            onEngineStop={handleEngineStop}
            nodeLabel={nodeLabel}
            backgroundColor="rgba(0,0,0,0)"
            showNavInfo={false}
          />
        ) : subgraph !== null ? (
          /* subgraph loaded but empty — user not in GNN graph */
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-6">
            <span className="text-3xl text-slate-600">&#128202;</span>
            <p className="text-slate-300 text-sm font-medium">此節點無圖形資料</p>
            <p className="text-slate-500 text-xs leading-relaxed">
              此用戶 ID 未包含於 GNN 圖結構中（未出現在 gnn_node_list 或 gnn_edge_list）。<br />
              請選擇節點列表中標有 <span className="text-emerald-400">&#9679;</span> 的用戶以查看圖形。
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}
