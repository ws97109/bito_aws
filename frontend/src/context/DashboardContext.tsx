import React, { createContext, useContext, useReducer, useCallback } from 'react';
import type { DashboardState, StatsResponse, FraudNode, SubgraphResponse, NodeDetailResponse, FpFnNode, DashboardMode, FpFnMode, ShapWaterfallResponse } from '../types/index';
import { getStats } from '../api/statsApi';
import { getFraudNodes } from '../api/fraudNodesApi';
import { getSubgraph } from '../api/subgraphApi';
import { getNodeDetail } from '../api/nodeApi';
import { getFpFnNodes } from '../api/fpFnApi';
import { getShapWaterfall } from '../api/shapApi';

// ── Action Types ──────────────────────────────────────────────────────────────

type Action =
  | { type: 'SET_STATS_LOADING' }
  | { type: 'SET_STATS_SUCCESS'; data: StatsResponse }
  | { type: 'SET_STATS_ERROR'; error: string }
  | { type: 'SET_FRAUD_NODES_LOADING' }
  | { type: 'SET_FRAUD_NODES_SUCCESS'; data: FraudNode[] }
  | { type: 'SET_FRAUD_NODES_ERROR'; error: string }
  | { type: 'SELECT_USER'; userId: number }
  | { type: 'SET_SUBGRAPH_LOADING' }
  | { type: 'SET_SUBGRAPH_SUCCESS'; data: SubgraphResponse }
  | { type: 'SET_SUBGRAPH_ERROR'; error: string }
  | { type: 'SET_NODE_DETAIL_LOADING' }
  | { type: 'SET_NODE_DETAIL_SUCCESS'; data: NodeDetailResponse }
  | { type: 'SET_NODE_DETAIL_ERROR'; error: string }
  | { type: 'SET_DASHBOARD_MODE'; mode: DashboardMode }
  | { type: 'SET_FPFN_MODE'; mode: FpFnMode }
  | { type: 'SET_FPFN_NODES_LOADING' }
  | { type: 'SET_FPFN_NODES_SUCCESS'; fp: FpFnNode[]; fn: FpFnNode[] }
  | { type: 'SET_FPFN_NODES_ERROR'; error: string }
  | { type: 'SET_SHAP_LOADING' }
  | { type: 'SET_SHAP_SUCCESS'; data: ShapWaterfallResponse }
  | { type: 'SET_SHAP_ERROR'; error: string };

// ── Initial State ─────────────────────────────────────────────────────────────

const initialState: DashboardState = {
  stats: null,
  fraudNodes: [],
  selectedUserId: null,
  subgraph: null,
  selectedNode: null,
  subgraphCache: new Map(),
  dashboardMode: 'fraud',
  fpFnMode: 'fp',
  fpNodes: [],
  fnNodes: [],
  shapWaterfall: null,
  loading: { stats: false, fraudNodes: false, subgraph: false, nodeDetail: false, fpFnNodes: false, shapWaterfall: false },
  error: { stats: null, fraudNodes: null, subgraph: null, nodeDetail: null, fpFnNodes: null, shapWaterfall: null },
};

// ── Reducer ───────────────────────────────────────────────────────────────────

function dashboardReducer(state: DashboardState, action: Action): DashboardState {
  switch (action.type) {
    case 'SET_STATS_LOADING':
      return { ...state, loading: { ...state.loading, stats: true }, error: { ...state.error, stats: null } };
    case 'SET_STATS_SUCCESS':
      return { ...state, stats: action.data, loading: { ...state.loading, stats: false }, error: { ...state.error, stats: null } };
    case 'SET_STATS_ERROR':
      return { ...state, loading: { ...state.loading, stats: false }, error: { ...state.error, stats: action.error } };

    case 'SET_FRAUD_NODES_LOADING':
      return { ...state, loading: { ...state.loading, fraudNodes: true }, error: { ...state.error, fraudNodes: null } };
    case 'SET_FRAUD_NODES_SUCCESS':
      return { ...state, fraudNodes: action.data, loading: { ...state.loading, fraudNodes: false }, error: { ...state.error, fraudNodes: null } };
    case 'SET_FRAUD_NODES_ERROR':
      return { ...state, loading: { ...state.loading, fraudNodes: false }, error: { ...state.error, fraudNodes: action.error } };

    case 'SELECT_USER': {
      const cached = state.subgraphCache.get(action.userId);
      return {
        ...state,
        selectedUserId: action.userId,
        subgraph: cached ?? state.subgraph,
        loading: { ...state.loading, subgraph: cached ? false : state.loading.subgraph },
      };
    }

    case 'SET_SUBGRAPH_LOADING':
      return { ...state, loading: { ...state.loading, subgraph: true }, error: { ...state.error, subgraph: null } };
    case 'SET_SUBGRAPH_SUCCESS':
      return {
        ...state,
        subgraph: action.data,
        subgraphCache: new Map(state.subgraphCache).set(state.selectedUserId!, action.data),
        loading: { ...state.loading, subgraph: false },
        error: { ...state.error, subgraph: null },
      };
    case 'SET_SUBGRAPH_ERROR':
      return { ...state, loading: { ...state.loading, subgraph: false }, error: { ...state.error, subgraph: action.error } };

    case 'SET_NODE_DETAIL_LOADING':
      return { ...state, loading: { ...state.loading, nodeDetail: true }, error: { ...state.error, nodeDetail: null } };
    case 'SET_NODE_DETAIL_SUCCESS':
      return { ...state, selectedNode: action.data, loading: { ...state.loading, nodeDetail: false }, error: { ...state.error, nodeDetail: null } };
    case 'SET_NODE_DETAIL_ERROR':
      return { ...state, loading: { ...state.loading, nodeDetail: false }, error: { ...state.error, nodeDetail: action.error } };

    case 'SET_DASHBOARD_MODE':
      return { ...state, dashboardMode: action.mode, selectedUserId: null, subgraph: null, selectedNode: null };
    case 'SET_FPFN_MODE':
      return { ...state, fpFnMode: action.mode, selectedUserId: null, subgraph: null, selectedNode: null };

    case 'SET_FPFN_NODES_LOADING':
      return { ...state, loading: { ...state.loading, fpFnNodes: true }, error: { ...state.error, fpFnNodes: null } };
    case 'SET_FPFN_NODES_SUCCESS':
      return { ...state, fpNodes: action.fp, fnNodes: action.fn, loading: { ...state.loading, fpFnNodes: false } };
    case 'SET_FPFN_NODES_ERROR':
      return { ...state, loading: { ...state.loading, fpFnNodes: false }, error: { ...state.error, fpFnNodes: action.error } };

    case 'SET_SHAP_LOADING':
      return { ...state, loading: { ...state.loading, shapWaterfall: true }, error: { ...state.error, shapWaterfall: null } };
    case 'SET_SHAP_SUCCESS':
      return { ...state, shapWaterfall: action.data, loading: { ...state.loading, shapWaterfall: false } };
    case 'SET_SHAP_ERROR':
      return { ...state, loading: { ...state.loading, shapWaterfall: false }, error: { ...state.error, shapWaterfall: action.error } };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────

interface DashboardContextValue {
  state: DashboardState;
  dispatch: React.Dispatch<Action>;
  loadStats: () => Promise<void>;
  loadFraudNodes: () => Promise<void>;
  loadSubgraph: (userId: number, hops?: number) => Promise<void>;
  loadNodeDetail: (userId: number) => Promise<void>;
  loadFpFnNodes: () => Promise<void>;
  loadShapWaterfall: (mode: FpFnMode, userId?: number) => Promise<void>;
}

export const DashboardContext = createContext<DashboardContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);

  const loadStats = useCallback(async () => {
    dispatch({ type: 'SET_STATS_LOADING' });
    try {
      const data = await getStats();
      dispatch({ type: 'SET_STATS_SUCCESS', data });
    } catch (err) {
      dispatch({ type: 'SET_STATS_ERROR', error: err instanceof Error ? err.message : 'Failed to load stats' });
    }
  }, []);

  const loadFraudNodes = useCallback(async () => {
    dispatch({ type: 'SET_FRAUD_NODES_LOADING' });
    try {
      const data = await getFraudNodes();
      dispatch({ type: 'SET_FRAUD_NODES_SUCCESS', data });
    } catch (err) {
      dispatch({ type: 'SET_FRAUD_NODES_ERROR', error: err instanceof Error ? err.message : 'Failed to load fraud nodes' });
    }
  }, []);

  const loadSubgraph = useCallback(async (userId: number, hops = 2) => {
    dispatch({ type: 'SET_SUBGRAPH_LOADING' });
    try {
      const data = await getSubgraph(userId, hops);
      dispatch({ type: 'SET_SUBGRAPH_SUCCESS', data });
    } catch (err) {
      dispatch({ type: 'SET_SUBGRAPH_ERROR', error: err instanceof Error ? err.message : 'Failed to load subgraph' });
    }
  }, []);

  const loadNodeDetail = useCallback(async (userId: number) => {
    dispatch({ type: 'SET_NODE_DETAIL_LOADING' });
    try {
      const data = await getNodeDetail(userId);
      dispatch({ type: 'SET_NODE_DETAIL_SUCCESS', data });
    } catch (err) {
      dispatch({ type: 'SET_NODE_DETAIL_ERROR', error: err instanceof Error ? err.message : 'Failed to load node detail' });
    }
  }, []);

  const loadFpFnNodes = useCallback(async () => {
    dispatch({ type: 'SET_FPFN_NODES_LOADING' });
    try {
      const { fp, fn } = await getFpFnNodes();
      dispatch({ type: 'SET_FPFN_NODES_SUCCESS', fp, fn });
    } catch (err) {
      dispatch({ type: 'SET_FPFN_NODES_ERROR', error: err instanceof Error ? err.message : 'Failed to load FP/FN nodes' });
    }
  }, []);

  const loadShapWaterfall = useCallback(async (mode: FpFnMode, userId?: number) => {
    dispatch({ type: 'SET_SHAP_LOADING' });
    try {
      const data = await getShapWaterfall(mode, userId);
      dispatch({ type: 'SET_SHAP_SUCCESS', data });
    } catch (err) {
      dispatch({ type: 'SET_SHAP_ERROR', error: err instanceof Error ? err.message : 'Failed to load SHAP data' });
    }
  }, []);

  return (
    <DashboardContext.Provider value={{ state, dispatch, loadStats, loadFraudNodes, loadSubgraph, loadNodeDetail, loadFpFnNodes, loadShapWaterfall }}>
      {children}
    </DashboardContext.Provider>
  );
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useDashboard(): DashboardContextValue {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be used within a DashboardProvider');
  return ctx;
}
