import { useEffect } from 'react';
import { DashboardProvider, useDashboard } from './context/DashboardContext';
import { Dashboard } from './components/layout/Dashboard';
import { CryptoBackground } from './components/common/CryptoBackground';

function AppContent() {
  const { loadStats, loadFraudNodes, loadFpFnNodes, loadPredictNodes } = useDashboard();

  useEffect(() => {
    Promise.all([loadStats(), loadFraudNodes(), loadFpFnNodes(), loadPredictNodes()]);
  }, [loadStats, loadFraudNodes, loadFpFnNodes, loadPredictNodes]);

  return <Dashboard />;
}

export default function App() {
  return (
    <DashboardProvider>
      <CryptoBackground />
      <AppContent />
    </DashboardProvider>
  );
}
