import { useEffect } from 'react';
import { DashboardProvider, useDashboard } from './context/DashboardContext';
import { Dashboard } from './components/layout/Dashboard';
import { CryptoBackground } from './components/common/CryptoBackground';

function AppContent() {
  const { loadStats, loadFraudNodes, loadFpFnNodes } = useDashboard();

  useEffect(() => {
    // Parallel initial data load (Requirement 7.2)
    Promise.all([loadStats(), loadFraudNodes(), loadFpFnNodes()]);
  }, [loadStats, loadFraudNodes, loadFpFnNodes]);

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
