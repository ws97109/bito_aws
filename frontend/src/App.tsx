import { useEffect } from 'react';
import { DashboardProvider, useDashboard } from './context/DashboardContext';
import { Dashboard } from './components/layout/Dashboard';

function AppContent() {
  const { loadStats, loadFraudNodes } = useDashboard();

  useEffect(() => {
    // Parallel initial data load (Requirement 7.2)
    Promise.all([loadStats(), loadFraudNodes()]);
  }, [loadStats, loadFraudNodes]);

  return <Dashboard />;
}

export default function App() {
  return (
    <DashboardProvider>
      <AppContent />
    </DashboardProvider>
  );
}
