'use client';

import { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress, SelectChangeEvent } from '@mui/material';
import { useSearchParams } from 'next/navigation';
import { nerService } from '@/lib/backend';
import { useHealth } from '@/contexts/HealthProvider';
import { useConditionalTelemetry } from '@/hooks/useConditionalTelemetry';
import { useLicense } from '@/hooks/useLicense';
import StatsCards from '@/components/stats/StatsCards';
import FilterDropdown from '@/components/ui/FilterDropdown';

const styles = {
  metricsHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '4rem',
    marginBottom: '36px',
  },
  metricsTitle: {
    fontWeight: 600,
  },
  filtersContainer: {
    display: 'flex',
    gap: '2.4rem',
    alignItems: 'center',
  },
};

const Dashboard = () => {
  const recordEvent = useConditionalTelemetry();
  useEffect(() => {
    recordEvent({
      UserAction: 'View usage stats dashboard',
      UIComponent: 'Usage Stats Dashboard',
      Page: 'Usage Stats Dashboard Page',
    });
  }, []);
  const { healthStatus } = useHealth();
  const { license } = useLicense();
  console.log('License:', license);
  const searchParams = useSearchParams();
  const deploymentId = searchParams.get('deploymentId');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Don't make API calls if health check hasn't passed
  if (!healthStatus) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Models for the dropdown
  const [days, setDays] = useState<number>(7);
  const handleDaysChange = (e: SelectChangeEvent<string | number>) => {
    const newDays = Number(e.target.value);
    setDays(newDays);
    recordEvent({
      UserAction: 'select',
      UIComponent: 'Days Filter',
      Page: 'Usage Stats Dashboard Page',
      Data: { days: newDays },
    });
  };
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const handleModelChange = (e: SelectChangeEvent<string | number>) => {
    const model = models.find((m) => m.Id === e.target.value) || null;
    setSelectedModel(model);
    recordEvent({
      UserAction: 'select',
      UIComponent: 'Model Filter',
      Page: 'Usage Stats Dashboard Page',
      Data: { model },
    });
  };

  useEffect(() => {
    const fetchModels = () => {
      nerService
        .listModels()
        .then((ms) => setModels(ms))
        .catch((err) => {
          console.error('Failed to load models:', err);
        });
    };

    // Initial fetch
    fetchModels();

    // Set up polling every 5 seconds
    const intervalId = setInterval(fetchModels, 5000);

    // Cleanup interval on unmount
    return () => clearInterval(intervalId);
  }, [healthStatus]);

  if (!healthStatus) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  const daysOptions = [
    { value: 7, label: '7 days' },
    { value: 30, label: '30 days' },
    { value: 90, label: '90 days' },
  ];

  const modelOptions = models.map((model) => ({
    value: model.Id,
    label: model.Name || model.Id,
  }));

  const [stats, setStats] = useState<InferenceMetrics | null>(null);
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const metrics = await nerService.getInferenceMetrics(selectedModel?.Id, days);
        setStats(metrics);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to fetch inference metrics:', err);
        setError('Failed to load metrics');
        setIsLoading(false);
      }
    };

    // Initial fetch
    fetchStats();

    // // Set up polling every 5 seconds
    // const intervalId = setInterval(fetchStats, 5000);

    // // Cleanup interval on unmount
    // return () => clearInterval(intervalId);
  }, [days, selectedModel, healthStatus]);

  return (
    <>
      <div className="container py-4">
        <div style={styles.metricsHeader}>
          <span style={styles.metricsTitle} className="text-gray-500 text-2xl">
            Metrics
          </span>
          <div style={styles.filtersContainer}>
            <FilterDropdown
              label="Days"
              value={days}
              options={daysOptions}
              onChange={handleDaysChange}
            />
            <FilterDropdown
              label="Model"
              value={selectedModel?.Id || 'all'}
              options={[{ value: 'all', label: 'All Models' }, ...modelOptions]}
              onChange={handleModelChange}
            />
          </div>
        </div>
        <StatsCards stats={stats} license={license} />
      </div>
    </>
  );
};

export default Dashboard;
