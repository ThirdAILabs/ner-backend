'use client';

import { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  FormControl,
  Select,
  MenuItem,
  SelectChangeEvent,
  Card,
  CardContent,
} from '@mui/material';
import { useSearchParams } from 'next/navigation';
import { nerService } from '@/lib/backend';
import MetricsDataViewer from './metrics/MetricsDataViewer';
import { useHealth } from '@/contexts/HealthProvider';
import useTelemetry from '@/hooks/useTelemetry';

import { useLicense } from '@/hooks/useLicense';

import MetricsDataViewerCard from '@/components/ui/MetricsDataViewerCard';
import { formatFileSize } from '@/lib/utils';

import Tooltip from '@mui/material/Tooltip';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import CustomisableCard from '@/components/ui/cards/customisableCard';
import StatsCards from '@/components/stats/StatsCards';

const Dashboard = () => {
  const recordEvent = useTelemetry();
  useEffect(() => {
    recordEvent({
      UserAction: 'View usage stats dashboard',
      UIComponent: 'Usage Stats Dashboard',
      Page: 'Usage Stats Dashboard Page',
    });
  }, []);
  const { healthStatus } = useHealth();
  const { license } = useLicense();
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
  const handleDaysChange = (e: SelectChangeEvent<number>) => {
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
  const handleModelChange = (e: SelectChangeEvent<string>) => {
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

  return (
    <>
      <div className="container py-4">
        <StatsCards />
      </div>
    </>
  );
};

export default Dashboard;
