'use client';

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import { useSearchParams } from 'next/navigation';
import { nerService } from '@/lib/backend';
import MetricsDataViewer from './metrics/MetricsDataViewer';
import TrainingResults from './metrics/TrainingResults';
import ExamplesVisualizer from './metrics/ExamplesVisualizer';
import ModelUpdate from './metrics/ModelUpdate';
import { useHealth } from '@/contexts/HealthProvider';
import useTelemetry from '@/hooks/useTelemetry';

const Dashboard = () => {
  const recordEvent = useTelemetry();
  React.useEffect(() => {
    recordEvent({
      UserAction: 'view',
      UIComponent: 'Usage Stats Dashboard Page',
      UI: 'Token Classification Page'
    });
  }, []);
  const { healthStatus } = useHealth();
  const searchParams = useSearchParams();
  const deploymentId = searchParams.get('deploymentId');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Models for the dropdown
  const [days, setDays] = useState<number>(7);
  const handleDaysChange = (e: SelectChangeEvent<number>) => {
    const newDays = Number(e.target.value);
    setDays(newDays);
    recordEvent({
      UserAction: 'select',
      UIComponent: 'Days Filter',
      UI: 'Usage Stats Dashboard Page',
      data: { days: newDays }
    });
  };
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const handleModelChange = (e: SelectChangeEvent<string>) => {
    const model = e.target.value;
    setSelectedModel(model);
    recordEvent({
      UserAction: 'select',
      UIComponent: 'Model Filter',
      UI: 'Usage Stats Dashboard Page',
      data: { model }
    });
  };

  useEffect(() => {
    nerService
      .listModels()
      .then((ms) => setModels(ms))
      .catch((err) => {
        console.error('Failed to load models:', err);
      });
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
    <Box
      className="bg-muted/60"
      sx={{
        padding: '24px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      }}
    >
      <div className="space-y-6">
        {/* <TrainingResults />
        <ExamplesVisualizer />
        <ModelUpdate 
          username="user"
          modelName={deploymentId as string}
          deploymentUrl={`/api/token-classification?deploymentId=${deploymentId}`}
          modelId={`user/${deploymentId}`}
        /> */}
        <Box mb={3} sx={{ display: 'flex', gap: 4, alignItems: 'flex-start' }}>
          {/* Days */}
          <Box>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
              Days
            </Typography>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <Select
                value={days}
                onChange={handleDaysChange}
                displayEmpty
              >
                <MenuItem value={1}>1 day</MenuItem>
                <MenuItem value={7}>7 days</MenuItem>
                <MenuItem value={30}>30 days</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Model Filter */}
          <Box flex={1} sx={{ maxWidth: 300 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 600 }}>
              Model
            </Typography>
            <FormControl size="small" fullWidth>
              <Select
                value={selectedModel}
                displayEmpty
                onChange={handleModelChange}
                renderValue={val =>
                  val === ''
                    ? 'All Models'
                    : models.find(m => m.Id === val)?.Name || val
                }
              >
                <MenuItem value="">
                  <em>All Models</em>
                </MenuItem>
                {models.map((m) => (
                  <MenuItem key={m.Id} value={m.Id}>
                    {m.Name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Box>

        {/* Your new metrics viewer */}
        <MetricsDataViewer modelId={selectedModel || undefined} days={days} />
      </div>
    </Box>
  );
};

export default Dashboard;
