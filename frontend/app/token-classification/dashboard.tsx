'use client';

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { useParams } from 'next/navigation';
import { nerService } from '@/lib/backend';
import MetricsDataViewer from './metrics/MetricsDataViewer';
import TrainingResults from './metrics/TrainingResults';
import ExamplesVisualizer from './metrics/ExamplesVisualizer';
import ModelUpdate from './metrics/ModelUpdate';

const Dashboard = () => {
  const { deploymentId } = useParams();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Models for the dropdown
  const [days, setDays] = useState<number>(7);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

  useEffect(() => {
    // Simulating loading for a short time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);

    nerService
      .listModels()
      .then(ms => setModels(ms))
      .catch(err => {
        console.error('Failed to load models:', err);
      });

    return () => {
      clearTimeout(timer);
    };
  }, []);

  if (isLoading) {
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
      sx={{
        padding: '24px',
        backgroundColor: '#F5F7FA',
        minHeight: 'calc(100vh - 112px)'
      }}
    >
      <div className="space-y-6">
      {/* <TrainingResults />
        <ExamplesVisualizer />
        <ModelUpdate 
          username="user"
          modelName={deploymentId as string}
          deploymentUrl={`/api/token-classification/${deploymentId}`}
          modelId={`user/${deploymentId}`}
        /> */}
        <Box mb={3} sx={{ display: 'flex', gap: 4, alignItems: 'flex-start' }}>
        {/* Days */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Days
          </Typography>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <Select
              value={days}
              onChange={e => setDays(Number(e.target.value))}
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
          <Typography variant="subtitle2" gutterBottom>
            Model
          </Typography>
          <FormControl size="small" fullWidth>
            <Select
              value={selectedModel}
              displayEmpty
              onChange={e => setSelectedModel(e.target.value)}
              renderValue={val =>
                val === ''
                  ? 'All Models'
                  : models.find(m => m.Id === val)?.Name || val
              }
            >
              <MenuItem value="">
                <em>All Models</em>
              </MenuItem>
              {models.map(m => (
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
