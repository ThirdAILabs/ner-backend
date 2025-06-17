'use client';

import React, { useEffect, useState } from 'react';
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

const Dashboard = () => {
  const recordEvent = useTelemetry();
  React.useEffect(() => {
    recordEvent({
      UserAction: 'view',
      UIComponent: 'Usage Stats Dashboard Page',
      UI: 'Token Classification Page',
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
      UI: 'Usage Stats Dashboard Page',
      data: { days: newDays },
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
      data: { model },
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
      <Card
        sx={{
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          bgcolor: 'white',
          borderRadius: '12px',
          mx: 'auto',
          maxWidth: '1400px',
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 4,
            }}
          >
            <Typography
              variant="h5"
              sx={{
                fontWeight: 600,
                fontSize: '1.5rem',
                color: '#4a5568',
              }}
            >
              Metrics Dashboard
            </Typography>
          </Box>

          <Box
            sx={{
              display: 'flex',
              gap: 4,
              alignItems: 'flex-start',
              mb: 4,
              '& .MuiFormControl-root': {
                bgcolor: 'white',
                borderRadius: '8px',
                '& .MuiSelect-select': {
                  py: 1.5,
                },
              },
            }}
          >
            {/* Days Filter */}
            <Box>
              <Typography
                variant="subtitle2"
                gutterBottom
                sx={{
                  fontWeight: 600,
                  color: '#475569',
                  mb: 1,
                }}
              >
                Days
              </Typography>
              <FormControl
                size="small"
                sx={{
                  minWidth: 120,
                  '& .MuiOutlinedInput-root': {
                    borderColor: 'grey.200',
                    '&:hover': {
                      borderColor: 'grey.300',
                    },
                  },
                }}
              >
                <Select
                  value={days}
                  onChange={(e) => setDays(Number(e.target.value))}
                  displayEmpty
                  sx={{
                    bgcolor: '#f8fafc',
                    '&:hover': {
                      bgcolor: '#f1f5f9',
                    },
                  }}
                >
                  <MenuItem value={1}>1 day</MenuItem>
                  <MenuItem value={7}>7 days</MenuItem>
                  <MenuItem value={30}>30 days</MenuItem>
                </Select>
              </FormControl>
            </Box>

            {/* Model Filter */}
            <Box flex={1} sx={{ maxWidth: 300 }}>
              <Typography
                variant="subtitle2"
                gutterBottom
                sx={{
                  fontWeight: 600,
                  color: '#475569',
                  mb: 1,
                }}
              >
                Model
              </Typography>
              <FormControl
                size="small"
                fullWidth
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderColor: 'grey.200',
                    '&:hover': {
                      borderColor: 'grey.300',
                    },
                  },
                }}
              >
                <Select
                  value={selectedModel}
                  displayEmpty
                  onChange={handleModelChange}
                  renderValue={(val) =>
                    val === ''
                      ? 'All Models'
                      : models.find((m) => m.Id === val)?.Name
                        ? models
                            .find((m) => m.Id === val)!
                            .Name.charAt(0)
                            .toUpperCase() + models.find((m) => m.Id === val)!.Name.slice(1)
                        : val
                  }
                  sx={{
                    bgcolor: '#f8fafc',
                    '&:hover': {
                      bgcolor: '#f1f5f9',
                    },
                  }}
                >
                  <MenuItem value="">
                    <em>All Models</em>
                  </MenuItem>
                  {models.map((m) => (
                    <MenuItem
                      key={m.Id}
                      value={m.Id}
                      sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexGrow: 1 }}>
                        {m.Name.charAt(0).toUpperCase() + m.Name.slice(1)}
                        {m.Status === 'TRAINING' && (
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <CircularProgress size={16} sx={{ ml: 1 }} />
                            <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                              Training...
                            </Typography>
                          </Box>
                        )}
                        {m.Status === 'QUEUED' && (
                          <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                            Queued
                          </Typography>
                        )}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          </Box>

          {/* Model Details */}
          {selectedModel && (
            <Box sx={{ mb: 4, ml: 4 }}>
              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                Started Training: {new Date('2025-06-10T12:00:00').toLocaleString()}
              </Typography>
              {models.find((m) => m.Id === selectedModel)?.BaseModelId && (
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Base Model:{' '}
                  {models.find(
                    (m) => m.Id === models.find((m) => m.Id === selectedModel)?.BaseModelId
                  )?.Name || 'Unknown'}
                </Typography>
              )}
            </Box>
          )}

          {/* Metrics Viewer */}
          <Box
            sx={{
              bgcolor: 'white',
              borderRadius: '12px',
              border: '1px solid',
              borderColor: 'grey.200',
              overflow: 'hidden',
            }}
          >
            <MetricsDataViewer modelId={selectedModel || undefined} days={days} />
          </Box>
        </CardContent>
      </Card>

      {license && license?.LicenseInfo?.LicenseType === 'free' && (
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            mx: 'auto',
            mt: 4,
            maxWidth: '1400px',
          }}
        >
          <CardContent sx={{ p: 4 }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 4,
              }}
            >
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 600,
                  fontSize: '1.5rem',
                  color: '#4a5568',
                }}
              >
                Free Tier Quota
              </Typography>
            </Box>

            <Box
              sx={{
                bgcolor: 'white',
                borderRadius: '12px',
                border: '1px solid',
                borderColor: 'grey.200',
                overflow: 'hidden',
              }}
            >
              <div style={{ padding: '16px' }}>
                <MetricsDataViewerCard
                  value={`${formatFileSize(license?.LicenseInfo?.Usage.UsedBytes)} / ${formatFileSize(license?.LicenseInfo?.Usage.MaxBytes)}`}
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      Quota Used
                      <Tooltip title="Resets on the 1st of each month.">
                        <InfoOutlinedIcon fontSize="inherit" sx={{ cursor: 'pointer' }} />
                      </Tooltip>
                    </Box>
                  }
                />
              </div>
            </Box>
          </CardContent>
        </Card>
      )}
    </>
  );
};

export default Dashboard;
