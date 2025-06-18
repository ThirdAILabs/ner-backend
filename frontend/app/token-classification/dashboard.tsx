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
import { nerService } from '@/lib/backend';
import MetricsDataViewer from './metrics/MetricsDataViewer';
import { useHealth } from '@/contexts/HealthProvider';
import useTelemetry from '@/hooks/useTelemetry';
import { useLicense } from '@/hooks/useLicense';
import { formatFileSize } from '@/lib/utils';
import Image from 'next/image';

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
  const { license, isEnterprise } = useLicense();
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
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const handleModelChange = (e: SelectChangeEvent<string>) => {
    const model = models.find((m) => m.Id === e.target.value) || null;
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
          setError('Failed to load models: ' + err);
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
                  value={selectedModel?.Name || ''}
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
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 1,
                          flexGrow: 1,
                        }}
                      >
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
              {selectedModel.CreationTime && (
                <Typography variant="body2" sx={{ color: 'text.secondary', mb: 1 }}>
                  Started Training: {new Date(selectedModel.CreationTime).toLocaleString()}
                </Typography>
              )}

              {selectedModel.BaseModelId && (
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  Base Model:{' '}
                  {(() => {
                    const baseModel = models.find((m) => m.Id === selectedModel?.BaseModelId);
                    if (baseModel?.Name) {
                      return baseModel.Name.charAt(0).toUpperCase() + baseModel.Name.slice(1);
                    }
                    return 'Unknown';
                  })()}
                </Typography>
              )}
            </Box>
          )}

          {/* Metrics Viewer */}
          <div className="mt-[-20px] p-0">
            <MetricsDataViewer modelId={selectedModel?.Id || undefined} days={days} />
          </div>
        </CardContent>
      </Card>

      {!isEnterprise && (
        <div className=" mt-[-60px]">
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                flex: '0 0 50%',
              }}
            >
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 600,
                  color: 'rgb(102,102,102)',
                  fontSize: '1.5rem',
                }}
              >
                Free Tier Quota
              </Typography>

              <Typography
                variant="body1"
                sx={{
                  color: '#64748b',
                  mb: 2,
                }}
              >
                {`${formatFileSize(license?.LicenseInfo?.Usage.UsedBytes || 0)} / ${formatFileSize(
                  license?.LicenseInfo?.Usage.MaxBytes || 0
                )} monthly quota used`}
              </Typography>

              <Box
                sx={{
                  width: '100%',
                  height: 8,
                  bgcolor: '#e2e8f0',
                  borderRadius: 4,
                  overflow: 'hidden',
                }}
              >
                <Box
                  sx={{
                    width: `${license ? (license?.LicenseInfo?.Usage.UsedBytes / license?.LicenseInfo?.Usage.MaxBytes) * 100 : 0}%`,
                    height: '100%',
                    bgcolor: '#60a5fa',
                    borderRadius: 4,
                    transition: 'width 0.5s ease-in-out',
                  }}
                />
              </Box>
            </Box>

            <Box
              sx={{
                flex: '0 0 50%',
                display: 'flex',
                justifyContent: 'flex-end',
              }}
            >
              <Image
                src="/image.png"
                alt="Background pattern"
                width={400}
                height={256}
                style={{
                  objectFit: 'contain',
                }}
              />
            </Box>
          </Box>
        </div>
      )}
    </>
  );
};

export default Dashboard;
