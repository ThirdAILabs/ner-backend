'use client';

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  FormControl,
  Select,
  MenuItem,
  Card,
  CardContent,
} from '@mui/material';
import { useSearchParams } from 'next/navigation';
import { nerService } from '@/lib/backend';
import MetricsDataViewer from './metrics/MetricsDataViewer';
import { useHealth } from '@/contexts/HealthProvider';

const Dashboard = () => {
  const { healthStatus } = useHealth();
  const searchParams = useSearchParams();
  const deploymentId = searchParams.get('deploymentId');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Models for the dropdown
  const [days, setDays] = useState<number>(7);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');

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
    <Card sx={{
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      bgcolor: 'white',
      borderRadius: '12px',
      mx: 'auto',
      maxWidth: '1400px'
    }}>
      <CardContent sx={{ p: 4 }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 4,
          }}
        >
          <Typography variant="h5" sx={{
            fontWeight: 600,
            fontSize: '1.5rem',
            color: '#111827'
          }}>
            Metrics Dashboard
          </Typography>
        </Box>

        <Box sx={{
          display: 'flex',
          gap: 4,
          alignItems: 'flex-start',
          mb: 4,
          '& .MuiFormControl-root': {
            bgcolor: 'white',
            borderRadius: '8px',
            '& .MuiSelect-select': {
              py: 1.5,
            }
          }
        }}>
          {/* Days Filter */}
          <Box>
            <Typography
              variant="subtitle2"
              gutterBottom
              sx={{
                fontWeight: 600,
                color: '#475569',
                mb: 1
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
                  }
                }
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
                  }
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
                mb: 1
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
                  }
                }
              }}
            >
              <Select
                value={selectedModel}
                displayEmpty
                onChange={(e) => setSelectedModel(e.target.value)}
                renderValue={(val) =>
                  val === '' ? 'All Models' : models.find((m) => m.Id === val)?.Name || val
                }
                sx={{
                  bgcolor: '#f8fafc',
                  '&:hover': {
                    bgcolor: '#f1f5f9',
                  }
                }}
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

        {/* Metrics Viewer */}
        <Box sx={{
          bgcolor: 'white',
          borderRadius: '12px',
          border: '1px solid',
          borderColor: 'grey.200',
          overflow: 'hidden'
        }}>
          <MetricsDataViewer modelId={selectedModel || undefined} days={days} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default Dashboard;
