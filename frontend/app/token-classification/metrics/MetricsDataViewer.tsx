'use client';

import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { nerService, InferenceMetrics, ThroughputMetrics } from '@/lib/backend';
import { formatFileSize, formatNumber } from '@/lib/utils';

interface MetricsDataViewerProps {
  modelId?: string;
  days: number;
}

const MetricsDataViewer: React.FC<MetricsDataViewerProps> = ({ modelId, days }) => {
  const [infMetrics, setInfMetrics] = useState<InferenceMetrics | null>(null);
  const [tpMetrics, setTpMetrics] = useState<ThroughputMetrics | null>(null);
  const [infSeries, setInfSeries] = useState<{ day: number; dataMB: number; tokens: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [throughput, setThroughput] = useState<string | null>('-');

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        // 1) fetch the summary for the full window
        const summary = await nerService.getInferenceMetrics(modelId, days);
        if (!mounted) return;
        setInfMetrics(summary);

        // 2) fetch throughput summary card
        if (modelId) {
          const tp = await nerService.getThroughputMetrics(modelId);
          if (!mounted) return;
          setTpMetrics(tp);
          setThroughput(
            tp.ThroughputMBPerHour !== undefined ? formatFileSize(tp.ThroughputMBPerHour, true) : '-'
          );
        } else {
          setTpMetrics(null);
        }

        // 3) build inference‐series by calling summary for each day = 1..days
        // const infPromises = Array.from({ length: days }, (_, i) =>
        //   nerService.getInferenceMetrics(modelId, i + 1)
        // );
        // const infResults = await Promise.all(infPromises);
        // if (!mounted) return;
        // setInfSeries(
        //   infResults.map((res, i) => ({
        //     day: i + 1,
        //     dataMB: parseFloat(res.DataProcessedMB.toFixed(2)),
        //     tokens: res.TokensProcessed
        //   }))
        // );
      } catch (e: any) {
        if (!mounted) return;
        setError(e.message || 'Failed to load metrics');
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [modelId, days]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography sx={{ color: 'error.main', fontSize: '0.875rem' }}>{error}</Typography>
      </Box>
    );
  }

  if (!infMetrics) {
    return null;
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: 'repeat(5, 1fr)',
          gap: 3,
        }}
      >
        {/* In-Progress Tasks */}
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            },
          }}
        >
          <CardContent
            sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              '&:last-child': { pb: 3 },
            }}
          >
            <Box
              sx={{
                height: '128px',
                width: '128px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <Typography
                sx={{
                  fontSize: '2rem',
                  fontWeight: 600,
                  color: '#1e293b',
                }}
              >
                {infMetrics.InProgress}
              </Typography>
            </Box>
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '0.875rem',
                color: '#64748b',
                fontWeight: 500,
              }}
            >
              In-Progress Reports
            </Typography>
          </CardContent>
        </Card>

        {/* Completed Tasks */}
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            },
          }}
        >
          <CardContent
            sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              '&:last-child': { pb: 3 },
            }}
          >
            <Box
              sx={{
                height: '128px',
                width: '128px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <Typography
                sx={{
                  fontSize: '2rem',
                  fontWeight: 600,
                  color: '#1e293b',
                }}
              >
                {infMetrics.Completed + infMetrics.Failed}
              </Typography>
            </Box>
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '0.875rem',
                color: '#64748b',
                fontWeight: 500,
              }}
            >
              Completed Reports
            </Typography>
          </CardContent>
        </Card>

        {/* Throughput */}
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            },
          }}
        >
          <CardContent
            sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              '&:last-child': { pb: 3 },
            }}
          >
            <Box
              sx={{
                height: '128px',
                width: '128px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <Typography
                sx={{
                  fontSize: (theme) =>
                    throughput && throughput.length > 7
                      ? '1.25rem'
                      : throughput && throughput.length > 5
                        ? '1.5rem'
                        : throughput && throughput.length > 3
                          ? '1.75rem'
                          : '2rem',
                  fontWeight: 600,
                  color: '#1e293b',
                  textAlign: 'center',
                }}
              >
                {throughput === '-' ? '-' : `${throughput}/Hour`}
              </Typography>
            </Box>
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '0.875rem',
                color: '#64748b',
                fontWeight: 500,
              }}
            >
              Throughput
            </Typography>
          </CardContent>
        </Card>

        {/* Data Processed */}
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            },
          }}
        >
          <CardContent
            sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              '&:last-child': { pb: 3 },
            }}
          >
            <Box
              sx={{
                height: '128px',
                width: '128px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <Typography
                sx={{
                  textAlign: 'center',
                  fontSize: '2rem',
                  fontWeight: 600,
                  color: '#1e293b',
                }}
              >
                {formatFileSize(infMetrics.DataProcessedMB * 1024 * 1024)}
              </Typography>
            </Box>
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '0.875rem',
                color: '#64748b',
                fontWeight: 500,
              }}
            >
              Data Processed
            </Typography>
          </CardContent>
        </Card>

        {/* Tokens Processed */}
        <Card
          sx={{
            boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
            bgcolor: 'white',
            borderRadius: '12px',
            transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            },
          }}
        >
          <CardContent
            sx={{
              p: 3,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              '&:last-child': { pb: 3 },
            }}
          >
            <Box
              sx={{
                height: '128px',
                width: '128px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mb: 2,
              }}
            >
              <Typography
                sx={{
                  fontSize: '2rem',
                  fontWeight: 600,
                  color: '#1e293b',
                  textAlign: 'center',
                }}
              >
                {formatNumber(infMetrics.TokensProcessed)}
              </Typography>
            </Box>
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '0.875rem',
                color: '#64748b',
                fontWeight: 500,
              }}
            >
              Tokens Processed
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default MetricsDataViewer;
