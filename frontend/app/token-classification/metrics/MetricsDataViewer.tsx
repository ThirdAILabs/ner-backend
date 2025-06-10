'use client';

import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { nerService } from '@/lib/backend';
import { formatFileSize, formatNumber } from '@/lib/utils';
import { useHealth } from '@/contexts/HealthProvider';
import MetricsDataViewerCard from '@/components/ui/MetricsDataViewerCard';

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
  const { healthStatus } = useHealth();

  function getFontSize(value: string) {
    if (!value) return '2rem';
    if (value.length > 7) return '1.25rem';
    if (value.length > 5) return '1.5rem';
    if (value.length > 3) return '1.75rem';

    return '2rem';
  }

  useEffect(() => {
    let mounted = true;

    // Don't make API calls if health check hasn't passed
    if (!healthStatus) {
      return;
    }

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
            tp.ThroughputMBPerHour !== undefined
              ? formatFileSize(tp.ThroughputMBPerHour, true)
              : '-'
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
  }, [modelId, days, healthStatus]);

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
        <MetricsDataViewerCard value={infMetrics.InProgress} label="In-Progress Reports" />

        {/* Completed Tasks */}
        <MetricsDataViewerCard
          value={infMetrics.Completed + infMetrics.Failed}
          label="Completed Reports"
        />

        {/* Throughput */}
        <MetricsDataViewerCard
          value={throughput === '-' ? '-' : `${throughput}/Hour`}
          label="Throughput"
        />

        {/* Data Processed */}
        <MetricsDataViewerCard
          value={formatFileSize(infMetrics.DataProcessedMB * 1024 * 1024)}
          label="Data Processed"
        />

        {/* Tokens Processed */}
        <MetricsDataViewerCard
          value={formatNumber(infMetrics.TokensProcessed)}
          label="Tokens Processed"
        />
      </Box>
    </Box>
  );
};

export default MetricsDataViewer;
